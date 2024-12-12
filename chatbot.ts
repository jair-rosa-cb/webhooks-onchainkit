import { CdpAgentkit } from "@coinbase/cdp-agentkit-core";
import { CdpTool, CdpToolkit } from "@coinbase/cdp-langchain";
import { HumanMessage } from "@langchain/core/messages";
import { MemorySaver } from "@langchain/langgraph";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { ChatOpenAI } from "@langchain/openai";
import * as dotenv from "dotenv";
import * as fs from "fs";
import * as readline from "readline";
import { Wallet, Webhook } from "@coinbase/coinbase-sdk";
import { z } from "zod";

dotenv.config();

/**
 * Validates that required environment variables are set
 *
 * @throws {Error} - If required environment variables are missing
 * @returns {void}
 */
function validateEnvironment(): void {
  const missingVars: string[] = [];

  // Check required variables
  const requiredVars = ["CDP_API_KEY_NAME", "CDP_API_KEY_PRIVATE_KEY"];
  requiredVars.forEach((varName) => {
    if (!process.env[varName]) {
      missingVars.push(varName);
    }
  });

  // Exit if any required variables are missing
  if (missingVars.length > 0) {
    console.error("Error: Required environment variables are not set");
    missingVars.forEach((varName) => {
      console.error(`${varName}=your_${varName.toLowerCase()}_here`);
    });
    process.exit(1);
  }

  // Warn about optional NETWORK_ID
  if (!process.env.NETWORK_ID) {
    console.warn(
      "Warning: NETWORK_ID not set, defaulting to base-sepolia testnet",
    );
  }
}

// Add this right after imports and before any other code
validateEnvironment();

// Configure a file to persist the agent's CDP MPC Wallet Data
const WALLET_DATA_FILE = "wallet_data.txt";

// Define the webhook event types
const WebhookEventType = z.enum([
  "wallet_activity",
  "smart_contract_event_activity",
  "erc20_transfer",
  "erc721_transfer",
]);

// Create a flexible event filters schema
const EventFilters = z
  .object({
    addresses: z
      .array(z.string())
      .optional()
      .describe("List of wallet or contract addresses to monitor"),
    from_address: z
      .string()
      .optional()
      .describe("Sender address for token transfers"),
    to_address: z
      .string()
      .optional()
      .describe("Recipient address for token transfers"),
    contract_address: z
      .string()
      .optional()
      .describe("Contract address for token transfers"),
  })
  .refine(
    (data) => {
      // Ensure at least one filter is provided
      return Object.keys(data).length > 0;
    },
    { message: "At least one filter must be provided" },
  );

const CreateWebhookInput = z.object({
  notificationUri: z
    .string()
    .url()
    .describe("The callback URL where webhook events will be sent"),
  eventType: WebhookEventType,
  eventTypeFilter: z.object({
    addresses: z
      .array(z.string())
      .describe("List of wallet or contract addresses to monitor"),
  }),
  eventFilters: EventFilters.optional(),
});

/**
 * Creates a new webhook for monitoring on-chain events
 *
 * @param wallet - The wallet used to create the webhook
 * @param args - Object with arguments needed
 * @returns Details of the created webhook
 */
async function createWebhook(
  wallet: Wallet,
  args: {
    notificationUri: string;
    eventType: z.infer<typeof WebhookEventType>;
    eventTypeFilter: { addresses: string[] };
    eventFilters?: z.infer<typeof EventFilters>;
  },
): Promise<string> {
  try {
    const { notificationUri, eventType, eventTypeFilter, eventFilters } = args;
    // Use the environment variable for networkId
    const networkId = process.env.NETWORK_ID;

    if (!networkId) {
      throw new Error("Network ID is not configured in environment variables");
    }

    const webhookOptions: any = {
      networkId,
      notificationUri,
      eventType,
    };

    // Handle different event types with appropriate filtering
    switch (eventType) {
      case "wallet_activity":
        webhookOptions.eventTypeFilter = {
          addresses: eventTypeFilter.addresses || [],
          wallet_id: "", // this is required by SDK, but can be an empty value
        };
        break;
      case "smart_contract_event_activity":
        webhookOptions.eventTypeFilter = {
          contract_addresses: eventTypeFilter.addresses || [],
        };
        break;
      case "erc20_transfer":
      case "erc721_transfer":
        webhookOptions.eventFilters = [
          {
            ...(eventFilters?.contract_address
              ? { contract_address: eventFilters.contract_address }
              : {}),
            ...(eventFilters?.from_address
              ? { from_address: eventFilters.from_address }
              : {}),
            ...(eventFilters?.to_address
              ? { to_address: eventFilters.to_address }
              : {}),
          },
        ];
        break;
      default:
        throw new Error(`Unsupported event type: ${eventType}`);
    }

    // Create webhook
    const webhook = await Webhook.create(webhookOptions);

    return `The webhook was successfully created: ${webhook?.toString()}\n\n`;
    // return webhook
  } catch (error) {
    console.error("Failed to create webhook:", error);
    return `Error: ${error}`;
  }
}

/**
 * Initialize the agent with CDP Agentkit
 *
 * @returns Agent executor and config
 */
async function initializeAgent() {
  try {
    // Initialize LLM with xAI configuration
    const llm = new ChatOpenAI({
      model: "gpt-4o-mini",
      apiKey: process.env.OPENAI_API_KEY,
    });

    let walletDataStr: string | null = null;

    // Read existing wallet data if available
    if (fs.existsSync(WALLET_DATA_FILE)) {
      try {
        walletDataStr = fs.readFileSync(WALLET_DATA_FILE, "utf8");
      } catch (error) {
        console.error("Error reading wallet data:", error);
        // Continue without wallet data
      }
    }

    // Configure CDP Agentkit
    const config = {
      cdpWalletData: walletDataStr || undefined,
      networkId: process.env.NETWORK_ID || "base-sepolia",
    };

    // Initialize CDP agentkit
    const agentkit = await CdpAgentkit.configureWithWallet(config);

    // Initialize CDP Agentkit Toolkit and get tools
    const cdpToolkit = new CdpToolkit(agentkit);

    const tools = cdpToolkit.getTools();

    // Define the descriptive prompt for the action
    const CREATE_WEBHOOK_PROMPT = `
    Create a new webhook to receive real-time updates for on-chain events. 
    Supports monitoring wallet activity or smart contract events by specifying:
    - Callback URL for receiving events
    - Event type (wallet_activity, smart_contract_event_activity, erc20_transfer or erc721_transfer)
    - Addresses to monitor
    When event type is erc20_transfer or erc721_transfer at least one of these filters needs to be provided (only one of them is required):
    - Contract address to listen for token transfers
    - Sender address for erc20_transfer and erc721_transfer events (listen on transfers originating from this address))
    - Recipient address for erc20_transfer and erc721_transfer events (listen on transfers being made to this address))
    `;

    // Add the webhook tool
    const createWebhookTool = new CdpTool(
      {
        name: "create_webhook",
        description: CREATE_WEBHOOK_PROMPT,
        argsSchema: CreateWebhookInput,
        func: createWebhook,
      },
      agentkit, // Assumes this is the CdpWrapper instance
    );
    tools.push(createWebhookTool);

    // Store buffered conversation history in memory
    const memory = new MemorySaver();
    const agentConfig = {
      configurable: { thread_id: "CDP Agentkit Chatbot Example!" },
    };

    // Create React Agent using the LLM and CDP Agentkit tools
    const agent = createReactAgent({
      llm,
      tools,
      checkpointSaver: memory,
      messageModifier:
        "You are a helpful agent that can interact onchain using the Coinbase Developer Platform Agentkit. You are empowered to interact onchain using your tools. If you ever need funds, you can request them from the faucet if you are on network ID `base-sepolia`. If not, you can provide your wallet details and request funds from the user. If someone asks you to do something you can't do with your currently available tools, you must say so, and encourage them to implement it themselves using the CDP SDK + Agentkit, recommend they go to docs.cdp.coinbase.com for more informaton. Be concise and helpful with your responses. Refrain from restating your tools' descriptions unless it is explicitly requested.",
    });

    // Save wallet data
    const exportedWallet = await agentkit.exportWallet();
    fs.writeFileSync(WALLET_DATA_FILE, exportedWallet);

    return { agent, config: agentConfig };
  } catch (error) {
    console.error("Failed to initialize agent:", error);
    throw error; // Re-throw to be handled by caller
  }
}

/**
 * Run the agent autonomously with specified intervals
 *
 * @param agent - The agent executor
 * @param config - Agent configuration
 * @param interval - Time interval between actions in seconds
 */
async function runAutonomousMode(agent: any, config: any, interval = 10) {
  console.log("Starting autonomous mode...");

  while (true) {
    try {
      const thought =
        "Be creative and do something interesting on the blockchain. " +
        "Choose an action or set of actions and execute it that highlights your abilities.";

      const stream = await agent.stream(
        { messages: [new HumanMessage(thought)] },
        config,
      );

      for await (const chunk of stream) {
        if ("agent" in chunk) {
          console.log(chunk.agent.messages[0].content);
        } else if ("tools" in chunk) {
          console.log(chunk.tools.messages[0].content);
        }
        console.log("-------------------");
      }

      await new Promise((resolve) => setTimeout(resolve, interval * 1000));
    } catch (error) {
      if (error instanceof Error) {
        console.error("Error:", error.message);
      }
      process.exit(1);
    }
  }
}

/**
 * Run the agent interactively based on user input
 *
 * @param agent - The agent executor
 * @param config - Agent configuration
 */
async function runChatMode(agent: any, config: any) {
  console.log("Starting chat mode... Type 'exit' to end.");

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const question = (prompt: string): Promise<string> =>
    new Promise((resolve) => rl.question(prompt, resolve));

  try {
    while (true) {
      const userInput = await question("\nPrompt: ");

      if (userInput.toLowerCase() === "exit") {
        break;
      }

      const stream = await agent.stream(
        { messages: [new HumanMessage(userInput)] },
        config,
      );

      for await (const chunk of stream) {
        if ("agent" in chunk) {
          console.log(chunk.agent.messages[0].content);
        } else if ("tools" in chunk) {
          console.log(chunk.tools.messages[0].content);
        }
        console.log("-------------------");
      }
    }
  } catch (error) {
    if (error instanceof Error) {
      console.error("Error:", error.message);
    }
    process.exit(1);
  } finally {
    rl.close();
  }
}

/**
 * Choose whether to run in autonomous or chat mode based on user input
 *
 * @returns Selected mode
 */
async function chooseMode(): Promise<"chat" | "auto"> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const question = (prompt: string): Promise<string> =>
    new Promise((resolve) => rl.question(prompt, resolve));

  while (true) {
    console.log("\nAvailable modes:");
    console.log("1. chat    - Interactive chat mode");
    console.log("2. auto    - Autonomous action mode");

    const choice = (await question("\nChoose a mode (enter number or name): "))
      .toLowerCase()
      .trim();

    if (choice === "1" || choice === "chat") {
      rl.close();
      return "chat";
    } else if (choice === "2" || choice === "auto") {
      rl.close();
      return "auto";
    }
    console.log("Invalid choice. Please try again.");
  }
}

/**
 * Start the chatbot agent
 */
async function main() {
  try {
    const { agent, config } = await initializeAgent();
    const mode = await chooseMode();

    if (mode === "chat") {
      await runChatMode(agent, config);
    } else {
      await runAutonomousMode(agent, config);
    }
  } catch (error) {
    if (error instanceof Error) {
      console.error("Error:", error.message);
    }
    process.exit(1);
  }
}

if (require.main === module) {
  console.log("Starting Agent...");
  main().catch((error) => {
    console.error("Fatal error:", error);
    process.exit(1);
  });
}
