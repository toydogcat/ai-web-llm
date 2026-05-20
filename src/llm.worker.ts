import { WebWorkerMLCEngineHandler } from "@mlc-ai/web-llm";

// Instantiate the handler to manage MLCEngine inside the Web Worker
const handler = new WebWorkerMLCEngineHandler();

// Bind the onmessage handler to receive commands from the main thread
self.onmessage = (msg: MessageEvent) => {
  handler.onmessage(msg);
};
