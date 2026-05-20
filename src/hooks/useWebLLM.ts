import { useState, useEffect, useRef, useCallback } from "react";
import { CreateWebWorkerMLCEngine } from "@mlc-ai/web-llm";
import type { MLCEngineInterface, InitProgressReport } from "@mlc-ai/web-llm";
import { evaluate } from "mathjs";

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

// ─── Module-Level Local Tool Helpers ─────────────────────────────────────────
// These are defined at module level (outside the hook) for stability.

function getCurrentTime(): string {
  return new Date().toLocaleString("zh-TW", { timeZoneName: "short" });
}

function runJsCalculation(expression: string): string {
  try {
    // Use mathjs.evaluate() — a proper sandboxed math expression parser
    // It supports: arithmetic, trigonometry, exponentiation, matrices, units, etc.
    // It does NOT have access to JS globals, window, fetch, etc.
    const result = evaluate(expression);
    if (result === undefined || result === null) {
      return "計算無法得出有效數值。";
    }
    return String(result);
  } catch (e) {
    return `計算錯誤: ${e instanceof Error ? e.message : String(e)}`;
  }
}

async function searchWikipedia(query: string): Promise<string> {
  try {
    const url = `https://zh.wikipedia.org/w/api.php?action=query&list=search&srsearch=${encodeURIComponent(query)}&format=json&origin=*`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    const data = await res.json();
    const searchResults = data.query?.search || [];
    if (searchResults.length === 0) {
      return `找不到與「${query}」相關的維基百科條目。`;
    }
    return searchResults
      .slice(0, 3)
      .map((item: { title: string; snippet: string }, i: number) => {
        const cleanSnippet = item.snippet.replace(/<\/?[^>]+(>|$)/g, "");
        return `[搜尋結果 ${i + 1}] 標題: ${item.title}\n內容摘要: ${cleanSnippet}\n`;
      })
      .join("\n");
  } catch (e) {
    return `搜尋發生錯誤: ${e instanceof Error ? e.message : String(e)}`;
  }
}
// ─────────────────────────────────────────────────────────────────────────────

export const useWebLLM = () => {
  const [webGpuSupported, setWebGpuSupported] = useState<boolean>(true);
  const [gpuName, setGpuName] = useState<string>("Detecting hardware...");
  const [engine, setEngine] = useState<MLCEngineInterface | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [progress, setProgress] = useState<InitProgressReport | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>("gemma-2-2b-it-q4f16_1-MLC");
  const [loadedModel, setLoadedModel] = useState<string | null>(null);

  const workerRef = useRef<Worker | null>(null);

  // 1. Detect WebGPU Support and get GPU hardware info
  useEffect(() => {
    const checkWebGpu = async () => {
      if (!navigator.gpu) {
        setWebGpuSupported(false);
        setGpuName("WebGPU is NOT supported in this browser.");
        return;
      }

      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
          setWebGpuSupported(false);
          setGpuName("WebGPU is supported, but no compatible GPU adapter was found.");
          return;
        }

        setWebGpuSupported(true);
        // Get GPU Adapter Info (standard in modern browsers)
        if ("requestAdapterInfo" in adapter) {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const info = await (adapter as any).requestAdapterInfo();
          const name = info.description || info.device || info.vendor || "WebGPU Graphics Device";
          setGpuName(name);
        } else {
          setGpuName("WebGPU Compatible Graphics Device");
        }
      } catch (err) {
        console.error("Error checking WebGPU:", err);
        setWebGpuSupported(false);
        setGpuName("Failed to initialize WebGPU adapter.");
      }
    };

    checkWebGpu();
  }, []);

  // 2. Terminate worker when hook unmounts
  useEffect(() => {
    return () => {
      if (workerRef.current) {
        workerRef.current.terminate();
      }
    };
  }, []);

  // 3. Load or reload the model
  const loadModel = useCallback(async (modelId: string) => {
    if (!webGpuSupported) return;

    setIsLoading(true);
    setProgress({ progress: 0, text: "Initializing Web Worker...", timeElapsed: 0 });

    try {
      // If a worker already exists, terminate it to start fresh
      if (workerRef.current) {
        workerRef.current.terminate();
        workerRef.current = null;
      }

      // Create a new Web Worker instance
      const worker = new Worker(
        new URL("../llm.worker.ts", import.meta.url),
        { type: "module" }
      );
      workerRef.current = worker;

      // Create the Web Worker Engine
      const mlcEngine = await CreateWebWorkerMLCEngine(
        worker,
        modelId,
        {
          initProgressCallback: (report) => {
            setProgress(report);
          },
        }
      );

      setEngine(mlcEngine);
      setLoadedModel(modelId);
      setIsLoading(false);
    } catch (err) {
      console.error("Failed to load model:", err);
      setProgress((prev) => ({
        progress: prev?.progress || 0,
        text: `Error: ${err instanceof Error ? err.message : String(err)}`,
        timeElapsed: prev?.timeElapsed || 0,
      }));
      setIsLoading(false);
    }
  }, [webGpuSupported]);

  // 4. Send a message list and receive streaming tokens with Agentic Tool Calling
  const sendMessage = useCallback(async (
    messages: ChatMessage[],
    onChunk: (text: string) => void
  ): Promise<string> => {
    if (!engine) {
      throw new Error("Engine is not initialized. Please load a model first.");
    }

    setIsGenerating(true);
    const currentMessages = [...messages];
    let fullResponse = "";
    let loopCount = 0;
    const maxLoops = 2; // Safeguard to prevent infinite tool call loops

    try {
      while (loopCount < maxLoops) {
        let responseText = "";
        const completion = await engine.chat.completions.create({
          messages: currentMessages,
          stream: true,
          max_tokens: 2048, // P10: prevent KV cache overflow
        });

        for await (const chunk of completion) {
          const text = chunk.choices[0]?.delta?.content || "";
          if (text) {
            responseText += text;
            onChunk(fullResponse + responseText);
          }
        }

        // Parse responseText for tool calls with high tolerance for whitespaces
        const timeMatch = responseText.match(/\[CALL:\s*get_current_time\s*\(\s*\)\s*\]/);
        const mathMatch = responseText.match(/\[CALL:\s*run_js_calculation\s*\(\s*expression\s*=\s*"([^"]+)"\s*\)\s*\]/);
        const searchMatch = responseText.match(/\[CALL:\s*search_wikipedia\s*\(\s*query\s*=\s*"([^"]+)"\s*\)\s*\]/);

        if (timeMatch || mathMatch || searchMatch) {
          let toolOutput = "";
          let matchedCall = "";

          if (timeMatch) {
            matchedCall = timeMatch[0];
            toolOutput = getCurrentTime();
          } else if (mathMatch) {
            matchedCall = mathMatch[0];
            const expression = mathMatch[1];
            toolOutput = runJsCalculation(expression);
          } else if (searchMatch) {
            matchedCall = searchMatch[0];
            const query = searchMatch[1];
            // Render fetch indicator
            onChunk(fullResponse + responseText + `\n\n*(🔍 正在從維基百科抓取「${query}」的最新新聞與資訊...)*`);
            toolOutput = await searchWikipedia(query);
          }

          // Accumulate generated text and append the clean visual divider
          fullResponse += responseText + `\n\n*(🛠️ 本機執行工具 ${matchedCall}，獲得輸出結果如下：)*\n\`\`\`text\n${toolOutput}\n\`\`\`\n\n<!-- ANSWER -->`;
          onChunk(fullResponse);

          // Push the tool run results into conversation context to guide the model's next response
          currentMessages.push({ role: "assistant", content: responseText });
          currentMessages.push({
            role: "user",
            content: `[SYSTEM TOOL CALL RESULT] The tool ${matchedCall} returned:\n${toolOutput}\n\nPlease analyze this returned text and write your final Traditional Chinese (繁體中文) response to the user.`,
          });

          loopCount++;
        } else {
          // No more tool calls found, append final response and break
          fullResponse += responseText;
          break;
        }
      }
    } catch (err) {
      console.error("Generation error in agent loop:", err);
      throw err;
    } finally {
      setIsGenerating(false);
    }

    return fullResponse;
  }, [engine]);

  // 5. Interrupt/stop active generation
  const interruptGeneration = useCallback(async () => {
    if (engine) {
      await engine.interruptGenerate();
      setIsGenerating(false);
    }
  }, [engine]);

  return {
    webGpuSupported,
    gpuName,
    engine,
    isLoading,
    isGenerating,
    progress,
    selectedModel,
    setSelectedModel,
    loadedModel,
    loadModel,
    sendMessage,
    interruptGeneration,
  };
};
