import { useState, useEffect, useRef, useCallback } from "react";
import { CreateWebWorkerMLCEngine } from "@mlc-ai/web-llm";
import type { MLCEngineInterface, InitProgressReport } from "@mlc-ai/web-llm";

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

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

  // 4. Send a message list and receive streaming tokens
  const sendMessage = useCallback(async (
    messages: ChatMessage[],
    onChunk: (text: string) => void
  ): Promise<string> => {
    if (!engine) {
      throw new Error("Engine is not initialized. Please load a model first.");
    }

    setIsGenerating(true);
    let fullResponse = "";

    try {
      const completion = await engine.chat.completions.create({
        messages: messages,
        stream: true,
      });

      for await (const chunk of completion) {
        const text = chunk.choices[0]?.delta?.content || "";
        if (text) {
          fullResponse += text;
          onChunk(fullResponse);
        }
      }
    } catch (err) {
      console.error("Generation error:", err);
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
