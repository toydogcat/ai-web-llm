import React, { useState, useEffect, useRef } from "react";
import { useWebLLM } from "./hooks/useWebLLM";
import type { ChatMessage } from "./hooks/useWebLLM";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Cpu,
  Send,
  Trash2,
  Plus,
  Sparkles,
  RefreshCw,
  AlertTriangle,
  CheckCircle,
  Terminal,
  HelpCircle,
  X,
  ShieldAlert,
  Check,
  Copy,
} from "lucide-react";

interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  systemPrompt: string;
  createdAt: number;
}

export default function App() {
  const {
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
  } = useWebLLM();

  // Chat sessions state
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string>("");
  const [inputMessage, setInputMessage] = useState<string>("");
  const [systemPrompt, setSystemPrompt] = useState<string>(
    "You are Gemma, a highly helpful, intelligent, and creative AI assistant running entirely locally in the browser via WebGPU."
  );
  
  // Visual states
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState<boolean>(true);

  // References
  const chatEndRef = useRef<HTMLDivElement | null>(null);

  // Load sessions from LocalStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem("gemma_chat_sessions");
    if (saved) {
      try {
        const parsed = JSON.parse(saved) as ChatSession[];
        if (parsed.length > 0) {
          setSessions(parsed);
          // Set active session to the most recently created or modified one
          setActiveSessionId(parsed[0].id);
          // Load system prompt for that session
          setSystemPrompt(parsed[0].systemPrompt || systemPrompt);
          return;
        }
      } catch (e) {
        console.error("Failed to parse saved sessions", e);
      }
    }

    // Create an initial default session if none exist
    const defaultSession: ChatSession = {
      id: "session_" + Date.now(),
      title: "Gemma Chat Session",
      messages: [],
      systemPrompt: systemPrompt,
      createdAt: Date.now(),
    };
    setSessions([defaultSession]);
    setActiveSessionId(defaultSession.id);
  }, []);

  // Save sessions to LocalStorage on changes
  useEffect(() => {
    if (sessions.length > 0) {
      localStorage.setItem("gemma_chat_sessions", JSON.stringify(sessions));
    }
  }, [sessions]);

  // Scroll to bottom when messages stream
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [sessions, activeSessionId, isGenerating]);

  // Get current active session
  const activeSession = sessions.find((s) => s.id === activeSessionId);

  // Handlers
  const handleCreateSession = () => {
    const newSession: ChatSession = {
      id: "session_" + Date.now(),
      title: `Session ${sessions.length + 1}`,
      messages: [],
      systemPrompt: systemPrompt,
      createdAt: Date.now(),
    };
    setSessions([newSession, ...sessions]);
    setActiveSessionId(newSession.id);
  };

  const handleDeleteSession = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const updated = sessions.filter((s) => s.id !== id);
    if (updated.length === 0) {
      const defaultSession: ChatSession = {
        id: "session_" + Date.now(),
        title: "Gemma Chat Session",
        messages: [],
        systemPrompt: systemPrompt,
        createdAt: Date.now(),
      };
      setSessions([defaultSession]);
      setActiveSessionId(defaultSession.id);
    } else {
      setSessions(updated);
      if (activeSessionId === id) {
        setActiveSessionId(updated[0].id);
      }
    }
  };

  const handleClearHistory = () => {
    if (!activeSessionId) return;
    setSessions(
      sessions.map((s) => {
        if (s.id === activeSessionId) {
          return { ...s, messages: [] };
        }
        return s;
      })
    );
  };

  const handleUpdateSystemPrompt = (newPrompt: string) => {
    setSystemPrompt(newPrompt);
    setSessions(
      sessions.map((s) => {
        if (s.id === activeSessionId) {
          return { ...s, systemPrompt: newPrompt };
        }
        return s;
      })
    );
  };

  const handleLoadModel = async () => {
    await loadModel(selectedModel);
  };

  const handleSendMessage = async (textToSend?: string) => {
    const text = (textToSend || inputMessage).trim();
    if (!text || !activeSessionId || !engine || isLoading || isGenerating) return;

    if (!textToSend) {
      setInputMessage("");
    }

    const currentSession = sessions.find((s) => s.id === activeSessionId);
    if (!currentSession) return;

    // Create user message object
    const userMsg: ChatMessage = { role: "user", content: text };

    // Format chat history to send to Gemma (inject system prompt at the beginning)
    const chatHistoryToSend: ChatMessage[] = [];
    if (currentSession.systemPrompt) {
      chatHistoryToSend.push({ role: "system", content: currentSession.systemPrompt });
    }
    // Append previous messages
    chatHistoryToSend.push(...currentSession.messages);
    // Append the new user message
    chatHistoryToSend.push(userMsg);

    // Update session locally with user message first
    const updatedSessionWithUser = {
      ...currentSession,
      title: currentSession.messages.length === 0 ? (text.slice(0, 20) + (text.length > 20 ? "..." : "")) : currentSession.title,
      messages: [...currentSession.messages, userMsg],
    };

    setSessions(
      sessions.map((s) => (s.id === activeSessionId ? updatedSessionWithUser : s))
    );

    // Setup an empty assistant bubble to receive stream
    const assistantMsgPlaceholder: ChatMessage = { role: "assistant", content: "" };
    const sessionWithPlaceholder = {
      ...updatedSessionWithUser,
      messages: [...updatedSessionWithUser.messages, assistantMsgPlaceholder],
    };

    setSessions(
      sessions.map((s) => (s.id === activeSessionId ? sessionWithPlaceholder : s))
    );

    try {
      // Call standard completions API
      await sendMessage(chatHistoryToSend, (streamedResponseText) => {
        // Stream chunk callback: update assistants message content in real time
        setSessions((prevSessions) =>
          prevSessions.map((s) => {
            if (s.id === activeSessionId) {
              const msgsCopy = [...s.messages];
              const lastMsg = msgsCopy[msgsCopy.length - 1];
              if (lastMsg && lastMsg.role === "assistant") {
                lastMsg.content = streamedResponseText;
              }
              return { ...s, messages: msgsCopy };
            }
            return s;
          })
        );
      });
    } catch (err) {
      console.error(err);
      // In case of failure, display error message inside the chat bubble
      setSessions((prevSessions) =>
        prevSessions.map((s) => {
          if (s.id === activeSessionId) {
            const msgsCopy = [...s.messages];
            const lastMsg = msgsCopy[msgsCopy.length - 1];
            if (lastMsg && lastMsg.role === "assistant") {
              lastMsg.content = `🛑 Generation interrupted or failed: ${err instanceof Error ? err.message : String(err)}`;
            }
            return { ...s, messages: msgsCopy };
          }
          return s;
        })
      );
    }
  };

  const handleCopyCode = (code: string, id: string) => {
    navigator.clipboard.writeText(code);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  // Quick Prompt Chips
  const promptSuggestions = [
    { label: "⚛️ Quantum computing", text: "Explain Quantum Computing in very simple terms for a high school student." },
    { label: "💻 React Custom Hook", text: "Write a React custom hook named useLocalStorage that handles local storage caching with proper TypeScript types." },
    { label: "✉️ Polite Email", text: "Draft a polite email declining an invitation to speak at a conference due to scheduling conflicts." },
    { label: "🥗 Healthy Plan", text: "Design a simple and healthy 3-day meal plan focusing on a Mediterranean diet with a shopping list." },
  ];

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-deep-space select-none text-slate-100">
      
      {/* 1. LEFT SIDEBAR PANEL */}
      <aside className="w-80 flex flex-col border-r border-purple-900/30 bg-black/40 backdrop-blur-xl shrink-0">
        
        {/* Sidebar Header */}
        <div className="p-4 border-b border-purple-900/30 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-cyber-purple to-cyber-cyan flex items-center justify-center shadow-lg shadow-cyber-purple/20">
              <Sparkles className="w-5 h-5 text-white animate-pulse" />
            </div>
            <div>
              <h1 className="text-md font-bold tracking-wider bg-gradient-to-r from-white via-slate-200 to-cyber-cyan bg-clip-text text-transparent">
                GEMMA GLASS
              </h1>
              <span className="text-[10px] text-purple-400 font-mono tracking-widest uppercase">
                WebGPU Engine
              </span>
            </div>
          </div>
          
          <button
            onClick={() => setShowSettings(!showSettings)}
            className={`p-1.5 rounded-lg border transition-all ${
              showSettings
                ? "bg-cyber-purple/20 border-cyber-purple/40 text-cyber-purple"
                : "border-purple-900/20 text-slate-400 hover:text-slate-200"
            }`}
            title="Toggle Settings & Diagnostics"
          >
            <Cpu className="w-4 h-4" />
          </button>
        </div>

        {/* Sidebar Body (Custom Scrollable Areas) */}
        <div className="flex-1 overflow-y-auto p-3 space-y-4">
          
          {/* Active Settings Panel */}
          {showSettings && (
            <div className="p-3 rounded-xl border border-purple-950/40 bg-purple-950/10 backdrop-blur-md space-y-3">
              <h3 className="text-xs font-semibold tracking-wider text-purple-300 uppercase flex items-center gap-1.5">
                <Terminal className="w-3.5 h-3.5 text-cyber-purple" />
                Settings & Core
              </h3>
              
              {/* Model selection */}
              <div className="space-y-1.5">
                <label className="text-[11px] text-slate-400 font-medium">Model ID</label>
                <select
                  disabled={isLoading}
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="w-full text-xs rounded-lg border border-purple-900/40 bg-slate-950 p-2 text-slate-200 focus:outline-none focus:border-cyber-cyan transition-colors"
                >
                  <option value="gemma-2-2b-it-q4f16_1-MLC">Gemma-2 2B (Recommended)</option>
                  <option value="gemma-2b-it-q4f16_1-MLC">Gemma-1 2B (Original)</option>
                </select>
              </div>

              {/* System Instruction */}
              <div className="space-y-1.5">
                <label className="text-[11px] text-slate-400 font-medium">System Instructions</label>
                <textarea
                  value={systemPrompt}
                  onChange={(e) => handleUpdateSystemPrompt(e.target.value)}
                  className="w-full text-xs rounded-lg border border-purple-900/40 bg-slate-950 p-2 h-20 text-slate-200 resize-none focus:outline-none focus:border-cyber-purple transition-colors"
                  placeholder="E.g. You are a code expert..."
                />
              </div>

              {/* Action Loader Button */}
              {(!loadedModel || loadedModel !== selectedModel) && (
                <button
                  onClick={handleLoadModel}
                  disabled={isLoading}
                  className="w-full py-2 px-3 text-xs font-semibold rounded-lg bg-gradient-to-r from-cyber-purple to-cyber-blue text-white hover:shadow-lg hover:shadow-cyber-purple/20 active:scale-95 disabled:scale-100 disabled:opacity-40 transition-all flex items-center justify-center gap-2"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? "animate-spin" : ""}`} />
                  {isLoading ? "Waking Model..." : "Initialize Gemma Model"}
                </button>
              )}
            </div>
          )}

          {/* WebGPU Status Check Panel */}
          <div className="p-3 rounded-xl border border-slate-900 bg-slate-950/20 backdrop-blur-md space-y-2.5">
            <h3 className="text-xs font-semibold tracking-wider text-slate-400 uppercase flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-cyber-cyan" />
              Hardware Diagnostic
            </h3>
            
            <div className="space-y-1.5 text-xs">
              <div className="flex items-center justify-between">
                <span className="text-slate-400">WebGPU Access:</span>
                {webGpuSupported ? (
                  <span className="flex items-center gap-1 text-[11px] font-semibold text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded">
                    <CheckCircle className="w-3 h-3" /> ACTIVE
                  </span>
                ) : (
                  <span className="flex items-center gap-1 text-[11px] font-semibold text-rose-400 bg-rose-500/10 px-1.5 py-0.5 rounded">
                    <ShieldAlert className="w-3 h-3" /> DISABLED
                  </span>
                )}
              </div>
              
              <div className="flex flex-col gap-0.5 border-t border-slate-900/60 pt-1.5">
                <span className="text-slate-500 text-[10px] uppercase font-mono">Adapter Info</span>
                <span className="text-slate-300 font-mono text-[10px] break-words line-clamp-2 leading-relaxed">
                  {gpuName}
                </span>
              </div>
              
              {loadedModel && (
                <div className="flex items-center justify-between border-t border-slate-900/60 pt-1.5">
                  <span className="text-slate-400">Loaded:</span>
                  <span className="text-cyber-cyan font-mono text-[11px]">
                    {loadedModel.split("-")[0].toUpperCase()} {loadedModel.includes("2b") ? "2B" : "1B"}
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Chat Sessions list */}
          <div className="space-y-2">
            <div className="flex items-center justify-between px-1">
              <h3 className="text-xs font-semibold tracking-wider text-slate-400 uppercase">
                Active Chats
              </h3>
              <button
                onClick={handleCreateSession}
                className="p-1 rounded-md text-slate-400 hover:text-white hover:bg-slate-900 transition-colors"
                title="New Chat Session"
              >
                <Plus className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-1 max-h-56 overflow-y-auto">
              {sessions.map((s) => (
                <button
                  key={s.id}
                  onClick={() => {
                    setActiveSessionId(s.id);
                    setSystemPrompt(s.systemPrompt || "");
                  }}
                  className={`w-full text-left p-2.5 rounded-lg flex items-center justify-between text-xs transition-all ${
                    activeSessionId === s.id
                      ? "bg-gradient-to-r from-purple-950/40 to-slate-900/40 border border-purple-900/35 text-slate-100 font-medium"
                      : "text-slate-400 hover:text-slate-200 hover:bg-slate-950/40 border border-transparent"
                  }`}
                >
                  <span className="truncate pr-2">{s.title}</span>
                  <span
                    onClick={(e) => handleDeleteSession(s.id, e)}
                    className="p-1 rounded text-slate-500 hover:text-rose-400 transition-colors shrink-0"
                    title="Delete Chat"
                  >
                    <Trash2 className="w-3 h-3" />
                  </span>
                </button>
              ))}
            </div>
          </div>

        </div>

        {/* Sidebar Footer */}
        <div className="p-3 border-t border-purple-900/20 text-center text-[10px] text-slate-500 font-mono">
          <span>Gemma Local Chat v1.0.0</span>
        </div>
      </aside>

      {/* 2. MAIN CHAT AREA */}
      <main className="flex-1 flex flex-col bg-slate-950/20 relative">
        
        {/* Core Layout Layer - Floating Background Starfield effect */}
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-purple-950/20 via-black/10 to-transparent pointer-events-none" />

        {/* Top Header info */}
        <header className="h-14 border-b border-purple-900/20 flex items-center justify-between px-6 bg-black/20 backdrop-blur-md z-10 shrink-0">
          <div className="flex items-center gap-3">
            <div>
              <h2 className="text-sm font-semibold text-slate-100 flex items-center gap-1.5">
                {activeSession ? activeSession.title : "Active Conversation"}
                {isGenerating && (
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping" />
                )}
              </h2>
              <span className="text-[10px] text-slate-400 font-mono">
                {loadedModel ? `Running Local ${loadedModel}` : "Not Initialized"}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleClearHistory}
              disabled={!activeSession || activeSession.messages.length === 0}
              className="px-2.5 py-1 text-xs text-slate-400 hover:text-white hover:bg-slate-900 disabled:opacity-40 disabled:hover:text-slate-400 disabled:hover:bg-transparent rounded-lg border border-slate-900 transition-colors flex items-center gap-1.5"
            >
              <Trash2 className="w-3.5 h-3.5" />
              Clear Current Chat
            </button>
          </div>
        </header>

        {/* Main Panel Content Routing based on state */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6 relative z-10">
          
          {/* CASE A: WEBGPU IS NOT SUPPORTED */}
          {!webGpuSupported && (
            <div className="max-w-xl mx-auto my-12 p-6 rounded-2xl border border-rose-500/20 bg-rose-950/10 backdrop-blur-xl space-y-4">
              <div className="w-12 h-12 rounded-full bg-rose-500/10 flex items-center justify-center text-rose-400 mx-auto">
                <AlertTriangle className="w-6 h-6 animate-pulse" />
              </div>
              <div className="text-center space-y-2">
                <h2 className="text-lg font-bold text-rose-300">WebGPU Access Denied</h2>
                <p className="text-sm text-slate-300">
                  Gemma cannot run completely in your browser because WebGPU is not supported or is disabled in your current setup.
                </p>
              </div>
              <div className="border-t border-rose-500/20 pt-4 space-y-2.5 text-xs text-slate-400">
                <p className="font-semibold text-slate-300">How to solve this:</p>
                <ul className="list-disc pl-4 space-y-1.5 leading-relaxed">
                  <li>
                    Ensure you are using a modern browser that natively supports WebGPU (e.g. <b>Chrome v113+</b>, <b>Edge v113+</b>, or <b>Opera</b>).
                  </li>
                  <li>
                    Firefox currently requires manual activation: Type <code className="text-cyan-400">about:config</code>, search for <code className="text-cyan-400">dom.webgpu.enabled</code> and set it to <b>true</b>.
                  </li>
                  <li>
                    If on Chrome/Edge, ensure hardware acceleration is active under Browser Settings {">"} System and Performance.
                  </li>
                </ul>
              </div>
            </div>
          )}

          {/* CASE B: MODEL LOADING / CACHE DOWNLOADING STATE */}
          {webGpuSupported && isLoading && progress && (
            <div className="max-w-md mx-auto my-16 p-6 rounded-2xl border border-purple-500/20 bg-purple-950/15 backdrop-blur-xl space-y-6 shadow-2xl shadow-purple-950/20">
              <div className="relative w-24 h-24 mx-auto">
                <div className="absolute inset-0 rounded-full border-2 border-purple-500/10" />
                {/* Glowing spinner background */}
                <div className="absolute inset-0 rounded-full border-2 border-t-cyber-purple border-r-cyber-cyan animate-spin-glow" />
                <div className="absolute inset-0 flex items-center justify-center text-sm font-semibold font-mono text-cyan-300">
                  {Math.round(progress.progress * 100)}%
                </div>
              </div>

              <div className="text-center space-y-2">
                <h3 className="text-sm font-semibold tracking-wider text-purple-300 uppercase">
                  Waking AI Gemma Engine
                </h3>
                <p className="text-[11px] text-slate-400 font-mono tracking-wide line-clamp-1">
                  {progress.text}
                </p>
              </div>

              {/* Progress bar */}
              <div className="space-y-1.5">
                <div className="w-full h-1.5 rounded-full bg-slate-900 overflow-hidden relative">
                  <div
                    className="h-full loader-glow-bar rounded-full transition-all duration-300 ease-out"
                    style={{ width: `${progress.progress * 100}%` }}
                  />
                </div>
                
                <div className="flex items-center justify-between text-[10px] text-slate-500 font-mono">
                  <span>Elapsed: {Math.round(progress.timeElapsed)}s</span>
                  <span>Size: ~1.4 GB</span>
                </div>
              </div>

              <div className="p-3 rounded-lg bg-black/40 text-[10.5px] text-slate-400 leading-relaxed border border-purple-950/50">
                💡 <b>IndexedDB Caching:</b> The initial initialization requires fetching weights (~1.4GB) from Hugging Face. Subsequent activations will be nearly instantaneous as assets will read from your local IndexedDB browser storage!
              </div>
            </div>
          )}

          {/* CASE C: CHAT READY, DISPLAY MODEL INITIAL STATUS (INTRO) OR CONVERSATIONS */}
          {webGpuSupported && !isLoading && (
            <>
              {/* Show welcome panel if there are no messages in the active session */}
              {activeSession && activeSession.messages.length === 0 ? (
                <div className="max-w-2xl mx-auto my-6 space-y-8">
                  <div className="text-center space-y-4 pt-4">
                    <div className="w-16 h-16 rounded-full bg-gradient-to-tr from-cyber-purple to-cyber-cyan p-[1.5px] mx-auto shadow-xl shadow-cyber-purple/20">
                      <div className="w-full h-full rounded-full bg-deep-space flex items-center justify-center">
                        <Sparkles className="w-8 h-8 text-cyber-cyan animate-pulse" />
                      </div>
                    </div>
                    <div className="space-y-1.5">
                      <h2 className="text-2xl font-bold bg-gradient-to-r from-white via-slate-100 to-cyan-300 bg-clip-text text-transparent tracking-wide animate-cyber-glow">
                        Gemma Client-Side AI
                      </h2>
                      <p className="text-sm text-slate-400 max-w-md mx-auto leading-relaxed">
                        Say goodbye to remote servers. This quantized Gemma LLM is running completely inside your browser via local WebGPU acceleration.
                      </p>
                    </div>
                  </div>

                  {/* Suggestion Prompt Chips */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
                    {promptSuggestions.map((s, index) => (
                      <button
                        key={index}
                        onClick={() => {
                          if (!loadedModel) {
                            handleLoadModel().then(() => handleSendMessage(s.text));
                          } else {
                            handleSendMessage(s.text);
                          }
                        }}
                        className="text-left p-4 rounded-xl border border-purple-950/20 bg-purple-950/5 hover:bg-purple-950/15 hover:border-purple-900/50 hover:-translate-y-0.5 transition-all duration-200 text-xs space-y-1"
                      >
                        <span className="font-semibold text-purple-300 block">{s.label}</span>
                        <p className="text-slate-400 line-clamp-2 leading-relaxed text-[11px]">
                          {s.text}
                        </p>
                      </button>
                    ))}
                  </div>

                  {/* Cache info / Wake trigger */}
                  {!loadedModel && (
                    <div className="p-4 rounded-xl border border-cyan-950/40 bg-cyan-950/5 backdrop-blur-md flex items-start gap-3 max-w-xl mx-auto shadow-md">
                      <HelpCircle className="w-5 h-5 text-cyan-400 shrink-0 mt-0.5" />
                      <div className="space-y-1.5 text-xs">
                        <span className="font-semibold text-slate-200">First-Time Setup Required</span>
                        <p className="text-slate-400 leading-relaxed text-[11.5px]">
                          To begin, click the button below to initialize and download the quantized weights. Once cached, Gemma works completely offline!
                        </p>
                        <button
                          onClick={handleLoadModel}
                          className="mt-2 py-1.5 px-3 rounded-lg bg-cyber-cyan text-slate-950 hover:bg-cyan-300 hover:shadow-md transition-all font-semibold"
                        >
                          Initialize Gemma Engine
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                /* Chat Conversation List */
                <div className="max-w-3xl mx-auto space-y-6 pb-20">
                  {activeSession?.messages.map((msg, index) => (
                    <div
                      key={index}
                      className={`flex gap-4 items-start ${
                        msg.role === "user" ? "flex-row-reverse" : "flex-row"
                      }`}
                    >
                      {/* Avatar */}
                      <div
                        className={`w-9 h-9 rounded-xl shrink-0 flex items-center justify-center font-bold text-xs ${
                          msg.role === "user"
                            ? "bg-slate-800 text-slate-200 shadow-md border border-slate-700/50"
                            : "bg-gradient-to-tr from-cyber-purple/20 to-cyber-cyan/20 text-cyan-300 border border-purple-900/30 shadow-md"
                        }`}
                      >
                        {msg.role === "user" ? "ME" : "G"}
                      </div>

                      {/* Chat bubble content */}
                      <div className="space-y-1 max-w-[80%]">
                        <span className="text-[10px] text-slate-500 font-mono tracking-wider">
                          {msg.role === "user" ? "USER" : "GEMMA AI"}
                        </span>
                        
                        <div
                          className={`rounded-2xl px-4 py-3 text-xs md:text-sm leading-relaxed border transition-all ${
                            msg.role === "user"
                              ? "bg-slate-900/60 border-slate-800/40 text-slate-100 rounded-tr-none"
                              : "bg-purple-950/10 backdrop-blur-md border-purple-900/25 text-slate-200 rounded-tl-none prose"
                          }`}
                        >
                          {msg.role === "user" ? (
                            <p className="whitespace-pre-wrap">{msg.content}</p>
                          ) : (
                            /* Markdown rendering + GFM + Custom code syntax with Copy Button */
                            <ReactMarkdown
                              remarkPlugins={[remarkGfm]}
                              components={{
                                code({ className, children, ...props }) {
                                  const match = /language-(\w+)/.exec(className || "");
                                  const codeString = String(children).replace(/\n$/, "");
                                  const codeBlockId = `code_${index}_${Math.random().toString(36).substr(2, 5)}`;
                                  
                                  return match ? (
                                    <div className="relative group my-2 border border-purple-950/50 rounded-lg overflow-hidden">
                                      <div className="flex justify-between items-center px-4 py-1.5 bg-black/60 border-b border-purple-900/20 text-[10px] text-slate-400 font-mono">
                                        <span className="uppercase tracking-wider">{match[1]}</span>
                                        <button
                                          type="button"
                                          onClick={() => handleCopyCode(codeString, codeBlockId)}
                                          className="hover:text-cyan-400 transition-colors flex items-center gap-1"
                                          title="Copy Code"
                                        >
                                          {copiedId === codeBlockId ? (
                                            <>
                                              <Check className="w-3 h-3 text-emerald-400" />
                                              <span className="text-emerald-400">Copied</span>
                                            </>
                                          ) : (
                                            <>
                                              <Copy className="w-3 h-3" />
                                              <span>Copy</span>
                                            </>
                                          )}
                                        </button>
                                      </div>
                                      <pre className="!mt-0 !bg-black/40 overflow-x-auto p-4 max-h-72">
                                        <code className="text-xs leading-relaxed" {...props}>
                                          {children}
                                        </code>
                                      </pre>
                                    </div>
                                  ) : (
                                    <code className="bg-purple-950/40 text-purple-300 px-1.5 py-0.5 rounded font-mono text-xs" {...props}>
                                      {children}
                                    </code>
                                  );
                                },
                              }}
                            >
                              {msg.content || "*(Generating response...)*"}
                            </ReactMarkdown>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}

                  {/* Scroll Anchor */}
                  <div ref={chatEndRef} />
                </div>
              )}
            </>
          )}

        </div>

        {/* Sticky floating bottom controls (Input field / Stop / Quick Chips) */}
        {webGpuSupported && !isLoading && (
          <footer className="absolute bottom-0 inset-x-0 p-6 bg-gradient-to-t from-deep-space via-deep-space/80 to-transparent z-10">
            <div className="max-w-3xl mx-auto space-y-3">
              
              {/* Stop Generation Floating Overlay */}
              {isGenerating && (
                <div className="flex justify-center">
                  <button
                    onClick={interruptGeneration}
                    className="py-1 px-3 text-xs bg-rose-500/10 hover:bg-rose-500/20 text-rose-300 border border-rose-500/30 rounded-full flex items-center gap-1.5 transition-all active:scale-95 shadow-md shadow-rose-950/20"
                  >
                    <X className="w-3.5 h-3.5" />
                    Stop Stream Generation
                  </button>
                </div>
              )}

              {/* Chat Text area input container */}
              <div className="relative rounded-2xl bg-black/40 border border-purple-900/30 backdrop-blur-md shadow-2xl focus-within:border-cyber-cyan transition-all">
                <form
                  onSubmit={(e) => {
                    e.preventDefault();
                    if (!loadedModel) {
                      handleLoadModel().then(() => handleSendMessage());
                    } else {
                      handleSendMessage();
                    }
                  }}
                  className="flex items-center p-2.5"
                >
                  <textarea
                    rows={1}
                    value={inputMessage}
                    onChange={(e) => setInputMessage(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        if (!loadedModel) {
                          handleLoadModel().then(() => handleSendMessage());
                        } else {
                          handleSendMessage();
                        }
                      }
                    }}
                    placeholder={
                      !loadedModel
                        ? "Wake up Gemma by typing a prompt or clicking the chip..."
                        : "Type a prompt entirely locally..."
                    }
                    className="flex-1 max-h-32 min-h-8 py-2 px-3 bg-transparent text-xs md:text-sm text-slate-100 placeholder-slate-500 focus:outline-none resize-none"
                  />

                  {/* Send Button */}
                  <button
                    type="submit"
                    disabled={!inputMessage.trim() || isGenerating}
                    className="w-9 h-9 shrink-0 rounded-xl bg-gradient-to-tr from-cyber-purple to-cyber-cyan hover:shadow-lg hover:shadow-cyber-purple/20 transition-all flex items-center justify-center text-white disabled:opacity-40 disabled:hover:shadow-none active:scale-95 disabled:scale-100"
                    title="Send message"
                  >
                    <Send className="w-4 h-4" />
                  </button>
                </form>
              </div>

              {/* Bottom helper guidelines */}
              <div className="flex justify-between items-center px-2 text-[10px] text-slate-500 font-mono">
                <span>Runs 100% locally via WebGPU</span>
                {loadedModel && (
                  <span className="flex items-center gap-1">
                    <CheckCircle className="w-3 h-3 text-cyan-400" /> Cache hit (Offline OK)
                  </span>
                )}
              </div>

            </div>
          </footer>
        )}

      </main>

    </div>
  );
}
