"""
Gemma Local Chat — Streamlit UI Version
========================================
Run with:
    conda run -n toby streamlit run app.py

Backend Priority (auto-detected):
    1. llama-cpp-python  → Fastest on CPU (AVX-512 optimized for Intel Ultra)
    2. Ollama REST API   → Easy setup fallback
    3. Hugging Face Transformers → Most compatible fallback
"""

import re
import math
import datetime
import urllib.parse
import urllib.request
import json
import os
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gemma Local Chat",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_SESSION_MESSAGES = 40
DEFAULT_MODEL_OLLAMA  = "gemma2:2b"
DEFAULT_MODEL_GGUF    = "gemma-2-2b-it-Q4_K_M.gguf"  # download from HuggingFace

SYSTEM_PROMPT = (
    "You are Gemma, a highly helpful, intelligent, and creative AI assistant "
    "running entirely locally on this machine.\n\n"
    "You have access to the following local tools. When you need to perform an action "
    "(get current time, calculate math, or search Wikipedia), output the tool call "
    "command on a SINGLE LINE BY ITSELF and IMMEDIATELY STOP:\n"
    "- To get the current local time: [CALL: get_current_time()]\n"
    '- To execute a safe mathematical calculation: [CALL: run_math_calculation(expression="sqrt(1523) + 45 * sin(30 * pi / 180)")]\n'
    '- To search info/news on Wikipedia: [CALL: search_wikipedia(query="2026 Winter Olympics")]\n\n'
    "Always write your final responses in Traditional Chinese (繁體中文)."
)

# ── Backend detection ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def detect_backend():
    """Auto-detect the best available backend."""
    # 1. Try llama-cpp-python
    try:
        from llama_cpp import Llama  # noqa: F401
        return "llama_cpp"
    except ImportError:
        pass

    # 2. Try Ollama REST API
    try:
        import urllib.request as _ur
        _ur.urlopen("http://localhost:11434/api/tags", timeout=2)
        return "ollama"
    except Exception:
        pass

    # 3. Transformers (CPU fallback)
    try:
        import transformers  # noqa: F401
        return "transformers"
    except ImportError:
        pass

    return None

# ── Llama.cpp loader ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="⚙️ Loading GGUF model with llama.cpp…")
def load_llama_cpp_model(gguf_path: str, n_threads: int, ctx_size: int):
    from llama_cpp import Llama
    return Llama(
        model_path=gguf_path,
        n_ctx=ctx_size,
        n_threads=n_threads,
        n_gpu_layers=0,      # CPU-only (set > 0 if you have Metal/CUDA/Vulkan)
        verbose=False,
    )

# ── Local Tools ───────────────────────────────────────────────────────────────

def tool_get_current_time() -> str:
    return datetime.datetime.now().strftime("%Y/%m/%d %A %H:%M:%S (本地時間)")


def tool_run_math_calculation(expression: str) -> str:
    cleaned = expression.strip().replace("^", "**")
    safe_ns = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    safe_ns["__builtins__"] = {}
    try:
        result = eval(cleaned, safe_ns)  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"計算錯誤: {exc}"


def tool_search_wikipedia(query: str) -> str:
    try:
        encoded = urllib.parse.quote(query)
        url = (
            f"https://zh.wikipedia.org/w/api.php"
            f"?action=query&list=search&srsearch={encoded}&format=json&origin=*"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "GemmaStreamlitChat/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
        results = data.get("query", {}).get("search", [])
        if not results:
            return f"找不到與「{query}」相關的維基百科條目。"
        lines = []
        for i, item in enumerate(results[:3], 1):
            snippet = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
            lines.append(f"**[結果 {i}]** {item['title']}\n{snippet}")
        return "\n\n".join(lines)
    except Exception as exc:
        return f"搜尋發生錯誤: {exc}"


_RE_TIME   = re.compile(r"\[CALL:\s*get_current_time\s*\(\s*\)\s*\]")
_RE_MATH   = re.compile(r'\[CALL:\s*run_math_calculation\s*\(\s*expression\s*=\s*"([^"]+)"\s*\)\s*\]')
_RE_SEARCH = re.compile(r'\[CALL:\s*search_wikipedia\s*\(\s*query\s*=\s*"([^"]+)"\s*\)\s*\]')


def dispatch_tool(text: str):
    if m := _RE_TIME.search(text):
        return tool_get_current_time(), m.group(0)
    if m := _RE_MATH.search(text):
        return tool_run_math_calculation(m.group(1)), m.group(0)
    if m := _RE_SEARCH.search(text):
        return tool_search_wikipedia(m.group(1)), m.group(0)
    return None, None

# ── LLM generation (streaming) ────────────────────────────────────────────────

def generate_ollama(messages: list[dict], model: str, max_tokens: int):
    """Generator: yields text tokens from Ollama streaming API."""
    import urllib.request, json
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"num_predict": max_tokens},
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        for line in resp:
            if line.strip():
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break


def generate_llama_cpp(messages: list[dict], gguf_path: str, n_threads: int,
                       ctx_size: int, max_tokens: int):
    """Generator: yields text tokens from llama-cpp-python."""
    llm = load_llama_cpp_model(gguf_path, n_threads, ctx_size)
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in response:
        token = chunk["choices"][0]["delta"].get("content", "")
        if token:
            yield token


def generate_transformers(messages: list[dict], model_id: str, max_tokens: int):
    """Generator: yields text tokens from HuggingFace Transformers."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
    import threading

    @st.cache_resource(show_spinner="Loading HuggingFace model…")
    def _load_model(mid):
        tok = AutoTokenizer.from_pretrained(mid)
        mdl = AutoModelForCausalLM.from_pretrained(mid, device_map="cpu")
        return tok, mdl

    tokenizer, model = _load_model(model_id)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    kwargs = {**inputs, "max_new_tokens": max_tokens, "streamer": streamer}
    thread = threading.Thread(target=model.generate, kwargs=kwargs)
    thread.start()
    for token in streamer:
        yield token

# ── Agentic loop (Streamlit-aware) ───────────────────────────────────────────

def agent_turn(messages: list[dict], cfg: dict) -> str:
    """
    Run the agentic tool loop inside Streamlit.
    Returns the full concatenated final response text.
    """
    current_messages = list(messages)
    full_response     = ""
    backend           = cfg["backend"]
    max_loops         = 3

    for loop_i in range(max_loops):
        response_text = ""
        container = st.empty()

        # ── Stream the model output ──────────────────────────────────────────
        if backend == "ollama":
            gen = generate_ollama(current_messages, cfg["ollama_model"], cfg["max_tokens"])
        elif backend == "llama_cpp":
            gen = generate_llama_cpp(
                current_messages, cfg["gguf_path"],
                cfg["n_threads"], cfg["ctx_size"], cfg["max_tokens"],
            )
        else:
            gen = generate_transformers(
                current_messages, cfg["hf_model_id"], cfg["max_tokens"]
            )

        for token in gen:
            response_text += token
            container.markdown(
                full_response + response_text + " ▌",
                unsafe_allow_html=False,
            )

        # ── Check for tool calls ─────────────────────────────────────────────
        tool_output, matched_call = dispatch_tool(response_text)

        if tool_output is not None:
            # Show thinking accordion
            thinking_md = f"{response_text}\n\n🛠️ **工具執行**: `{matched_call}`\n\n```\n{tool_output}\n```"
            container.expander("🔍 思考與工具調用 (Thinking & Tools)", expanded=True).markdown(thinking_md)

            full_response += thinking_md + "\n\n---\n\n"

            # Inject tool result back
            current_messages.append({"role": "assistant", "content": response_text})
            current_messages.append({
                "role": "user",
                "content": (
                    f"[SYSTEM TOOL CALL RESULT] The tool {matched_call} returned:\n"
                    f"{tool_output}\n\n"
                    "Please analyze this result and write your final response in Traditional Chinese (繁體中文)."
                ),
            })
        else:
            container.markdown(full_response + response_text)
            full_response += response_text
            break

    return full_response

# ── Streamlit UI ──────────────────────────────────────────────────────────────

# Custom CSS for dark cyberpunk theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Fira+Code&display=swap');

    .stApp { background: #07070e; font-family: 'Outfit', sans-serif; }
    .stChatMessage { border-radius: 16px; }
    .stChatMessage[data-testid="stChatMessage-assistant"] {
        background: linear-gradient(135deg, rgba(157,78,221,0.06), rgba(6,182,212,0.04));
        border: 1px solid rgba(157,78,221,0.2);
    }
    .stChatMessage[data-testid="stChatMessage-user"] {
        background: rgba(15,15,30,0.7);
        border: 1px solid rgba(255,255,255,0.06);
    }
    code, pre { font-family: 'Fira Code', monospace !important; }
    .status-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 600;
        font-family: monospace;
    }
    .badge-green  { background: rgba(52,211,153,.15); color: #34d399; border: 1px solid rgba(52,211,153,.3); }
    .badge-yellow { background: rgba(251,191,36,.15);  color: #fbbf24; border: 1px solid rgba(251,191,36,.3); }
    .badge-red    { background: rgba(248,113,113,.15); color: #f87171; border: 1px solid rgba(248,113,113,.3); }
    [data-testid="stSidebar"] { background: rgba(0,0,0,.5) !important; border-right: 1px solid rgba(157,78,221,.2); }
    .stButton button {
        background: linear-gradient(135deg, #9d4edd, #06b6d4);
        color: white; border: none; border-radius: 12px;
        font-family: 'Outfit', sans-serif; font-weight: 600;
    }
    .stButton button:hover { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)


def main():
    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🧠 **GEMMA LOCAL**")
        st.caption("100% Local · Agentic AI · Streamlit UI")
        st.divider()

        # Backend detection
        backend = detect_backend()
        if backend == "llama_cpp":
            st.markdown('<span class="status-badge badge-green">⚡ llama.cpp (Fastest)</span>', unsafe_allow_html=True)
        elif backend == "ollama":
            st.markdown('<span class="status-badge badge-yellow">🦙 Ollama</span>', unsafe_allow_html=True)
        elif backend == "transformers":
            st.markdown('<span class="status-badge badge-yellow">🤗 Transformers</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge badge-red">❌ No backend found</span>', unsafe_allow_html=True)
            st.error("Please install one of: llama-cpp-python, ollama, or transformers")
            st.stop()

        st.divider()
        st.markdown("### ⚙️ Settings")

        cfg: dict = {"backend": backend}

        if backend == "llama_cpp":
            cfg["gguf_path"] = st.text_input(
                "GGUF model path",
                value=str(os.path.expanduser(f"~/.cache/huggingface/{DEFAULT_MODEL_GGUF}")),
                help="Download from: https://huggingface.co/bartowski/gemma-2-2b-it-GGUF",
            )
            cfg["n_threads"] = st.slider("CPU threads", 1, 22, 16,
                help="Intel Ultra 7 155H has 22 logical CPUs. 14-18 is optimal.")
            cfg["ctx_size"]  = st.select_slider("Context window (tokens)",
                options=[1024, 2048, 4096, 8192], value=4096)

        elif backend == "ollama":
            cfg["ollama_model"] = st.text_input("Ollama model", value=DEFAULT_MODEL_OLLAMA)

        else:  # transformers
            cfg["hf_model_id"] = st.text_input(
                "HuggingFace model ID",
                value="google/gemma-2-2b-it",
            )

        cfg["max_tokens"] = st.slider("Max tokens per response", 256, 4096, 2048)

        st.divider()
        st.markdown("### 📝 System Prompt")
        system_prompt = st.text_area(
            "System Instructions",
            value=SYSTEM_PROMPT,
            height=200,
            label_visibility="collapsed",
        )

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col2:
            msg_count = len(st.session_state.get("messages", []))
            st.metric("Messages", msg_count)

        st.caption(f"Context limit: {MAX_SESSION_MESSAGES} msgs")

    # ── Main chat area ────────────────────────────────────────────────────────
    st.markdown("# 💬 Gemma Client-Side AI")
    st.caption("Say goodbye to remote servers. Running entirely locally on your machine.")

    # Init session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧠" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])

    # Quick prompt chips
    if not st.session_state.messages:
        st.markdown("#### ⚡ 快速提示")
        chips = [
            ("⏰ 現在幾點？", "幫我查詢現在的本地時間"),
            ("🧮 計算 √1523 + 45×sin(30°)", 'run_math_calculation(expression="sqrt(1523) + 45 * sin(30 * pi / 180)")'),
            ("🔍 查台灣新聞", "幫我從維基百科查一下最近台灣的重要新聞主題"),
            ("💻 寫 Python 排序", "用 Python 寫一個快速排序（QuickSort）算法，需要包含詳細的繁體中文註解"),
        ]
        cols = st.columns(2)
        for i, (label, text) in enumerate(chips):
            with cols[i % 2]:
                if st.button(label, use_container_width=True, key=f"chip_{i}"):
                    st.session_state._pending_chip = text
                    st.rerun()

    # Handle pending chip input
    if hasattr(st.session_state, "_pending_chip"):
        pending = st.session_state._pending_chip
        del st.session_state._pending_chip
        user_input = pending
    else:
        user_input = st.chat_input("在這裡輸入您的問題... (完全本地端運算)")

    if user_input:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        # Build messages list with context truncation
        history = st.session_state.messages[-MAX_SESSION_MESSAGES:]
        messages_to_send = [{"role": "system", "content": system_prompt}] + [
            {"role": m["role"], "content": m["content"]} for m in history
        ]

        # Generate response
        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("🔮 Gemma is thinking locally…"):
                full_reply = agent_turn(messages_to_send, cfg)

        # Save to session state
        st.session_state.messages.append({"role": "assistant", "content": full_reply})
        st.rerun()


if __name__ == "__main__":
    main()
