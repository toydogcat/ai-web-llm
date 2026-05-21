#!/usr/bin/env python3
"""
Gemma Local Chat — Python Terminal Version
==========================================
A fully local, offline-capable AI chat client that mirrors the browser WebGPU
app but runs in your terminal via Ollama (or Hugging Face Transformers).

Requirements:
  1. Install Ollama: https://ollama.com/
  2. Pull a Gemma model:
       ollama pull gemma2:2b
  3. Install Python dependencies:
       pip install -r requirements.txt
  4. Run:
       python chat.py

Supported Agentic Tools (same as the web version):
  - get_current_time()          → Returns system clock
  - run_math_calculation(expr)  → Runs mathjs-equivalent via Python's math module
  - search_wikipedia(query)     → Fetches from Wikipedia API (requires internet)
"""

import re
import sys
import math
import datetime
import urllib.parse
import urllib.request
import json
import readline  # noqa: F401  (enables arrow-key history in terminal)
from dotenv import load_dotenv
load_dotenv()
from rag.vector_store import search_docs

# ── Optional colour support ──────────────────────────────────────────────────
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False

    class _Noop:
        def __getattr__(self, _): return ""
    Fore = Style = _Noop()

# ── Try to import ollama client ───────────────────────────────────────────────
try:
    import ollama
    BACKEND = "ollama"
except ImportError:
    BACKEND = None

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME    = "gemma2:2b"        # Change to any model you've pulled
CONTEXT_LIMIT = 20                 # Max messages kept in history (same as web app)
MAX_TOKENS    = 2048               # Max tokens per response

SYSTEM_PROMPT = (
    "You are Gemma, a highly helpful, intelligent, and creative AI assistant "
    "running entirely locally via Ollama.\n\n"
    "You have access to the following local tools. When you need to perform an action "
    "(get current time, calculate math, search Wikipedia, or search local documents), "
    "output the tool call command on a single line by itself and IMMEDIATELY STOP:\n"
    "- To get the current local time: [CALL: get_current_time()]\n"
    '- To execute a safe mathematical calculation: [CALL: run_math_calculation(expression="sqrt(1523) + 45 * sin(30 * pi / 180)")]\n'
    '- To search info/news on Wikipedia: [CALL: search_wikipedia(query="2026 Winter Olympics")]\n'
    '- To search YOUR OWN LOCAL DOCUMENTS (PDFs, notes): [CALL: search_local_documents(query="calculus derivatives")]\n\n'
    "Always write your final responses in Traditional Chinese (繁體中文)."
)

# ── Tool implementations ──────────────────────────────────────────────────────

def tool_get_current_time() -> str:
    now = datetime.datetime.now()
    return now.strftime("%Y/%m/%d %A %H:%M:%S (本地時間)")


def tool_run_math_calculation(expression: str) -> str:
    """
    Safe math evaluator using Python's math module.
    Allowed: numbers, operators (+−*/^), math functions (sin, cos, sqrt, log, …),
    pi, e, parentheses, spaces.
    Everything else is blocked.
    """
    # Whitelist: only safe characters + known math functions
    allowed = re.compile(
        r'^[\d\s\+\-\*/\^\(\)\.\,]'
        r'|sqrt|sin|cos|tan|log|log10|log2|exp|pow|abs|ceil|floor|round'
        r'|pi|e|inf|nan'
    )
    cleaned = expression.strip()
    # Replace ^ with ** for Python power syntax
    cleaned = cleaned.replace("^", "**")

    # Build a restricted namespace with only math functions
    safe_ns = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    safe_ns["__builtins__"] = {}  # Block builtins entirely

    try:
        result = eval(cleaned, safe_ns)  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"計算錯誤: {exc}"


def tool_search_wikipedia(query: str) -> str:
    """Fetch search results from the Chinese Wikipedia API."""
    try:
        encoded = urllib.parse.quote(query)
        url = (
            f"https://zh.wikipedia.org/w/api.php"
            f"?action=query&list=search&srsearch={encoded}&format=json&origin=*"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "GemmaLocalChat/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
        results = data.get("query", {}).get("search", [])
        if not results:
            return f"找不到與「{query}」相關的維基百科條目。"
        lines = []
        for i, item in enumerate(results[:3], 1):
            snippet = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
            lines.append(f"[搜尋結果 {i}] 標題: {item['title']}\n內容摘要: {snippet}")
        return "\n\n".join(lines)
    except Exception as exc:
        return f"搜尋發生錯誤: {exc}"


def tool_search_local_docs(query: str) -> str:
    """Semantic search in indexed local PDF/MD files."""
    return search_docs(query)


# ── Tool regex patterns (mirrors the TypeScript version) ─────────────────────
_RE_TIME   = re.compile(r"\[CALL:\s*get_current_time\s*\(\s*\)\s*\]")
_RE_MATH   = re.compile(r'\[CALL:\s*run_math_calculation\s*\(\s*expression\s*=\s*"([^"]+)"\s*\)\s*\]')
_RE_SEARCH = re.compile(r'\[CALL:\s*search_wikipedia\s*\(\s*query\s*=\s*"([^"]+)"\s*\)\s*\]')
_RE_LOCAL  = re.compile(r'\[CALL:\s*search_local_documents\s*\(\s*query\s*=\s*"([^"]+)"\s*\)\s*\]')


def dispatch_tool(response_text: str) -> tuple[str | None, str | None]:
    """
    Check if response_text contains a tool call.
    Returns (tool_output, matched_call_string) or (None, None).
    """
    if m := _RE_TIME.search(response_text):
        return tool_get_current_time(), m.group(0)
    if m := _RE_MATH.search(response_text):
        return tool_run_math_calculation(m.group(1)), m.group(0)
    if m := _RE_SEARCH.search(response_text):
        print(f"\n{Fore.CYAN}🔍 正在從維基百科搜尋「{m.group(1)}」...{Style.RESET_ALL}")
        return tool_search_wikipedia(m.group(1)), m.group(0)
    if m := _RE_LOCAL.search(response_text):
        print(f"\n{Fore.CYAN}📂 正在搜尋本地文檔「{m.group(1)}」...{Style.RESET_ALL}")
        return tool_search_local_docs(m.group(1)), m.group(0)
    return None, None


# ── Ollama chat backend ───────────────────────────────────────────────────────

def chat_with_ollama(messages: list[dict]) -> str:
    """Stream a response from Ollama and return full text."""
    full = ""
    print(f"\n{Fore.MAGENTA}Gemma AI ▸{Style.RESET_ALL} ", end="", flush=True)
    stream = ollama.chat(
        model=MODEL_NAME,
        messages=messages,
        stream=True,
        options={"num_predict": MAX_TOKENS},
    )
    for chunk in stream:
        token = chunk["message"]["content"]
        print(token, end="", flush=True)
        full += token
    print()  # newline after streamed output
    return full


# ── Agent loop (mirrors the TypeScript sendMessage loop) ─────────────────────

def agent_send(messages: list[dict]) -> str:
    """
    Run the agentic tool-calling loop:
    1. Get a response from the model.
    2. If it contains a CALL directive, execute the tool.
    3. Inject the result and call the model again for the final answer.
    """
    current_messages = list(messages)
    full_response    = ""
    max_loops        = 3

    for _ in range(max_loops):
        response_text = chat_with_ollama(current_messages)

        tool_output, matched_call = dispatch_tool(response_text)

        if tool_output is not None:
            print(f"\n{Fore.YELLOW}🛠  工具執行: {matched_call}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}   輸出結果:{Style.RESET_ALL} {tool_output}")
            print(f"\n{Fore.CYAN}⟳  分析工具結果，產生最終回答…{Style.RESET_ALL}\n")

            full_response += response_text + f"\n\n[工具輸出: {tool_output}]\n\n"

            # Feed tool result back to the model
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
            # No tool call — this is the final answer
            full_response += response_text
            break

    return full_response


# ── Main chat loop ────────────────────────────────────────────────────────────

def print_banner():
    print(f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════╗
║         Gemma Local Chat — Python Terminal Client         ║
║    100%% local · Powered by Ollama · Agentic Tools        ║
╚══════════════════════════════════════════════════════════╝{Style.RESET_ALL}
Model : {Fore.GREEN}{MODEL_NAME}{Style.RESET_ALL}
Type  : {Fore.YELLOW}exit / quit{Style.RESET_ALL} to leave  |  {Fore.YELLOW}clear{Style.RESET_ALL} to reset history
""")


def main():
    if BACKEND != "ollama":
        print(
            f"{Fore.RED}[Error]{Style.RESET_ALL} 'ollama' Python package not found.\n"
            "Install it with:  pip install ollama\n"
            "And make sure Ollama is running:  ollama serve\n"
            f"Then pull the model:  ollama pull {MODEL_NAME}"
        )
        sys.exit(1)

    print_banner()

    history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input(f"{Fore.WHITE}You ▸ {Style.RESET_ALL}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{Fore.CYAN}Bye!{Style.RESET_ALL}")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print(f"{Fore.CYAN}Bye!{Style.RESET_ALL}")
            break
        if user_input.lower() == "clear":
            history = [{"role": "system", "content": SYSTEM_PROMPT}]
            print(f"{Fore.YELLOW}✓ 對話歷史已清除。{Style.RESET_ALL}\n")
            continue

        history.append({"role": "user", "content": user_input})

        # Truncate history to prevent context overflow (same logic as web app)
        system_msg  = history[:1]
        chat_msgs   = history[1:]
        if len(chat_msgs) > CONTEXT_LIMIT:
            chat_msgs = chat_msgs[-CONTEXT_LIMIT:]
        messages_to_send = system_msg + chat_msgs

        final_reply = agent_send(messages_to_send)
        history.append({"role": "assistant", "content": final_reply})
        print()


if __name__ == "__main__":
    main()
