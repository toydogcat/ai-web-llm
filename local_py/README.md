# Gemma Local Chat — Python Terminal Version

A Python terminal chat client that mirrors the browser WebGPU app, but runs entirely locally via **Ollama**.

Supports the same 3 agentic tools as the web version:
- ⏰ **Time** — `get_current_time()`  
- 🧮 **Math** — `run_math_calculation(expression="...")`  
- 🔍 **Search** — `search_wikipedia(query="...")`

---

## Setup

### 1. Install Ollama
```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows — download from https://ollama.com/
```

### 2. Pull the Gemma model
```bash
ollama pull gemma2:2b
# or for a smaller/faster model:
ollama pull gemma:2b
```

### 3. Create a virtual environment and install dependencies
```bash
cd local_py
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Start Ollama (if not already running)
```bash
ollama serve
```

### 5. Run the chat client
```bash
python chat.py
```

---

## Usage

```
╔══════════════════════════════════════════════════════════╗
║         Gemma Local Chat — Python Terminal Client         ║
║    100% local · Powered by Ollama · Agentic Tools        ║
╚══════════════════════════════════════════════════════════╝
Model : gemma2:2b

You ▸ 現在幾點？
Gemma AI ▸  [CALL: get_current_time()]
🛠  工具執行: [CALL: get_current_time()]
   輸出結果: 2026/05/20 Wednesday 16:52:00 (本地時間)
⟳  分析工具結果，產生最終回答…
Gemma AI ▸  現在是2026年5月20日，星期三，下午四點五十二分。
```

### Commands
| Input | Action |
|-------|--------|
| Any text | Chat with Gemma |
| `clear` | Reset conversation history |
| `exit` or `quit` | Close the client |
| `Ctrl+C` | Force quit |

---

## Change Model

Edit `MODEL_NAME` at the top of `chat.py`:

```python
MODEL_NAME = "gemma2:2b"   # Change to any Ollama model
# Other options:
# MODEL_NAME = "gemma:2b"
# MODEL_NAME = "llama3.2:3b"
# MODEL_NAME = "phi3:mini"
```
