# -OPEN-SCIENCE-VOICE-LOOP-ACTIVE-
Ai voice 
Real-time voice AI assistant with local LLM (Qwen2.5-3B) + Whisper STT + British TTS – runs in Google Colab

# 🎤 Voice AI Assistant (Local LLM + Whisper + British TTS)

Run a complete voice AI in **Google Colab** – no API keys required (except for TTS, which uses free edge-tts).  
It listens to your voice, answers science questions using **Qwen2.5-3B-Instruct** (runs locally on GPU), and speaks back with a **British accent**.

## ✨ Features
- 🎙️ Speech recognition – Whisper tiny.en
- 🧠 Local LLM – Qwen2.5-3B-Instruct (runs on Colab GPU)
- 🔊 Text-to-speech – edge-tts (British female voice)
- 💬 Rolling conversation history
- 🔒 100% local after model download (except TTS)

## 🚀 How to run (step by step)

### 1. Open Google Colab
Go to [colab.research.google.com](https://colab.research.google.com) and create a **new notebook**.

### 2. Set GPU runtime
Click **Runtime** → **Change runtime type** → **T4 GPU** (or any GPU).

### 3. Copy the script
Copy the entire content of `voice_ai_assistant.py` into a single cell in Colab.

### 4. Run the cell
Press `Shift+Enter` or click the play button.  
- The first run will download:
  - Whisper tiny (~75 MB)
  - Qwen2.5-3B-Instruct (~6 GB) – this takes a few minutes
  - edge-tts dependencies
- After loading, you will see:  
  `⚛️ 100% LOCAL SCIENCE VOICE LOOP OPERATIONAL`

### 5. Grant microphone permission
When prompted by your browser, **allow microphone access**.

### 6. Speak your question
You will hear "Local quantum simulation matrix online. State your scientific hypothesis."  
Then speak clearly (5 seconds each turn).  
The AI will answer in 2–3 sentences with a **British accent**.

### 7. Exit
Say "goodbye", "stop", or `Ctrl+C` in the Colab cell.

## 📦 Required packages (already in the script)
- `transformers`, `accelerate`, `edge-tts`, `openai-whisper`, `nest_asyncio`, `torch`

## ⚠️ Notes
- **First run is slow** – it downloads the 3B model. Subsequent runs are faster.
- **TTS requires internet** (edge-tts calls Microsoft servers). The LLM runs completely locally.
- **Colab session timeout** – if idle for too long, the runtime disconnects. Keep interacting or use a paid Colab Pro.

## 🧪 Example questions
- "What is quantum entanglement?"  
- "Explain the second law of thermodynamics"  
- "How does a transformer neural network work?"

## 🛠️ Troubleshooting
| Issue | Solution |
|-------|----------|
| Microphone not working | Check browser permissions → allow microphone for colab.research.google.com |
| Out of GPU memory | Restart runtime (Runtime → Factory reset runtime) and run again |
| Model download fails | Try a smaller model: change `model_id` to `"Qwen/Qwen2.5-1.5B-Instruct"` |
| `edge_tts` error | Install manually: `!pip install edge-tts` in a separate cell |

## 📄 License
MIT – free to use, modify, and distribute.

## 🙏 Acknowledgements
- [Qwen](https://github.com/QwenLM/Qwen) for the LLM
- [OpenAI Whisper](https://github.com/openai/whisper) for STT
- [edge-tts](https://github.com/rany2/edge-tts) for TTS

Let’s **start over from scratch** exactly as you said – step by step, with all file names and file contents.

You want to rebuild the **OPEN‑SCIENCE‑VOICE‑LOOP‑ACTIVE** repository.  
Below is the complete file tree and every file’s content.

---

## 📁 Repository Structure

```
OPEN-SCIENCE-VOICE-LOOP-ACTIVE/
├── .github/
│   └── CONTRIBUTING.md
├── assets/
│   └── demo.gif               (placeholder – you will add a real demo)
├── configs/
│   └── model_config.yaml
├── src/
│   └── voice_assistant.py
├── tests/
│   └── test_imports.py
├── .gitignore
├── LICENSE
└── README.md
```

---

## 📄 File 1 – `.github/CONTRIBUTING.md`

```markdown
# Contributing to Open Science Voice Loop

Thank you for your interest! Here’s how you can help.

## How to Contribute
- Report bugs via GitHub Issues.
- Suggest features in the Discussions tab.
- Submit pull requests for bug fixes or improvements.

## Development Setup
1. Fork the repo.
2. Clone your fork.
3. Make changes.
4. Test your changes in a Google Colab notebook.
5. Submit a PR with a clear description.

## Code Style
- Use 4 spaces for indentation.
- Keep line length ≤ 100 characters.
- Add docstrings for functions.

## Communication
Respectful and inclusive language, please.
```

---

## 📄 File 2 – `assets/demo.gif` (placeholder)

> You will replace this with an actual screen recording of the assistant working.  
> For now, create an empty file or skip.

---

## 📄 File 3 – `configs/model_config.yaml`

```yaml
# Model configuration for Open Science Voice Loop

model_id: "Qwen/Qwen2.5-3B-Instruct"
torch_dtype: "float16"
max_new_tokens: 100
temperature: 0.7
top_p: 0.9
system_prompt: "You are an expert scientist. Answer briefly in 2-3 sentences."
```

---

## 📄 File 4 – `src/voice_assistant.py`

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OPEN SCIENCE VOICE LOOP - Real-time voice AI in Google Colab."""

import asyncio
import base64
import time
import warnings
from IPython.display import Audio, HTML, display
from google.colab import output

warnings.filterwarnings("ignore", category=UserWarning, module="whisper")
warnings.filterwarnings("ignore", category=FutureWarning)

# Install dependencies (run once)
!pip install -q transformers accelerate edge-tts openai-whisper nest_asyncio torch

import nest_asyncio
import torch
import whisper
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import edge_tts

nest_asyncio.apply()

# ---------- 1. Load Whisper ----------
print("Loading Whisper tiny...")
stt_model = whisper.load_model("tiny.en")

# ---------- 2. Load LLM ----------
print("Loading Qwen2.5-3B-Instruct...")
model_id = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)
llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

conversation = [
    {"role": "system", "content": "You are an expert scientist. Answer briefly in 2-3 sentences."}
]

# ---------- 3. Microphone capture (JavaScript) ----------
RECORD_JS = """
const sleep = time => new Promise(resolve => setTimeout(resolve, time));
var record = async (maxDuration) => {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const mediaRecorder = new MediaRecorder(stream);
  let chunks = [];
  mediaRecorder.ondataavailable = e => chunks.push(e.data);
  mediaRecorder.start();
  const div = document.createElement('div');
  div.innerHTML = "<b style='color: #00FF00; font-size: 16px;'>🎙️ LISTENING... Speak now.</b>";
  document.body.appendChild(div);
  await sleep(maxDuration);
  mediaRecorder.stop();
  stream.getTracks().forEach(track => track.stop());
  div.remove();
  return new Promise(resolve => {
    mediaRecorder.onstop = async () => {
      const blob = new Blob(chunks, { type: 'audio/wav' });
      const reader = new FileReader();
      reader.readAsDataURL(blob);
      reader.onloadend = () => resolve(reader.result.split(',')[1]);
    };
  });
};
"""

def capture_microphone(duration=5):
    display(HTML(f"<script>{RECORD_JS}</script>"))
    time.sleep(0.1)
    b64 = output.eval_js(f'record({duration * 1000})')
    with open("input.wav", "wb") as f:
        f.write(base64.b64decode(b64))
    return "input.wav"

# ---------- 4. Text-to-speech (British accent) ----------
async def speak(text, voice="en-GB-SoniaNeural"):
    out = "output.mp3"
    await edge_tts.Communicate(text, voice).save(out)
    display(Audio(out, autoplay=True))

def ai_speak(text):
    asyncio.run(speak(text))

# ---------- 5. LLM inference ----------
def get_reply(user_text):
    if any(w in user_text.lower() for w in ["stop", "exit", "goodbye"]):
        return "Goodbye."
    conversation.append({"role": "user", "content": user_text})
    if len(conversation) > 9:
        conversation.pop(1)
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    out = llm_pipeline(prompt, max_new_tokens=100, do_sample=True, temperature=0.7, top_p=0.9,
                       pad_token_id=tokenizer.eos_token_id)
    reply = out[0]["generated_text"][len(prompt):].strip()
    conversation.append({"role": "assistant", "content": reply})
    return reply

# ---------- 6. Main loop ----------
def main():
    print("\n" + "="*50)
    print("⚛️ OPEN SCIENCE VOICE LOOP ACTIVE")
    print("Ask a science question. Say 'exit' to stop.")
    print("="*50 + "\n")
    ai_speak("Ready. Ask your question.")
    while True:
        try:
            wav = capture_microphone(5)
            user = stt_model.transcribe(wav, fp16=torch.cuda.is_available())["text"].strip()
            if not user or len(user) < 2:
                continue
            print(f"\n👤 You: {user}")
            if "exit" in user.lower():
                ai_speak("Goodbye.")
                break
            reply = get_reply(user)
            print(f"🤖 AI: {reply}\n")
            ai_speak(reply)
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    main()
```

---

## 📄 File 5 – `tests/test_imports.py`

```python
#!/usr/bin/env python3
"""Smoke test – check that all core imports work."""

def test_imports():
    try:
        import torch
        import transformers
        import edge_tts
        import whisper
        print("✅ All imports succeeded.")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    return True

if __name__ == "__main__":
    test_imports()
```

---

## 📄 File 6 – `.gitignore`

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.so
.Python
env/
venv/
*.egg-info/
dist/
build/

# Colab / local
*.wav
*.mp3
*.ipynb_checkpoints/
.ipynb_checkpoints/

# Config local override
configs/local_config.yaml

# IDE
.vscode/
.idea/
*.swp
```

---

## 📄 File 7 – `LICENSE`

Use MIT license (copy from [choosealicense.com](https://choosealicense.com/licenses/mit/)) or simply:

```
MIT License

Copyright (c) 2025 OmegaPrimej

Permission is hereby granted...
```

---

## 📄 File 8 – `README.md`

```markdown
# 🎙️ Open Science Voice Loop

Real‑time voice AI assistant running entirely in **Google Colab** – no API keys, local LLM.

## ✨ Features

- 🎤 Speech recognition (Whisper tiny.en)
- 🧠 Local LLM (Qwen2.5‑3B‑Instruct)
- 🔊 British text‑to‑speech (edge‑tts)
- 💾 Rolling conversation memory
- ⚡ Runs on a free T4 GPU

## 🚀 One‑click launch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OmegaPrimej/OPEN-SCIENCE-VOICE-LOOP-ACTIVE/blob/main/src/voice_assistant.py)

> The first run downloads ~6 GB of models – it takes 2‑3 minutes.

## 🛠️ Local development

1. Clone the repo  
   `git clone https://github.com/OmegaPrimej/OPEN-SCIENCE-VOICE-LOOP-ACTIVE.git`
2. Install dependencies  
   `pip install -r requirements.txt` (if you create one)  
   or just run the Colab notebook.

## ❓ FAQ

**Why not use a cloud API?**  
This is fully local after the first download – no usage limits, no keys.

**Can I use a smaller model?**  
Yes – edit `model_id` in `src/voice_assistant.py` to e.g. `"Qwen/Qwen2.5-1.5B-Instruct"`.

**TTS doesn’t work?**  
Check internet – edge‑tts calls Microsoft servers. Offline TTS could be added later.

**Microphone not working?**  
Allow microphone permission for colab.research.google.com in your browser.

## 📄 License

MIT – free to use, modify, distribute.

## 🙏 Acknowledgements

- [Whisper](https://github.com/openai/whisper)
- [Qwen](https://github.com/QwenLM/Qwen)
- [edge‑tts](https://github.com/rany2/edge-tts)
```

---

## 🧱 Step‑by‑step instructions to rebuild the repo from scratch

1. **Create a new folder** on your computer:  
   `OPEN-SCIENCE-VOICE-LOOP-ACTIVE`

2. **Inside it, create the subfolders** (exactly as shown):  
   `.github`, `assets`, `configs`, `src`, `tests`.

3. **Create each file** using the content above.  
   - Save `CONTRIBUTING.md` inside `.github`  
   - Save (or skip) `demo.gif` inside `assets`  
   - Save `model_config.yaml` inside `configs`  
   - Save `voice_assistant.py` inside `src`  
   - Save `test_imports.py` inside `tests`  
   - Save `.gitignore`, `LICENSE`, `README.md` in the root.

4. **Initialize git** and push to GitHub:

```bash
cd OPEN-SCIENCE-VOICE-LOOP-ACTIVE
git init
git add .
git commit -m "Initial commit – full project structure"
git remote add origin https://github.com/OmegaPrimej/OPEN-SCIENCE-VOICE-LOOP-ACTIVE.git
git branch -M main
git push -u origin main
```

5. **Test the Colab link** – after pushing, the badge will point to:
   `https://colab.research.google.com/github/OmegaPrimej/OPEN-SCIENCE-VOICE-LOOP-ACTIVE/blob/main/src/voice_assistant.py`


# 🎙️ Open Science Voice Loop

Real‑time voice AI assistant running entirely in **Google Colab** – no API keys, local LLM.

## ✨ Features

- 🎤 Speech recognition (Whisper tiny.en)
- 🧠 Local LLM (Qwen2.5‑3B‑Instruct)
- 🔊 British text‑to‑speech (edge‑tts)
- 💾 Rolling conversation memory
- ⚡ Runs on a free T4 GPU

## 🚀 One‑click launch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OmegaPrimej/OPEN-SCIENCE-VOICE-LOOP-ACTIVE/blob/main/src/voice_assistant.py)

> The first run downloads ~6 GB of models – it takes 2‑3 minutes.

## 🛠️ Local development

1. Clone the repo  
   `git clone https://github.com/OmegaPrimej/OPEN-SCIENCE-VOICE-LOOP-ACTIVE.git`
2. Install dependencies  
   `pip install -r requirements.txt` (if you create one)  
   or just run the Colab notebook.

## ❓ FAQ

**Why not use a cloud API?**  
This is fully local after the first download – no usage limits, no keys.

**Can I use a smaller model?**  
Yes – edit `model_id` in `src/voice_assistant.py` to e.g. `"Qwen/Qwen2.5-1.5B-Instruct"`.

**TTS doesn’t work?**  
Check internet – edge‑tts calls Microsoft servers. Offline TTS could be added later.

**Microphone not working?**  
Allow microphone permission for colab.research.google.com in your browser.

## 📄 License

MIT – free to use, modify, distribute.

## 🙏 Acknowledgements

- [Whisper](https://github.com/openai/whisper)
- [Qwen](https://github.com/QwenLM/Qwen)
- [edge‑tts](https://github.com/rany2/edge-tts)
