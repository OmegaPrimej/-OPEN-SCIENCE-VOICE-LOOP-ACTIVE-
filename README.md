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
