#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OPEN SCIENCE VOICE LOOP - Real-time voice AI in Google Colab.

Uses Whisper for STT, Qwen2.5-3B-Instruct for LLM, and edge-tts for TTS.
"""

import asyncio
import base64
import time
import warnings
from IPython.display import Audio, HTML, display
from google.colab import output

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="whisper")
warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------
# 1. Install & Import Dependencies
# ----------------------------
!pip install -q transformers accelerate edge-tts openai-whisper nest_asyncio torch

import nest_asyncio
import torch
import whisper
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

nest_asyncio.apply()

print("Loading Whisper tiny...")
stt_model = whisper.load_model("tiny.en")

print("Loading Qwen2.5-3B-Instruct model...")
model_id = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

conversation_history = [
    {
        "role": "system",
        "content": "You are an expert scientist. Answer briefly in 2-3 sentences."
    }
]

# ----------------------------
# 2. JavaScript Microphone Capture
# ----------------------------
RECORD_JS = """
const sleep = time => new Promise(resolve => setTimeout(resolve, time));
var record = async (maxDuration) => {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const mediaRecorder = new MediaRecorder(stream);
  let chunks = [];

  mediaRecorder.ondataavailable = e => chunks.push(e.data);
  mediaRecorder.start();

  const div = document.createElement('div');
  div.innerHTML = "<b style='color: #00FF00; font-size: 16px;'>🎙️ SYSTEM LISTENING... Speak now.</b>";
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
      reader.onloadend = () => {
        resolve(reader.result.split(',')[1]);
      };
    };
  });
};
"""

def capture_microphone_input(duration_seconds: int = 5) -> str:
    display(HTML(f"<script>{RECORD_JS}</script>"))
    time.sleep(0.1)
    b64_audio_data = output.eval_js(f'record({duration_seconds * 1000})')
    output_filename = "user_input.wav"
    with open(output_filename, "wb") as f:
        f.write(base64.b64decode(b64_audio_data))
    return output_filename

# ----------------------------
# 3. Text-to-Speech (British Accent)
# ----------------------------
import edge_tts

async def speak_response(text: str, voice: str = "en-GB-SoniaNeural"):
    output_file = "ai_voice.mp3"
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)
    display(Audio(output_file, autoplay=True))

def ai_speak(text: str):
    asyncio.run(speak_response(text))

# ----------------------------
# 4. LLM Inference Engine
# ----------------------------
def get_ai_response(user_text: str) -> str:
    global conversation_history
    text_clean = user_text.lower().strip()
    if any(word in text_clean for word in ["stop", "exit", "goodbye"]):
        return "Understood. Local science core standing down. Goodbye."

    conversation_history.append({"role": "user", "content": user_text})
    if len(conversation_history) > 9:
        conversation_history = [conversation_history[0]] + conversation_history[-6:]

    try:
        prompt = tokenizer.apply_chat_template(
            conversation_history,
            tokenize=False,
            add_generation_prompt=True
        )
        outputs = llm_pipeline(
            prompt,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        full_text = outputs[0]["generated_text"]
        response = full_text[len(prompt):].strip()
        conversation_history.append({"role": "assistant", "content": response})
        return response
    except Exception as e:
        return f"Local engine execution lag. I captured your input text: {user_text}"

# ----------------------------
# 5. Main Conversation Loop
# ----------------------------
def start_live_voice_mode():
    print("\n=======================================================")
    print("⚛️ 100% LOCAL SCIENCE VOICE LOOP OPERATIONAL")
    print("Running natively on GPU memory. Talk freely.")
    print("=======================================================\n")

    ai_speak("Local quantum simulation matrix online. State your scientific hypothesis.")
    time.sleep(4.0)

    while True:
        try:
            audio_file = capture_microphone_input(duration_seconds=5)
            result = stt_model.transcribe(audio_file, fp16=torch.cuda.is_available())
            user_speech = result["text"].strip()

            if not user_speech or len(user_speech) < 3:
                continue

            print(f"👤 Question: {user_speech}")
            ai_reply = get_ai_response(user_speech)
            print(f"🤖 AI Core: {ai_reply}\n")
            ai_speak(ai_reply)

            if any(word in user_speech.lower() for word in ["goodbye", "stop", "exit"]):
                break

            time.sleep(max(4.0, len(ai_reply.split()) / 2.2))
        except KeyboardInterrupt:
            print("\nVoice environment stopped manually.")
            break
        except Exception as e:
            print(f"Loop processing error: {e}")
            break

if __name__ == "__main__":
    start_live_voice_mode()
