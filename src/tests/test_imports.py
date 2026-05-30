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
    test_imports()  let chunks = [];
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
