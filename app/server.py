"""Live Khmer captioning + rolling summary (Flask + Socket.IO).

Run from the project root:
    khmer-stt
    python app/server.py                                  # seanghay checkpoint
    ASR_MODEL=results/run1/model python app/server.py     # our fine-tuned model
Then open http://localhost:5050

The browser sends ~3 s chunks of mic audio (PLAN.md D6). Each chunk is transcribed
and pushed back as a caption. The summary refreshes every ~30 s when the transcript
has grown, or immediately on "Summarize now" (D7).
"""

import queue
import threading
import time

import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO

from asr import Transcriber
from summarize import Summarizer

SUMMARY_EVERY_S = 30
PORT = 5050

app = Flask(__name__)
socketio = SocketIO(app, async_mode="threading", max_http_buffer_size=10_000_000)

print("loading models...")
asr = Transcriber()
summarizer = Summarizer()
print(f"ASR: {asr.name} on {asr.device} | summarizer: {summarizer.name} on {summarizer.device}")

# Warm up both models: the first MPS call is slow (first summary took 37.6 s vs 0.6 s warm).
print("warming up...")
for _ in range(2):
    asr(np.random.default_rng(0).normal(0, 0.1, 48000).astype(np.float32))
    summarizer("ស្ពាន កំពង់ ចម្លង អ្នកលឿង នៅ ព្រៃវែង ជា ស្ពាន វែង ជាង គេ ក្នុង ព្រះ រាជាណាចក្រ កម្ពុជា។")

chunks = queue.Queue()
state_lock = threading.Lock()
summary_lock = threading.Lock()
state = {"captions": [], "summarized_upto": 0}


def transcript():
    with state_lock:
        return " ".join(state["captions"])


@app.route("/")
def index():
    return render_template("index.html", asr_model=asr.name, sum_model=summarizer.name, device=asr.device)


@socketio.on("audio_chunk")
def on_audio_chunk(msg):
    pcm = np.frombuffer(msg["pcm"], dtype=np.int16).astype(np.float32) / 32768.0
    chunks.put((pcm, int(msg["sr"])))


@socketio.on("summarize_now")
def on_summarize_now():
    socketio.start_background_task(run_summary, True)


@socketio.on("clear")
def on_clear():
    with state_lock:
        state["captions"].clear()
        state["summarized_upto"] = 0
    socketio.emit("cleared")


def asr_worker():
    while True:
        audio, sr = chunks.get()
        start = time.time()
        try:
            text = asr(audio, sr)
        except Exception as e:  # keep the demo alive on a bad chunk
            socketio.emit("status", {"error": f"ASR failed: {e}"})
            continue
        latency = time.time() - start
        rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
        print(f"chunk {len(audio) / sr:.1f}s @ {sr} Hz, level {rms:.4f} -> {latency:.2f}s: {text or '(silence/empty)'}",
              flush=True)
        if text:
            with state_lock:
                state["captions"].append(text)
        socketio.emit("caption", {"text": text, "latency": round(latency, 2),
                                  "audio_s": round(len(audio) / sr, 1), "queued": chunks.qsize()})


def run_summary(force=False):
    if not summary_lock.acquire(blocking=False):
        return  # one summary at a time
    try:
        with state_lock:
            n = len(state["captions"])
            stale = n > state["summarized_upto"]
        if not n or not (stale or force):
            return
        socketio.emit("summary_started")
        start = time.time()
        summary = summarizer(transcript())
        with state_lock:
            state["summarized_upto"] = n
        socketio.emit("summary", {"text": summary, "seconds": round(time.time() - start, 1),
                                  "at": time.strftime("%H:%M:%S")})
    except Exception as e:
        socketio.emit("status", {"error": f"Summary failed: {e}"})
    finally:
        summary_lock.release()


def summary_timer():
    while True:
        time.sleep(SUMMARY_EVERY_S)
        run_summary()


if __name__ == "__main__":
    threading.Thread(target=asr_worker, daemon=True).start()
    threading.Thread(target=summary_timer, daemon=True).start()
    print(f"open http://localhost:{PORT}")
    socketio.run(app, host="127.0.0.1", port=PORT, allow_unsafe_werkzeug=True)
