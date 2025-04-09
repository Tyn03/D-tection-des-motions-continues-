import numpy as np
import cv2
import tensorflow as tf
import json
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib import dates as mdates
from PIL import Image
import io
import tempfile
import os
import time
import threading
import queue

# === 1. Setup ===
TIMEOUT = 30
FRAME_SKIP = 3
LOG_INTERVAL = 5
MAX_LOG_LENGTH = 300

emotion_map = {
    "Happy":     {"valence": 0.9,  "arousal": 0.7, "emoji": "😄"},
    "Neutral":   {"valence": 0.0,  "arousal": 0.0, "emoji": "😐"},
    "Sad":       {"valence": -0.7, "arousal": -0.4, "emoji": "😢"},
    "Anger":     {"valence": -0.8, "arousal": 0.6, "emoji": "😠"},
    "Surprise":  {"valence": 0.4,  "arousal": 0.9, "emoji": "😲"},
    "Relaxed":   {"valence": 0.5,  "arousal": -0.5, "emoji": "😌"},
    "Contempt":  {"valence": -0.9, "arousal": -0.3, "emoji": "😒"},
    "Fear":      {"valence": -0.4, "arousal": 0.9, "emoji": "😨"}
}

labels = ['Surprise', 'Neutral', 'Anger', 'Happy', 'Sad', 'Relaxed', 'Contempt', 'Fear']

face_detection = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = tf.keras.models.load_model('network-5Labels.h5', compile=False)
settings = {'scaleFactor': 1.3, 'minNeighbors': 5, 'minSize': (50, 50)}

# === 2. Circumplex chart ===
def plot_circumplex(valence, arousal):
    fig, ax = plt.subplots(figsize=(2.75, 2.75))
    plt.style.use("dark_background")
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1)
    emotions = {
        "Exciting": (0, 1), "Surprise": (0.5, 0.9), "Happy": (0.8, 0.5),
        "Relaxed": (0.5, -0.5), "Sad": (-0.6, -0.6), "Contempt": (-0.9, -0.3),
        "Disgust": (-0.8, 0.3), "Angry": (-0.7, 0.6), "Fear": (-0.4, 0.9)
    }
    for emo, (x, y) in emotions.items():
        ax.text(x, y, emo, fontsize=8, ha='center', va='center', color='violet')
    ax.plot(valence, arousal, 'yo', markersize=8)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_title("Emotion Circumplex", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    return fig

# === 3. Render timeline chart as image ===
def render_emotion_plot_as_image(emotion_log):
    if len(emotion_log) < 2:
        return None
    timestamps = [datetime.fromisoformat(e["timestamp"]) for e in emotion_log[-100:]]
    valences = [e["valence"] for e in emotion_log[-100:]]
    arousals = [e["arousal"] for e in emotion_log[-100:]]
    intensities = [abs(v) + abs(a) for v, a in zip(valences, arousals)]

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(timestamps, valences, label="Valence", color='orange', marker='o', markersize=3)
    ax.plot(timestamps, arousals, label="Arousal", color='dodgerblue', marker='o', markersize=3)
    #ax.plot(timestamps, intensities, label="Intensity", color='yellow', marker='o', markersize=3)
    ax.set_ylim(-1.1, 1.1)
    ax.set_ylabel("Value")
    ax.set_xlabel("Time")
    ax.set_title("Emotion Dynamics Over Time")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=45)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.tight_layout(pad=1.0)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img

# === 4. Threaded prediction ===
def emotion_predictor_worker(q_in, q_out):
    while True:
        frame, gray = q_in.get()
        if frame is None:
            break
        detected = face_detection.detectMultiScale(gray, **settings)
        logs = []
        for x, y, w, h in detected:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48)) / 255.0
            prediction = model.predict(np.array([face.reshape((48, 48, 1))]), verbose=0)
            label = labels[np.argmax(prediction)]
            conf = float(np.max(prediction))
            logs.append({
                "label": label,
                "conf": conf,
                "valence": emotion_map[label]['valence'],
                "arousal": emotion_map[label]['arousal'],
                "emoji": emotion_map[label]['emoji'],
                "rect": (x, y, w, h)
            })
        q_out.put((frame, logs))

# === 5. Main streaming ===
def process_stream(cap, key_prefix="default"):
    emotion_log = []
    layout_col1, layout_col2 = st.columns([1.5, 1])
    cam_box = layout_col1.container()
    frame_placeholder = cam_box.empty()
    status_placeholder = cam_box.empty()
    timeline_placeholder = cam_box.empty()
    right_block = layout_col2.container()
    circumplex_placeholder = right_block.empty()
    stop_button = right_block.button("🔴 Stop", key=f"{key_prefix}_stop_button")
    stop_state_key = f"{key_prefix}_stop"

    if stop_state_key not in st.session_state:
        st.session_state[stop_state_key] = False
    if stop_button:
        st.session_state[stop_state_key] = True
        st.session_state[f"{key_prefix}_log"] = emotion_log
        st.rerun()

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 15)
    q_in, q_out = queue.Queue(), queue.Queue()
    threading.Thread(target=emotion_predictor_worker, args=(q_in, q_out), daemon=True).start()

    frame_count = 0
    last_save_time = time.time()
    start_time = time.time()

    while cap.isOpened():
        

        ret, img = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        frame_count += 1
        if frame_count % FRAME_SKIP == 0:
            q_in.put((img.copy(), gray))

        if not q_out.empty():
            frame, results = q_out.get()
            for res in results:
                x, y, w, h = res["rect"]
                label = res["label"]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (245, 135, 66), 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", width=460)

            for res in results:
                log = {
                    "timestamp": datetime.now().isoformat(),
                    "emotion": res['label'],
                    "confidence": res['conf'],
                    "valence": res['valence'],
                    "arousal": res['arousal']
                }
                emotion_log.append(log)
                if len(emotion_log) > MAX_LOG_LENGTH:
                    emotion_log = emotion_log[-MAX_LOG_LENGTH:]

                status_placeholder.markdown(f"""
<div style='font-size: 36px; font-weight: bold; text-align: center;'>
    {res['emoji']} Current Emotion: <span style='color: #FDE68A;'>{res['label']}</span>
</div>
""", unsafe_allow_html=True)

                circumplex_placeholder.pyplot(plot_circumplex(res['valence'], res['arousal']))

        now = time.time()
        if now - last_save_time > LOG_INTERVAL and emotion_log:
            try:
                with open("emotion_log.json", "w", encoding='utf-8') as f:
                    json.dump(emotion_log, f, indent=4, ensure_ascii=False)
                timeline_img = render_emotion_plot_as_image(emotion_log)
                if timeline_img:
                    timeline_placeholder.image(timeline_img)
            except:
                pass
            last_save_time = now

    q_in.put((None, None))
    cap.release()
    cv2.destroyAllWindows()
    st.session_state[stop_state_key] = False

    if emotion_log:
        st.success("✅ Analysis complete.")
    else:
        st.warning("⚠️ No face detected during the analysis.")

# === 6. Streamlit UI ===
st.set_page_config(page_title="Facial Emotion Analysis", layout="wide")
st.title("🎭 Facial Emotion Analysis App")

mode = st.radio("Choose input method:", ["Use Webcam", "Upload Video File (.mp4)"], key="mode_select")

if mode == "Use Webcam":
    if st.button("Start Webcam Analysis", key="start_webcam"):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        process_stream(cap, key_prefix="webcam")

elif mode == "Upload Video File (.mp4)":
    video_file = st.file_uploader("Upload your video", type=["mp4"], key="upload_file")
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        st.video(tfile.name)
        if st.button("Analyze Uploaded Video", key="start_video_analysis"):
            cap = cv2.VideoCapture(tfile.name)
            process_stream(cap, key_prefix="video")
