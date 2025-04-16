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
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
import os
os.environ["STREAMLIT_WATCH_FILE_SYSTEM"] = "false"
import math
import mediapipe as mp

# === Setup ===
TIMEOUT = 30
FRAME_SKIP = 3
LOG_INTERVAL = 5
MAX_LOG_LENGTH = 300
SAVE_INTERVAL = 10  # Save data every 10 seconds
DETECT_EVERY_N_FRAMES = 2  # Chỉ dự đoán mỗi n frame để giảm tải cho CPU

# Khởi tạo MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Khởi tạo biến toàn cục cho theo dõi chuyển động
prev_head_pose = None
prev_shoulders_pos = None
agitation_window = []

# Định nghĩa lại vị trí các cảm xúc trên không gian valence-arousal
# Sử dụng các giá trị đã được nghiên cứu kỹ lưỡng hơn
emotion_map = {
    # Vùng phải trên - cảm xúc tích cực, kích thích cao
    "Happy":     {"valence": 0.8, "arousal": 0.4, "emoji": "😄"},      # Hạnh phúc: tích cực, phấn khích vừa phải
    "Surprise":  {"valence": 0.4, "arousal": 0.8, "emoji": "😲"},      # Ngạc nhiên: tích cực vừa phải, kích thích cao
    "Excited":   {"valence": 0.7, "arousal": 0.7, "emoji": "🤩"},      # Hào hứng: tích cực cao, kích thích cao
    
    # Vùng trái trên - cảm xúc tiêu cực, kích thích cao
    "Angry":     {"valence": -0.7, "arousal": 0.6, "emoji": "😠"},     # Giận dữ: tiêu cực cao, kích thích cao
    "Fear":      {"valence": -0.5, "arousal": 0.7, "emoji": "😨"},     # Sợ hãi: tiêu cực, kích thích cao
    "Disgust":   {"valence": -0.6, "arousal": 0.3, "emoji": "🤢"},     # Ghê tởm: tiêu cực, kích thích vừa phải
    
    # Vùng trái dưới - cảm xúc tiêu cực, kích thích thấp
    "Sad":       {"valence": -0.6, "arousal": -0.3, "emoji": "😢"},    # Buồn bã: tiêu cực, kích thích thấp
    "Depressed": {"valence": -0.7, "arousal": -0.7, "emoji": "😞"},    # Trầm cảm: tiêu cực cao, kích thích rất thấp
    "Bored":     {"valence": -0.3, "arousal": -0.7, "emoji": "😒"},    # Chán nản: tiêu cực vừa phải, kích thích thấp
    
    # Vùng phải dưới - cảm xúc tích cực, kích thích thấp
    "Relaxed":   {"valence": 0.6, "arousal": -0.5, "emoji": "😌"},     # Thư giãn: tích cực, kích thích thấp
    "Sleepy":    {"valence": 0.3, "arousal": -0.7, "emoji": "😴"},     # Buồn ngủ: tích cực vừa phải, kích thích rất thấp
    "Calm":      {"valence": 0.4, "arousal": -0.4, "emoji": "😊"},     # Bình tĩnh: tích cực vừa phải, kích thích thấp
    
    # Các vị trí trung tính và bổ sung
    "Neutral":   {"valence": 0.0, "arousal": 0.0, "emoji": "😐"},      # Trung tính: không tích cực/tiêu cực, kích thích trung bình
    "Tired":     {"valence": -0.2, "arousal": -0.6, "emoji": "😩"}     # Mệt mỏi: hơi tiêu cực, kích thích thấp
}

# Bản đồ tên cảm xúc từ mô hình để khớp với emotion_map
emotion_label_map = {
    "Happy": "Happy",
    "Sad": "Sad",
    "Angry": "Angry", 
    "Fear": "Fear",
    "Surprise": "Surprise",
    "Disgust": "Disgust",
    "Neutral": "Neutral"
}

labels = list(emotion_map.keys())
face_detection = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = tf.keras.models.load_model('network-5Labels.h5', compile=False)
settings = {'scaleFactor': 1.3, 'minNeighbors': 5, 'minSize': (50, 50)}

# === Heatmap Plot ===




from matplotlib.colors import ListedColormap

from matplotlib.colors import ListedColormap

def plot_circumplex(emotion_log):
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='black')
    plt.style.use("dark_background")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axhline(0, color='blue')
    ax.axvline(0, color='blue')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.set_title("Emotion Heatmap in Circumplex", color='white')
    ax.add_artist(plt.Circle((0, 0), 1.0, color='white', fill=False, lw=1))

    # Định nghĩa lại vị trí các cảm xúc trên không gian valence-arousal
    directions = {
        # Phần tư thứ nhất (trên phải)
        "Happy":    (0.8, 0.5),    # Vị trí hợp lý hơn cho Happy
        "Excited":  (0.7, 0.7),    # Cảm xúc phấn khích
        "Surprise": (0.5, 0.8),    # Bất ngờ (trên phải)
        
        # Phần tư thứ hai (trên trái)
        "Angry":    (-0.7, 0.7),   # Giận dữ
        "Fear":     (-0.5, 0.8),   # Sợ hãi
        "Disgust":  (-0.8, 0.3),   # Ghê tởm (trên trái)
        
        # Phần tư thứ ba (dưới trái)
        "Sad":      (-0.8, -0.6),  # Buồn bã 
        "Depressed":(-0.7, -0.8),  # Trầm cảm
        "Negative": (-0.9, -0.2),  # Tiêu cực (dưới trái)
        
        # Phần tư thứ tư (dưới phải)
        "Relaxed":  (0.7, -0.7),   # Thư giãn
        "Sleepy":   (0.5, -0.8),   # Buồn ngủ
        "Tired":    (0.3, -0.9),   # Mệt mỏi
        
        # Các vị trí trên trục
        "Positive": (0.9, 0.0),    # Tích cực (phải)
        "Neutral":  (0.0, 0.0),    # Trung tính (giữa)
        "Calming":  (0.0, -0.9),   # Bình tĩnh (dưới)
        "Exciting": (0.0, 0.9),    # Hào hứng (trên)
        
        # Các cảm xúc khác
        "Bored":    (-0.3, -0.9)   # Chán nản (dưới)
    }
    
    for label, (x, y) in directions.items():
        color = 'white'
        if "Negative" in label or "Sad" in label or "Angry" in label or "Fear" in label or "Disgust" in label:
            color = '#FF6B6B'
        elif "Positive" in label or "Happy" in label:
            color = '#90EE90'
        elif "Exciting" in label or "Excited" in label or "Surprise" in label:
            color = '#FFA07A'
        elif "Calming" in label or "Relaxed" in label or "Sleepy" in label:
            color = '#87CEFA'
        ax.text(x, y, label, ha='center', va='center', fontsize=9, color=color, fontweight='bold')

    if len(emotion_log) >= 2:
        # Tạo lưới (grid) để đếm số lần đi qua
        grid_size = 200 # Tăng độ phân giải để đường đi chính xác hơn
        visit_count = np.zeros((grid_size, grid_size))
        color_mask = np.zeros((grid_size, grid_size, 4)) # RGBA, mặc định trong suốt

        # Duyệt qua từng đoạn đường đi
        for i in range(1, len(emotion_log)):
            x0, y0 = emotion_log[i - 1]["valence"], emotion_log[i - 1]["arousal"]
            x1, y1 = emotion_log[i]["valence"], emotion_log[i]["arousal"]
            
            # Tính số điểm nội suy dựa trên khoảng cách
            distance = np.sqrt((x1-x0)**2 + (y1-y0)**2)
            steps = max(5, int(distance * 150)) # Nhiều điểm nội suy hơn để mịn
            
            # Nội suy và cập nhật visit_count theo kiểu "bút tô màu"
            for alpha in np.linspace(0, 1, steps):
                x = x0 * (1 - alpha) + x1 * alpha
                y = y0 * (1 - alpha) + y1 * alpha
                
                # Ánh xạ vào grid
                grid_x = int((x + 1) / 2 * (grid_size - 1))
                grid_y = int((y + 1) / 2 * (grid_size - 1))
                grid_x = np.clip(grid_x, 0, grid_size - 1)
                grid_y = np.clip(grid_y, 0, grid_size - 1)
                
                # Tăng visit_count cho ô chính
                visit_count[grid_y, grid_x] += 1.0

                # Làm dày đường vẽ một cách có kiểm soát (chỉ các ô liền kề)
                thickness_radius = 1 # Chỉ tô màu các ô ngay sát cạnh
                for dx in [-thickness_radius, 0, thickness_radius]:
                    for dy in [-thickness_radius, 0, thickness_radius]:
                        if dx == 0 and dy == 0: continue # Bỏ qua ô trung tâm
                        nx, ny = grid_x + dx, grid_y + dy
                        if 0 <= nx < grid_size and 0 <= ny < grid_size:
                            # Thêm một phần nhỏ vào ô lân cận để tạo độ dày - Giảm xuống
                            visit_count[ny, nx] += 0.15 # Giảm từ 0.3 xuống 0.15

        # Chuẩn hóa và tạo màu
        max_count = np.max(visit_count)
        if max_count > 0:
            # Ngưỡng đỏ tương đối - Giảm xuống 15% của max_count để nhạy hơn
            red_threshold = 0.15 * max_count + 1e-6 # Thêm epsilon nhỏ để tránh chia cho 0
            
            # Hệ số để đảm bảo màu đạt đỉnh
            green_intensity_factor = 1.0 / red_threshold
            red_intensity_factor = 1.0 / max(1e-6, max_count - red_threshold)

            for i in range(grid_size):
                for j in range(grid_size):
                    count = visit_count[j, i] # visit_count dùng [y, x]
                    if count > 0:
                        real_x = i / (grid_size - 1) * 2 - 1
                        real_y = j / (grid_size - 1) * 2 - 1
                        
                        if real_x**2 + real_y**2 <= 1:
                            # Tính toán màu sắc
                            if count > red_threshold:
                                # Chuyển sang đỏ
                                red_ratio = min(1.0, (count - red_threshold) * red_intensity_factor)
                                r = 0.3 + 0.6 * red_ratio # Đỏ tăng từ 0.3 đến 0.9
                                g = 0.7 * (1 - red_ratio) # Xanh lá giảm từ 0.7 xuống 0
                                b = 0.1
                            else:
                                # Giữ màu xanh lá
                                green_ratio = count * green_intensity_factor
                                r = 0.1 + 0.2 * green_ratio # Đỏ thấp
                                g = 0.3 + 0.6 * green_ratio # Xanh lá tăng từ 0.3 đến 0.9
                                b = 0.1
                            
                            # Tính alpha (độ trong suốt) - CỐ ĐỊNH để cường độ màu đồng đều
                            alpha = 0.85 # Đặt alpha cố định cho mọi điểm được vẽ
                            
                            color_mask[j, i] = [r, g, b, alpha]
            
            # Làm mượt heatmap bằng Gaussian Filter - Giảm sigma
            sigma_blur = 1.2 # Giảm từ 1.5 xuống 1.2
            for c in range(4):
                color_mask[:, :, c] = gaussian_filter(color_mask[:, :, c], sigma=sigma_blur)
            
            # Hiển thị heatmap
            ax.imshow(color_mask, extent=[-1, 1, -1, 1], origin='lower', interpolation='bilinear')

    # Vẽ vị trí hiện tại (điểm màu vàng)
    if len(emotion_log) > 0:
        last_entry = emotion_log[-1]
        last_x, last_y = last_entry["valence"], last_entry["arousal"]
        ax.plot(last_x, last_y, 'yo', markersize=8) # Giảm kích thước điểm vàng một chút
        ax.text(0, -1.15, f"Valence: {last_x:.2f}   Arousal: {last_y:.2f}",
                ha='center', va='top', fontsize=10, color='white', fontweight='bold')
    else:
        # Khi chưa có dữ liệu, chỉ hiển thị text chờ
        ax.text(0, 0, "Waiting for data...", ha='center', va='center', color='gray', fontsize=10)
        ax.text(0, -1.15, f"Valence: 0.00   Arousal: 0.00",
                ha='center', va='top', fontsize=10, color='white', fontweight='bold')

    return fig

# === Emotion Data Management ===
def load_emotion_data():
    try:
        with open('emotion_log.json', 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"emotions": {}, "metadata": {"last_updated": datetime.now().isoformat(), "version": "1.0"}}

def save_emotion_data(emotion_data):
    with open('emotion_log.json', 'w') as f:
        json.dump(emotion_data, f, indent=4)

def update_emotion_stats(emotion_log, current_emotion):
    emotion_data = load_emotion_data()
    
    if current_emotion not in emotion_data["emotions"]:
        emotion_data["emotions"][current_emotion] = {
            "count": 0,
            "duration": 0,
            "last_detected": None,
            "average_valence": 0,
            "average_arousal": 0
        }
    
    stats = emotion_data["emotions"][current_emotion]
    stats["count"] += 1
    stats["last_detected"] = datetime.now().isoformat()
    
    # Update averages
    if len(emotion_log) > 0:
        current_entry = emotion_log[-1]
        stats["average_valence"] = (stats["average_valence"] * (stats["count"] - 1) + current_entry["valence"]) / stats["count"]
        stats["average_arousal"] = (stats["average_arousal"] * (stats["count"] - 1) + current_entry["arousal"]) / stats["count"]
    
    emotion_data["metadata"]["last_updated"] = datetime.now().isoformat()
    save_emotion_data(emotion_data)

# === Real-time Video Capture and Emotion Detection ===
def process_stream(cap, custom_frame_skip):
    global prev_head_pose, prev_shoulders_pos, agitation_window
    
    # Reset hoàn toàn trạng thái
    if 'emotion_log' in st.session_state:
        del st.session_state.emotion_log
    if 'smooth_points' in st.session_state:
        del st.session_state.smooth_points
    
    # Khởi tạo lại từ đầu
    st.session_state.emotion_log = []
    st.session_state.smooth_points = []
    st.session_state.last_save_time = time.time()
    st.session_state.prev_emotion = None
    st.session_state.first_detect = True

    emotion_log = st.session_state.emotion_log
    col1, col2 = st.columns([3, 2])
    frame_placeholder = col1.empty()
    chart_placeholder = col2.empty()
    status_placeholder = st.empty()
    stop_btn = st.button("\U0001F534 Stop")

    # Khởi tạo biểu đồ trống
    chart_placeholder.pyplot(plot_circumplex([]))

    # Tối ưu xử lý video
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    
    # Tạo worker threads
    frame_count = 0
    q_in, q_out = queue.Queue(maxsize=2), queue.Queue(maxsize=2)
    
    NUM_WORKERS = 2
    workers = []
    for _ in range(NUM_WORKERS):
        worker = threading.Thread(target=emotion_predictor_worker, args=(q_in, q_out), daemon=True)
        worker.start()
        workers.append(worker)
    
    last_chart_time = time.time()
    last_frame_time = time.time()
    fps_counter = 0
    fps = 0

    def update_fps():
        nonlocal fps_counter, fps, last_frame_time
        fps_counter += 1
        if time.time() - last_frame_time >= 1.0:
            fps = fps_counter
            fps_counter = 0
            last_frame_time = time.time()
            
    frame_buffer = None
    results_buffer = []
    
    # Khởi tạo MediaPipe Holistic
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        try:
            while cap.isOpened():
                if stop_btn:
                    if emotion_log:
                        update_emotion_stats(emotion_log, emotion_log[-1]["emotion"])
                        break

                ret, img = cap.read()
                if not ret:
                    break
                        
                update_fps()
                
                if img.shape[1] > 640:
                    scale_factor = 640 / img.shape[1]
                    img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                frame_count += 1
                
                # Xử lý với MediaPipe
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = holistic.process(img_rgb)
                
                # Tính điểm kích động
                agitation_score = calculate_agitation_score(
                    results,
                    prev_head_pose,
                    prev_shoulders_pos,
                    agitation_window
                )
                
                # Cập nhật vị trí trước đó
                if results.pose_landmarks:
                    prev_head_pose = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
                    left_shoulder = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
                    right_shoulder = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
                    prev_shoulders_pos = (
                        (left_shoulder.x + right_shoulder.x) / 2,
                        (left_shoulder.y + right_shoulder.y) / 2
                    )
                
                # Vẽ landmarks
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        img,
                        results.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                
                # Hiển thị điểm kích động và thông tin debug
                debug_text = f"Agitation: {agitation_score:.2f}"
                if results.pose_landmarks:
                    debug_text += f" | Landmarks: {len(results.pose_landmarks.landmark)}"
                else:
                    debug_text += " | No landmarks detected"
                
                cv2.putText(img, debug_text, 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Chỉ gửi frame để dự đoán mỗi DETECT_EVERY_N_FRAMES
                if frame_count % (custom_frame_skip * DETECT_EVERY_N_FRAMES) == 0:
                    try:
                        q_in.put((img.copy(), gray), block=False)
                    except queue.Full:
                        pass

                # Kiểm tra và xử lý kết quả
                frame_to_show = img.copy()
                
                cv2.putText(frame_to_show, f"FPS: {fps}", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                if not q_out.empty():
                    frame, results = q_out.get()
                    frame_buffer = frame.copy()
                    results_buffer = results
                    
                    for res in results:
                        x, y, w, h = res['rect']
                        label = res['label']
                        val = res['valence']
                        aro = res['arousal']
                        conf = res['conf']
                        emoji = res['emoji']
                        
                        cv2.rectangle(frame_to_show, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.rectangle(frame_to_show, (x, y-30), (x+w, y), (0, 0, 0, 0.7), -1)
                        label_text = f"{label} {emoji}"
                        cv2.putText(frame_to_show, label_text, (x+5, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Thêm điểm kích động vào log
                        current_time = datetime.now().isoformat()
                        emotion_log.append({
                            "timestamp": current_time,
                            "emotion": label,
                            "confidence": conf,
                            "valence": val,
                            "arousal": aro,
                            "agitation": agitation_score
                        })
                        
                        if len(emotion_log) > MAX_LOG_LENGTH:
                            del emotion_log[0]
                
                # Hiển thị frame
                frame_placeholder.image(cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB), channels="RGB")
                
                # Cập nhật biểu đồ
                if time.time() - last_chart_time > 1.5:
                    if emotion_log:
                        chart_placeholder.pyplot(plot_circumplex(emotion_log))
                    last_chart_time = time.time()
                    
                status_placeholder.text(f"Processing... FPS: {fps}")
                time.sleep(0.001)
                
        finally:
            # Dọn dẹp tài nguyên
            for _ in range(NUM_WORKERS):
                try:
                    q_in.put((None, None), block=False)
                except queue.Full:
                    pass
            
            for worker in workers:
                worker.join(timeout=1.0)

            cap.release()
            cv2.destroyAllWindows()
            status_placeholder.text("Processing completed")

# === Threaded Prediction Worker ===
def emotion_predictor_worker(q_in, q_out):
    while True:
        try:
            frame, gray = q_in.get(timeout=3.0)
            if frame is None:
                break

            is_first_detection = 'first_detect' in st.session_state and st.session_state.first_detect

            scale_factor = 0.5
            small_gray = cv2.resize(gray, (0, 0), fx=scale_factor, fy=scale_factor)
            small_gray = cv2.equalizeHist(small_gray)

            if is_first_detection:
                scaleFactor = 1.08
                minNeighbors = 2
                minSize = (20, 20)
            else:
                scaleFactor = 1.1
                minNeighbors = 3
                minSize = (25, 25)

            faces = face_detection.detectMultiScale(
                small_gray, 
                scaleFactor=scaleFactor,
                minNeighbors=minNeighbors,
                minSize=minSize
            )

            logs = []
            for (x, y, w, h) in faces:
                x, y, w, h = int(x/scale_factor), int(y/scale_factor), int(w/scale_factor), int(h/scale_factor)

                padding = int(0.05 * w)
                x_padded = max(0, x - padding)
                y_padded = max(0, y - padding)
                w_padded = min(gray.shape[1] - x_padded, w + 2*padding)
                h_padded = min(gray.shape[0] - y_padded, h + 2*padding)

                face = gray[y_padded:y_padded+h_padded, x_padded:x_padded+w_padded]

                if face.size == 0 or face.shape[0] == 0 or face.shape[1] == 0:
                    continue

                try:
                    face_resized = cv2.resize(face, (48, 48))
                    face_normalized = face_resized / 255.0

                    prediction = model.predict(np.array([face_normalized.reshape((48, 48, 1))]), verbose=0)
                    label = labels[np.argmax(prediction)]
                    confidence = float(np.max(prediction))

                    if confidence > 0.3:
                        if is_first_detection:
                            st.session_state.first_detect = False

                        # Phương pháp tính valence và arousal
                        # Chọn tất cả cảm xúc có xác suất trên ngưỡng để có sự pha trộn tốt hơn
                        threshold = 0.1  # Chỉ xét cảm xúc có xác suất từ 10% trở lên
                        significant_emotions = [(i, prob) for i, prob in enumerate(prediction[0]) if prob >= threshold]
                        
                        # Nếu không có cảm xúc nào vượt ngưỡng, lấy cảm xúc có xác suất cao nhất
                        if not significant_emotions:
                            significant_emotions = [(np.argmax(prediction[0]), np.max(prediction[0]))]
                        
                        # Chuẩn hóa lại trọng số để tổng bằng 1
                        total_weight = sum(prob for _, prob in significant_emotions)
                        normalized_emotions = [(i, prob/total_weight) for i, prob in significant_emotions]
                        
                        # Tính valence và arousal dựa trên vị trí trung bình có trọng số
                        valence = 0
                        arousal = 0
                        intensity = sum(prob for _, prob in normalized_emotions)
                        
                        for i, weight in normalized_emotions:
                            emotion_name = labels[i]
                            # Ánh xạ tên cảm xúc từ model sang emotion_map
                            mapped_emotion = emotion_label_map.get(emotion_name, "Neutral")
                            
                            # Lấy giá trị valence và arousal từ bản đồ cảm xúc - giữ nguyên giá trị
                            valence += emotion_map[mapped_emotion]['valence'] * weight
                            arousal += emotion_map[mapped_emotion]['arousal'] * weight
                        
                        # Thêm nhiễu nhỏ để tạo biến động nhẹ
                        noise_factor = 0.02  # Giảm nhiễu xuống để không làm méo mó vị trí qua mỗi khung hình
                        valence += np.random.normal(0, noise_factor)
                        arousal += np.random.normal(0, noise_factor)
                        
                        # Chỉ giới hạn trong khoảng [-1, 1] để tránh vượt quá giới hạn của biểu đồ
                        # Không áp dụng hệ số scaling nào - giữ nguyên vị trí theo tính toán
                        valence = np.clip(valence, -1, 1)
                        arousal = np.clip(arousal, -1, 1)
                        
                        # Làm tròn đến 2 chữ số thập phân
                        valence = round(valence, 2)
                        arousal = round(arousal, 2)

                        logs.append({
                            "label": label,
                            "conf": confidence,
                            "valence": valence,
                            "arousal": arousal,
                            "intensity": intensity,
                            "emoji": emotion_map[label]['emoji'],
                            "rect": (x, y, w, h)
                        })
                except Exception:
                    continue

            try:
                q_out.put((frame, logs), block=False)
            except queue.Full:
                pass

        except queue.Empty:
            continue
        except Exception:
            continue

def calculate_agitation_score(landmarks, prev_head_pose=None, prev_shoulders_pos=None, agitation_window=None):
    """
    Tính toán điểm kích động dựa trên chuyển động của đầu và vai sử dụng MediaPipe Holistic
    """
    if agitation_window is None:
        agitation_window = []
        
    if landmarks.pose_landmarks:
        # Lấy các điểm mốc quan trọng
        head_pose = landmarks.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
        left_shoulder = landmarks.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        
        # Tính vị trí trung bình của vai
        shoulders_pos = (
            (left_shoulder.x + right_shoulder.x) / 2,
            (left_shoulder.y + right_shoulder.y) / 2
        )
        
        # Tính chuyển động của đầu
        if prev_head_pose is not None:
            head_movement = math.sqrt(
                (head_pose.x - prev_head_pose.x) ** 2 +
                (head_pose.y - prev_head_pose.y) ** 2 +
                (head_pose.z - prev_head_pose.z) ** 2
            )
        else:
            head_movement = 0
            
        # Tính chuyển động của vai
        if prev_shoulders_pos is not None:
            shoulder_movement = math.sqrt(
                (shoulders_pos[0] - prev_shoulders_pos[0]) ** 2 +
                (shoulders_pos[1] - prev_shoulders_pos[1]) ** 2
            )
        else:
            shoulder_movement = 0
            
        # Tính điểm kích động dựa trên chuyển động
        # Tăng độ nhạy với chuyển động đầu (nhân 2)
        agitation_score = (head_movement * 2 + shoulder_movement) / 3
        
        # Thêm điểm vào cửa sổ
        agitation_window.append(agitation_score)
        
        # Giới hạn kích thước cửa sổ (30 frames)
        if len(agitation_window) > 30:
            agitation_window.pop(0)
            
        # Tính điểm kích động trung bình
        avg_agitation = sum(agitation_window) / len(agitation_window)
        
        # Chuẩn hóa về khoảng [0, 1]
        return min(max(avg_agitation, 0), 1)
        
    return 0

# === Streamlit UI ===
st.set_page_config(page_title="Facial Emotion Heatmap", layout="wide")
st.title("🎭 Real-time Facial Emotion Recognition with Heatmap")

mode = st.radio("Select Mode:", ["Use Webcam", "Upload Video", "Test Video"])

if mode == "Use Webcam":
    if st.button("Start Webcam Analysis"):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        process_stream(cap, FRAME_SKIP)

elif mode == "Upload Video":
    video_file = st.file_uploader("Upload your .mp4 video", type=["mp4"])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        st.video(tfile.name)
        if st.button("Analyze Uploaded Video"):
            cap = cv2.VideoCapture(tfile.name)
            process_stream(cap, FRAME_SKIP)

elif mode == "Test Video":
    st.subheader("Test Video Settings")
    
    # Video selection
    test_videos = ["test_video1.mp4", "test_video2.mp4", "emotions_test.mp4"]
    
    # Kiểm tra xem video nào tồn tại
    existing_videos = []
    for video in test_videos:
        if os.path.exists(video):
            existing_videos.append(video)
    
    # Nếu không tìm thấy video nào, hiển thị thông báo
    if not existing_videos:
        st.warning("No test videos found. Please place test videos in the same directory.")
        
        # Cung cấp tùy chọn để tải lên video test
        custom_test = st.file_uploader("Upload a test video", type=["mp4"])
        if custom_test:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(custom_test.read())
            custom_path = tfile.name
            st.success(f"Test video uploaded successfully: {os.path.basename(custom_path)}")
            existing_videos = [custom_path]
    
    if existing_videos:
        selected_video = st.selectbox("Select a test video", existing_videos)
        
        # Cài đặt phát video
        col1, col2 = st.columns(2)
        with col1:
            speed = st.slider("Playback Speed", min_value=0.25, max_value=2.0, value=1.0, step=0.25)
            frame_skip = st.slider("Frame Skip", min_value=1, max_value=10, value=3, step=1)
        
        with col2:
            path_thickness = st.slider("Path Thickness", min_value=0.3, max_value=0.6, value=0.45, step=0.05)
            smoothness = st.slider("Path Smoothness", min_value=1.0, max_value=2.0, value=1.4, step=0.1)
        
        # Hiển thị preview và nút chạy
        st.video(selected_video)
        
        if st.button("Run Test Analysis"):
            # Áp dụng các thiết lập từ người dùng
            # Sử dụng biến từ slider mà không cần biến global
            custom_frame_skip = frame_skip
            
            # Tiện ích để thiết lập các thông số cho path visualization
            st.session_state.path_thickness = path_thickness
            st.session_state.path_smoothness = smoothness
            
            # Mở và xử lý video
            cap = cv2.VideoCapture(selected_video)
            
            # Điều chỉnh tốc độ phát video
            if speed != 1.0:
                original_fps = cap.get(cv2.CAP_PROP_FPS)
                st.info(f"Original video FPS: {original_fps:.1f}, Adjusted FPS: {original_fps * speed:.1f}")
            
            # Truyền frame_skip dưới dạng tham số cho process_stream
            process_stream(cap, custom_frame_skip)