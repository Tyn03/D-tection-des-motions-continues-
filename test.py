import numpy as np
import cv2
import tensorflow as tf
import json
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
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
FRAME_SKIP = 1
LOG_INTERVAL = 5 
MAX_LOG_LENGTH = 300
SAVE_INTERVAL = 10  # Save data every 10 seconds
DETECT_EVERY_N_FRAMES = 1  # Only predict every n frames to reduce CPU load

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize global variables for movement tracking
prev_head_pose = None
prev_shoulders_pos = None
agitation_window = []

# Redefine emotion positions in valence-arousal space
# Using more thoroughly researched values
emotion_map = {
    # Top right quadrant - positive emotions, high arousal
    "Happy":     {"valence": 0.8, "arousal": 0.4, "emoji": "😄"},      # Happiness: positive, moderate excitement
    "Surprise":  {"valence": 0.4, "arousal": 0.8, "emoji": "😲"},      # Surprise: moderately positive, high excitement
    "Excited":   {"valence": 0.7, "arousal": 0.7, "emoji": "🤩"},      # Excitement: highly positive, high excitement
    
    # Top left quadrant - negative emotions, high arousal
    "Angry":     {"valence": -0.7, "arousal": 0.6, "emoji": "😠"},     # Anger: highly negative, high excitement
    "Fear":      {"valence": -0.5, "arousal": 0.7, "emoji": "😨"},     # Fear: negative, high excitement
    "Disgust":   {"valence": -0.6, "arousal": 0.3, "emoji": "🤢"},     # Disgust: negative, moderate excitement
    
    # Bottom left quadrant - negative emotions, low arousal
    "Sad":       {"valence": -0.6, "arousal": -0.3, "emoji": "😢"},    # Sadness: negative, low excitement
    "Depressed": {"valence": -0.7, "arousal": -0.7, "emoji": "😞"},    # Depression: highly negative, very low excitement
    "Bored":     {"valence": -0.3, "arousal": -0.7, "emoji": "😒"},    # Boredom: moderately negative, low excitement
    
    # Bottom right quadrant - positive emotions, low arousal
    "Relaxed":   {"valence": 0.6, "arousal": -0.5, "emoji": "😌"},     # Relaxation: positive, low excitement
    "Sleepy":    {"valence": 0.3, "arousal": -0.7, "emoji": "😴"},     # Sleepiness: moderately positive, very low excitement
    "Calm":      {"valence": 0.4, "arousal": -0.4, "emoji": "😊"},     # Calmness: moderately positive, low excitement
    
    # Neutral and additional positions
    "Neutral":   {"valence": 0.0, "arousal": 0.0, "emoji": "😐"},      # Neutral: neither positive/negative, moderate excitement
    "Tired":     {"valence": -0.2, "arousal": -0.6, "emoji": "😩"}     # Tiredness: slightly negative, low excitement
}

# Emotion name mapping from model to match emotion_map
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
def calculate_average_emotions(emotion_log):
    """
    Calculate average valence and arousal values for each second
    :param emotion_log: List of emotion entries
    :return: List of averaged emotion entries
    """
    if not emotion_log:
        return []
    
    # Group emotions by second
    emotions_by_second = {}
    for entry in emotion_log:
        timestamp = datetime.fromisoformat(entry["timestamp"])
        second_key = timestamp.replace(microsecond=0)
        
        if second_key not in emotions_by_second:
            emotions_by_second[second_key] = []
        emotions_by_second[second_key].append(entry)
    
    # Calculate averages for each second
    averaged_emotions = []
    for second_key, entries in emotions_by_second.items():
        if entries:
            avg_valence = sum(entry["valence"] for entry in entries) / len(entries)
            avg_arousal = sum(entry["arousal"] for entry in entries) / len(entries)
            
            # Use the emotion with highest confidence from the entries
            best_entry = max(entries, key=lambda x: x["confidence"])
            
            averaged_emotions.append({
                "timestamp": second_key.isoformat(),
                "emotion": best_entry["emotion"],
                "valence": round(avg_valence, 2),
                "arousal": round(avg_arousal, 2),
                "confidence": best_entry["confidence"],
                "agitation": 0
            })
    
    return averaged_emotions

def plot_circumplex(emotion_log):
    """
    Plot emotion heatmap in circumplex model
    :param emotion_log: List of emotion entries with valence and arousal values
    :return: matplotlib figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.ndimage import gaussian_filter

    # Calculate averaged emotions
    averaged_emotions = calculate_average_emotions(emotion_log)

    # Initialize heatmap memory in session_state if not exists
    grid_size = 100
    if 'heatmap_memory' not in st.session_state:
        # Store a dictionary at each grid cell to track emotions by second
        st.session_state.heatmap_memory = np.empty((grid_size, grid_size), dtype=object)
        for i in range(grid_size):
            for j in range(grid_size):
                st.session_state.heatmap_memory[i, j] = {}
        st.session_state.last_heatmap_points = []
    
    # Get reference to the heatmap memory
    emotion_counters = st.session_state.heatmap_memory
    
    # Find new points to add to the heatmap
    if averaged_emotions:
        new_points = []
        # If we have previous points, start from there
        if st.session_state.last_heatmap_points:
            last_idx = 0
            # Find where the new data starts
            for i, emotion in enumerate(averaged_emotions):
                if i < len(st.session_state.last_heatmap_points) and emotion["timestamp"] == st.session_state.last_heatmap_points[i]["timestamp"]:
                    last_idx = i + 1
                else:
                    break
            # Add only new points
            new_points = averaged_emotions[last_idx:]
        else:
            # First time, add all points
            new_points = averaged_emotions
        
        # Store current points for next comparison
        st.session_state.last_heatmap_points = averaged_emotions.copy()
        
        # Process new points with improved logic for continuity
        kernel_radius = int(0.05 * grid_size)  # Điều chỉnh bán kính lên 0.05 theo yêu cầu ban đầu
        
        for i in range(1, len(new_points)):
            x0, y0 = new_points[i - 1]["valence"], new_points[i - 1]["arousal"]
            x1, y1 = new_points[i]["valence"], new_points[i]["arousal"]
            emotion = new_points[i]["emotion"]  # Get current emotion
            timestamp = new_points[i]["timestamp"]  # Get timestamp
            
            # Extract second from timestamp
            second_key = timestamp.split('.')[0]  # Remove milliseconds
            
            # Calculate distance between points
            distance = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
            
            # Significantly increase steps for better interpolation, especially for diagonal movements
            steps = max(100, int(distance * 300))  # Increased from 30/200 to 50/300 for more precision
            
            # Pre-calculate all interpolated points with higher precision
            interp_points = []
            for alpha in np.linspace(0, 1, steps):
                x = x0 * (1 - alpha) + x1 * alpha
                y = y0 * (1 - alpha) + y1 * alpha
                
                # Convert coordinates to grid with floating point precision first
                # Only convert to integer at the very end to maintain precision
                grid_x_float = (x + 1) / 2 * (grid_size - 1)
                grid_y_float = (y + 1) / 2 * (grid_size - 1)
                
                grid_x = int(round(grid_x_float))  # Use round instead of int truncation
                grid_y = int(round(grid_y_float))  # Use round instead of int truncation
                
                interp_points.append((grid_x, grid_y))
            
            # Apply a modified Bresenham-inspired algorithm to ensure continuous paths
            # This adds extra points between any gaps in the interpolated path
            continuous_points = []
            for j in range(len(interp_points)):
                continuous_points.append(interp_points[j])
                
                # Add extra points between any gaps (except at the end of the list)
                if j < len(interp_points) - 1:
                    x_curr, y_curr = interp_points[j]
                    x_next, y_next = interp_points[j + 1]
                    
                    # Calculate Manhattan distance between consecutive points
                    manhattan_dist = abs(x_next - x_curr) + abs(y_next - y_curr)
                    
                    # If points aren't adjacent, fill in the gaps
                    if manhattan_dist > 1:
                        # Get intermediate points using Bresenham's line algorithm logic
                        dx = abs(x_next - x_curr)
                        dy = abs(y_next - y_curr)
                        sx = 1 if x_curr < x_next else -1
                        sy = 1 if y_curr < y_next else -1
                        err = dx - dy
                        
                        # Start from current point but don't include it (already added)
                        x, y = x_curr, y_curr
                        
                        while (x != x_next or y != y_next):
                            e2 = 2 * err
                            if e2 > -dy:
                                err -= dy
                                x += sx
                            if e2 < dx:
                                err += dx
                                y += sy
                            
                            # Add intermediate point but avoid duplicates
                            if (x, y) != interp_points[j] and (x, y) != interp_points[j + 1]:
                                continuous_points.append((x, y))
            
            # Apply circular kernel effect to each point in the continuous path
            for grid_x, grid_y in continuous_points:
                # Apply circular kernel effect with radius
                for dx in range(-kernel_radius, kernel_radius + 1):
                    for dy in range(-kernel_radius, kernel_radius + 1):
                        nx, ny = grid_x + dx, grid_y + dy
                        
                        # Check if within grid bounds and within circular kernel
                        if (0 <= nx < grid_size and 0 <= ny < grid_size and 
                            dx*dx + dy*dy <= kernel_radius*kernel_radius):
                            
                            # Calculate falloff based on distance from center
                            dist_sq = dx*dx + dy*dy
                            falloff = np.exp(-dist_sq / (2 * (kernel_radius/2)**2))
                            
                            # Add emotion to tracker dictionary - ensure it's a dictionary first
                            if emotion_counters[ny, nx] is None or not isinstance(emotion_counters[ny, nx], dict):
                                emotion_counters[ny, nx] = {}
                                
                            if emotion not in emotion_counters[ny, nx]:
                                emotion_counters[ny, nx][emotion] = set()
                            
                            # Add second to the set for this emotion
                            emotion_counters[ny, nx][emotion].add(second_key)
            
            # Also add the direct endpoints to ensure they're properly marked
            # Add starting point (previous point)
            grid_x0 = int(round((x0 + 1) / 2 * (grid_size - 1)))  # Use round for more precision
            grid_y0 = int(round((y0 + 1) / 2 * (grid_size - 1)))  # Use round for more precision
            if i > 1:  # Not the first point in the sequence
                prev_emotion = new_points[i - 1]["emotion"]
                prev_timestamp = new_points[i - 1]["timestamp"]
                prev_second_key = prev_timestamp.split('.')[0]
                
                for dx in range(-kernel_radius, kernel_radius + 1):
                    for dy in range(-kernel_radius, kernel_radius + 1):
                        nx, ny = grid_x0 + dx, grid_y0 + dy
                        
                        if (0 <= nx < grid_size and 0 <= ny < grid_size and 
                            dx*dx + dy*dy <= kernel_radius*kernel_radius):
                            
                            dist_sq = dx*dx + dy*dy
                            falloff = np.exp(-dist_sq / (2 * (kernel_radius/2)**2))
                            
                            if emotion_counters[ny, nx] is None or not isinstance(emotion_counters[ny, nx], dict):
                                emotion_counters[ny, nx] = {}
                                
                            if prev_emotion not in emotion_counters[ny, nx]:
                                emotion_counters[ny, nx][prev_emotion] = set()
                            
                            emotion_counters[ny, nx][prev_emotion].add(prev_second_key)
            
            # Add ending point (current point)
            grid_x1 = int(round((x1 + 1) / 2 * (grid_size - 1)))  # Use round for more precision
            grid_y1 = int(round((y1 + 1) / 2 * (grid_size - 1)))  # Use round for more precision
            
            for dx in range(-kernel_radius, kernel_radius + 1):
                for dy in range(-kernel_radius, kernel_radius + 1):
                    nx, ny = grid_x1 + dx, grid_y1 + dy
                    
                    if (0 <= nx < grid_size and 0 <= ny < grid_size and 
                        dx*dx + dy*dy <= kernel_radius*kernel_radius):
                        
                        dist_sq = dx*dx + dy*dy
                        falloff = np.exp(-dist_sq / (2 * (kernel_radius/2)**2))
                        
                        if emotion_counters[ny, nx] is None or not isinstance(emotion_counters[ny, nx], dict):
                            emotion_counters[ny, nx] = {}
                            
                        if emotion not in emotion_counters[ny, nx]:
                            emotion_counters[ny, nx][emotion] = set()
                        
                        emotion_counters[ny, nx][emotion].add(second_key)

    # Create the figure
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

    directions = {
        "Happy":    (0.8, 0.5),
        "Excited":  (0.7, 0.7),
        "Surprise": (0.5, 0.8),
        "Angry":    (-0.7, 0.7),
        "Fear":     (-0.5, 0.8),
        "Disgust":  (-0.8, 0.3),
        "Sad":      (-0.8, -0.6),
        "Depressed":(-0.7, -0.8),
        "Negative": (-0.9, -0.2),
        "Relaxed":  (0.7, -0.7),
        "Sleepy":   (0.5, -0.8),
        "Tired":    (0.3, -0.9),
        "Positive": (0.9, 0.0),
        "Neutral":  (0.0, 0.0),
        "Calming":  (0.0, -0.9),
        "Exciting": (0.0, 0.9),
        "Bored":    (-0.3, -0.9)
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

    # Create multiple color masks for each level/threshold
    color_masks = []
    
    # Define color thresholds based on number of unique seconds
    thresholds = [
        1,      # Blue for 1-4 unique seconds
        5,      # Green for 5-24 unique seconds
        25,     # Yellow for 25-124 unique seconds
        125,    # Orange for 125-624 unique seconds
        625     # Red for 625+ unique seconds
    ]
    
    # Define colors
    colors = [
        [0.0, 0.5, 1.0, 0.7],    # Blue with alpha
        [0.0, 0.8, 0.2, 0.7],    # Green with alpha
        [1.0, 1.0, 0.0, 0.7],    # Yellow with alpha
        [1.0, 0.5, 0.0, 0.7],    # Orange with alpha
        [1.0, 0.0, 0.0, 0.7]     # Red with alpha
    ]
    
    # Create visit_count array representing number of unique seconds for each emotion
    visit_count = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            if emotion_counters[i, j] is not None and isinstance(emotion_counters[i, j], dict):
                # Count number of unique seconds for each emotion
                total_unique_seconds = 0
                for emotion, seconds in emotion_counters[i, j].items():
                    total_unique_seconds += len(seconds)
                visit_count[i, j] = total_unique_seconds

    if np.sum(visit_count) > 0:
        # Create separate masks for each threshold level (in reverse order)
        for idx in range(len(thresholds)-1, -1, -1):
            threshold = thresholds[idx]
            color = colors[idx]
            
            mask = np.zeros((grid_size, grid_size, 4))
            
            # Apply color based on visit count
            for i in range(grid_size):
                for j in range(grid_size):
                    count = visit_count[j, i]
                    
                    # Calculate position in real coordinates
                    real_x = i / (grid_size - 1) * 2 - 1
                    real_y = j / (grid_size - 1) * 2 - 1
                    
                    # Only apply if count meets this threshold and is within unit circle
                    if count >= threshold and (real_x**2 + real_y**2 <= 1):
                        mask[j, i] = color
            
            # Apply Gaussian blur for smoother appearance
            # Use slightly larger sigma for better blending and continuity
            sigma_blur = 1.0  # Reduced sigma for less blur, making faint paths more visible
            for c in range(4):
                mask[:, :, c] = gaussian_filter(mask[:, :, c], sigma=sigma_blur)
            
            # Add to list of masks
            color_masks.append(mask)
        
        # Display the heatmap masks in reverse order (lowest threshold first)
        for mask in reversed(color_masks):
            ax.imshow(mask, extent=[-1, 1, -1, 1], origin='lower', interpolation='bilinear')

    if len(averaged_emotions) > 0:
        last_entry = averaged_emotions[-1]
        last_x, last_y = last_entry["valence"], last_entry["arousal"]
        ax.plot(last_x, last_y, 'yo', markersize=8)
        ax.text(0, -1.15, f"Valence: {last_x:.2f}   Arousal: {last_y:.2f}",
                ha='center', va='top', fontsize=10, color='white', fontweight='bold')
    else:
        ax.text(0, 0, "Waiting for data...", ha='center', va='center', color='gray', fontsize=10)
        ax.text(0, -1.15, f"Valence: 0.00   Arousal: 0.00",
                ha='center', va='top', fontsize=10, color='white', fontweight='bold')

    return fig

# === Emotion Data Management ===
def load_emotion_data():
    """
    Load emotion data from JSON file
    :return: Dictionary containing emotion data
    """
    try:
        with open('emotion_log.json', 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "emotions": {},
            "sequential_log": [],
            "metadata": {
                "last_updated": datetime.now().isoformat(),
                "version": "1.0"
            }
        }

def save_emotion_data(emotion_data):
    """
    Save emotion data to JSON file
    :param emotion_data: Dictionary containing emotion data
    """
    with open('emotion_log.json', 'w') as f:
        json.dump(emotion_data, f, indent=4)

def update_emotion_stats(emotion_log, current_emotion):
    """
    Update emotion statistics in the log
    :param emotion_log: List of emotion entries
    :param current_emotion: Current detected emotion
    """
    try:
        # Read current data
        try:
            with open('emotion_log.json', 'r') as f:
                emotion_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            emotion_data = {
                "emotions": {},
                "sequential_log": [],
                "metadata": {
                    "last_updated": datetime.now().isoformat(),
                    "version": "1.0"
                }
            }
        
        # Add new log to sequential_log
        if len(emotion_log) > 0:
            current_entry = emotion_log[-1]
            new_log_entry = {
                "timestamp": current_entry["timestamp"],
                "emotion": current_emotion,
                "valence": current_entry["valence"],
                "arousal": current_entry["arousal"],
                "agitation": 0,
                "confidence": current_entry["confidence"]
            }
            
            # Check if log already exists
            if not any(entry["timestamp"] == new_log_entry["timestamp"] for entry in emotion_data["sequential_log"]):
                emotion_data["sequential_log"].append(new_log_entry)
                
                # Limit number of logs
                if len(emotion_data["sequential_log"]) > 1000:
                    emotion_data["sequential_log"] = emotion_data["sequential_log"][-1000:]
        
        # Update aggregate statistics
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
        
        # Save data
        with open('emotion_log.json', 'w') as f:
            json.dump(emotion_data, f, indent=4)
            
    except Exception as e:
        print(f"Error updating emotion stats: {e}")

# === Real-time Video Capture and Emotion Detection ===
def log_emotions_per_second(emotion_log, filename="emotion_log_second.json"):
    """
    Log emotions detected in each second to a JSON file
    :param emotion_log: List of emotion entries
    :param filename: Name of the log file
    """
    if not emotion_log:
        return
        
    # Group emotions by second
    emotions_by_second = {}
    for entry in emotion_log:
        timestamp = datetime.fromisoformat(entry["timestamp"])
        second_key = timestamp.replace(microsecond=0)
        
        if second_key not in emotions_by_second:
            emotions_by_second[second_key] = []
        emotions_by_second[second_key].append(entry)
    
    # Create JSON structure
    log_data = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "total_seconds": len(emotions_by_second),
            "total_emotions": len(emotion_log)
        },
        "emotions_by_second": {}
    }
    
    # Add emotions for each second
    for second_key, entries in sorted(emotions_by_second.items()):
        second_data = {
            "timestamp": second_key.isoformat(),
            "emotions": [],
            "averages": {
                "valence": round(sum(entry["valence"] for entry in entries) / len(entries), 2),
                "arousal": round(sum(entry["arousal"] for entry in entries) / len(entries), 2)
            }
        }
        
        # Add individual emotions
        for entry in entries:
            second_data["emotions"].append({
                "emotion": entry["emotion"],
                "valence": entry["valence"],
                "arousal": entry["arousal"],
                "confidence": entry["confidence"]
            })
            
        log_data["emotions_by_second"][second_key.isoformat()] = second_data
    
    # Write to JSON file
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=4, ensure_ascii=False)
        print(f"Successfully saved emotion log to {filename}")
    except Exception as e:
        print(f"Error saving emotion log: {e}")

def print_emotions_per_second(emotion_log, filename="emotion_log_second.json", mode='w'):
    """
    Print emotions detected in each second to a JSON file
    :param emotion_log: List of emotion entries
    :param filename: Name of the output file
    :param mode: File open mode ('w' for write, 'a' for append)
    """
    if not emotion_log:
        st.write("No emotions to log")
        return
        
    st.write(f"Processing {len(emotion_log)} emotions for logging...")
        
    # Group emotions by second
    emotions_by_second = {}
    for entry in emotion_log:
        timestamp = datetime.fromisoformat(entry["timestamp"])
        second_key = timestamp.replace(microsecond=0)
        
        if second_key not in emotions_by_second:
            emotions_by_second[second_key] = []
        emotions_by_second[second_key].append(entry)
    
    st.write(f"Grouped into {len(emotions_by_second)} seconds")
    
    # Create JSON structure
    log_data = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "total_seconds": len(emotions_by_second),
            "total_emotions": len(emotion_log)
        },
        "emotions_by_second": {}
    }
    
    # Add emotions for each second
    for second_key, entries in sorted(emotions_by_second.items()):
        second_data = {
            "timestamp": second_key.isoformat(),
            "emotions": [],
            "averages": {
                "valence": round(sum(entry["valence"] for entry in entries) / len(entries), 2),
                "arousal": round(sum(entry["arousal"] for entry in entries) / len(entries), 2)
            }
        }
        
        # Add individual emotions
        for entry in entries:
            second_data["emotions"].append({
                "emotion": entry["emotion"],
                "valence": entry["valence"],
                "arousal": entry["arousal"],
                "confidence": entry["confidence"]
            })
            
        log_data["emotions_by_second"][second_key.isoformat()] = second_data
    
    # Write to JSON file
    try:
        st.write(f"Writing to file: {filename}")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=4, ensure_ascii=False)
        st.write(f"Successfully saved emotions per second to {filename}")
        
        # Verify file was written
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            st.write(f"Verified file contains {len(saved_data['emotions_by_second'])} seconds of data")
        except Exception as e:
            st.write(f"Error verifying file: {e}")
            
    except Exception as e:
        st.write(f"Error saving emotions per second: {e}")

def save_emotions_per_second(emotion_log, filename="emotion_log_second.json"):
    """
    Save emotions detected in each second to a JSON file
    :param emotion_log: List of emotion entries
    :param filename: Name of the output file
    """
    if not emotion_log:
        return
        
    # Group emotions by second
    emotions_by_second = {}
    for entry in emotion_log:
        timestamp = datetime.fromisoformat(entry["timestamp"])
        second_key = timestamp.replace(microsecond=0)
        
        if second_key not in emotions_by_second:
            emotions_by_second[second_key] = []
        emotions_by_second[second_key].append(entry)
    
    # Get video start and end time
    if emotions_by_second:
        start_time = min(emotions_by_second.keys())
        end_time = max(emotions_by_second.keys())
        video_duration = (end_time - start_time).total_seconds()
        
        # Find missing seconds
        all_seconds = set()
        current = start_time
        while current <= end_time:
            all_seconds.add(current)
            current += timedelta(seconds=1)
        
        missing_seconds = all_seconds - set(emotions_by_second.keys())
    else:
        start_time = end_time = datetime.now()
        video_duration = 0
        missing_seconds = set()
    
    # Create JSON structure
    log_data = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "total_seconds": len(emotions_by_second),
            "total_emotions": len(emotion_log),
            "video_duration": video_duration,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "missing_seconds": [s.isoformat() for s in sorted(missing_seconds)]
        },
        "emotions_by_second": {}
    }
    
    # Add emotions for each second
    for second_key, entries in sorted(emotions_by_second.items()):
        # Calculate averages
        avg_valence = sum(entry["valence"] for entry in entries) / len(entries)
        avg_arousal = sum(entry["arousal"] for entry in entries) / len(entries)
        
        # Get most common emotion
        emotions = [entry["emotion"] for entry in entries]
        most_common_emotion = max(set(emotions), key=emotions.count)
        
        # Count emotions
        emotion_counts = {}
        for emotion in emotions:
            if emotion not in emotion_counts:
                emotion_counts[emotion] = 0
            emotion_counts[emotion] += 1
        
        log_data["emotions_by_second"][second_key.isoformat()] = {
            "timestamp": second_key.isoformat(),
            "emotions": emotions,
            "emotion_counts": emotion_counts,
            "total_emotions": len(emotions),
            "averages": {
                "valence": round(avg_valence, 2),
                "arousal": round(avg_arousal, 2)
            },
            "most_common_emotion": most_common_emotion
        }
    
    # Write to JSON file
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=4, ensure_ascii=False)
        print(f"Saved {len(emotions_by_second)} seconds of emotion data")
        if missing_seconds:
            print(f"Missing {len(missing_seconds)} seconds: {[s.strftime('%H:%M:%S') for s in sorted(missing_seconds)]}")
    except Exception as e:
        st.write(f"Error saving emotions per second: {e}")

def process_stream(cap, custom_frame_skip):
    """
    Process video stream and detect emotions
    :param cap: Video capture object
    :param custom_frame_skip: Number of frames to skip between detections
    """
    global prev_head_pose, prev_shoulders_pos, agitation_window
    
    # Reset state completely
    if 'emotion_log' in st.session_state:
        del st.session_state.emotion_log
    if 'smooth_points' in st.session_state:
        del st.session_state.smooth_points
    
    # Reset heatmap memory when starting a new session - Use correct initialization
    if 'heatmap_memory' in st.session_state:
        grid_size = 100
        st.session_state.heatmap_memory = np.empty((grid_size, grid_size), dtype=object)
        for i in range(grid_size):
            for j in range(grid_size):
                st.session_state.heatmap_memory[i, j] = {}  # Initialize with empty dict
        st.session_state.last_heatmap_points = []
    
    # Initialize from scratch
    st.session_state.emotion_log = []
    st.session_state.smooth_points = []
    st.session_state.last_save_time = time.time()
    st.session_state.prev_emotion = None
    st.session_state.first_detect = True
    
    # Set video capture properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    emotion_log = st.session_state.emotion_log
    col1, col2 = st.columns([3, 2])
    frame_placeholder = col1.empty()
    chart_placeholder = col2.empty()
    status_placeholder = st.empty()
    stop_btn = st.button("\U0001F534 Stop")

    # Add placeholder for log
    log_placeholder = st.empty()
    log_container = st.container()

    # Initialize empty chart
    chart_placeholder.pyplot(plot_circumplex([]))

    # Optimize video processing
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    
    # Create worker threads
    frame_count = 0
    q_in, q_out = queue.Queue(maxsize=100), queue.Queue(maxsize=100)  # Tăng queue size lên 100
    
    NUM_WORKERS = 32  # Tăng số worker lên 32
    workers = []
    for _ in range(NUM_WORKERS):
        worker = threading.Thread(target=emotion_predictor_worker, args=(q_in, q_out), daemon=True)
        worker.start()
        workers.append(worker)
    
    last_chart_time = time.time()
    last_frame_time = time.time()
    last_log_time = time.time()
    last_save_time = time.time()
    fps_counter = 0
    fps = 0
    target_fps = 25  # Set target frame rate to 25 (Step 7)
    frame_interval = 1.0 / target_fps
    frame_count = 0
    processing_time = 0
    skip_frame = False

    def display_realtime_log():
        nonlocal last_log_time, last_save_time
        current_time = time.time()
        if current_time - last_log_time >= 1.0:  # Update log every second
            with log_container:
                st.subheader("Real-time Emotion Timeline")
                if emotion_log:
                    # Display last 10 logs
                    for entry in emotion_log[-10:]:
                        timestamp = datetime.fromisoformat(entry["timestamp"])
                        formatted_time = timestamp.strftime("%H:%M:%S")
                        st.write(f"{formatted_time} - {entry['emotion']} (V: {entry['valence']:.2f}, A: {entry['arousal']:.2f}, Ag: {entry.get('agitation', 0):.2f})")
            
            # Save emotions per second every 5 seconds
            if current_time - last_save_time >= 5.0:
                save_emotions_per_second(emotion_log)
                last_save_time = current_time
                
            last_log_time = current_time
            
    frame_buffer = None
    results_buffer = []
    
    # Variable to store the last successful MediaPipe results
    last_holistic_results = None

    # Initialize MediaPipe Holistic
    with mp_holistic.Holistic(
        min_detection_confidence=0.4,  # Lower threshold for faster detection
        min_tracking_confidence=0.4,   # Lower threshold for faster tracking
        model_complexity=0             # Use the lightest model (0 instead of default 1)
    ) as holistic: # Apply Step 6 optimizations
        try:
            while cap.isOpened():
                if stop_btn:
                    if emotion_log:
                        update_emotion_stats(emotion_log, emotion_log[-1]["emotion"])
                        # Save final emotions per second
                        save_emotions_per_second(emotion_log)
                        
                        # Reset heatmap memory when stopping - Use correct initialization
                        if 'heatmap_memory' in st.session_state:
                            grid_size = 100
                            st.session_state.heatmap_memory = np.empty((grid_size, grid_size), dtype=object)
                            for i in range(grid_size):
                                for j in range(grid_size):
                                    st.session_state.heatmap_memory[i, j] = {}  # Initialize with empty dict
                            st.session_state.last_heatmap_points = []
                        
                        break

                # --- FPS Limiting Logic Start ---
                start_time = time.time()
                
                read_start = time.time()
                ret, img = cap.read()
                
                read_duration = time.time() - read_start
                print(f"[DEBUG] cap.read() time: {read_duration:.4f}s") # Measure read time

                if not ret:
                    break
                        
                #display_realtime_log()  # Display real-time log
                
                # Resize image to a fixed size for consistent processing speed
                target_width = 640
                if img.shape[1] != target_width:
                    scale_factor = target_width / img.shape[1]
                    img = cv2.resize(img, (target_width, int(img.shape[0] * scale_factor)))

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                frame_count += 1
                
                # Process with MediaPipe (less frequently)
                results = last_holistic_results # Assume using previous results initially
                if frame_count % 2 == 0: # Process only every 2nd frame
                    process_start_time = time.time()
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    current_results = holistic.process(img_rgb)
                    if current_results.pose_landmarks: # Store results only if landmarks were found
                        last_holistic_results = current_results
                        results = current_results # Use the new results for this frame
                    print(f"[DEBUG] MediaPipe Process time (executed): {time.time() - process_start_time:.3f}s")
                else:
                    print(f"[DEBUG] MediaPipe Process time (skipped)")

                # Ensure we have results before proceeding with calculations/drawing
                if results is None:
                    continue # Skip the rest of the loop if no results yet
                
                # Calculate agitation score
                agitation_score = 0 # Set to 0 as it's disabled
                
                # Update previous positions
                if results.pose_landmarks:
                    prev_head_pose = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
                    left_shoulder = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
                    right_shoulder = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
                    prev_shoulders_pos = (
                        (left_shoulder.x + right_shoulder.x) / 2,
                        (left_shoulder.y + right_shoulder.y) / 2
                    )
                
                # Draw landmarks
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        img,
                        results.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                
                # Send every frame for prediction with retry
                max_retries = 5  # Tăng số lần retry
                for retry in range(max_retries):
                    try:
                        q_in.put((img.copy(), gray), block=False)
                        break
                    except queue.Full:
                        if retry < max_retries - 1:
                            print(f"[DEBUG] Queue full, retrying frame {frame_count} (attempt {retry + 1})")
                            time.sleep(0.05)  # Giảm thời gian chờ
                        else:
                            print(f"[DEBUG] Queue full, skipping frame {frame_count} after {max_retries} attempts")
                            continue

                # Check and process results
                frame_to_show = img.copy()
                
                # Display Actual FPS
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
                        label_text = f"{label} {emoji}"
                        
                        current_time = datetime.now().isoformat()
                        new_entry = {
                            "timestamp": current_time,
                            "emotion": label,
                            "confidence": conf,
                            "valence": val,
                            "arousal": aro,
                            "agitation": 0 # Store 0 as agitation is disabled
                        }
                        
                        # Add new entry to log
                        emotion_log.append(new_entry)
                        
                        # Save immediately to JSON file
                        update_emotion_stats(emotion_log, label)
                        
                        if len(emotion_log) > MAX_LOG_LENGTH:
                            del emotion_log[0]
                
                # Display frame
                frame_placeholder.image(cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB), channels="RGB", width=320,)  # Đặt chiều rộng là 400px
                
                # Update chart
                if time.time() - last_chart_time > 1.5: # Revert to update chart every 1.5 seconds
                    if emotion_log:
                        chart_placeholder.pyplot(plot_circumplex(emotion_log))
                    last_chart_time = time.time()
                    
                # --- FPS Limiting Logic End ---
                end_time = time.time()
                processing_time = end_time - start_time
                sleep_time = frame_interval - processing_time
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Calculate actual FPS based on total frame time (processing + sleep)
                actual_frame_time = time.time() - start_time
                if actual_frame_time > 0:
                    fps = 1.0 / actual_frame_time
                else:
                    fps = float('inf') # Avoid division by zero if processing is extremely fast
                
                status_placeholder.text(f"Processing... FPS: {fps:.1f}") # Display actual FPS (now more stable)
                print(f"[DEBUG] Full frame time: {actual_frame_time:.3f}s | FPS: {fps}") # Re-enable this

        finally:
            # Clean up resources
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
            
            # Display complete log after finishing
            with log_container:
                st.subheader("Complete Emotion Timeline")
                if emotion_log:
                    for entry in emotion_log:
                        timestamp = datetime.fromisoformat(entry["timestamp"])
                        formatted_time = timestamp.strftime("%H:%M:%S")
                        st.write(f"{formatted_time} - {entry['emotion']} (V: {entry['valence']:.2f}, A: {entry['arousal']:.2f}, Ag: {entry.get('agitation', 0):.2f})")

# === Threaded Prediction Worker ===
def emotion_predictor_worker(q_in, q_out):
    """
    Worker thread for emotion prediction
    :param q_in: Input queue for frames
    :param q_out: Output queue for results
    """
    # Enable TensorFlow optimizations
    tf.config.optimizer.set_jit(True)
    
    while True:
        try:
            frame, gray = q_in.get(timeout=1.0)
            if frame is None:
                break
                
            # Skip frames if processing is taking too long
            if not q_in.empty():
                continue

            is_first_detection = 'first_detect' in st.session_state and st.session_state.first_detect

            # Fixed size for face detection
            target_size = (240, 180)  # Make the target detection size smaller (Step 5)
            small_gray = cv2.resize(gray, target_size)
            small_gray = cv2.equalizeHist(small_gray)

            # Use fixed parameters for more stable detection
            scaleFactor = 1.1
            minNeighbors = 3 # Keep at 3 (Step 5)
            minSize = (25, 25) # Keep at (25, 25) (Step 5)
            maxSize = (200, 200)  # Add maximum face size

            faces = face_detection.detectMultiScale(
                small_gray, 
                scaleFactor=scaleFactor,
                minNeighbors=minNeighbors,
                minSize=minSize,
                maxSize=maxSize,
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            logs = []
            for (x, y, w, h) in faces:
                # Calculate scaling ratios based on original gray image size and target_size
                orig_h, orig_w = gray.shape[:2]
                scale_w = orig_w / target_size[0]  # target_size is (width, height)
                scale_h = orig_h / target_size[1]
                
                # Scale coordinates back to original image dimensions
                x = int(x * scale_w)
                y = int(y * scale_h)
                w = int(w * scale_w)
                h = int(h * scale_h)

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
                    predict_start = time.time()
                    prediction = model.predict(np.array([face_normalized.reshape((48, 48, 1))]), verbose=0)
                    predict_duration = time.time() - predict_start
                    print(f"[DEBUG] Predict time: {predict_duration:.3f}s") # Re-enable this
                    label = labels[np.argmax(prediction)]
                    confidence = float(np.max(prediction))

                    if confidence > 0.3:
                        if is_first_detection:
                            st.session_state.first_detect = False

                        # Method for calculating valence and arousal
                        # Select all emotions with probability above threshold for better blending
                        threshold = 0.1  # Only consider emotions with probability >= 10%
                        significant_emotions = [(i, prob) for i, prob in enumerate(prediction[0]) if prob >= threshold]
                        
                        # If no emotions exceed threshold, take the one with highest probability
                        if not significant_emotions:
                            significant_emotions = [(np.argmax(prediction[0]), np.max(prediction[0]))]
                        
                        # Normalize weights to sum to 1
                        total_weight = sum(prob for _, prob in significant_emotions)
                        normalized_emotions = [(i, prob/total_weight) for i, prob in significant_emotions]
                        
                        # Calculate valence and arousal based on weighted average position
                        valence = 0
                        arousal = 0
                        intensity = sum(prob for _, prob in normalized_emotions)
                        
                        for i, weight in normalized_emotions:
                            emotion_name = labels[i]
                            # Map emotion name from model to emotion_map
                            mapped_emotion = emotion_label_map.get(emotion_name, "Neutral")
                            
                            # Get valence and arousal values from emotion map - keep original values
                            valence += emotion_map[mapped_emotion]['valence'] * weight
                            arousal += emotion_map[mapped_emotion]['arousal'] * weight
                        
                        # Add small noise for slight variation
                        noise_factor = 0.02  # Reduce noise to avoid distorting position across frames
                        valence += np.random.normal(0, noise_factor)
                        arousal += np.random.normal(0, noise_factor)
                        
                        # Only limit to range [-1, 1] to avoid exceeding chart limits
                        # Don't apply any scaling factor - keep position as calculated
                        valence = np.clip(valence, -1, 1)
                        arousal = np.clip(arousal, -1, 1)
                        
                        # Round to 2 decimal places
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
    Calculate agitation score based on head and shoulder movements using MediaPipe Holistic
    :param landmarks: MediaPipe Holistic landmarks
    :param prev_head_pose: Previous head position
    :param prev_shoulders_pos: Previous shoulders position
    :param agitation_window: Window of previous agitation scores
    :return: Agitation score between 0 and 1
    """
    if agitation_window is None:
        agitation_window = []
        
    if landmarks.pose_landmarks:
        # Get key landmark points
        head_pose = landmarks.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
        left_shoulder = landmarks.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        
        # Calculate average shoulder position
        shoulders_pos = (
            (left_shoulder.x + right_shoulder.x) / 2,
            (left_shoulder.y + right_shoulder.y) / 2
        )
        
        # Calculate head movement
        if prev_head_pose is not None:
            head_movement = math.sqrt(
                (head_pose.x - prev_head_pose.x) ** 2 +
                (head_pose.y - prev_head_pose.y) ** 2 +
                (head_pose.z - prev_head_pose.z) ** 2
            )
        else:
            head_movement = 0
            
        # Calculate shoulder movement
        if prev_shoulders_pos is not None:
            shoulder_movement = math.sqrt(
                (shoulders_pos[0] - prev_shoulders_pos[0]) ** 2 +
                (shoulders_pos[1] - prev_shoulders_pos[1]) ** 2
            )
        else:
            shoulder_movement = 0
            
        # Calculate agitation score based on movements
        # Increase sensitivity to head movement (multiply by 2)
        agitation_score = (head_movement * 2 + shoulder_movement) / 3
        
        # Add score to window
        agitation_window.append(agitation_score)
        
        # Limit window size (30 frames)
        if len(agitation_window) > 30:
            agitation_window.pop(0)
            
        # Calculate average agitation score
        avg_agitation = sum(agitation_window) / len(agitation_window)
        
        # Normalize to range [0, 1]
        return min(max(avg_agitation, 0), 1)
        
    return 0

# === Streamlit UI ===
st.set_page_config(page_title="Facial Emotion Heatmap", layout="wide")
st.title("🎭 Real-time Facial Emotion Recognition with Heatmap")

mode = st.radio("Select Mode:", ["Use Webcam", "Upload Video", "Test Video"])

if mode == "Use Webcam":
    if st.button("Start Webcam Analysis"):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
        process_stream(cap, FRAME_SKIP)
        display_sequential_log()  # Display log after finishing

elif mode == "Upload Video":
    video_file = st.file_uploader("Upload your .mp4 video", type=["mp4"])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        st.video(tfile.name)
        if st.button("Analyze Uploaded Video"):
            cap = cv2.VideoCapture(tfile.name)
            process_stream(cap, FRAME_SKIP)
            display_sequential_log()  # Display log after finishing

elif mode == "Test Video":
    st.subheader("Test Video Settings")
    
    # Video selection
    test_videos = ["test_video1.mp4", "test_video2.mp4", "emotions_test.mp4"]
    
    # Check which videos exist
    existing_videos = []
    for video in test_videos:
        if os.path.exists(video):
            existing_videos.append(video)
    
    # If no videos found, show warning
    if not existing_videos:
        st.warning("No test videos found. Please place test videos in the same directory.")
        
        # Provide option to upload test video
        custom_test = st.file_uploader("Upload a test video", type=["mp4"])
        if custom_test:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(custom_test.read())
            custom_path = tfile.name
            st.success(f"Test video uploaded successfully: {os.path.basename(custom_path)}")
            existing_videos = [custom_path]
    
    if existing_videos:
        selected_video = st.selectbox("Select a test video", existing_videos)
        
        # Video playback settings
        col1, col2 = st.columns(2)
        with col1:
            speed = st.slider("Playback Speed", min_value=0.25, max_value=2.0, value=1.0, step=0.25)
            frame_skip = st.slider("Frame Skip", min_value=1, max_value=10, value=3, step=1)
        
        with col2:
            path_thickness = st.slider("Path Thickness", min_value=0.3, max_value=0.6, value=0.45, step=0.05)
            smoothness = st.slider("Path Smoothness", min_value=1.0, max_value=2.0, value=1.4, step=0.1)
        
        # Display preview and run button
        st.video(selected_video)
        
        if st.button("Run Test Analysis"):
            # Apply user settings
            # Use variable from slider without needing global variable
            custom_frame_skip = frame_skip
            
            # Utility to set parameters for path visualization
            st.session_state.path_thickness = path_thickness
            st.session_state.path_smoothness = smoothness
            
            # Open and process video
            cap = cv2.VideoCapture(selected_video)
            
            # Adjust video playback speed
            if speed != 1.0:
                original_fps = cap.get(cv2.CAP_PROP_FPS)
                st.info(f"Original video FPS: {original_fps:.1f}, Adjusted FPS: {original_fps * speed:.1f}")
            
            # Pass frame_skip as parameter to process_stream
            process_stream(cap, custom_frame_skip)
            display_sequential_log()  # Display log after finishing

# Add function to display time-based log
def display_sequential_log():
    """
    Display emotion log with timestamps
    """
    emotion_data = load_emotion_data()
    if "sequential_log" in emotion_data and emotion_data["sequential_log"]:
        st.subheader("Emotion Timeline")
        for entry in emotion_data["sequential_log"]:
            timestamp = datetime.fromisoformat(entry["timestamp"])
            formatted_time = timestamp.strftime("%H:%M:%S")
            st.write(f"{formatted_time} - {entry['emotion']} (V: {entry['valence']:.2f}, A: {entry['arousal']:.2f}, Ag: {entry.get('agitation', 0):.2f})")



# Export emotion_log_second.json thủ công
if st.button("📁 Export Emotion Log (.json)"):
    try:
        save_emotions_per_second(st.session_state.emotion_log)
        with open("emotion_log_second.json", "r", encoding="utf-8") as f:
            json_data = f.read()
        st.success("Emotion log exported successfully!")
        st.download_button(
            label="📥 Download emotion_log_second.json",
            data=json_data,
            file_name="emotion_log_second.json",
            mime="application/json"
        )
    except Exception as e:
        st.error(f"Error exporting emotion log: {e}")