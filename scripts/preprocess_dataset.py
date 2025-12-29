import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from PIL import Image

# ================= CONFIG =================
DATASET_DIR = "data/dataset/SL"       # Your downloaded dataset
SEQUENCE_DIR = os.path.join(DATASET_DIR, "sequences")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")  # Optional: save frames
MAX_FRAMES = 40                   # Pad/truncate sequences to this length
SAVE_IMAGES = False               # Set True if you want frame images

# =========================================

# Create sequences folder
if not os.path.exists(SEQUENCE_DIR):
    os.makedirs(SEQUENCE_DIR)
# Create gesture subfolders
for gesture in os.listdir(DATASET_DIR):
    gesture_path = os.path.join(DATASET_DIR, gesture)
    if os.path.isdir(gesture_path) and gesture != "sequences":
        seq_path = os.path.join(SEQUENCE_DIR, gesture)
        if not os.path.exists(seq_path):
            os.makedirs(seq_path)
        if SAVE_IMAGES:
            img_path = os.path.join(IMAGES_DIR, gesture)
            if not os.path.exists(img_path):
                os.makedirs(img_path)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Helper function to save frame images
def save_frame_image(frame, save_dir, video_name, frame_idx):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    filename = f"{video_name}_frame_{frame_idx:03d}.jpg"
    pil_image.save(os.path.join(save_dir, filename))

# ================= PROCESS VIDEOS =================
for gesture in os.listdir(DATASET_DIR):
    gesture_path = os.path.join(DATASET_DIR, gesture)
    if not os.path.isdir(gesture_path) or gesture == "sequences":
        continue

    seq_path = os.path.join(SEQUENCE_DIR, gesture)
    video_files = [f for f in os.listdir(gesture_path) if f.endswith((".mp4", ".avi", ".mov"))]

    for video_file in tqdm(video_files, desc=f"Processing {gesture}"):
        video_path = os.path.join(gesture_path, video_file)
        cap = cv2.VideoCapture(video_path)
        landmarks_list = []
        frame_idx = 0
        video_name = os.path.splitext(video_file)[0]

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR -> RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                frame_landmarks = []
                for lm in hand_landmarks.landmark:
                    frame_landmarks.extend([lm.x, lm.y])
                landmarks_list.append(frame_landmarks)
            else:
                # No hand detected → fill zeros
                landmarks_list.append([0]*42)

            if SAVE_IMAGES:
                save_frame_image(frame, os.path.join(IMAGES_DIR, gesture), video_name, frame_idx)

            frame_idx += 1

        cap.release()

        # Convert to numpy array
        landmarks_array = np.array(landmarks_list)

        # Pad or truncate to MAX_FRAMES
        if landmarks_array.shape[0] < MAX_FRAMES:
            padding = np.zeros((MAX_FRAMES - landmarks_array.shape[0], 42))
            landmarks_array = np.vstack([landmarks_array, padding])
        elif landmarks_array.shape[0] > MAX_FRAMES:
            landmarks_array = landmarks_array[:MAX_FRAMES]

        # Save .npy sequence
        save_path = os.path.join(seq_path, video_file.replace(".mp4", ".npy"))
        np.save(save_path, landmarks_array)

# ================= DONE =================
print("✅ Preprocessing complete!")
print(f"Sequences saved in: {SEQUENCE_DIR}")
if SAVE_IMAGES:
    print(f"Frame images saved in: {IMAGES_DIR}")
