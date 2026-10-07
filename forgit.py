import cv2
import mediapipe as mp
import numpy as np
import os
import time
from PIL import Image

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

mp_face = mp.solutions.face_detection 
face_detection = mp_face.FaceDetection(min_detection_confidence=0.7)

mp_draw = mp.solutions.drawing_utils

gif_frames = []
if not os.path.exists('ears.gif'):
    print("ears.gif not found. Please ensure the GIF file is in the same directory as this script.")
else:
    gif = Image.open('ears.gif')
    try:
        while True:
            frame_rgba = gif.copy().convert("RGBA")
            frame_cv = cv2.cvtColor(np.array(frame_rgba), cv2.COLOR_RGBA2BGRA)
            gif_frames.append(frame_cv)
            gif.seek(len(gif_frames))
    except EOFError:
        pass 

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None
is_recording = False

def overlay_transparent(background, overlay, x, y):
    bg_h, bg_w, bg_channels = background.shape
    ov_h, ov_w, ov_channels = overlay.shape

    y1, y2 = max(0, y), min(bg_h, y + ov_h)
    x1, x2 = max(0, x), min(bg_w, x + ov_w)
    
    y1o, y2o = max(0, -y), min(ov_h, bg_h - y)
    x1o, x2o = max(0, -x), min(ov_w, bg_w - x)

    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return background

    alpha = overlay[y1o:y2o, x1o:x2o, 3] / 255.0
    alpha_inv = 1.0 - alpha

    for c in range(0, 3):
        background[y1:y2, x1:x2, c] = (alpha * overlay[y1o:y2o, x1o:x2o, c] +
                                       alpha_inv * background[y1:y2, x1:x2, c])
    return background

def recognize_gesture(hand_landmarks, label):
    fingers = []
    
    if label == 'Right':
        fingers.append(1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[2].x else 0)
    else:
        fingers.append(1 if hand_landmarks.landmark[4].x > hand_landmarks.landmark[2].x else 0)
        
    tips = [8, 12, 16, 20]
    for tip in tips:
        fingers.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y else 0)
        
    if fingers[1:] == [1, 1, 0, 0]:
        return "Peace"
    if fingers[1:] == [0, 1, 0, 0]:
        return "Middle Finger"
        
    if sum(fingers[1:]) == 0:
        if hand_landmarks.landmark[4].y < hand_landmarks.landmark[5].y - 0.05:
            return "Thumbs Up"
        elif hand_landmarks.landmark[4].y > hand_landmarks.landmark[0].y + 0.05:
            return "Thumbs Down"
        else:
            return "0 fingers"
            
    return f"{sum(fingers)} fingers"

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape
    
    hands_results = hands.process(rgb_frame)
    face_results = face_detection.process(rgb_frame)
    
    peace_count = 0
    
    if hands_results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
            label = results_handedness = hands_results.multi_handedness[idx].classification[0].label
            gesture_text = recognize_gesture(hand_landmarks, label)
            
            if gesture_text == "Peace":
                peace_count += 1
                
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            if label == 'Left':
                cv2.putText(frame, f"Left: {gesture_text}", (20, 80), 
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"Right: {gesture_text}", (w - 350, 80), 
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)

    if peace_count == 2 and face_results.detections:
        for detection in face_results.detections:
            bbox = detection.location_data.relative_bounding_box
            face_x = int(bbox.xmin * w)
            face_y = int(bbox.ymin * h)
            face_w = int(bbox.width * w)
            face_h = int(bbox.height * h)
            
            if gif_frames:
                frame_index = int(time.time() * 10) % len(gif_frames)
                current_ear = gif_frames[frame_index]
                
                ears_w = int(face_w * 1.3)
                ears_h = int(current_ear.shape[0] * (ears_w / current_ear.shape[1]))
                resized_ears = cv2.resize(current_ear, (ears_w, ears_h))
                
                ears_x = face_x - int((ears_w - face_w) / 2)
                ears_y = face_y - int(ears_h * 0.95) + int(face_h * 0.05)
                
                frame = overlay_transparent(frame, resized_ears, ears_x, ears_y)
                
                cv2.putText(frame, "NYA :3", (face_x, ears_y - 20), 
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 200, 255), 3)


    if is_recording:
        out.write(frame)
        cv2.circle(frame, (30, 35), 10, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (50, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Press 'R' to record", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow('Smart Camera', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        is_recording = not is_recording
        if is_recording:
            filename = f"capture_{int(time.time())}.avi"
            out = cv2.VideoWriter(filename, fourcc, 20.0, (w, h))
            print(f"Запись начата: {filename}")
        else:
            out.release()
            print("Запись сохранена!")

if out is not None:
    out.release()
cap.release()
cv2.destroyAllWindows()