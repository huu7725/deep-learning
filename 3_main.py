import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

try:
    model = load_model("hand_sign_model.h5")
    print("Đã tải model!")
except:
    print("Chưa có model. Hãy chạy bước 2.")
    exit()

# --- SỬA TÊN 3 CỬ CHỈ Ở ĐÂY ---
class_names = [
    "Xin Chao",  # Tương ứng folder 0
    "Tam biet",  # Tương ứng folder 1
    "tymm"  # Tương ứng folder 2
]

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    img_white = np.zeros([h, w, 3], dtype=np.uint8)
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    prediction_text = "..."
    confidence = 0.0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            mp_draw.draw_landmarks(img_white, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                   mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

        # Dự đoán
        img_input = cv2.resize(img_white, (128, 128))
        img_input = img_input.astype("float32") / 255.0
        img_input = np.expand_dims(img_input, axis=0)

        prediction = model.predict(img_input, verbose=0)
        idx = np.argmax(prediction)
        confidence = np.max(prediction)

        if confidence > 0.7:
            if idx < len(class_names):
                prediction_text = class_names[idx]
            else:
                prediction_text = f"Class {idx}"

    # Hiển thị
    cv2.rectangle(frame, (0, 0), (300, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"Doan: {prediction_text}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Ti le: {confidence * 100:.1f}%", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Main", frame)
    cv2.imshow("Input", img_white)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()