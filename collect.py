import cv2
import numpy as np
import os
import mediapipe as mp

# --- CẤU HÌNH: CHỈ 3 CỬ CHỈ ---
NUMBER_OF_CLASSES = 3

# Tạo thư mục data/0, data/1, data/2
if not os.path.exists("data"):
    os.makedirs("data")

for i in range(NUMBER_OF_CLASSES):
    if not os.path.exists(f"data/{i}"):
        os.makedirs(f"data/{i}")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
counters = [0] * NUMBER_OF_CLASSES

print("Hệ thống thu thập 3 cử chỉ:")
print("  Phím 0: Cử chỉ 1")
print("  Phím 1: Cử chỉ 2")
print("  Phím 2: Cử chỉ 3")
print("  Phím q: Thoát")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    img_white = np.zeros([h, w, 3], dtype=np.uint8)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img_white, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                   mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    img_input = cv2.resize(img_white, (128, 128))

    cv2.putText(frame, f"0:{counters[0]} | 1:{counters[1]} | 2:{counters[2]}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Camera", frame)
    cv2.imshow("Input", img_input)

    key = cv2.waitKey(1)

    # Logic chỉ nhận phím 0, 1, 2
    if ord('0') <= key < ord('0') + NUMBER_OF_CLASSES:
        class_id = key - ord('0')
        counters[class_id] += 1
        cv2.imwrite(f"data/{class_id}/img_{counters[class_id]}.jpg", img_input)
        print(f"Lưu ảnh vào class {class_id}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()