import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# --- CẤU HÌNH ---
DATA_DIR = 'data'
IMG_SIZE = 128
BATCH_SIZE = 32

# 1. Kiểm tra dữ liệu và số lượng Class
if not os.path.exists(DATA_DIR):
    print(f"LỖI: Không tìm thấy thư mục '{DATA_DIR}'. Hãy chạy bước 1 trước!")
    exit()

# Đếm số thư mục con (số class)
class_folders = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))]
num_classes = len(class_folders)
print(f"--> Tìm thấy {num_classes} động tác (classes): {class_folders}")

if num_classes < 2:
    print("LỖI: Cần ít nhất 2 thư mục động tác để train.")
    exit()

# Đếm tổng số ảnh
total_images = sum([len(files) for r, d, files in os.walk(DATA_DIR)])
print(f"--> Tổng số ảnh tìm thấy: {total_images}")

# 2. Thiết lập Data Generators (Tự động điều chỉnh validation)
# Nếu ít hơn 50 ảnh thì không chia tập validation để tránh lỗi crash
if total_images < 50:
    print("CẢNH BÁO: Ít dữ liệu (<50 ảnh). Sẽ dùng toàn bộ để train (không validation).")
    val_split = 0.0
else:
    val_split = 0.2  # Dùng 20% ảnh để kiểm tra

datagen = ImageDataGenerator(rescale=1. / 255, validation_split=val_split)

train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='training'
)

val_generator = None
if val_split > 0:
    val_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='validation'
    )

# 3. Xây dựng mô hình CNN
model = Sequential([
    # Lớp Input tường minh để tránh Warning
    Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    # Layer 1
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # Layer 2
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # Layer 3
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),

    # --- CHỈNH SỬA TỪ 0.5 THÀNH 0.3 ---
    Dropout(0.3),

    # Output Layer: Tự động chỉnh số neuron theo số class tìm thấy
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 4. Huấn luyện (Train)
print("\nBắt đầu train model...")

# --- CHỈNH SỬA SỐ EPOCHS THÀNH 3 ---
epochs = 30  # Chạy 30 vòng để train chính xác

try:
    if val_generator and val_generator.samples > 0:
        model.fit(train_generator, epochs=epochs, validation_data=val_generator)
    else:
        model.fit(train_generator, epochs=epochs)

    # 5. Lưu model
    model.save("hand_sign_model.h5")
    print("\n--------------------------------------")
    print("THÀNH CÔNG! Đã lưu file: hand_sign_model.h5")
    print(f"Model này dùng được cho {num_classes} động tác.")
    print("--------------------------------------")

except Exception as e:
    print(f"\nLỖI KHI TRAIN: {e}")
    print("Hãy đảm bảo bạn đã chụp đủ ít nhất 20-30 ảnh cho mỗi động tác!")