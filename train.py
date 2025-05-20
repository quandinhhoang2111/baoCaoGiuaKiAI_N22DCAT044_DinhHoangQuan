import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization
import matplotlib.pyplot as plt
import pickle
import face_recognition 

# Cấu hình GPU nếu có
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Đường dẫn dataset chính xác
DATASET_PATH = r"D:\project\archive\lfw-deepfunneled\lfw-deepfunneled"
IMAGE_SIZE = (100, 100)

# Đếm số lượng ảnh mỗi người
print("Đang đếm số lượng ảnh của từng người...")
person_image_count = defaultdict(int)
for person_name in os.listdir(DATASET_PATH):
    person_folder = os.path.join(DATASET_PATH, person_name)
    if not os.path.isdir(person_folder):
        continue
    count = len([name for name in os.listdir(person_folder) if os.path.isfile(os.path.join(person_folder, name))])
    if count >= 30:
        person_image_count[person_name] = count

# Load ảnh và nhãn từ người có >=30 ảnh
X, y = [], []
print(f"Đang load dữ liệu từ {len(person_image_count)} người có ≥ 30 ảnh...")

for person_name in tqdm(person_image_count.keys()):
    person_folder = os.path.join(DATASET_PATH, person_name)
    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        try:
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)

            if face_locations:
                top, right, bottom, left = face_locations[0]  # chỉ lấy khuôn mặt đầu tiên
                face_image = image[top:bottom, left:right]
                img_resized = Image.fromarray(face_image).resize(IMAGE_SIZE)

                X.append(np.array(img_resized))
                y.append(person_name)
        except Exception as e:
            continue

# In ra danh sách tên các class (người) đã chọn
print(f"Danh sách các class được load ({len(person_image_count)} người):")
for name in person_image_count.keys():
    print(name)
print(f"Tổng số ảnh đã load: {len(X)}")

X = np.array(X)
y = np.array(y)

# Encode nhãn thành số
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Chia X_test hiện tại thành 50% validation và 50% test (tức là 10% - 10% từ toàn bộ)
X_val, X_test_final, y_val, y_test_final = train_test_split(
    X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)
print(f"Số ảnh dùng để huấn luyện (Train): {len(X_train)}")
print(f"Số ảnh dùng để kiểm tra (Validation): {len(X_val)}")
print(f"Số ảnh dùng để đánh giá (Test): {len(X_test_final)}")

# Tiền xử lý ảnh
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test_final = X_test_final / 255.0

# One-hot encode label
num_classes = len(np.unique(y_encoded))
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes)
y_test_cat = tf.keras.utils.to_categorical(y_test_final, num_classes)

# Model CNN 
model = Sequential([
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(100, 100, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.3),

    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.3),

    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.3),

    Conv2D(256, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.3),

    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.6),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.6),

    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

# Huấn luyện
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    brightness_range=[0.8,1.2],
    horizontal_flip=True
)

# Sau khi fit xong label_encoder
with open("D:/project/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)  

datagen.fit(X_train)
history = model.fit(X_train, y_train_cat,
                    epochs=60,
                    batch_size=64,
                    validation_data=(X_val, y_val_cat)
                    )

# Đánh giá
loss, acc = model.evaluate(X_test_final, y_test_cat)
print(f"Accuracy trên tập test: {acc*100:.2f}%")

# Lưu model nếu muốn
model.save("D:/project/lfw_cnn_model.h5")

# Plot accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



np.save("D:/project/X_test_final.npy", X_test_final)
np.save("D:/project/y_test_cat.npy", y_test_cat)



