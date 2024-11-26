import os
import cv2
import numpy as np
import pickle  # Để lưu trữ đặc trưng

# Đường dẫn thư mục ảnh shop
SHOP_IMAGES_PATH = r"C:\Users\Acer\Dropbox\PC\Desktop\python\sift-main\images"
FEATURES_PATH = r"C:\Users\Acer\Dropbox\PC\Desktop\python\sift-main\sift_features.pkl"

# Hàm trích xuất đặc trưng SIFT
def extract_sift_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return descriptors

# Lưu đặc trưng của tất cả ảnh trong shop
shop_features = {}

for filename in os.listdir(SHOP_IMAGES_PATH):
    image_path = os.path.join(SHOP_IMAGES_PATH, filename)
    descriptors = extract_sift_features(image_path)
    if descriptors is not None:
        shop_features[filename] = descriptors

# Lưu đặc trưng vào file
with open(FEATURES_PATH, "wb") as f:
    pickle.dump(shop_features, f)

print(f"Đã lưu trữ đặc trưng của {len(shop_features)} ảnh.")
