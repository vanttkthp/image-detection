from scipy.spatial.distance import cdist
import cv2
import numpy as np
import pickle  # Để lưu trữ đặc trưng

FEATURES_PATH = r"C:\Users\Acer\Dropbox\PC\Desktop\python\sift-main\sift_features.pkl"

def extract_sift_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return descriptors

# Hàm tính khoảng cách giữa query và shop features
def find_similar_images(query_image_path, shop_features):
    # Trích xuất đặc trưng từ query image
    query_descriptors = extract_sift_features(query_image_path)
    if query_descriptors is None:
        print("Không tìm thấy đặc trưng trong ảnh query.")
        return []

    # So khớp query descriptors với shop descriptors
    scores = []
    for filename, descriptors in shop_features.items():
        # Tính khoảng cách Euclidean giữa các vector đặc trưng
        distances = cdist(query_descriptors, descriptors, metric="euclidean")
        # Lấy trung bình khoảng cách nhỏ nhất
        score = np.mean(np.min(distances, axis=1))
        scores.append((filename, score))

    # Sắp xếp ảnh theo độ tương đồng (điểm thấp là tương đồng hơn)
    scores.sort(key=lambda x: x[1])
    return scores

# Load đặc trưng của shop
with open(FEATURES_PATH, "rb") as f:
    shop_features = pickle.load(f)

# Ảnh query của khách
QUERY_IMAGE_PATH = r"C:\Users\Acer\Dropbox\PC\Desktop\python\sift-main\test-images\s-l400.jpg"  # Đường dẫn ảnh query

# Tìm ảnh tương tự
similar_images = find_similar_images(QUERY_IMAGE_PATH, shop_features)

# Hiển thị kết quả
print("Top 5 ảnh tương tự:")
for filename, score in similar_images[:5]:
    print(f"Ảnh: {filename}, Điểm: {score}")
