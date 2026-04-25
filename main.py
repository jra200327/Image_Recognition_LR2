import csv
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog


def load_dataset(csv_path):
    dataset = []

    with open(csv_path, newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        for row in reader:
            dataset.append({
                "path": row["path"],
                "class": row["class"],
                "R_mean": float(row["R_mean"]),
                "G_mean": float(row["G_mean"]),
                "B_mean": float(row["B_mean"]),
                "R_std": float(row["R_std"]),
                "G_std": float(row["G_std"]),
                "B_std": float(row["B_std"])
            })

    return dataset

def load_image(path):
    image = io.imread(path)

    if len(image.shape) == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    return image

def crop_center(image, crop_ratio=0.6):
    h, w = image.shape[:2]

    ch = int(h * crop_ratio)
    cw = int(w * crop_ratio)

    start_h = (h - ch) // 2
    start_w = (w - cw) // 2

    return image[start_h:start_h+ch, start_w:start_w+cw]

def normalize_rgb(r, g, b):
    s = r + g + b
    if s == 0:
        return 0, 0, 0
    return r/s, g/s, b/s

def get_features(image):
    image = crop_center(image)

    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    R_mean, G_mean, B_mean = np.mean(R), np.mean(G), np.mean(B)
    R_std, G_std, B_std = np.std(R), np.std(G), np.std(B)

    R_mean, G_mean, B_mean = normalize_rgb(R_mean, G_mean, B_mean)

    R_std = R_std / 255
    G_std = G_std / 255
    B_std = B_std / 255

    return R_mean, G_mean, B_mean, R_std, G_std, B_std

def distance(f1, f2):
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(f1, f2)))

def classify(image, dataset, k=5):
    features = get_features(image)

    distances = []

    for item in dataset:
        db_features = (
            item["R_mean"], item["G_mean"], item["B_mean"],
            item["R_std"], item["G_std"], item["B_std"]
        )

        d = distance(features, db_features)
        distances.append((d, item["class"]))

    # сортируем по расстоянию
    distances.sort(key=lambda x: x[0])

    # берём k ближайших
    top_k = distances[:k]

    # голосование
    votes = {}

    for _, cls in top_k:
        if cls not in votes:
            votes[cls] = 0
        votes[cls] += 1

    # выбираем класс с максимальным числом голосов
    best_class = max(votes.items(), key=lambda x: x[1])[0]

    # можно также вернуть "уверенность"
    confidence = votes[best_class] / k

    return best_class, confidence

def visualize(image, predicted_class):
    plt.imshow(image)
    plt.title(f"Predicted: {predicted_class}")
    plt.axis('off')
    plt.show()

def test_model(dataset):
    correct = 0
    total = 0

    for item in dataset:
        if not os.path.exists(item["path"]):
            continue

        image = load_image(item["path"])
        predicted, _ = classify(image, dataset)

        print(f"True: {item['class']} | Predicted: {predicted}")

        if predicted == item["class"]:
            correct += 1

        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"\nAccuracy: {accuracy:.2f}")

def select_image_file():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Выберите изображение",
        filetypes=[("Image files", "*.png *.jpg *.jpeg")]
    )

    return file_path

if __name__ == "__main__":
    CSV_PATH = "dataset.csv"

    dataset = load_dataset(CSV_PATH)

    print("=== Тестирование модели ===")
    test_model(dataset)

    print("\n=== Распознавание нового изображения ===")
    test_image_path = select_image_file()

    if test_image_path:
        img = load_image(test_image_path)

        predicted, confidence = classify(img, dataset, k=5)

        print(f"Результат: {predicted}")
        print(f"Уверенность: {confidence:.2f}")

        visualize(img, predicted)
    else:
        print("Файл не выбран")