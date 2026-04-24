import os
import csv
import numpy as np
from skimage import io

def load_image(path):
    image = io.imread(path)

    # если RGBA → убираем альфа-канал
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

    # нормализация среднего
    R_mean, G_mean, B_mean = normalize_rgb(R_mean, G_mean, B_mean)

    return R_mean, G_mean, B_mean, R_std, G_std, B_std

def get_class_name(filename):
    return os.path.splitext(filename)[0]

def create_csv(dataset_path, output_file="dataset.csv"):
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Новый заголовок (6 признаков!)
        writer.writerow([
            "path", "class",
            "R_mean", "G_mean", "B_mean",
            "R_std", "G_std", "B_std"
        ])

        for root, _, files in os.walk(dataset_path):
            for filename in files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(root, filename)

                    try:
                        image = load_image(path)

                        # защита от серых изображений
                        if len(image.shape) < 3:
                            print(f"Пропущено (не RGB): {filename}")
                            continue

                        features = get_features(image)
                        class_name = get_class_name(filename)

                        writer.writerow([path, class_name, *features])

                        print(f"Обработано: {filename}")

                    except Exception as e:
                        print(f"Ошибка: {filename} → {e}")

    print(f"\nCSV файл создан: {output_file}")

DATASET_PATH = "images"
create_csv(DATASET_PATH)