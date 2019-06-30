"""
Remove tiles that contain a high percentage of black pixels
"""
import json
import numpy as np
import conventional_classifier as cc;


def black_ratio(img):
    blacks = np.invert(np.any(img, axis=-1))
    x, y = blacks.shape
    return np.sum(blacks) / (x * y)


def filter_black_pixels(dataset, ratio, remove=False):
    black_tiles = 0
    all_len = len(dataset.data)
    for i, imgpath in enumerate(dataset.data):
        img = imgpath.load_data()
        img_ratio = black_ratio(img)
        if i % 100 == 0:
            print(f"{i}/{all_len}: {black_tiles}")
        if img_ratio > ratio:
            if remove:
                imgpath.path.unlink()
            black_tiles += 1
    print("Number of black tiles", black_tiles)


def main():
    black_pixel_cutoff_ratio = 0.5
    print("Checking images for completely black images")
    dataset = cc.Dataset("../sentinel-data/final-dataset-cleaned/2019")
    filter_black_pixels(dataset, black_pixel_cutoff_ratio, remove=True)
    dataset = cc.Dataset("../sentinel-data/final-dataset-cleaned/2018")
    filter_black_pixels(dataset, black_pixel_cutoff_ratio, remove=True)


if __name__ == "__main__":
    main()
