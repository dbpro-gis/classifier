import os
import re
from pathlib import Path
import collections

import numpy as np

import matplotlib as mpl
mpl.use("Agg")
import seaborn as sns
import matplotlib.pyplot as plt

import imageio
import sklearn as sk
from sklearn import neighbors, model_selection


sns.set()
DATASET_PATH = os.environ["DBPRO_DATASET"]
REGEX_PERCENTAGE = re.compile(r"^t.*_p(\d\.\d{2})$")


CORINE_MAP = {
    "11": "Urban fabric",
    "12": "Transport",
    "13": "Mine, dump, construction sites"
}
CORINE_MAP_L1 = {
    "1": "Artificial",
    "2": "Agricultural",
    "3": "Forest",
    "4": "Wetlands",
    "5": "Water",
}


def overview_data(path: Path, name: str):
    classes = collections.defaultdict(list)
    for class_dir in path.iterdir():
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for image_path in class_dir.glob("t*.png"):
            percentage = float(REGEX_PERCENTAGE.match(image_path.stem)[1])
            classes[class_name].append(percentage)

    l2_classes = collections.defaultdict(list)
    for name, data in classes.items():
        l2_classes[name[:2]] += data

    l1_classes = collections.defaultdict(list)
    for name, data in classes.items():
        l1_classes[name[:1]] += data
    for name, data in l1_classes.items():
        print(name, len(data))
        sns.kdeplot(data, label=CORINE_MAP_L1[name], shade=True)
    plt.xlim(0, 1)
    plt.title("Distribution of major CLC label in patches")
    plt.xlabel("Ratio of highest class")
    plt.tight_layout()
    plt.savefig(f"l1_{name}.png")
    plt.close()

    # for prefix in ["1", "2", "3", "4", "5"]:
    #     for name, data in l2_classes.items():
    #         if not name.startswith(prefix):
    #             continue
    #         print(name, len(data))
    #         sns.kdeplot(data, label=name)
    #     plt.tight_layout()
    #     plt.savefig(f"test_{prefix}.png")
    #     plt.close()


class Image:
    def __init__(self, path):
        if isinstance(path, self.__class__):
            path = path.path

        self.percentage = float(REGEX_PERCENTAGE.match(path.stem)[1])
        self.label = str(path.parent.name)
        self.l1 = self.label[0]
        self.l2 = self.label[0:2]
        self.l3 = self.label
        self.year = str(path.parent.parent.name)

        self.path = path

    def load_data(self):
        return imageio.imread(self.path)


class Dataset:
    def __init__(self, path):
        if isinstance(path, list):
            self.data = path
        else:
            classes = collections.defaultdict(list)
            self.data = [Image(p) for p in path.glob("**/t*.png")]

    @property
    def l1(self):
        return sorted(list(self.l1_counts.keys()))

    @property
    def l2(self):
        return sorted(list(self.l2_counts.keys()))

    @property
    def l3(self):
        return sorted(list(self.l3_counts.keys()))

    @property
    def l1_counts(self):
        return collections.Counter(i.l1 for i in self.data)

    @property
    def l2_counts(self):
        return collections.Counter(i.l2 for i in self.data)

    @property
    def l3_counts(self):
        return collections.Counter(i.l3 for i in self.data)

    def filter(self, percentage):
        data = [i for i in self.data if i.percentage >= percentage]
        return self.__class__(data)

    def sample_each(self, count, level="l2", classes=None):
        data = []
        if classes is None:
            classes = getattr(self, level)
        counts = {c: 0 for c in classes}
        for image in self.data:
            image_level = getattr(image, level)
            if image_level not in counts:
                continue
            if counts[image_level] < count:
                data.append(image)
                counts[image_level] += 1
        return self.__class__(data)


def create_np_data(dataset: Dataset, level="l2"):
    imgs = []
    labels = []
    for image_data in dataset.data:
        imgs.append(image_data.load_data())
        labels.append(getattr(image_data, level))
    return imgs, labels


def preprocess_svm(x_data, y_data):
    x_data = [np.ravel(i) for i in x_data]
    return x_data, y_data


def main():
    dataset_dir = Path(DATASET_PATH)
    # overview_data(dataset_dir / "2018", "2018v2")
    # overview_data(dataset_dir / "2019", "2019v2")
    ds_2018 = Dataset(dataset_dir / "2018")
    ds_2018_high = ds_2018.filter(0.4)
    print(ds_2018_high.l2_counts)

    ds_2019 = Dataset(dataset_dir / "2019")
    ds_2019_high = ds_2019.filter(0.4)
    print(ds_2019_high.l2_counts)


    # l2_2018 = ds_2018_high.l2
    # l2_2019 = ds_2019_high.l2
    # for l in l2_2018:
    #     if l not in l2_2019:
    #         print("Not in 2019", l)
    # for l in l2_2019:
    #     if l not in l2_2018:
    #         print("Not in 2018", l)

    # Classification classes initial
    classes = [
        "11", "12", "13", "14",
        "21", "22", "23", "24",
        "31", "32",
        "41", "42",
        "51", "52"
    ]

    sampled_2018 = ds_2018_high.sample_each(100, classes=classes)
    sampled_2019 = ds_2019_high.sample_each(100, classes=classes)

    x_train, y_train = create_np_data(sampled_2018)
    x_train, y_train = preprocess_svm(x_train, y_train)

    x_test, y_test = preprocess_svm(*create_np_data(sampled_2019))

    for n in [2, 5, 10]:
        print("Training model")
        model = neighbors.KNeighborsClassifier(n_neighbors=n)
        model.fit(x_train, y_train)

        print("Predicting using model")
        result = model.predict(x_test)
        correct = 0
        for pred, actual in zip(result, y_test):
            if pred == actual:
                correct += 1
        print(f"Acc (n = {n})", correct / len(result))


if __name__ == "__main__":
    main()
