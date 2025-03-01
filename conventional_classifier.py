import os
import re
from pathlib import Path
import collections
import random
from argparse import ArgumentParser

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("Agg")
import seaborn as sns
import matplotlib.pyplot as plt

import imageio
from sklearn import neighbors, model_selection, preprocessing, ensemble, svm


sns.set()
DATASET_PATH = os.environ["DBPRO_DATASET"]
REGEX_PERCENTAGE = re.compile(r"^t.*_p(\d\.\d{2})$")
REGEX_NAME = re.compile(r"^(t.*)_p\d\.\d{2}$")


CORINE_MAP_L2 = {
    "11": "Urban fabric",
    "12": "Industrial, comercial and transport units",
    "13": "Mine, dump and construction sites",
    "14": "Artificial, non-agricultural vegetated areas",
    "21": "Arable land",
    "22": "Permanent crops",
    "23": "Pastures",
    "24": "Heterogenous agricultural areas",
    "31": "Forest",
    "32": "Shrub and/or herbaceous vegetation associations",
    "33": "Open spaces with little or no vegetation",
    "41": "Inland wetlands",
    "42": "Coastal wetlands",
    "51": "Inland waters",
    "52": "Marine waters",
}

CORINE_MAP_L1 = {
    "1": "Artificial",
    "2": "Agricultural",
    "3": "Forest",
    "4": "Wetlands",
    "5": "Water",
}


def overview_data(path: Path, name: str):
    """Plot distribution of CLC label ratio in tiles."""
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


class Image:
    """Wrapper for loading images."""
    def __init__(self, path):
        if isinstance(path, self.__class__):
            path = path.path

        self.name = REGEX_NAME.match(path.stem)[1]
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
    """Collections of images."""
    def __init__(self, path):
        if isinstance(path, list):
            self.data = path
        else:
            path = Path(path)
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
        """Filter dataset on percentage in major class."""
        data = [i for i in self.data if i.percentage >= percentage]
        return self.__class__(data)

    def sample_each(self, count, level="l2", classes=None):
        """Return a count number of images for the given classes in the
        specified level.  If no classes are given return count for all classes
        in the given level.
        """
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

    def save_prediction(self, preds, destination):
        """Export predictions to csv"""
        pred_json = []
        for img_obj, pred in zip(self.data, preds):
            pred_json.append({"name": img_obj.name, "pred": pred})
        pred_df = pd.DataFrame(pred_json)
        pred_df.to_csv(str(destination) + ".csv", index=False)

    def split(self, train):
        """Split dataset into train and test dataset."""
        data = random.sample(self.data, len(self.data))
        return self.__class__(data[:train]), self.__class__(data[train:])


def create_np_data(dataset: Dataset, level="l2"):
    """Convert image to numpy array of images."""
    imgs = []
    labels = []
    for image_data in dataset.data:
        imgs.append(image_data.load_data())
        labels.append(getattr(image_data, level))
    return imgs, labels


def preprocess_svm(x_data, y_data, scaler=None):
    """Put 2-D image into continuous 1-D sequence."""
    x_data = [np.ravel(i) for i in x_data]
    return x_data, y_data, None


def preprocess_randomforest(x_data, y_data, scaler=None):
    """Scale images between 1 and 0."""
    x_data = [np.ravel(i) for i in x_data]
    if scaler is None:
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(x_data)
    x_data = scaler.transform(x_data)
    return x_data, y_data, scaler


def batched_generate(dataset, batch_size=100):
    """Generate batches of data for predicting on the entire dataset."""
    num_files = len(dataset.data)
    for i in range(0, num_files, batch_size):
        start = i
        end = min(num_files, start + batch_size)
        batch = Dataset(dataset.data[start:end])
        yield batch


def count_tables(path, name):
    """Generate latex tables with counts per Corine class."""
    l1_counts = {}
    for year in ["2018", "2019"]:
        dataset = Dataset(path / year)
        final_dict = {}
        l2_counts = dataset.l2_counts
        for l1, count in dataset.l1_counts.items():
            matched_keys = sorted([k for k in l2_counts if k.startswith(l1)])
            for k in matched_keys:
                final_dict[".".join(list(k)) + " " + CORINE_MAP_L2[k]] = l2_counts[k]
            final_dict[l1 + " " + CORINE_MAP_L1[l1]] = count
        l1_counts[year] = final_dict

    l1_table = pd.DataFrame.from_dict(l1_counts)
    print(l1_table)
    with open(f"{name}.tex", "w") as lfile:
        lfile.write(l1_table.to_latex())


def main(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = Path(args.input)
    ds_2018 = Dataset(dataset_dir / "2018")

    # Classification classes initial
    classes = {
        "l1": ["1", "2", "3", "4", "5"],
        "l2": [
            "11", "12", "13", "14",
            "21", "22", "23", "24",
            "31", "32",
            "41", "42",
            "51", "52"
        ],
    }

    level = args.level

    sampled_2018 = ds_2018.sample_each(300, classes=classes[level], level=level)
    train, test = sampled_2018.split(train=200)

    models = [
        [
            "knn",
            "kNN (n = {})",
            preprocess_svm,
            lambda n: neighbors.KNeighborsClassifier(n_neighbors=n, n_jobs=10),
            [2, 5, 10, 20]
        ],
        [
            "randomforest",
            "RandomForest (estimators = {})",
            preprocess_randomforest,
            lambda n: ensemble.RandomForestClassifier(n_estimators=n),
            [10, 100, 1000],
        ],
        [
            "svm",
            "Linear SVM (C = {})",
            preprocess_randomforest,
            lambda n: svm.LinearSVC(C=n),
            [0.1, 1, 10]
        ],
    ]
    tests = {}
    best_model = ()
    for model_name, label, preprocess, modelfun, grid in models:
        results = []

        x_train, y_train, scaler = preprocess(*create_np_data(train, level=level))
        x_test, y_test, _ = preprocess(*create_np_data(test, level=level), scaler=scaler)
        for n in grid:
            print("Training model")
            model = modelfun(n)
            model.fit(x_train, y_train)

            print("Predicting using model")
            result = model.predict(x_test)
            correct = 0
            for pred, actual in zip(result, y_test):
                if pred == actual:
                    correct += 1
            test.save_prediction(result, output_dir / f"final_pred_{model_name}")
            print(f"Acc (n = {n})", correct / len(result))
            acc = correct / len(result)
            if not best_model or acc > best_model[0]:
                best_model = (acc, model_name, preprocess, scaler, model)
            results.append((n, acc))
        best_n, best_acc = max(results, key=lambda r: r[1])
        fmt_label = label.format(best_n)
        tests[fmt_label] = {"Accuracy": best_acc}

    if args.predict:
        print("Using best model to predict 2019 data")
        ds_2019 = Dataset(dataset_dir / "2019")
        all_preds = []
        for batch in batched_generate(ds_2019, batch_size=100):
            test_2019, _, _ = best_model[2](*create_np_data(batch), scaler=best_model[3])
            preds = best_model[4].predict(test_2019)
            all_preds += list(preds)
        ds_2019.save_prediction(all_preds, output_dir / "pred_2019")

    result_df = pd.DataFrame.from_dict(tests, orient="index")
    print(result_df)
    with open(str(output_dir / "results.tex"), "w") as cfile:
        cfile.write(result_df.to_latex())


if __name__ == "__main__":
    PARSER = ArgumentParser()
    PARSER.add_argument("--predict", help="Predict 2019 data", action="store_true")
    PARSER.add_argument("--level", choices=("l1", "l2"), default="l1")
    PARSER.add_argument("input", help="Input dataset")
    PARSER.add_argument("output", help="Output classification result tables")
    main(PARSER.parse_args())
