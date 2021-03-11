import pickle
import os
from pure_sklearn.map import convert_estimator


def load_classifier(path_to_sav):
    """Load saved classifier"""
    file = open(path_to_sav, "rb")
    classifier = pickle.load(file)
    file.close()
    return classifier


def convert_classifier(path):
    # convert to pure python estimator
    print("Loading classifier...")
    clf = load_classifier(path)
    dir_path = os.path.dirname(path)
    filename = os.path.basename(path)
    filename, _ = filename.split(".")
    clf_pure_predict = convert_estimator(clf)
    with open(dir_path + "/" + filename + "_pure.sav", "wb") as f:
        pickle.dump(clf_pure_predict, f)
    print(f"Converted Classifier {filename}")


if __name__ == "__main__":
    path_to_classifier = r"D:\SimBa\Jens_models\pursuit_prediction_11.sav"
    convert_classifier(path_to_classifier)
