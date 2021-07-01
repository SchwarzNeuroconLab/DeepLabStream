import pickle
import joblib
import os
from pure_sklearn.map import convert_estimator


def load_classifier_SIMBA(path_to_sav):
    """Load saved classifier"""
    file = open(path_to_sav, "rb")
    classifier = pickle.load(file)
    file.close()
    return classifier

def load_classifier_BSOID(path_to_sav):
    """Load saved classifier"""
    file = open(path_to_sav, "rb")
    clf = joblib.load(file)
    file.close()
    return clf


def convert_classifier(path, origin: str):
    # convert to pure python estimator
    dir_path = os.path.dirname(path)
    filename = os.path.basename(path)
    filename, _ = filename.split(".")

    print("Loading classifier...")
    if origin.lower() == 'simba':
        clf = load_classifier_SIMBA(path)
        clf_pure_predict = convert_estimator(clf)
        with open(dir_path + "/" + filename + "_pure.sav", "wb") as f:
            pickle.dump(clf_pure_predict, f)

    elif origin.lower() == 'bsoid':
        clf_pack = load_classifier_BSOID(path)
        # bsoid exported classfier has format [a, b, c, clf, d, e]
        clf_pure_predict = convert_estimator(clf_pack[3])
        clf_pack[3] =clf_pure_predict
        with open(dir_path + "/" + filename + "_pure.sav", "wb") as f:
            joblib.dump(clf_pack, f)
    else:
        raise ValueError(f'{origin} is not a valid classifier origin.')

    print(f"Converted Classifier {filename}")


if __name__ == "__main__":

    """Converted BSOID Classifiers are not integrated yet, although you can already convert them here"""
    path_to_classifier = "PATH_TO_CLASSIFIER"
    convert_classifier(path_to_classifier, origin= 'SIMBA')
