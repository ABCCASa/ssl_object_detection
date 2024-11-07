import common_utils
import os
import json
def get_default():
    return TrainConfig()


class TrainConfig:
    def __init__(self, load_dir=None):

        if load_dir is not None:
            with open(load_dir) as file:
                data = json.load(file)

            self.LABELED_SAMPLE = data["LABELED_SAMPLE"]
            self.UNLABELED_SAMPLE = data["UNLABELED_SAMPLE"]
            self.VALID_SAMPLE = data["VALID_SAMPLE"]

            self.SEMI_SUPERVISED_TRAIN_START = data["SEMI_SUPERVISED_TRAIN_START"]
            self.PSEUDO_LABEL_THRESHOLD = data["PSEUDO_LABEL_THRESHOLD"]
            self.EMA_UPDATE_BETA = data["EMA_UPDATE_BETA"]
            self.UNSUPERVISED_WEIGHT = data["UNSUPERVISED_WEIGHT"]
            return

        print("Train Config Init:")
        self.LABELED_SAMPLE = common_utils.get_exist_dir("Please enter labeled samples txt: ")
        self.UNLABELED_SAMPLE = common_utils.get_exist_dir("Please enter unlabeled samples txt: ")
        self.VALID_SAMPLE = common_utils.get_exist_dir("Please enter valid samples txt: ")

        # Semi Supervised Learning
        self.SEMI_SUPERVISED_TRAIN_START = 100  # start semi-supervised learning are x epoch
        self.PSEUDO_LABEL_THRESHOLD = 0.8
        self.EMA_UPDATE_BETA = 0.9999
        self.UNSUPERVISED_WEIGHT = 0.5

        print(f"SEMI_SUPERVISED_TRAIN_START = {self.SEMI_SUPERVISED_TRAIN_START}",
              f"PSEUDO_LABEL_THRESHOLD = {self.PSEUDO_LABEL_THRESHOLD}",
              f"EMA_UPDATE_BETA = {self.EMA_UPDATE_BETA}",
              f"UNSUPERVISED_WEIGHT = {self.UNSUPERVISED_WEIGHT}",
              sep="\n")
        if input("Enter Y to change the default value: ").lower() == "y":
            self.SEMI_SUPERVISED_TRAIN_START = common_utils.input_int("Please enter semi -supervised start epoch: ", 1)
            self.PSEUDO_LABEL_THRESHOLD = common_utils.input_float("please enter pseudo label threshold: ", 0, 1)
            self.EMA_UPDATE_BETA = common_utils.input_float("please enter ema beta: ", 0, 1)
            self.UNSUPERVISED_WEIGHT = common_utils.input_float("please enter unsupervised weight: ", 0, 100)

    def print_out(self):
        print("Train Config:",
              f"LABELED_SAMPLE = {self.LABELED_SAMPLE}",
              f"UNLABELED_SAMPLE = {self.UNLABELED_SAMPLE}",
              f"VALID_SAMPLE = {self.VALID_SAMPLE}",
              f"SEMI_SUPERVISED_TRAIN_START = {self.SEMI_SUPERVISED_TRAIN_START}",
              f"PSEUDO_LABEL_THRESHOLD = {self.PSEUDO_LABEL_THRESHOLD}",
              f"EMA_UPDATE_BETA = {self.EMA_UPDATE_BETA}",
              f"UNSUPERVISED_WEIGHT = {self.UNSUPERVISED_WEIGHT}",
              sep="\n")


    def save(self, save_folder):
        config = {
            "LABELED_SAMPLE":self.LABELED_SAMPLE,
            "UNLABELED_SAMPLE":self.UNLABELED_SAMPLE,
            "VALID_SAMPLE": self.VALID_SAMPLE,
            "SEMI_SUPERVISED_TRAIN_START":self.SEMI_SUPERVISED_TRAIN_START,
            "PSEUDO_LABEL_THRESHOLD": self.PSEUDO_LABEL_THRESHOLD,
            "EMA_UPDATE_BETA": self.EMA_UPDATE_BETA,
            "UNSUPERVISED_WEIGHT": self.UNSUPERVISED_WEIGHT
        }
        with open(save_folder, 'w', encoding='utf-8') as file:
            json.dump(config, file)


