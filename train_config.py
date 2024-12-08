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
            for k, v in data.items():
                setattr(self, k, v)
            return

        print("Train Config Init:")
        self.LABELED_SAMPLE = common_utils.get_exist_dir("labeled samples txt: ")
        self.UNLABELED_SAMPLE = common_utils.get_exist_dir("unlabeled samples txt: ")
        self.VALID_SAMPLE = common_utils.get_exist_dir("enter valid samples txt: ")
        self.SEMI_SUPERVISED_TRAIN_START = common_utils.input_int("semi-supervised start epoch (200000): ",
                                                                  1, default_value=200000)
        self.SEMI_SUPERVISED_TRAIN_END = common_utils.input_int("semi-supervised start epoch (200000): ",
                                                                self.SEMI_SUPERVISED_TRAIN_START, default_value=250000)
        self.PSEUDO_LABEL_THRESHOLD = common_utils.input_float("pseudo label threshold (0.8): ", 0,
                                                               1, 0.8)
        self.EMA_UPDATE_BETA = common_utils.input_float("ema beta (0.9999): ", 0, 1, 0.9999)
        self.UNSUPERVISED_WEIGHT = common_utils.input_float("unsupervised weight (1): ", 0, 100, 1)
        self.FUSE_COUNT = common_utils.input_int("fuse count (5): ", 1, 100)
        self.PSEUDO_LABEL_DECAY = common_utils.input_float("pseudo label decay (1): ", 0, 1, 1)

    def print_out(self):
        print("Train Config:")
        for k, v in self.__dict__.items():
            print(f"{k} = {v}")

    def save(self, save_folder):
        with open(save_folder, 'w', encoding='utf-8') as file:
            json.dump(self.__dict__, file)


