import os.path

import matplotlib.pyplot as plt


class ModelLog:
    def __init__(self):
        self.iter_num = 0
        self.epoch_num = 0
        self.evals = {}
        self.states = {}
        self.is_ssl_init = False

    def get_ssl_init(self):
        return self.is_ssl_init

    def set_ssl_init(self):
        self.is_ssl_init = True

    def one_epoch(self):
        self.epoch_num += 1

    def one_iter(self, data):
        self.iter_num += 1
        self.states[self.iter_num] = data

    def update_eval(self, data):
        self.evals[self.iter_num] = data

    def plot_eval(self, save_folder=None, index=0):
        y_labels = ["Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]",
                    "Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]",
                    "Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]",
                    "Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]",
                    "Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]",
                    "Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]",
                    "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]",
                    "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]",
                    "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]",
                    "Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]",
                    "Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]",
                    "Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]"
                    ]
        x1, y1 = [], []
        x2, y2 = [0], [0]
        x3, y3 = [0], [0]
        for k, v in self.evals.items():
            if self.states[k]["full_supervised"]:
                x1.append(k)
                y1.append(v["supervised"][index])
                x2[0] = k
                y2[0] = v["supervised"][index]
                x3[0] = k
                y3[0] = v["supervised"][index]
            else:
                if "teacher" in v.keys():
                    x2.append(k)
                    y2.append(v["teacher"][index])
                x3.append(k)
                y3.append(v["student"][index])

        plt.plot(x3, y3)
        plt.plot(x2, y2)
        plt.plot(x1, y1)
        plt.xlabel("Iterations")
        plt.ylabel(y_labels[index])


        if save_folder is None:
            plt.show()
        else:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            plt.savefig(os.path.join(save_folder, "eval.png"))
        plt.clf()
