import matplotlib.pyplot as plt


class ModelLog:
    def __init__(self):
        self.iter_num = 0
        self.epoch_num = 0
        self.current_epoch_iter = 0
        self.iter_per_epoch = {}
        self.evals = {}
        self.states = {}
        self.is_ssl_init = False

    def get_ssl_init(self):
        return self.is_ssl_init

    def set_ssl_init(self):
        self.is_ssl_init = True

    def one_epoch(self):
        self.epoch_num += 1
        self.iter_per_epoch[self.epoch_num] = self.current_epoch_iter
        self.current_epoch_iter = 0

    def get_current_epoch_iter(self):
        return self.current_epoch_iter

    def one_iter(self, train_state):
        self.iter_num += 1
        self.current_epoch_iter += 1
        self.states[self.iter_num] = train_state

    def update_eval(self, data):
        self.evals[self.iter_num] = data

    def plot_eval(self, index=0):
        x1 = []
        y1 = []
        x2 = [0]
        y2 = [0]
        x3 = [0]
        y3 = [0]


        for k, v in self.evals.items():
            if self.states[k]["supervised"]:
                x1.append(k)
                y1.append(v["supervised"][index])
                x2[0] = k
                y2[0] = v["supervised"][index]
                x3[0] = k
                y3[0] = v["supervised"][index]
            else:
                if len(v) == 2:
                    if "teacher" in v.keys():
                        x2.append(k)
                        y2.append(v["teacher"][index])
                    x3.append(k)
                    y3.append(v["student"][index])
                else:
                    x2.append(k)
                    y2.append(v["student"][index])

        plt.plot(x3, y3)
        plt.plot(x2, y2)
        plt.plot(x1, y1)
        plt.show()

