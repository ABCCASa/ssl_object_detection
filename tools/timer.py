import time


class Timer:
    def __init__(self):
        self.is_start = False
        self.saved_time = 0
        self.start_time = None

    def get_total_time(self) -> float:
        total_time = self.saved_time
        if self.is_start:
            total_time += (time.time() - self.start_time)
        return total_time

    def start(self):
        if not self.is_start:
            self.is_start = True
            self.start_time = time.time()

    def stop(self):
        if self.is_start:
            self.is_start = False
            self.saved_time += (time.time() - self.start_time)

    def reset(self):
        self.is_start = False
        self.saved_time = 0
        self.start_time = None
