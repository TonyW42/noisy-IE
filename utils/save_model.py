import pickle
from datetime import datetime


def current_time():
    now = datetime.now()  # current date and time
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    return date_time


class ModelSaver:
    def __init__(self, path="./model_save/") -> None:
        self.path = path

    def save_model(self, model):
        date_time = current_time()
        with open(self.path + date_time + ".pickle", "wb") as pickle_out:
            pickle.dump(model, pickle_out)

    def load_model(self, name):
        pickle_in = open(self.path + name + ".pickle", "rb")
        return pickle.load(pickle_in)
