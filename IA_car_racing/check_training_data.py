import os
import pickle
import textwrap
import numpy as np
import argparse

available_actions = {
    "left": [-1, 0, 0],
    "left + break": [-1, 0, 1],
    "aceleration + left + break": [-1, 1, 1],
    "aceleration + left": [-1, 1, 0],
    "aceleration + right + break": [1, 1, 1],
    "aceleration + right": [1, 1, 0],
    "right + break": [1, 0, 1],
    "right": [1, 0, 0],
    "aceleration + break": [0, 1, 1],
    "acceleration": [0, 1, 0],
    "break": [0, 0, 1],
    "no action": [0, 0, 0],
}

parser = argparse.ArgumentParser()
parser.add_argument("--file", help="File path with data to check")
args = parser.parse_args()


def read_data(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError("{} doesn't exist".format(filename))

    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data


if __name__ == "__main__":

    data = read_data(args.file)
    states, actions = map(np.array, zip(*data))

    print("states shape = {}".format(states.shape))
    print("actions shape = {}".format(actions.shape))

    act_classes = np.full((len(actions)), -1, dtype=np.int32)
    for i, a in enumerate(list(available_actions.values())):
        act_classes[np.all(actions == a, axis=1)] = i

    for i, a in enumerate(list(available_actions.keys())):
        if available_actions[a][0] == -1:
            st = "Actions of type {} - {:<30}:".format(available_actions[a], str(a))
        else:
            st = "Actions of type {}  - {:<30}:".format(available_actions[a], str(a))
        print("{} {}".format(st, str(act_classes[act_classes == i].size)))
