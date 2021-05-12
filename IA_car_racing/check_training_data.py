import os
import pickle
import textwrap
import numpy as np
import argparse
from util import available_actions, read_data

parser = argparse.ArgumentParser()
parser.add_argument(
    "--file", default="data\data.pkl", help="File path with data to check"
)
args = parser.parse_args()


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
