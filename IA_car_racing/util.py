import pickle
import os
from torchvision import transforms


def read_data(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError("{} doesn't exist".format(filename))

    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data


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

data_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Grayscale(1),
        transforms.Pad((12, 12, 12, 0)),
        transforms.CenterCrop(84),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ]
)

test_seeds = [1024, 1069, 1120, 1188, 1257, 1489, 1575]
