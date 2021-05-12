import argparse
import numpy as np
from PIL import Image
from util import read_data, data_transform, available_actions
from torchvision import transforms
import torch
import random


class CarV0GymDataset(torch.utils.data.Dataset):
    """
    Helper class to allow transformations
    by default TensorDataset doesn't support them
    """

    def __init__(self, pkl_file, transform=True, rare_events=5, classification=False):
        self.pkl_file = pkl_file
        self.transform = transform
        self.data = read_data(pkl_file)
        data_copy = self.data.copy()
        for da in self.data:
            for act in [[-1, 0, 1], [1, 0, 1], [0, 0, 1], [1, 0, 0], [-1, 0, 0]]:
                if np.array_equal(da[1], act):
                    data_copy += (da,) * rare_events
        self.data = data_copy
        random.shuffle(self.data)

        self.states, self.actions = map(np.array, zip(*self.data))

        if classification:
            act_classes = np.full((len(self.actions)), -1, dtype=np.int32)
            for i, act in enumerate(available_actions):
                act_classes[np.all(self.actions == available_actions[act], axis=1)] = i

            self.states = np.array(self.states)
            self.states = self.states[act_classes != -1]
            act_classes = act_classes[act_classes != -1]

            non_accel = act_classes != list(available_actions.values()).index([0, 1, 0])
            drop_mask = np.random.rand(act_classes[~non_accel].size) > 0.7
            non_accel[~non_accel] = drop_mask
            self.states = self.states[non_accel]
            act_classes = act_classes[non_accel]

            non_act = act_classes != list(available_actions.values()).index([0, 0, 0])
            drop_mask = np.random.rand(act_classes[~non_act].size) > 0.3
            non_act[~non_act] = drop_mask
            self.states = self.states[non_act]
            act_classes = act_classes[non_act]

            self.actions = act_classes

    def __getitem__(self, idx):
        y = data_transform(self.states[idx])
        return self.actions[idx], y

    def __len__(self):
        assert len(self.states) == len(self.actions)
        return len(self.states)


class CarV0GymClassificationDataset(torch.utils.data.Dataset):
    """
    Helper class to allow transformations
    by default TensorDataset doesn't support them
    """

    def __init__(self, pkl_file, transform=True):
        self.pkl_file = pkl_file
        self.transform = transform
        self.data = read_data(pkl_file)
        self.states, self.actions = map(np.array, zip(*self.data))

    def __getitem__(self, idx):
        return self.actions[idx], data_transform(self.states[idx])

    def __len__(self):
        return len(states)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        default="data\data_09052021_2204.pkl",
        help="File path with data to check",
    )
    parser.add_argument(
        "--save-file",
        default="data\prepared_data.pkl",
        help="File path with data to check",
    )
    parser.add_argument(
        "--show-example",
        default=False,
        help="Either to show an example of the transformation or not",
    )
    args = parser.parse_args()

    data = read_data(args.file)
    states, actions = map(np.array, zip(*data))
    dataset = CarV0GymDataset(args.file, classification=True)
    if args.show_example:
        selected_state = states[int(len(states) / 2)]
        selected_state = transforms.ToPILImage()(selected_state)
        selected_state.show(title="Original Image")
        selected_state = transforms.Grayscale(1)(selected_state)
        selected_state.show(title="Grayscale")
        selected_state = transforms.Pad((12, 12, 12, 0))(selected_state)
        selected_state.show(title="Padding")
        selected_state = transforms.CenterCrop(84)(selected_state)
        selected_state.show(title="CenterCrop")
        selected_state = transforms.ToTensor()(selected_state)
        selected_state = transforms.Normalize((0,), (1,))(selected_state)
        selected_state = transforms.ToPILImage()(selected_state)
        selected_state.show(title="Final Normalized")
