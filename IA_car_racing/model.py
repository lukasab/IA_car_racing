from util import available_actions
import numpy as np
import torch


class BehavioralCloningModel(torch.nn.Module):
    def __init__(self, chIn=1, ch=2):
        super(BehavioralCloningModel, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=chIn, out_channels=ch * 8, kernel_size=7),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=ch * 8, out_channels=ch * 16, kernel_size=5, stride=2
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=ch * 16, out_channels=ch * 32, kernel_size=3, stride=2
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=ch * 32, out_channels=ch * 32, kernel_size=3, stride=2
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=ch * 32, out_channels=ch * 64, kernel_size=3, stride=2
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=ch * 64, out_channels=ch * 64, kernel_size=3, stride=2
            ),
            torch.nn.ReLU(),
        )
        self.v = torch.nn.Sequential(
            torch.nn.Linear(64 * ch * 1 * 1, 256), torch.nn.ReLU()
        )
        self.fc = torch.nn.Linear(256, 3)
        self.ch = ch

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.v(x)
        x = self.fc(x)

        x[:, 0] = torch.tanh(x[:, 0])
        x[:, 1] = torch.sigmoid(x[:, 1])
        x[:, 2] = torch.sigmoid(x[:, 2])
        return x


class BehavioralCloningClassificationModel(torch.nn.Module):
    def __init__(
        self,
    ):
        super(BehavioralCloningClassificationModel, self).__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 8, 4),
            torch.nn.BatchNorm2d(32),
            torch.nn.ELU(),
            torch.nn.Dropout2d(0.5),
            torch.nn.Conv2d(32, 64, 4, 2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ELU(),
            torch.nn.Dropout2d(0.5),
            torch.nn.Conv2d(64, 64, 3, 1),
            torch.nn.ELU(),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.BatchNorm1d(64 * 7 * 7),
            torch.nn.Dropout(),
            torch.nn.Linear(64 * 7 * 7, 120),
            torch.nn.ELU(),
            torch.nn.BatchNorm1d(120),
            torch.nn.Dropout(),
            torch.nn.Linear(120, len(available_actions)),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

    def take_actions(self, state):
        with torch.set_grad_enabled(False):
            outputs = self(state)

        normalized = torch.nn.functional.softmax(outputs, dim=1)
        max_action = np.argmax(normalized.cpu().numpy()[0])
        action = available_actions[max_action]
        return action


if __name__ == "__main__":
    print("Printing BahavioralCloningModel")
    print(BehavioralCloningModel())

    print("Printing BahavioralClassificationCloningModel")
    print(BehavioralCloningClassificationModel())
