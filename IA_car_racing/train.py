import argparse
from model import BehavioralCloningModel
from prepare_data import CarV0GymDataset
from util import available_actions, data_transform
import torch
import os
import time
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


def save_model(model, path_to_save="models"):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path_to_save)
    file_name = "model_{}.pkl".format(time.strftime("%d%m%Y_%H%M", time.localtime()))
    model_dir = os.path.join(path, file_name)
    torch.save(model.state_dict(), model_dir)
    print("Model saved in file: {}".format(model_dir))


def train(filename, epochs = 2, train_val_split = 0.85, batch_size = 64):
    dataset = CarV0GymDataset(filename)
    train_size = int(train_val_split * len(dataset))
    val_size = int(len(dataset) - train_size)
    trainset, valset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=True,
    )

    model = BehavioralCloningModel()

    dataiter = iter(train_loader)
    _, images = dataiter.next()
    writer.add_graph(model, images)

    loss_function = torch.nn.CrossEntropyLoss()
    loss_function = torch.nn.MSELoss()  # MSE loss
    optimizer = torch.optim.Adam(model.parameters())

    train_loss, valid_loss = 0, 0
    train_epoch_loss, valid_epoch_loss = 0, 0
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        model.train()
        for i, (actions, states) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(states)
            loss = loss_function(outputs.float(), actions.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_epoch_loss += train_loss

            if i % 100 == 99:
                print(
                    "[Epoch: {}, Mini-batch: {}] loss:{}".format(
                        epoch + 1, i + 1, train_loss / 100
                    )
                )
                writer.add_scalar(
                    "Training loss", train_loss / 100, epoch * len(train_loader) + i + 1
                )
                train_loss = 0
        print(
            "Average train epoch loss: {}".format(
                (train_epoch_loss / len(train_loader))
            )
        )
        writer.add_scalar(
            "Epoch training loss", train_epoch_loss / len(train_loader), epoch + 1
        )
        train_epoch_loss = 0

        model.eval()
        with torch.no_grad():
            for i, (actions, states) in enumerate(val_loader):
                outputs = model(states)
                valid_loss += loss_function(outputs.float(), actions.float()).item()
                valid_epoch_loss += valid_loss
                if i % 10 == 9:
                    print(
                        "[Epoch: {}, Mini-batch: {}] val_loss:{}".format(
                            epoch + 1, i + 1, valid_loss / 10
                        )
                    )
                    writer.add_scalar(
                        "Validation loss", valid_loss / 10, epoch * len(val_loader) + i + 1
                    )
                    valid_loss = 0
            print("Average validation epoch loss: {}".format(valid_epoch_loss/len(val_loader)))
            writer.add_scalar(
                "Epoch validation loss", valid_epoch_loss / len(val_loader), epoch + 1
            )
            valid_epoch_loss = 0
    save_model(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file", default="data\data_09052021_2204.pkl", help="File path with data to check"
    )
    parser.add_argument("--epochs", default=2, help="Number of epochs")
    parser.add_argument("--split", default=0.85, help="Train/val ratio")
    parser.add_argument("--batch-size", default=64, help="Batch size")
    args = parser.parse_args()
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.file)
    epochs = int(args.epochs)
    train_val_split = args.split
    batch_size = args.batch_size
    train(path, epochs, train_val_split, batch_size)
