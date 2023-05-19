import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLASSES = 10


class CNN(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES):
        super(CNN, self).__init__()

        # 1st conv layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=32)

        # 2nd conv layer
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=32)

        # 3rd conv layer
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=2, padding=1
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)

        # flatten output and feed it into dense layer
        self.fc1 = nn.Linear(in_features=32 * 17 * 34, out_features=64)
        self.dropout = nn.Dropout(p=0.3)

        # output layer
        self.fc2 = nn.Linear(in_features=64, out_features=n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1st conv stack
        x = F.relu(self.bn1(self.pool1(self.conv1(x))))
        # 2nd conv stack
        x = F.relu(self.bn2(self.pool2(self.conv2(x))))
        # 3rd conv stack
        x = F.relu(self.bn3(self.pool3(self.conv3(x))))
        # flatten
        x = torch.flatten(x, 1)
        # FC + ReLU + Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # FC + softmax
        x = self.fc2(x)
        probabilities = F.softmax(x, dim=1)
        return probabilities
