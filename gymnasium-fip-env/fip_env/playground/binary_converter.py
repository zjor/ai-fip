import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

"""
Convert a binary number to decimal using a neural network

Input:
    one of [0, 0, 0] -> ... -> [1, 1, 1]
Output:
    One of [0, ..., 7]
"""


class BinaryConverterNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BinaryConverterNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


def dec_to_bin(a: int, length: int = 3) -> list[int]:
    result = []
    while a > 0:
        result.append(a % 2)
        a //= 2
    while len(result) < length:
        result.append(0)
    result.reverse()
    return result


def get_training_data(size: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    data = []
    for i in range(size):
        y = np.zeros(8)
        y[i % 8] = 1
        data.append(dec_to_bin(i % 8) + y.tolist())
    np.random.shuffle(data)
    data = np.array(data)
    x_train = data[:, :3]
    y_train = data[:, 3:]
    return (torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32))


def main() -> None:
    input_size = 3
    hidden_size = 16
    output_size = 8
    num_epochs = 50000
    learning_rate = 0.01

    x_train, y_train = get_training_data(500)
    model = BinaryConverterNet(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        outputs = model(x_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights

        if (epoch + 1) % 1000 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # testing the model
    with torch.no_grad():
        for i in range(8):
            binary = dec_to_bin(i)
            predicted = model(torch.tensor([binary], dtype=torch.float32))
            print(f'Input: {binary}; Predicted: {np.argmax(predicted[0])}')


if __name__ == '__main__':
    main()
