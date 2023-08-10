# When taking sqrt for initialization you might want to use math package,
# since torch.sqrt requires a tensor, and math.sqrt is ok with integer
import math
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem


class F1(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h: int, d: int, k: int):
        """Create a F1 model as described in pdf.

        Args:
            h (int): Hidden dimension.
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        self.h = h
        self.d = d
        self.k = k

        alpha0 = 1 / math.sqrt(self.d)
        l1 = Uniform(-alpha0, alpha0)
        self.w0 = Parameter(l1.sample((self.h, self.d)))
        self.b0 = Parameter(l1.sample((self.h, 1)))

        alpha1 = 1 / math.sqrt(self.h)
        l2 = Uniform(-alpha1, alpha1)
        self.w1 = Parameter(l2.sample((self.k, self.h)))
        self.b1 = Parameter(l2.sample((self.k, 1)))


    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F1 model.

        It should perform operation:
        W_1(sigma(W_0*x + b_0)) + b_1

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: LongTensor of shape (n, k). Prediction.
        """
        return (self.w1 @ relu(self.w0 @ x.t() + self.b0) + self.b1).t()
class F2(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """Create a F2 model as described in pdf.

        Args:
            h0 (int): First hidden dimension (between first and second layer).
            h1 (int): Second hidden dimension (between second and third layer).
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        self.h0 = h0
        self.h1 = h1
        self.d = d
        self.k = k

        alpha0 = 1 / math.sqrt(self.d)
        l1 = Uniform(-alpha0, alpha0)
        self.w0 = Parameter(l1.sample((self.h0, self.d)))
        self.b0 = Parameter(l1.sample((self.h0, 1)))

        alpha1 = 1 / math.sqrt(self.h0)
        l2 = Uniform(-alpha1, alpha1)
        self.w1 = Parameter(l2.sample((self.h1, self.h0)))
        self.b1 = Parameter(l2.sample((self.h1, 1)))

        alpha2 = 1 / math.sqrt(self.h1)
        l3 = Uniform(-alpha2, alpha2)
        self.w2 = Parameter(l3.sample((self.k, self.h1)))
        self.b2 = Parameter(l3.sample((self.k, 1)))

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        It should perform operation:
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: LongTensor of shape (n, k). Prediction.
        """
        return (self.w2 @ relu(self.w1 @ relu(self.w0 @ x.t() + self.b0) + self.b1) + self.b2).t()


@problem.tag("hw3-A")
def train(model: Module, optimizer: Adam, train_loader: DataLoader) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of training crossentropy losses for each epochs.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce tuples (x, y),
            where x is FloatTensor of shape (n, d) and y is LongTensor of shape (n,).

    Returns:
        List[float]: List containing average loss for each epoch.
    """
    losses = []
    acc = 0
    while acc < 0.99:
        for x, y in train_loader:
            optimizer.zero_grad()
            y_hat = model.forward(x)
            loss = cross_entropy(y_hat, y)
            loss.backward()
            optimizer.step()

        avg_loss, acc = loss_and_accuracy(model, train_loader)
        print(f"Loss: {avg_loss}, Accuracy: {acc}")
        losses.append(avg_loss)
        
    return losses

@problem.tag("hw3-A", start_line=5)
def main():
    """
    Main function of this problem.
    For both F1 and F2 models it should:
        1. Train a model
        2. Plot per epoch losses
        3. Report accuracy and loss on test set
        4. Report total number of parameters for each network

    Note that we provided you with code that loads MNIST and changes x's and y's to correct type of tensors.
    We strongly advise that you use torch functionality such as datasets, but as mentioned in the pdf you cannot use anything from torch.nn other than what is imported here.
    """
    (x, y), (x_test, y_test) = load_dataset("mnist")
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()

    train_loader = DataLoader(TensorDataset(x, y), batch_size=len(x), shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=len(x_test), shuffle=True)

    # 1. Train a model
    f1_model = F1(64, 784, 10)
    f1_optimizer = Adam(f1_model.parameters(), lr=0.1)
    f1_losses = train(f1_model, f1_optimizer, train_loader)

    f2_model = F2(32, 32, 784, 10)
    f2_optimizer = Adam(f2_model.parameters(), lr=0.1)
    f2_losses = train(f2_model, f2_optimizer, train_loader)

    # 2. Plot per epoch losses
    plt.title("Training loss vs epoch (f1 model, batch size = 256, learningRate = 0.001)")
    plt.plot(f1_losses)
    plt.xlabel("Number of epochs")
    plt.ylabel("f1 losses")
    plt.show()

    plt.title("Training loss vs epoch (f2 model, batch size = 256, learningRate = 0.001)")
    plt.plot(f2_losses)
    plt.xlabel("Number of epochs")
    plt.ylabel("f2 losses")
    plt.show()

    # 3. Report accuracy and loss on test set
    f1_loss, f1_accuracy = loss_and_accuracy(f1_model, test_loader)
    f2_loss, f2_accuracy = loss_and_accuracy(f2_model, test_loader)

    print(f"F1: loss: {f1_loss}, accuracy: {f1_accuracy}")
    print(f"F2: loss: {f2_loss}, accuracy: {f2_accuracy}")

    # 4. Report total number of parameters for each network
    f1_params = sum(p.numel() for p in f1_model.parameters())
    f2_params = sum(p.numel() for p in f2_model.parameters())
    print(f"F1: {f1_params} parameters")
    print(f"F2: {f2_params} parameters")

def loss_and_accuracy(model: Module, data_loader: DataLoader) -> float:
    correct = 0
    loss = 0
    for x, y in data_loader:
        y_hat = model.forward(x)
        loss += cross_entropy(y_hat, y).item()
        correct += (y_hat.argmax(dim=1) == y).sum().item()
    return (loss / len(data_loader)), (correct / len(data_loader.dataset))

if __name__ == "__main__":
    main()
