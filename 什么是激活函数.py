# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import matplotlib.pyplot as plt


# Define the activation functions using PyTorch
def sigmoid(x):
    return torch.sigmoid(x)


def tanh(x):
    return torch.tanh(x)


def relu(x):
    return torch.relu(x)


def leaky_relu(x, alpha=0.01):
    return torch.nn.functional.leaky_relu(x, negative_slope=alpha)


def softmax(x):
    return torch.softmax(x, dim=0)


def swish(x):
    return x * torch.sigmoid(x)


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
    # Define the x-axis range
    x = torch.linspace(-5, 5, 500)

    # Plot the functions
    plt.figure(figsize=(14, 10))

    # Sigmoid
    plt.subplot(2, 3, 1)
    plt.plot(x, sigmoid(x), label='Sigmoid')
    plt.title("Sigmoid Function")
    plt.grid(True)

    # Tanh
    plt.subplot(2, 3, 2)
    plt.plot(x, tanh(x), label='Tanh')
    plt.title("Tanh Function")
    plt.grid(True)

    # ReLU
    plt.subplot(2, 3, 3)
    plt.plot(x, relu(x), label='ReLU')
    plt.title("ReLU Function")
    plt.grid(True)

    # Leaky ReLU
    plt.subplot(2, 3, 4)
    plt.plot(x, leaky_relu(x), label='Leaky ReLU')
    plt.title("Leaky ReLU Function")
    plt.grid(True)

    # Softmax
    plt.subplot(2, 3, 5)
    softmax_values = softmax(x)  # Normalize over the range
    plt.plot(x, softmax_values, label='Softmax (element-wise)', color='orange')
    plt.title("Softmax Function")
    plt.grid(True)

    # Swish
    plt.subplot(2, 3, 6)
    plt.plot(x, swish(x), label='Swish')
    plt.title("Swish Function")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('什么是激活函数')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
