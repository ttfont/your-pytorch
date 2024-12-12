# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import matplotlib.pyplot as plt


# 使用 PyTorch 定义激活函数

def sigmoid(x):
    # Sigmoid 函数：将输入压缩到 (0, 1) 范围内
    return torch.sigmoid(x)


def tanh(x):
    # Tanh 函数：将输入压缩到 (-1, 1) 范围内
    return torch.tanh(x)


def relu(x):
    # ReLU 函数：将负值截断为 0，正值保持不变
    return torch.relu(x)


def leaky_relu(x, alpha=0.01):
    # Leaky ReLU 函数：允许负值通过一个小斜率 α
    return torch.nn.functional.leaky_relu(x, negative_slope=alpha)


def softmax(x):
    # Softmax 函数：将输入转化为概率分布，所有输出的和为 1
    return torch.softmax(x, dim=0)


def swish(x):
    # Swish 函数：平滑的激活函数，定义为 x * sigmoid(x)
    return x * torch.sigmoid(x)


def softplus(x):
    # Softplus 函数：ReLU 的平滑版本，定义为 ln(1 + e^x)
    return torch.nn.functional.softplus(x)


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
    # 定义 x 轴范围
    x = torch.linspace(-5, 5, 500)

    # 绘制激活函数
    plt.figure(figsize=(14, 12))

    # Sigmoid 函数
    plt.subplot(3, 3, 1)
    plt.plot(x.numpy(), sigmoid(x).numpy(), label='Sigmoid')
    plt.title("Sigmoid Function")
    plt.grid(True)

    # Tanh 函数
    plt.subplot(3, 3, 2)
    plt.plot(x.numpy(), tanh(x).numpy(), label='Tanh')
    plt.title("Tanh Function")
    plt.grid(True)

    # ReLU 函数
    plt.subplot(3, 3, 3)
    plt.plot(x.numpy(), relu(x).numpy(), label='ReLU')
    plt.title("ReLU Function")
    plt.grid(True)

    # Leaky ReLU 函数
    plt.subplot(3, 3, 4)
    plt.plot(x.numpy(), leaky_relu(x).numpy(), label='Leaky ReLU')
    plt.title("Leaky ReLU Function")
    plt.grid(True)

    # Softmax 函数
    plt.subplot(3, 3, 5)
    softmax_values = softmax(x)  # 归一化处理
    plt.plot(x.numpy(), softmax_values.numpy(), label='Softmax (element-wise)', color='orange')
    plt.title("Softmax Function")
    plt.grid(True)

    # Swish 函数
    plt.subplot(3, 3, 6)
    plt.plot(x.numpy(), swish(x).numpy(), label='Swish')
    plt.title("Swish Function")
    plt.grid(True)

    # Softplus 函数
    plt.subplot(3, 3, 7)
    plt.plot(x.numpy(), softplus(x).numpy(), label='Softplus')
    plt.title("Softplus Function")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('什么是激活函数')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
