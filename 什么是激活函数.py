# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import matplotlib.pyplot as plt


# 使用 NumPy 定义激活函数

def sigmoid(x):
    # Sigmoid 函数：将输入压缩到 (0, 1) 范围内
    return 1 / (1 + np.exp(-x))


def tanh(x):
    # Tanh 函数：将输入压缩到 (-1, 1) 范围内
    return np.tanh(x)


def relu(x):
    # ReLU 函数：将负值截断为 0，正值保持不变
    return np.maximum(0, x)


def leaky_relu(x, alpha=0.01):
    # Leaky ReLU 函数：允许负值通过一个小斜率 α
    return np.where(x > 0, x, alpha * x)


def softmax(x):
    # Softmax 函数：将输入转化为概率分布，所有输出的和为 1
    e_x = np.exp(x - np.max(x))  # 数值稳定性处理
    return e_x / e_x.sum()


def swish(x):
    # Swish 函数：平滑的激活函数，定义为 x * sigmoid(x)
    return x / (1 + np.exp(-x))


def softplus(x):
    # Softplus 函数：ReLU 的平滑版本，定义为 ln(1 + e^x)
    return np.log(1 + np.exp(x))


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
    # 定义 x 轴范围
    x = np.linspace(-5, 5, 500)

    # 绘制激活函数
    plt.figure(figsize=(14, 12))

    # Sigmoid 函数
    plt.subplot(3, 3, 1)
    plt.plot(x, sigmoid(x), label='Sigmoid')
    plt.title("Sigmoid Function")
    plt.grid(True)

    # Tanh 函数
    plt.subplot(3, 3, 2)
    plt.plot(x, tanh(x), label='Tanh')
    plt.title("Tanh Function")
    plt.grid(True)

    # ReLU 函数
    plt.subplot(3, 3, 3)
    plt.plot(x, relu(x), label='ReLU')
    plt.title("ReLU Function")
    plt.grid(True)

    # Leaky ReLU 函数
    plt.subplot(3, 3, 4)
    plt.plot(x, leaky_relu(x), label='Leaky ReLU')
    plt.title("Leaky ReLU Function")
    plt.grid(True)

    # Softmax 函数
    plt.subplot(3, 3, 5)
    softmax_values = softmax(x)  # 归一化处理
    plt.plot(x, softmax_values, label='Softmax (element-wise)', color='orange')
    plt.title("Softmax Function")
    plt.grid(True)

    # Swish 函数
    plt.subplot(3, 3, 6)
    plt.plot(x, swish(x), label='Swish')
    plt.title("Swish Function")
    plt.grid(True)

    # Softplus 函数
    plt.subplot(3, 3, 7)
    plt.plot(x, softplus(x), label='Softplus')
    plt.title("Softplus Function")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('什么是激活函数')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
