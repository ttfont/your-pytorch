# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import torch
from torch.autograd import Variable  # torch 中 Variable 模块


def variable_old():
    # 创建一个 Float 类型的张量
    tensor = torch.FloatTensor([[1, 2], [3, 4]])
    # 使用 Variable 封装张量并设置 requires_grad=True
    variable = Variable(tensor, requires_grad=True)
    print("Tensor:", tensor)  # 输出张量
    print("Variable:", variable)  # 输出 Variable

    # 计算 x^2 的均值
    v_out = torch.mean(variable * variable)  # x^2
    print("Output:", v_out)  # 输出结果: 7.5

    # 进行反向传播，计算梯度
    v_out.backward()
    print("Gradient of Variable:", variable.grad)  # 输出初始 Variable 的梯度

    print("Variable:", variable)  # 输出 Variable 形式
    print("Variable data:", variable.data)  # 获取 Variable 中的张量数据
    print("Variable data as NumPy:", variable.data.numpy())  # 转换为 NumPy 数据


def tensor_new():
    # 创建支持梯度计算的张量
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    print("Data:", tensor)  # 输出张量的值

    # 定义一个计算过程 (均值平方)
    output = torch.mean(tensor * tensor)  # 相当于 x^2 的均值

    # 输出计算结果
    print("Output of mean(square):", output.item())  # 输出标量值

    # 进行反向传播，计算梯度
    output.backward()

    # 查看梯度
    print("Gradient of Tensor:", tensor.grad)  # 输出梯度

    # 直接获取 Tensor 的数据
    print("Tensor:", tensor)

    # 去除梯度信息，获取纯数据
    print("Tensor data (no grad):", tensor.detach())

    # 转换为 NumPy 数据
    print("Tensor as NumPy array:", tensor.detach().numpy())


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

    # 老的写法
    variable_old()

    # 新的写法
    tensor_new()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('变量 Variable 和 Tensor')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
