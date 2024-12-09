# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import numpy as np


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

    # 用 Numpy 还是 Torch
    np_data = np.arange(6).reshape((2, 3))  # 创建一个 2x3 的 NumPy 数组
    torch_data = torch.from_numpy(np_data)  # 将 NumPy 数组转换为 Torch 张量
    tensor2array = torch_data.numpy()  # 将 Torch 张量转换回 NumPy 数组
    print(
        '\nnumpy array:', np_data,  # NumPy 数组内容: [[0 1 2], [3 4 5]]
        '\ntorch tensor:', torch_data,  # Torch 张量内容: 0  1  2 \n 3  4  5    [torch.LongTensor of size 2x3]
        '\ntensor to array:', tensor2array,  # 转换回的 NumPy 数组内容: [[0 1 2], [3 4 5]]
    )

    # Torch 中的数学运算
    # abs 绝对值计算
    data = [-1, -2, 1, 2]  # 输入数据
    tensor = torch.FloatTensor(data)  # 转换成 32 位浮点张量
    print(
        '\nabs',
        '\nnumpy: ', np.abs(data),  # NumPy 计算绝对值结果: [1 2 1 2]
        '\ntorch: ', torch.abs(tensor)  # Torch 计算绝对值结果: [1 2 1 2]
    )

    # sin 三角函数 sin
    print(
        '\nsin',
        '\nnumpy: ', np.sin(data),  # NumPy 计算正弦值结果: [-0.84147098 -0.90929743  0.84147098  0.90929743]
        '\ntorch: ', torch.sin(tensor)  # Torch 计算正弦值结果: [-0.8415 -0.9093  0.8415  0.9093]
    )

    # mean 均值
    print(
        '\nmean',
        '\nnumpy: ', np.mean(data),  # NumPy 计算均值结果: 0.0
        '\ntorch: ', torch.mean(tensor)  # Torch 计算均值结果: 0.0
    )

    # 矩阵运算
    # matrix multiplication 矩阵乘法
    data = [[1, 2], [3, 4]]  # 输入矩阵
    tensor = torch.FloatTensor(data)  # 转换成 32 位浮点张量
    print(
        '\nmatrix multiplication (matmul)',
        '\nnumpy: ', np.matmul(data, data),  # NumPy 计算矩阵乘法结果: [[7, 10], [15, 22]]
        '\ntorch: ', torch.mm(tensor, tensor)  # Torch 计算矩阵乘法结果: [[7, 10], [15, 22]]
    )


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('Torch 或 NumPy')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
