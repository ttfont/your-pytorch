import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 定义隐藏层，输入维度为 n_feature，输出维度为 n_hidden
        self.predict = torch.nn.Linear(n_hidden, n_output)  # 定义输出层，输入维度为 n_hidden，输出维度为 n_output

    def forward(self, x):
        x = F.relu(self.hidden(x))  # 使用 ReLU 激活函数处理隐藏层的输出
        x = self.predict(x)  # 计算最终输出
        return x


def print_hi(name):
    print(f'Hi, {name}')
    # 创建保存图片的目录
    # target_directory = "/Users/your/Desktop/001"
    # if not os.path.exists(target_directory):
    #     os.makedirs(target_directory)
    # 创建数据集
    # 生成一维的线性空间数据，并增加一维使其形状为 (100, 1)
    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
    y = x.pow(2) + 0.2 * torch.rand(x.size())  # 生成对应的 y 数据，加上噪声模拟真实情况

    # 初始化神经网络
    net = Net(n_feature=1, n_hidden=10, n_output=1)  # 定义网络，输入输出各为 1，隐藏层有 10 个神经元
    print(net)  # 打印网络结构

    # 定义优化器和损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=0.2)  # 使用随机梯度下降法优化网络参数，学习率为 0.2
    loss_func = torch.nn.MSELoss()  # 定义均方误差损失函数

    plt.ion()  # 开启交互模式，允许动态更新图像

    for epoch in range(200):
        prediction = net(x)  # 前向传播，使用当前网络计算预测值
        loss = loss_func(prediction, y)  # 计算预测值与真实值之间的误差

        optimizer.zero_grad()  # 清空上一步的梯度信息
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 根据梯度更新网络参数

        if epoch % 5 == 0:  # 每 5 个周期更新一次图像
            plt.cla()  # 清除当前图像内容
            plt.scatter(x.data.numpy(), y.data.numpy(), label='True Data')
            plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=2, label='Prediction')
            plt.text(0.5, 0, f'Loss={loss.item():.4f}', fontdict={'size': 20, 'color': 'red'})
            plt.legend()  # 添加图例

            # 保存当前图像
            # file_path = os.path.join(target_directory, f'epoch_{epoch}.png')
            # plt.savefig(file_path)
            # print(f"图像已保存: {file_path}")
            plt.pause(0.1)  # 暂停以更新图像

    plt.ioff()  # 关闭交互模式
    plt.show()  # 显示最终图像


if __name__ == '__main__':
    print_hi('关系拟合')
