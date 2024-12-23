import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F  # 激励函数都在这


class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()  # 继承 __init__ 功能
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 隐藏层线性输出
        self.out = torch.nn.Linear(n_hidden, n_output)  # 输出层线性输出

    def forward(self, x):
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))  # 激励函数(隐藏层的线性值)
        x = self.out(x)  # 输出值, 但是这个不是预测值, 预测值还需要再另外计算
        return x


def generate_data():
    """生成模拟数据"""
    n_data = torch.ones(100, 2)  # 数据的基本形态
    x0 = torch.normal(2 * n_data, 1)  # 类型0 x data
    y0 = torch.zeros(100)  # 类型0 y data

    x1 = torch.normal(-2 * n_data, 1)  # 类型1 x data
    y1 = torch.ones(100)  # 类型1 y data

    # 合并数据
    x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
    y = torch.cat((y0, y1), ).type(torch.LongTensor)

    return x, y


def train_model(net, x, y, optimizer, loss_func):
    """训练模型并绘制结果"""
    plt.ion()  # 打开交互模式绘图
    for t in range(100):
        out = net(x)  # 喂给 net 训练数据 x, 输出分析值

        loss = loss_func(out, y)  # 计算误差
        optimizer.zero_grad()  # 清空上一步的残余更新参数值
        loss.backward()  # 误差反向传播，计算参数更新值
        optimizer.step()  # 更新模型参数

        # 每隔 2 次训练迭代绘制一次图
        if t % 2 == 0:
            plt.cla()
            prediction = torch.max(F.softmax(out, dim=1), 1)[1]  # 获取预测类别
            # print("prediction = ", prediction)
            pred_y = prediction.data.numpy().squeeze()
            # print("pred_y = ", pred_y)
            target_y = y.data.numpy()
            # print("target_y = ", target_y)

            plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
            accuracy = sum(pred_y == target_y) / 200.  # 计算准确度

            # print("sum() = ", sum(pred_y == target_y))
            plt.text(1.5, -4, f'Accuracy={accuracy:.2f}', fontdict={'size': 20, 'color': 'red'})
            plt.pause(0.1)

    plt.ioff()  # 关闭交互模式绘图
    plt.show()


def print_hi(name):
    """打印欢迎信息并展示数据"""
    print(f'Hi, {name}')

    # 生成数据
    x, y = generate_data()

    # # 可视化数据分布
    plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
    plt.show()
    #
    # 初始化神经网络模型
    net = Net(n_feature=2, n_hidden=10, n_output=2)

    print(net)  # 输出网络结构

    # 初始化优化器和损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=0.02)  # 使用 SGD 优化器
    loss_func = torch.nn.CrossEntropyLoss()  # 使用交叉熵损失函数

    # 开始训练
    train_model(net, x, y, optimizer, loss_func)


# 主函数
if __name__ == '__main__':
    print_hi('区分类型-分类')
