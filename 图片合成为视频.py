import cv2
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


def save_as_mp4(image_folder, output_file, frame_rate=5):
    # 获取所有图片路径
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # 按文件名中的 epoch 排序

    # 获取图片尺寸
    first_image_path = os.path.join(image_folder, images[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape

    # 初始化视频写入器
    video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video_writer.write(frame)  # 写入帧到视频

    video_writer.release()
    print(f"视频已保存为 {output_file}")


def main():
    target_directory = "./images"  # 保存图片的目录
    video_output = "training_visualization.mp4"  # 输出视频文件名

    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
    y = x.pow(2) + 0.2 * torch.rand(x.size())

    net = Net(n_feature=1, n_hidden=10, n_output=1)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
    loss_func = torch.nn.MSELoss()

    for epoch in range(200):
        prediction = net(x)
        loss = loss_func(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            plt.cla()
            plt.scatter(x.detach().numpy(), y.detach().numpy(), label='True Data')
            plt.plot(x.detach().numpy(), prediction.detach().numpy(), 'r-', lw=1, label='Prediction')
            plt.text(0.5, 0, f'Loss={loss.item():.4f}', fontdict={'size': 12, 'color': 'red'})
            plt.legend()

            file_path = os.path.join(target_directory, f'epoch_{epoch}.png')
            plt.savefig(file_path)
            print(f"图像已保存: {file_path}")

    plt.ioff()
    plt.show()

    # 将图片合成为 MP4 视频
    save_as_mp4(target_directory, video_output, frame_rate=5)


if __name__ == "__main__":
    main()
