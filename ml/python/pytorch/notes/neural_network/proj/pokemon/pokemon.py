import torch
import os, glob
import random, csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

"""
5个子文件夹, 每个文件夹下保存着图片
文件夹的名字就是分类的名称
"""


class PokemonDataset(Dataset):
    def __init__(
        self,
        path,
        mode,
        resize=None,
        to_tensor=False,
        transformers=None,
        write_to_csv=False,
    ):
        self.path = path
        self.resize = resize
        self.mode = mode
        self.write_to_csv = write_to_csv
        self.to_tensor = to_tensor
        self.transformers = transformers
        ## 1. 对每一个label进行编码
        self.name2label = {}  # 将label进行编码
        for name in sorted(os.listdir(os.path.join(path))):
            if not os.path.isdir(os.path.join(path, name)):
                continue
            # 使用当前key的长度为id
            # {'bulbasaur': 0, 'charmander': 1, 'mewtwo': 2, 'pikachu': 3, 'squirtle': 4}
            self.name2label[name] = len(self.name2label.keys())
        ## 2. 对每一个image获取一个label
        self.images_total, self.labels_total = self.load_csv("files.csv")

        ## 3. 训练集和测试集
        if mode == "train":  # 60%
            self.images = self.images_total[: int(0.6 * len(self.images_total))]
            self.labels = self.labels_total[: int(0.6 * len(self.labels_total))]
        if mode == "val":  # 20%
            self.images = self.images[
                int(0.6 * len(self.images_total)) : int(0.8 * len(self.images_total))
            ]
            self.labels = self.images[
                int(0.6 * len(self.labels_total)) : int(0.8 * len(self.labels_total))
            ]
        if mode == "test":  # 20%
            self.images = self.images_total[int(0.8 * len(self.images_total)) :]
            self.labels = self.labels_total[int(0.8 * len(self.labels_total)) :]

    def load_csv(self, filename):
        output, labels = [], []
        if self.write_to_csv and os.path.exists(os.path.join(self.path, filename)):
            with open(os.path.join(self.path, filename)) as f:
                reader = csv.reader(f)
                for (output_path, label) in reader:
                    label = int(label)
                    output.append(output_path)
                    labels.append(label)
        else:
            images = []
            for name in self.name2label.keys():
                folder_images = []
                folder_images += glob.glob(os.path.join(self.path, name, "*.png"))
                folder_images += glob.glob(os.path.join(self.path, name, "*.jpg"))
                folder_images += glob.glob(os.path.join(self.path, name, "*.jpeg"))
                for image in folder_images:
                    images.append((image, name))
            random.shuffle(images)
            for (output_path, label) in images:
                output.append(os.path.abspath(output_path))
                labels.append(self.name2label[label])
            if self.write_to_csv:
                with open(os.path.join(self.path, filename), mode="w", newline="") as f:
                    writer = csv.writer(f)
                    for (path, label) in images:
                        writer.writerow([os.path.abspath(path), self.name2label[label]])
        return output, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img, label = self.images[index], self.labels[index]
        ## 将路径读取进来, 然后使用torchvision输出. 这里考试构建转换器
        transformers = [lambda x: Image.open(x).convert("RGB")]
        if self.resize is not None:
            if isinstance(self.resize, int):
                transformers.append(transforms.Resize((self.resize, self.resize)))
            elif isinstance(self.resize, tuple):
                transformers.append(transforms.Resize(self.resize))
        if self.to_tensor:
            transformers.append(transforms.ToTensor())
            label = torch.tensor(label)

        if self.transformers is not None:
            if isinstance(self.transformers, bool) and self.transformers:
                transformers.append(transforms.RandomRotation(15))
                # 这里使用的是imagenet的统计结果, 一般就直接使用
                transformers.append(
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    )
                )
            elif not isinstance(self.transformers, bool):
                transformers = self.transformers
        # 转换
        tf = transforms.Compose(transformers)
        return (tf(img), label)

    def denormalize(self, x_hat):
        if not self.to_tensor:
            raise Exception("To tensor is required to use denormalization")
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # x: [c,h,w]
        # mean : [3] => [3,1,1] # 自动broadcast
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        return x_hat * std + mean


"""
测试
"""


def main():
    import visdom
    import time

    # 开启docker服务
    # docker run -it -p 8097:8097 --name visdom -d hypnosapos/visdom
    viz = visdom.Visdom()
    db = PokemonDataset(
        "data/pokeman", "train", resize=224, augmentation=True, to_tensor=True
    )
    x, y = next(iter(db))
    print("sample:", x.shape, y.shape, y)

    # 然后 localhost:8097 建立连接
    viz.image(db.denormalize(x), win="sample_x", opts=dict(title="simple_x"))

    loader = DataLoader(db, batch_size=32, shuffle=True)


if __name__ == "__main__":
    main()
