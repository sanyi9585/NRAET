# from PIL import Image
# import torch
# from torch.utils.data import Dataset
#
#
# class MyDataSet(Dataset):
#     """自定义数据集"""
#
#     def __init__(self, images_path: list, images_class: list, transform=None):
#         self.images_path = images_path
#         self.images_class = images_class
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.images_path)
#
#     def __getitem__(self, item):
#         img = Image.open(self.images_path[item])
#         # RGB为彩色图片，L为灰度图片
#         img = img.convert('RGB')
#         if img.mode != 'RGB':
#             raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
#         label = self.images_class[item]
#
#         if self.transform is not None:
#             img = self.transform(img)
#
#         return img, label
#
#     @staticmethod
#     def collate_fn(batch):
#         # 官方实现的default_collate可以参考
#         # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
#         images, labels = tuple(zip(*batch))
#
#         images = torch.stack(images, dim=0)
#         labels = torch.as_tensor(labels)
#         return images, labels
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset

# 允许加载截断的图像
ImageFile.LOAD_TRUNCATED_IMAGES = True

class MyDataSet(Dataset):
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        try:
            # 尝试打开图像
            img = Image.open(self.images_path[item])
            # 确保图像为RGB格式
            if img.mode != 'RGB':
                img = img.convert('RGB')
        except Exception as e:
            print(f"警告: 无法加载图像 {self.images_path[item]}: {e}")
            # 返回一个默认图像或跳过该图像
            # 这里我们创建一个全黑的图像作为替代
            img = Image.new('RGB', (224, 224), color='black')

        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
