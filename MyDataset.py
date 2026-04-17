import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from einops import rearrange
import warnings
import time
Image.MAX_IMAGE_PIXELS = None

import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import clip
class MyDataset(Dataset):
    def __init__(self, root, csv_path, mode='train', transform=None):
        self.root = root
        self.csv_path = csv_path
        # 读取并随机打乱数据
        #best is state 100
        # self.data = pd.read_csv(csv_path).sample(frac=1,random_state=109).reset_index(drop=True)
        self.data = pd.read_csv(csv_path, usecols=['dis', 'mos']).sample(frac=1, random_state=109).reset_index(drop=True)

        l = len(self.data)
        train_end_idx = int(0.8 * l)

        # 划分训练集与验证集
        train_data = self.data.iloc[:train_end_idx]
        val_data = self.data.iloc[train_end_idx:]
        self.name_to_label = dict(zip(self.data['dis'], self.data['mos']))
        if mode == 'train':
            self.image_names = train_data['dis'].tolist()
        elif mode == 'test':
            self.image_names = val_data['dis'].tolist()
        else:
            raise ValueError("Invalid mode. Use 'train' or 'test'.")

        self.transforms = transform

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.root, image_name)
        image = Image.open(image_path).convert('RGB')
        label = self.name_to_label[image_name]

        # # ==== 1/8极区裁剪逻辑（PIL + NumPy） ====
        # img_np = np.array(image)
        # H, W = img_np.shape[:2]
        # cut_ratio = 1/6
        # top = int(H * cut_ratio)
        # bottom = int(H * (1 - cut_ratio))
        # cropped_img = img_np[top:bottom, :]
        # 转回PIL Image对象
        # image = Image.fromarray(cropped_img)

        # ==== 后续变换 ====
        if self.transforms:
            image = self.transforms(image)
        return image, label
    def __len__(self):
        return len(self.image_names)
if __name__ == "__main__":
    #split_dataset_oiqa('/home/d310/10t/wkc/Database/Csv_file/oiq_10k_info.csv')
    test_transform = transforms.Compose([
        transforms.Resize((1024,1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    star_time = time.time()
    test_dataset = MyDataset('/mnt/10T/Yanzhu/Databases/OIQ-10k_1024','/mnt/10T/liziyi/LargeKernel/OIQ-10k/oiq_10k_info_alter.csv',mode='train',transform=test_transform)
    print(len(test_dataset), "train data has been load!")
    star_time = time.time()
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=16,
        num_workers=0,
        shuffle=False,
    )
    end_time = time.time()
    print("消耗的时间是",(end_time-star_time)/60)
    star_time = time.time()
    print(test_loader)
    for imgs,mos in test_loader:
        print(imgs.shape)
        break
    end_time = time.time()
    print("消耗的时间是",(end_time-star_time)/60)
    