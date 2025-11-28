import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import pandas as pd
import numpy as np

# CLASS_NAMES = ['Brain', 'Liver', 'Retina_RESC', 'Retina_OCT2017', 'Chest', 'Histopathology']
CLASS_INDEX = {'Brain': -3, 'Liver': -2, 'Retina_RESC': -1, 'Retina_OCT2017': -1, 'Chest': -2, 'Histopathology': -3, 'Knee': -4}


class MedDataset(Dataset):
    def __init__(self,
                 dataset_path='/data/',
                 class_name='Knee',
                 resize=320,
                 shot=12,
                 iterate=0
                 ):
        assert shot>0, 'shot number : {}, should be positive integer'.format(shot)

        # self.dataset_path = os.path.join(dataset_path, f'{class_name}_AD')
        self.dataset_path =dataset_path  # 有Legion、NonLegion、study_label三个子文件夹
        self.resize = resize
        self.shot = shot
        self.iterate = iterate
        self.class_name = class_name

        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()


        self.transform_x = transforms.Compose([
            transforms.Resize((resize,resize), Image.BICUBIC),
            transforms.ToTensor(),
            ])


        self.transform_mask = transforms.Compose([
            transforms.Resize((resize,resize), Image.NEAREST),
            transforms.ToTensor()
            ])


        self.fewshot_norm_img = self.get_few_normal()
        self.fewshot_abnorm_img, self.fewshot_abnorm_mask = self.get_few_abnormal()
        
            

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]

        n = np.abs(np.load(x, 'r')['n'])
        nx = np.abs(np.load(x, 'r')['nx'])
        ny = np.abs(np.load(x, 'r')['ny'])
        nx_length = np.abs(np.load(x, 'r')['nx_length'])
        ny_length = np.abs(np.load(x, 'r')['ny_length'])

        x = np.abs(np.load(x, 'r')['org'])
        map = np.abs(np.zeros_like(x))
        x = torch.from_numpy(x)
        x_img = x.repeat(3, 1, 1)

        # 伪造mask，其中map为二维的

        for ii in range(int(n)):
            map[0, int(ny[ii]):int(ny[ii]) + int(ny_length[ii]),
                 int(nx[ii]):int(nx[ii]) + int(nx_length[ii])] = 1


        # x = Image.open(x).convert('RGB')
        # x_img = self.transform_x(x)

        return x_img, y, map, n, nx, ny, nx_length, ny_length

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        x, y, mask = [], [], []

        normal_img_dir = os.path.join(self.dataset_path, 'test', 'NonLegion')
        img_fpath_list = sorted([os.path.join(normal_img_dir, f) for f in os.listdir(normal_img_dir)])
        x.extend(img_fpath_list)
        y.extend([0] * len(img_fpath_list))
        mask.extend([None] * len(img_fpath_list))


        abnormal_img_dir = os.path.join(self.dataset_path, 'test', 'Legion')
        img_fpath_list = sorted([os.path.join(abnormal_img_dir, f) for f in os.listdir(abnormal_img_dir)])
        x.extend(img_fpath_list)
        y.extend([1] * len(img_fpath_list))

        mask.extend([None] * len(img_fpath_list))

        assert len(x) == len(y), 'number of x and y should be same'
        return list(x), list(y), list(mask)


    def get_few_normal(self):
        x = []
        img_dir = os.path.join(self.dataset_path, 'train', 'NonLegion')
        normal_names = os.listdir(img_dir)

        # select images
        if self.iterate < 0:
            random_choice = random.sample(normal_names, self.shot)
        else:
            random_choice = []
            with open(f'PD-12.txt', 'r', encoding='utf-8') as infile:
                for line in infile:
                    data_line = line.strip("\n").split()  # 去除首尾换行符，并按空格划分
                    if data_line[0] == f'n-{self.iterate}:':
                        random_choice = data_line[1:]
                        break

        for f in random_choice:
            if f.endswith('.npz'):
                x.append(os.path.join(img_dir, f))  # 文件路径

        fewshot_img = []
        for idx in range(self.shot):
            image = x[idx]

            image = np.abs(np.load(image, 'r')['org'])
            image = torch.from_numpy(image)
            image = image.repeat(3, 1, 1)

            # image = Image.open(image).convert('RGB')
            # image = self.transform_x(image)
            fewshot_img.append(image.unsqueeze(0))

        fewshot_img = torch.cat(fewshot_img)
        return fewshot_img


    def get_few_abnormal(self):
        x = []
        y = []
        img_dir = os.path.join(self.dataset_path, 'train', 'Legion')
        # mask_dir = os.path.join(self.dataset_path, 'valid', 'Ungood', 'anomaly_mask')

        abnormal_names = os.listdir(img_dir)

        # select images
        if self.iterate < 0:
            random_choice = random.sample(abnormal_names, self.shot)
        else:
            random_choice = []
            with open(f'PD-12.txt', 'r', encoding='utf-8') as infile:
                for line in infile:
                    data_line = line.strip("\n").split()  # 去除首尾换行符，并按空格划分
                    if data_line[0] == f'a-{self.iterate}:':
                        random_choice = data_line[1:]
                        break

        for f in random_choice:
            if f.endswith('.npz'):
                x.append(os.path.join(img_dir, f))
                # y.append(os.path.join(mask_dir, f))

        fewshot_img = []
        fewshot_mask = []
        for idx in range(self.shot):
            image = x[idx]

            n = np.abs(np.load(image, 'r')['n'])
            nx = np.abs(np.load(image, 'r')['nx'])
            ny = np.abs(np.load(image, 'r')['ny'])
            nx_length = np.abs(np.load(image, 'r')['nx_length'])
            ny_length = np.abs(np.load(image, 'r')['ny_length'])

            image = np.abs(np.load(image, 'r')['org'])
            map = np.abs(np.zeros_like(image))
            image = torch.from_numpy(image)
            image = image.repeat(3, 1, 1)

            # 伪造mask，其中map为二维的
            for ii in range(int(n)):
                map[0, int(ny[ii]):int(ny[ii]) + int(ny_length[ii]),
                int(nx[ii]):int(nx[ii]) + int(nx_length[ii])] = 1

            # image = Image.open(image).convert('RGB')
            # image = self.transform_x(image)
            fewshot_img.append(image.unsqueeze(0))
            fewshot_mask.append(torch.from_numpy(map).unsqueeze(0))

        fewshot_img = torch.cat(fewshot_img)

        if len(fewshot_mask) == 0:
            return fewshot_img, None
        else:
            fewshot_mask = torch.cat(fewshot_mask)
            return fewshot_img, fewshot_mask

