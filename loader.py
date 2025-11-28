import random
import torch
import csv
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import os,time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import torch.utils.data as data
import numpy as np

def getloader_dir1_classify(dir):
    number = len(dir)
    Legion = []
    dataset = Folder(dir,number)
    loader = DataLoader(dataset=dataset,batch_size=number,shuffle=True)
    for i in range(number):
        data = np.load(dir[i], allow_pickle=True)
        Legion.append(data['Legion'])
    Legion = np.array(Legion)
    return loader,Legion

def getloader_dir_kspace(a):
    number = len(a)
    dataset = Folder_kspace(a,number)
    loader = DataLoader(dataset=dataset,batch_size=number,shuffle=True,drop_last=True)
    return loader

class Folder(data.Dataset):
    def __init__(self,data_dir,number,train=True):
        org=[]
        csm=[]
        label=[]
        n=[]
        nx=[]
        ny=[]
        nx_length=[]
        ny_length=[]
        org_old=[]
        # Legion1=[]
        for i in range(number):
            data = np.load(data_dir[i],allow_pickle=True)
            org.append(data['org'])
            csm.append(data['csm'])
            label.append(data['label'])
            n.append(data['n'])
            nx.append(data['nx'])
            ny.append(data['ny'])
            nx_length.append(data['nx_length'])
            ny_length.append(data['ny_length'])

        org = torch.from_numpy(np.array(org))
        csm = torch.from_numpy(np.array(csm))
        label = torch.from_numpy(np.array(label))
        org_old = torch.from_numpy(np.array(org_old))
        n = np.array(n)
        nx = np.array(nx)
        ny = np.array(ny)
        nx_length = np.array(nx_length)
        ny_length = np.array(ny_length)
        self.org=org
        self.csm=csm
        self.label=label
        self.org_old = org_old
        self.n=n
        self.nx=nx
        self.ny=ny
        self.nx_length = nx_length
        self.ny_length = ny_length
        self.train = train

    def __getitem__(self,index):
        org = self.org[index]
        csm = self.csm[index]
        label = self.label[index]
        n = self.n[index]
        nx = self.nx[index]
        ny = self.ny[index]
        nx_length = self.nx_length[index]
        ny_length = self.ny_length[index]

        return org,csm,label,n,nx,ny,nx_length,ny_length
    def __len__(self):
        return len(self.org)

class Folder_kspace(data.Dataset):
    def __init__(self,data_dir,number,train=True):
        org=[]
        csm=[]
        label=[]
        for i in range(number):
            data = np.load(data_dir[i])
            bb=data['csm']
            org.append(data['org'])
            csm.append(bb)
            label.append(data['label'])

        org = torch.from_numpy(np.array(org))
        csm = torch.from_numpy(np.array(csm))
        label = torch.from_numpy(np.array(label))
        self.org=org
        self.csm=csm
        self.label=label
        self.train = train

    def __getitem__(self,index):
        org = self.org[index]
        csm = self.csm[index]
        label = self.label[index]
        return org,csm,label

    def __len__(self):
        return len(self.org)