# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os

identity = lambda x:x
class SimpleDataset:
    def __init__(self, data_file, transform, target_transform=identity):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.data = self.meta['image_names']
        self.label = self.meta['image_labels']

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self,i):
        image_path = os.path.join(self.meta['image_names'][i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.meta['image_labels'][i])
        return img, target

    def __len__(self):
        return len(self.meta['image_names'])


class MultiModalDataset:
    def __init__(self, img_file, attr_file, transform, target_transform=identity):
        with open(img_file, 'r') as f:
            self.meta = json.load(f)

        self.data = self.meta['image_names']
        self.label = self.meta['image_labels']

        self.attr_all = torch.load(attr_file) # CUB(200, 134)
        self.n_attr = self.attr_all.shape[1]
        self.transform = transform
        self.target_transform = target_transform  # for label y


    def __getitem__(self, i):
        image_path = self.data[i]
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)

        # attr = self.attr_all[self.meta['image_labels'][i]]   # only a category  
        attr = self.attr_all[self.label[i]]
        target = self.target_transform(self.label[i])
        return img, attr, target

    def __len__(self):
        return len(self.data)
        



class SetDataset:
    def __init__(self, data_file, batch_size, transform):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
 
        self.cl_list = np.unique(self.meta['image_labels']).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
            self.sub_meta[y].append(x)

        self.sub_dataloader = [] 
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform = transform )
            self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

    def __getitem__(self,i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)

class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self,i):
        #print( '%d -%d' %(self.cl,i))
        image_path = os.path.join( self.sub_meta[i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]





class EpisodicMultiModalSampler(object):
    def __init__(self, label, n_way, n_per, n_episodes):   
        self.n_episodes = n_episodes
        self.n_way = n_way
        self.n_per = n_per
        label = np.array(label)
        self.m_ind = []
        
        for i in np.unique(label):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)     

    def __len__(self):
        return self.n_episodes


    def __iter__(self):
        """ iterater of each episode
        """
        for i_batch in range(self.n_episodes):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_way]
            for c in classes:
                l = self.m_ind[c]  # all samples of c class
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            # batch = torch.stack(batch).t().reshape(-1)
            batch = torch.stack(batch).reshape(-1)
            yield batch