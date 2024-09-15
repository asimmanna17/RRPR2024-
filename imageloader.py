import random
import torch
import numpy as np
import torch.utils.data as data
import os
import PIL.Image as Image

def sample_return(root):
    newdataset = []
    for image in os.listdir(root):
        if 'Hernia' not in image:
            label=[]
            #print(image)
            path = os.path.join(root, image)
            #print(path)
            labels = image.split('_')[2].split('|')
            #(labels)

            dis_total = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
                            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
                            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
                            'Pleural']
            #print(labels)
            #if 'Hernia' not in labels:
            for dis in dis_total:
                #print(dis)
                if dis in labels:
                    label.append(1)
                else:
                    label.append(0)
        
            label_array = np.array(label)
            label_tensor = torch.FloatTensor(label_array)
            #print(label_tensor)
            item = (path, label_tensor)
            #print(item)
            newdataset.append(item)
    #print(len(newdataset))
    return newdataset

class customDataset(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
       
        self.root = root
        samples = sample_return(root)
        samples1 = random.sample(samples, len(samples))
        
        self.samples = samples
        self.samples1 = samples1

        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, index):
        #print('index:',(index))
        img, label= self.samples[index]
        img1, label1= self.samples1[index]
        #print(img)
        #print(img1)
        img = np.load(img)
        img1 = np.load(img1)
        #print(label)
        #print(label1)
        
        if len(img.shape)!= 2:
            img = np.mean(img,axis=-1)
        if len(img1.shape)!= 2:
            img1 = np.mean(img1,axis=-1)
        
        
        img = Image.fromarray(img)
        img1 = Image.fromarray(img1)
        if self.transform is not None:
            img = self.transform(img)
            img1 = self.transform(img1)

        if self.target_transform is not None:
            label = self.target_transform(label)
            label1 = self.target_transform(label1)
        return img, label, index, img1, label1
    
    def __len__(self):
        return len(self.samples)


