from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

#landmarks_frame = pd.read_csv('faces/face_landmarks.csv')
#n = 65

#img_name = landmarks_frame.iloc[n,0]
#landmarks = landmarks_frame.iloc[n,1:].as_matrix()
#landmarks = landmarks.astype('float').reshape(-1,2)

#print(img_name)
#print(landmarks.shape)
#print(landmarks[:4])

def show_landmarks(image, landmarks):
    plt.imshow(image)
    plt.scatter(landmarks[:,0], landmarks[:,1], s=10, marker='.', c='r')
    plt.pause(0.001)

#plt.figure()

#show_landmarks(io.imread(os.path.join('faces/', img_name)),landmarks)
#plt.show()

class FaceLandmarksDataset(Dataset):
    def __init__(self,csv_file, root_dir, transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    # 数据集的长度
    def __len__(self):
        return len(self.landmarks_frame)

    # 分别读取图像
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx,0])

        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx,1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1,2)
        sample = {'image':image,'landmarks':landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h,w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                h_new, w_new = self.output_size * h / w, self.output_size
            else:
                h_new, w_new = self.output_size, self.output_size * h / w
        else:
            h_new, w_new = self.output_size

        h_new, w_new = int(h_new), int(w_new)

        img = transform.resize(image, (h_new, w_new))

        landmarks = landmarks * [w_new / w, h_new / h]

        return {'image':img, 'landmarks':landmarks}

class Totensor(object):
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        image = image.transpose((2, 0, 1))

        return {'image':torch.from_numpy(image),
                'landmarks':torch.from_numpy(landmarks)}


class RandomCrop(object):
    def __init__(self,output_size):
        assert isinstance(output_size,(int, tuple))

        if isinstance(output_size, int):
            self.output_size = (output_size,output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        h_new, w_new = self.output_size

        top = np.random.randint(0, h - h_new)
        left = np.random.randint(0, w - w_new)

        image = image[top: top + h_new,
                        left: left + w_new]

        landmarks = landmarks - [left, top]

        return {'image':image, 'landmarks':landmarks}


composed = transforms.Compose([Rescale((256,256)),
                               RandomCrop(224),
                               Totensor()])

composed_vision = transforms.Compose([transforms.Scale(256),
                                      transforms.RandomCrop(224),
                                      transforms.ToTensor()])

face_dataset = FaceLandmarksDataset(csv_file='faces/face_landmarks.csv',
                                    root_dir='faces/',
                                    transform = composed)

dataloder = DataLoader(face_dataset, batch_size=4,
                       shuffle=True, num_workers=4)


for i_batch, sample_batch in enumerate(dataloder):
    print(i_batch, sample_batch['image'].size(),
          sample_batch['landmarks'].size())
