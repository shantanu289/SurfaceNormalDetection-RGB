import torch
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import random
import skimage.io as sio
import skimage.transform as tr
import pickle
import numpy as np
import scipy.misc as misc
import cv2
import os
from tqdm import tqdm

def extract_data(data_info):
    to_tensor = transforms.ToTensor()
    input_tensor_list = []
    orient_tensor_list = []
#     orient_vert_tensor_list = []
    orient_mask_tensor_list = []
    X_list = []
    Y_list = []
    
    root = './dataset/scannet-frames'
    idx = [i for i in range(len(data_info[0]))]
    intrinsics = [577.591, 318.905, 578.73, 242.684]
    xx, yy = np.meshgrid(np.array([i for i in range(640)]), np.array([i for i in range(480)]))
    meshx = tr.resize((xx - intrinsics[1])/intrinsics[0], (240,320))
    meshy = tr.resize((yy - intrinsics[3])/intrinsics[2], (240,320))
    meshx = meshx.astype('float32')
    meshy = meshy.astype('float32')
    for index in tqdm(range(len(idx))):
        color_split = data_info[0][idx[index]].split('/')
        orient_split = data_info[1][idx[index]][:-4].split('/')
        mask_split = data_info[2][idx[index]].split('/')

        color_info = root + '/' + color_split[-2] + '/' + color_split[-1]
        orient_info_X = root + '/' + orient_split[-2] + '/' + orient_split[-1] + '-X.png'
        orient_info_Y = root + '/' + orient_split[-2] + '/' + orient_split[-1] + '-Y.png'
        mask_info = root + '/' + mask_split[-2] + '/' + mask_split[-1]

        orient_mask_tensor = tr.resize(sio.imread(mask_info), (240,320))
        
        color_img = tr.resize(sio.imread(color_info), (240,320,3))
        color_tensor = to_tensor(color_img)
        input_tensor = np.zeros((5, color_img.shape[0], color_img.shape[1]), dtype='float32')
        input_tensor[0:3,:,:] = color_tensor
        input_tensor[3,:,:] = meshx
        input_tensor[4,:,:] = meshy

        orient_x = tr.resize(sio.imread(orient_info_X), (240, 320,3))
        orient_x = (orient_x*2 - 1).astype('float32')
        lx = np.linalg.norm(orient_x, axis=2, keepdims=True)
        orient_x = orient_x/(lx+1e-9)
        X = torch.from_numpy(np.transpose(orient_x, (2,0,1)))
        orient_x[:,:,0] = orient_x[:,:,0] - meshx*orient_x[:,:,2]
        orient_x[:,:,1] = orient_x[:,:,1] - meshy*orient_x[:,:,2]

        orient_y = tr.resize(sio.imread(orient_info_Y), (240,320,3))
        orient_y = (orient_y*2 - 1).astype('float32')
        ly = np.linalg.norm(orient_y, axis=2, keepdims=True)
        orient_y = orient_y/(ly+1e-9)
        Y = torch.from_numpy(np.transpose(orient_y, (2,0,1)))
        orient_y[:,:,0] = orient_y[:,:,0] - meshx * orient_y[:,:,2]
        orient_y[:,:,1] = orient_y[:,:,1] - meshy * orient_y[:,:,2]

        orient_img = np.zeros((orient_x.shape[0], orient_x.shape[1], 4), dtype='float32')
        orient_img[:,:,0:2] = orient_x[:,:,0:2]*(lx > 0.5)
        orient_img[:,:,2:4] = orient_x[:,:,2:4]*(ly > 0.5)

        orient_img_vertical = orient_img.copy()
        orient_img_vertical[:,:,0:2] = orient_img[:,:,2:4]
        orient_img_vertical[:,:,2:4] = -orient_img[:,:,0:2]

        orient_tensor = torch.from_numpy(np.transpose(orient_img, (2,0,1)))
#         orient_vert_tensor = torch.from_numpy(np.transpose(orient_img_vertical,(2,0,1)))
        orient_mask_tensor = torch.Tensor(orient_mask_tensor)
        
        input_tensor_list.append(input_tensor)
        orient_tensor_list.append(orient_tensor)
#         orient_vert_tensor_list.append(orient_vert_tensor)
        orient_mask_tensor_list.append(orient_mask_tensor)
        X_list.append(X)
        Y_list.append(Y)
        
    return input_tensor_list, orient_tensor_list, orient_mask_tensor_list, X_list, Y_list

def setpaths(root, usage):    
    data_info = pickle.load(open(root + '/train_test_split.pkl', 'rb'))[usage]
    root = root + '/scannet-frames'
    index_list = []
    for i in range(len(data_info[0])):
        path_split = data_info[0][i].split('/')
        img_path = root + '/' + path_split[-2] + '/' + path_split[-1]
        imgExists = os.path.exists(img_path)
        if imgExists == False:
            index_list.append(i)
    for ind in sorted(index_list, reverse=True):
        del data_info[0][ind]
        del data_info[1][ind]
        del data_info[2][ind]
    return data_info

class AffineDataset(Dataset):
    def __init__(self, root='./dataset', usage='test'):
#         super(AffineDataset, self).__init__()
        self.root = root
        self.to_tensor = transforms.ToTensor()
        # self.data_info = pickle.load(open(self.root + '/train_test_split.pkl', 'rb'))[usage]
        self.data_info = setpaths(root, usage)
        self.idx = [i for i in range(len(self.data_info[0]))]
        self.data_len = len(self.data_info[0])

#         self.intrinsics = [577.591, 318.905, 578.73, 242.684]
#         xx, yy = np.meshgrid(np.array([i for i in range(640)]), np.array([i for i in range(480)]))
#         self.meshx = tr.resize((xx - self.intrinsics[1])/self.intrinsics[0], (240,320))
#         self.meshy = tr.resize((yy - self.intrinsics[3])/self.intrinsics[2], (240,320))
#         self.meshx = self.meshx.astype('float32')
#         self.meshy = self.meshy.astype('float32')
        self.root = self.root + '/scannet-frames'
        self.input_tensor_list, self.orient_tensor_list, self.orient_mask_tensor_list, self.X_list, self.Y_list = extract_data(self.data_info)   
        
    
    def __getitem__(self,index):

        #get the right image path

#         color_split = self.data_info[0][self.idx[index]].split('/')
#         orient_split = self.data_info[1][self.idx[index]][:-4].split('/')
#         mask_split = self.data_info[2][self.idx[index]].split('/')

#         color_info = self.root + '/' + color_split[-2] + '/' + color_split[-1]
#         orient_info_X = self.root + '/' + orient_split[-2] + '/' + orient_split[-1] + '-X.png'
#         orient_info_Y = self.root + '/' + orient_split[-2] + '/' + orient_split[-1] + '-Y.png'
#         mask_info = self.root + '/' + mask_split[-2] + '/' + mask_split[-1]

#         orient_mask_tensor = tr.resize(sio.imread(mask_info), (240,320))

#         #extract image from the location

#         color_img = tr.resize(sio.imread(color_info), (240,320,3))
#         color_tensor = self.to_tensor(color_img)
#         input_tensor = np.zeros((5, color_img.shape[0], color_img.shape[1]), dtype='float32')
#         input_tensor[0:3,:,:] = color_tensor
#         input_tensor[3,:,:] = self.meshx
#         input_tensor[4,:,:] = self.meshy

#         orient_x = tr.resize(sio.imread(orient_info_X), (240, 320,3))
#         orient_x = (orient_x*2/255.0 - 1).astype('float32')
#         lx = np.linalg.norm(orient_x, axis=2, keepdims=True)
#         orient_x = orient_x/(lx+1e-9)
#         X = torch.from_numpy(np.transpose(orient_x, (2,0,1)))
#         orient_x[:,:,0] = orient_x[:,:,0] - self.meshx*orient_x[:,:,2]
#         orient_x[:,:,1] = orient_x[:,:,1] - self.meshy*orient_x[:,:,2]

#         orient_y = tr.resize(sio.imread(orient_info_Y), (240,320,3))
#         orient_y = (orient_y*2/255.0 - 1).astype('float32')
#         ly = np.linalg.norm(orient_y, axis=2, keepdims=True)
#         orient_y = orient_y/(ly+1e-9)
#         Y = torch.from_numpy(np.transpose(orient_y, (2,0,1)))
#         orient_y[:,:,0] = orient_y[:,:,0] - self.meshx * orient_y[:,:,2]
#         orient_y[:,:,1] = orient_y[:,:,1] - self.meshy * orient_y[:,:,2]

#         orient_img = np.zeros((orient_x.shape[0], orient_x.shape[1], 4), dtype='float32')
#         orient_img[:,:,0:2] = orient_x[:,:,0:2]*(lx > 0.5)
#         orient_img[:,:,2:4] = orient_x[:,:,2:4]*(ly > 0.5)

#         orient_img_vertical = orient_img.copy()
#         orient_img_vertical[:,:,0:2] = orient_img[:,:,2:4]
#         orient_img_vertical[:,:,2:4] = -orient_img[:,:,0:2]

#         orient_tensor = torch.from_numpy(np.transpose(orient_img, (2,0,1)))
#         orient_vert_tensor = torch.from_numpy(np.transpose(orient_img_vertical,(2,0,1)))
#         orient_mask_tensor = torch.Tensor(orient_mask_tensor/255.0)
        input_tensor = self.input_tensor_list[self.idx[index]]
        orient_tensor = self.orient_tensor_list[self.idx[index]]
#         orient_vert_tensor = self.orient_vert_tensor_list[self.idx[index]]
        orient_mask_tensor = self.orient_mask_tensor_list[self.idx[index]]
        X = self.X_list[self.idx[index]]
        Y = self.Y_list[self.idx[index]]
        

        return {'image':input_tensor, 'label':orient_tensor, 'mask':orient_mask_tensor, 'X':X, 'Y':Y}
    
    def __len__(self):
        return self.data_len
    
# if __name__ == "__main__":
#     sample_data = AffineDataset()

#     print(len(sample_data.data_info))
#     print(len(sample_data.data_info[0]))
#     print(len(sample_data.data_info[1]))
#     print(len(sample_data.data_info[2]))
#     print(sample_data.data_info[0][0])
#     print(sample_data.data_info[1][0])
#     print(sample_data.data_info[2][0])
    

    
    # input_tensor, orient_tensor, orient_vert_tensor, orient_mask_tensor ,X, Y = sample_data.__getitem__(0)



        



