import torch
import os
import torch.nn as nn
import numpy as np
import sys
import skimage.io as sio
import math
import logging
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from time import time
from dorn import DORN
from dataset_parser_scannet import AffineDataset
from utils import *
from tqdm import tqdm

logging.basicConfig(filename='training-run2.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

num_epochs = 200
batch_size = 4
use_min = 1
training_loss_run2 = []
validation_loss_run2 = []

print("Fetching Training Data ...")
train_dataset = AffineDataset(usage='train', root='./dataset')
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
logging.info("Got Train Data")
print("Fetching Validation/Test Data ...")
test_dataset = AffineDataset(usage='test', root='./dataset')
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
# val_dataset = AffineDataset(usage='test', root='./dataset')
# val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
logging.info("Got Val/Test Data")
train = True


cnn = DORN(channels=5, output_channels=13)
# cnn.load_state_dict(torch.load('./model-run2/model-epoch-2.pth'))
optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-3)
cnn = cnn.cuda()

n_iter = 0
m_iter = 0

test_iter = iter(test_dataloader)
# val_iter = iter(val_dataloader)

def train_one_iter(i, sampled_batch, evaluate=0, use_min=0, train=True):
    global n_iter, m_iter

    cnn.train()
    if evaluate > 0: 
        cnn.eval()
    
    images = sampled_batch['image']
    labels = sampled_batch['label']
    mask_alt = sampled_batch['mask']
    tmask = mask_alt.clone()
    X = sampled_batch['X']
    Y = sampled_batch['Y']

    images_tensor = Variable(images.float()); images_tensor = images_tensor.cuda()
    labels_tensor = Variable(labels); labels_tensor = labels_tensor.cuda()
    mask_alt_tensor = Variable(mask_alt); mask_alt_tensor = mask_alt_tensor.cuda()

    mask = (labels_tensor[:,0:1,:,:] * labels_tensor[:,0:1,:,:] + labels_tensor[:,1:2,:,:] * labels_tensor[:,1:2,:,:]) > 0.2
    mask = mask.float()
    X = X.cuda()
    Y = Y.cuda()

    if torch.sum(mask).item()*2 == 0:
        return
    
    optimizer.zero_grad()

    temp_output = cnn(images_tensor)
    output1 = temp_output[:,0:4,:,:]
    output2 = temp_output[:,4:10,:,:]
    norm_output = temp_output[:,10:13,:,:]
    dirx = Normalize(output2[:,0:3,:,:])
    diry = Normalize(output2[:,3:6,:,:])
    preds = ConvertToAngle(output1) 

    l0 = labels_tensor
    a0 = ConvertToAngle(l0)
    l1 = Rotate90(l0)
    a1 = ConvertToAngle(l1)
    l2 = Rotate90(l1)
    a2 = ConvertToAngle(l2)
    l3 = Rotate90(l2)
    a3 = ConvertToAngle(l3) 


    # mse_loss_1 and mse_loss_2
    
    d01 = preds - a0
    d02 = torch.min(torch.abs(d01), torch.min(torch.abs(d01 + 360), torch.abs(d01 - 360)))
    d03 = torch.sum(d02, dim=1)
    if use_min == 0:
       
        d04 = d03.view(d03.shape[0], 1, d03.shape[1], d03.shape[2])
        loss = torch.sum(d04*mask)
        diff11 = (output1-l0)**2
        diff12 = torch.sum(diff11, dim=1).view(output1.shape[0], 1, output1.shape[2], output1.shape[3])
        diff13 = diff12*mask
        mse_loss_1 = torch.sum(diff13)

        diffa = torch.sum((dirx - X)**2, dim=1).view(output1.shape[0], 1, output1.shape[2], output1.shape[3])
        diffb = torch.sum((diry - Y)**2, dim=1).view(output1.shape[0], 1, output1.shape[2], output1.shape[3])
        mse_loss_2 = torch.sum((diffa + diffb)*mask)
    elif use_min == 1:
        
        d11 = preds - a1
        d12 = torch.min(torch.abs(d11), torch.min(torch.abs(d11 + 360), torch.abs(d11 - 360)))
        d13 = torch.sum(d12, dim=1)
        d21 = preds - a2
        d22 = torch.min(torch.abs(d21), torch.min(torch.abs(d21 + 360), torch.abs(d21 - 360)))
        d23 = torch.sum(d22, dim=1)
        d31 = preds - a3
        d32 = torch.min(torch.abs(d31), torch.min(torch.abs(d31 + 360), torch.abs(d31 - 360)))
        d33 = torch.sum(d32, dim=1)
        dtemp = torch.min(d03, torch.min(d13, torch.min(d23, d33)))
        
        d = dtemp.view(dtemp.shape[0], 1, dtemp.shape[1], dtemp.shape[2])
        
        loss = torch.sum(d*mask)

        diff0 = torch.sum((output1 - l0)**2, dim=1)
        diff1 = torch.sum((output1 - l1)**2, dim=1)
        diff2 = torch.sum((output1 - l2)**2, dim=1)
        diff3 = torch.sum((output1 - l3)**2, dim=1)
        diff = torch.min(diff0, torch.min(diff1, torch.min(diff2, diff3)))
        diff = diff.view(diff.shape[0], 1, diff.shape[1], diff.shape[2])
        mse_loss_1 = torch.sum(diff*mask)

        diff_a1 = torch.sum((dirx - X)**2, dim=1).view(output1.shape[0], 1, output1.shape[2], output1.shape[3])
        diff_b1 = torch.sum((diry - Y)**2, dim=1).view(output1.shape[0], 1, output1.shape[2], output1.shape[3])
        diff_x1 = diff_a1 + diff_b1

        diff_a2 = torch.sum((dirx - Y)**2, dim=1).view(output1.shape[0], 1, output1.shape[2], output1.shape[3])
        diff_b2 = torch.sum((diry + X)**2, dim=1).view(output1.shape[0], 1, output1.shape[2], output1.shape[3])
        diff_x2 = diff_a2 + diff_b2

        diff_a3 = torch.sum((dirx + X)**2, dim=1).view(output1.shape[0], 1, output1.shape[2], output1.shape[3])
        diff_b3 = torch.sum((diry + Y)**2, dim=1).view(output1.shape[0], 1, output1.shape[2], output1.shape[3])
        diff_x3 = diff_a3 + diff_b3

        diff_a4 = torch.sum((dirx + Y)**2, dim=1).view(output1.shape[0], 1, output1.shape[2], output1.shape[3])
        diff_b4 = torch.sum((diry - X)**2, dim=1).view(output1.shape[0], 1, output1.shape[2], output1.shape[3])
        diff_x4 = diff_a4 + diff_b4

        diff_x = torch.min(diff_x1, torch.min(diff_x2, torch.min(diff_x3, diff_x4)))
        mse_loss_2 = torch.sum(diff_x*mask)

    # mse_loss_proj
    c1 = dirx[:,0,:,:] - images_tensor[:,3,:,:]*dirx[:,2,:,:] - output1[:,0,:,:]
    c2 = dirx[:,1,:,:] - images_tensor[:,4,:,:]*dirx[:,2,:,:] - output1[:,1,:,:]
    c3 = diry[:,0,:,:] - images_tensor[:,3,:,:]*diry[:,2,:,:] - output1[:,2,:,:]
    c4 = diry[:,1,:,:] - images_tensor[:,4,:,:]*diry[:,2,:,:] - output1[:,3,:,:]
    mse_loss_proj = torch.sum((c1**2 + c2**2 + c3**2 + c4**2).view(mask.shape[0],1,mask.shape[2],mask.shape[3])*mask)

    # mse_norm_loss
    norm0 = Normalize(torch.cross(dirx, diry, dim=1))
    norm1 = Normalize(norm_output)
    norm2 = Normalize(torch.cross(X,Y,dim=1))
    mse_loss_norm = torch.sum(torch.sum((norm1 - norm0)**2,dim=1).view(mask.shape[0],1,mask.shape[2],mask.shape[3])*mask)

    # angular_loss    
    angular_loss = torch.sum(torch.sum((norm1 - norm2)**2, dim=1).view(mask.shape[0],1,mask.shape[2],mask.shape[3])*mask)

    if train == True:
        if evaluate == 0:
            losses = mse_loss_1 + mse_loss_2 + angular_loss + mse_loss_proj*5 + mse_loss_norm*5
            losses = losses/(240*320*images.shape[0])
            losses.backward()
            optimizer.step()
    if train == False:
        losses = mse_loss_1 + mse_loss_2 + angular_loss + mse_loss_proj*5 + mse_loss_norm*5
        losses = losses/(240*320*images.shape[0])   
    
    del images_tensor, labels_tensor, loss, output1, mask
    return losses, images.shape[0]

#------------------------------------ training loop ---------------------------------------------#
print("Training DORN ...")
for num_ in tqdm(range(num_epochs)):
    train_loss = 0
    if train == True:
        for i, sampled_batch in enumerate(train_dataloader):
#             print("In epoch : ", num_, " iter : ", i)            
            losses, num_imgs = train_one_iter(i, sampled_batch, 0, use_min=use_min, train=train)            
            train_loss += losses.item()*num_imgs
    logging.info(f"Epoch : {num_} Avg training loss : {train_loss/len(train_dataloader)}")
    training_loss_run2.append(train_loss/len(train_dataloader))
    np.save("training_loss_run2", np.array(training_loss_run2))
    if (num_ % 5 == 0) and (num_ % 10 == 0):
        path = f'./model-run2/model-epoch-1.pth'
        logging.info(f"Storing model after epoch : {num_} in path 1")
        torch.save(cnn.state_dict(), path)
    if (num_ % 5 == 0) and (num_ % 10 != 0):
        path = f'./model-run2/model-epoch-2.pth'
        logging.info(f"Storing model after epoch : {num_} in path 2")
        torch.save(cnn.state_dict(), path)
    
    if num_% 10 == 0:
        val_loss = 0
        logging.info(f"Validating after Epoch : {num_}")
        for it, sample in enumerate(test_dataloader):
            losses, num_imgs = train_one_iter(it, sample, 0, use_min=use_min, train=False)
            val_loss += losses.item()*num_imgs
        logging.info(f"Avg Validation Loss after Epoch {num_} = {val_loss/len(test_dataloader)}")
        validation_loss_run2.append(val_loss/len(test_dataloader))
        np.save("validation_loss_run2", np.array(validation_loss_run2))
                        
    if train == False:
        for i, sampled_batch in enumerate(test_dataloader):
            losses, num_imgs = train_one_iter(i, sampled_batch, 0, use_min=use_min, train=train)
            m_iter += 1
        break
    if num_ == 2:
        use_min = 1



logging.info("Finished Training")
logging.info("Saving Final Model ... ")
finalpath = './model-run2/final_model.pth'
torch.save(cnn.state_dict(), finalpath)
logging.info("Done")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            






    







    
