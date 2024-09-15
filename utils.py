import torch
import numpy as np

def Normalize(dir):
    dir1 = torch.sqrt(torch.sum(dir**2,dim=1) + 1e-7).view(dir.shape[0],1,dir.shape[2],dir.shape[3])
    dir2 = torch.cat([dir1, dir1, dir1], dim=1)
    return dir/dir2

def ConvertToAngle(q):
    a1 = torch.atan2(q[:,1:2,:,:], q[:,0:1,:,:])*180/np.pi
    a2 = torch.atan2(q[:,3:4,:,:], q[:,2:3,:,:])*180/np.pi
    return torch.cat([a1,a2], dim=1)

def Rotate90(q):
    qq = q.clone()
    qq[:,0,:,:] = q[:,2,:,:]
    qq[:,1,:,:] = q[:,3,:,:]
    qq[:,2,:,:] = -q[:,0,:,:]
    qq[:,3,:,:] = -q[:,1,:,:]
    return qq

