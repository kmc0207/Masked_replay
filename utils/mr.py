import torch
import numpy as np
import random

def masking(output,labels,device,num_classes=2):
   # print(len(output))
    for i in range(len(output)):
        #print(output[i],output[i].size())
        t = ((labels[i]//num_classes)*num_classes).to('cpu')
        mask = [i for i in range(t,t+num_classes)]
        not_mask = np.setdiff1d(np.arange(len(output[0])),mask)
        not_mask = torch.tensor(not_mask, dtype = torch.int64).to(device)
        output[i] = output[i].index_fill(dim=0,index=not_mask,value=float('-inf'))
    return output