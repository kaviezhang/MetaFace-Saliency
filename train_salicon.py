import os
import time
import copy
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import cv2
from utils.loss_function import *
from utils.data_process import MyDataset
import math
from sklearn.metrics import roc_auc_score

if __name__ == '__main__':
    flag = 0
    batch_size = 12
    lr = 1e-3
    if flag == 0:
        from HATModel import *
        model = Swin_cpas_offset()
    train_ids = pd.read_csv(r'D:\mnt\SALICON\SALICON/train_ids.csv')
    val_ids = pd.read_csv(r'D:\mnt\SALICON\SALICON/val_ids.csv')
    print(train_ids.iloc[1])
    print(val_ids.iloc[1])

    dataset_sizes = {'train': len(train_ids), 'val': len(val_ids)}
    print(dataset_sizes)

    train_set = MyDataset(ids=train_ids,
                          stimuli_dir=r'D:\mnt\SALICON\SALICON\train\train_stimuli/',
                          saliency_dir=r'D:\mnt\SALICON\SALICON\train\train_saliency/',
                          fixation_dir=r'D:\mnt\SALICON\SALICON\train\train_fixation/',
                          transform=transforms.Compose([
                              transforms.Resize((288, 384)),
                              transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                          ]))

    val_set = MyDataset(ids=val_ids,
                        stimuli_dir=r'D:\mnt\SALICON\SALICON\val\val_stimuli/',
                        saliency_dir=r'D:\mnt\SALICON\SALICON\val\val_saliency/',
                        fixation_dir=r'D:\mnt\SALICON\SALICON\val\val_fixation/',
                        transform=transforms.Compose([
                            transforms.Resize((288, 384)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ]))

    dataloaders = {'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1),
                   'val': DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=1)}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Training paras=" + str(trainable_num))

    # single learning rate
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # multi learning rate
    backbone_params = list(map(id, model.backbone.parameters()))
    model_params = filter(lambda p: id(p) not in backbone_params, model.parameters())
    params = [
        {'params': model.backbone.parameters(), 'lr': lr * 0.1},
        {'params': model_params, 'lr': lr}
    ]
    optimizer = optim.Adam(lr=lr, params=params)
    fun_1 = lambda epoch: 0.1 ** (epoch // 10)
    fun_2 = lambda epoch: 0.1 ** (epoch // 10)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=[fun_1, fun_2], verbose=True)

    loss_fn = SaliencyLoss()
    # criterion = EdgeSaliencyLoss(torch.device(device='cuda'))
    criterion = nn.L1Loss()

    '''Training'''
    best_model_wts = copy.deepcopy(model.state_dict())
    num_epochs = 30
    best_loss = 100
    # for k, v in model.named_parameters():
    #     print('{}: {}'.format(k, v.requires_grad))

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            total_cc = 0.0
            total_sim = 0.0
            total_kld = 0.0
            total_nss = 0.0

            # Iterate over data.
            auc_all = []
            aucjudd_all = []
            for i_batch, sample_batched in tqdm(enumerate(dataloaders[phase])):
                stimuli, smap, fmap = sample_batched['image'], sample_batched['saliency'], sample_batched['fixation']
                stimuli, smap, fmap = \
                    stimuli.type(torch.cuda.FloatTensor), \
                    smap.type(torch.cuda.FloatTensor), \
                    fmap.type(torch.cuda.FloatTensor)
                stimuli, smap, fmap = stimuli.to(device), smap.to(device), fmap.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(stimuli)

                    loss_cc = loss_fn(outputs, smap, loss_type='cc')
                    loss_sim = loss_fn(outputs, smap, loss_type='sim')
                    loss_kldiv = loss_fn(outputs, smap, loss_type='kldiv')
                    loss_nss = loss_fn(outputs, fmap, loss_type='nss')
                    # loss = -1 * loss_cc - 1 * loss_sim + 10 * loss_kldiv - 0.1 * loss_nss
                    loss = loss_kldiv
                    # loss = criterion(outputs, smap)
                    print('Batch{} {} Loss: {:.4f}'.format(i_batch, phase, loss.item()))
                    act = loss_cc.item(), loss_sim.item(), loss_kldiv.item(), loss_nss.item()
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    if phase == 'val':
                        for index in range(outputs.shape[0]):
                            pred = outputs[index, 0]
                            pred = pred.cpu().detach().numpy()
                            output = np.clip(pred * 255, 0, 255)
                            cv2.imwrite("result/{}.png".format(i_batch*batch_size+index), output)
                            pred = pred
                            label = fmap[index, 0].cpu().numpy().astype(dtype=bool)
                            if epoch == num_epochs - 1:
                                auc_all.append(roc_auc_score(label.flatten(), pred.flatten()))
                                aucjudd_all.append(AUC_Judd(pred/pred.max(), label))
                # statistics
                running_loss += loss.item() * stimuli.size(0)
                total_cc += loss_cc.item() * stimuli.size(0)
                total_sim += loss_sim.item() * stimuli.size(0)
                total_kld += loss_kldiv.item() * stimuli.size(0)
                total_nss += loss_nss.item() * stimuli.size(0)

            if phase == 'train':
                scheduler.step()
            if phase == 'val' and epoch == num_epochs - 1:
                print("AUC:{:.4f} AUC-Judd:{:.4f}".format(sum(auc_all)/len(auc_all), sum(aucjudd_all)/len(aucjudd_all)))

            epoch_loss = running_loss / dataset_sizes[phase]
            total_cc = total_cc / dataset_sizes[phase]
            total_sim = total_sim / dataset_sizes[phase]
            total_kld = total_kld / dataset_sizes[phase]
            total_nss = total_nss / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))
            print('Average {},cc={},sim={},kld={},nss={}'.format(phase, total_cc, total_sim, total_kld, total_nss))
            if not math.isnan(total_kld):
                print("保存模型")
                torch.save(model.state_dict(), "my-model_salicon.pth")
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                counter = 0
            elif phase == 'val' and epoch_loss >= best_loss:
                counter += 1
                if counter == 10:
                    print('early stop!')
                    # break

    print('Best val loss: {:4f}'.format(best_loss))
    model.load_state_dict(best_model_wts)
    savepath = r'my-model_salicon.pth'
    torch.save(model.state_dict(), savepath)
