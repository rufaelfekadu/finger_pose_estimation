from typing import Any, Optional, Sequence, Union
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim

from lightning.pytorch import callbacks
from torch.optim.optimizer import Optimizer

import matplotlib.pyplot as plt
import numpy as np

from hpe.models import build_model, build_backbone, make_test
from hpe.data import build_dataloader
from hpe.data.EmgDataset import build_dataloaders
from hpe.loss import make_loss, NTXentLoss_poly  


class MLP(nn.Module):
    def __init__(self, infeatures=128, outfeatures=16):
        super().__init__()
        self.mlp_head = nn.Sequential(
            nn.Linear(infeatures, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, outfeatures))
    def forward(self, x):
        return self.mlp_head(x)
        

class PlotSample(callbacks.Callback):
    def __init__(self, cfg, dataloaders, device):
        super().__init__()
        self.cfg = cfg
    
    def on_validation_epoch_end(self, trainer, pl_module):
        pass


class EmgNet(pl.LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__()
        #  get cfg from kwargs
        cfg = kwargs['cfg']
        self.cfg = cfg
        dataloaders = build_dataloader(cfg, shuffle=True, visualize=False)
        self.train_loader = dataloaders['train']
        self.val_loader = dataloaders['val']
        self.test_loader = dataloaders['test']

        self.backbone = build_backbone(cfg).to(self.device)
        self.loss_fn = make_loss(cfg)
        self.criterion = torch.nn.L1Loss()

        self.plot_output = 10
        self.train_step_output = []
        self.validation_step_output = []
        self.validation_step_target = []
        self.test_step_output = []

        self.save_hyperparameters()

    def forward(self, x, target=None):
        x = self.backbone(x)
        if target is not None:
            loss = self.loss_fn(x, target)
            return x, loss
        return x, None

    def training_step(self, batch, batch_idx):

        inputs, labels, gestures = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        outputs, losses = self.forward(inputs, labels)
        # loss = self.criterion(outputs, labels)

        loss_dict = {i: v for i, v in zip(self.loss_fn.keypoints, losses[0])}
        self.log_dict({'train_loss': losses[1] })
        return losses[1]

    def validation_step(self, batch, batch_idx):
        inputs, labels, gestures = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        outputs, losses = self.forward(inputs, labels)
        if self.current_epoch % self.plot_output == 0:
            self.validation_step_output.append(outputs.detach().cpu())
            self.validation_step_target.append(labels.detach().cpu())
        # loss = self.criterion(outputs, labels[:,-1,:])
        loss_dict = {i: v for i, v in zip(self.loss_fn.keypoints, losses[0])}
        self.log_dict({'val_loss': losses[1], **loss_dict})
        return losses[1]
    
    def test_step(self, batch, batch_idx):
        inputs, labels, gestures = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        outputs, losses = self.forward(inputs, labels)
        # compute the 90% percentile of the loss
        # loss = self.criterion(outputs, labels[:,-1,:])

        loss_dict = {i: v for i, v in zip(self.cfg.DATA.LABEL_COLUMNS,losses[0])}
        self.test_step_output.append(losses[0].detach().cpu())
        self.log_dict({'test_loss': losses[1], **loss_dict})
        return losses[1]
    
    
    def on_test_end(self) -> None:
        #  plot the scalar values of the logger as bar chart
        fig, ax = plt.subplots(1, 1, figsize=(20,5))
        #  stack the values
        out = torch.stack(self.test_step_output, dim=0).mean(dim=0)
        fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
        for i, c in enumerate(fingers):
            idx = [j for j in range(len(self.loss_fn.keypoints)) if c in self.loss_fn.keypoints[j].lower()]
            ax.bar(idx, out[idx], label=c)
            #  set xticks to be the cfg.label_columns
        ax.set_xticks(range(len(out)))
        ax.set_xticklabels(self.cfg.DATA.LABEL_COLUMNS, rotation=45)
        ax.legend()
        #  bar plot the values
        # ax.bar(range(len(out)), out)
        #  set xticks to be the cfg.label_columns
        # ax.set_xticks(range(len(out)))
        # ax.set_xticklabels(self.cfg.DATA.LABEL_COLUMNS, rotation=45)

        self.logger.experiment.add_figure('final results', fig, self.current_epoch)

    def on_validation_epoch_end(self) -> None:
        # plot a sample output
        from matplotlib import pyplot as plt 
        
        if len(self.validation_step_output)!=0 and len(self.validation_step_output[0].shape) < 3 and self.current_epoch % self.plot_output == 0:
            pred = torch.concatenate(self.validation_step_output, dim=0).view(-1, self.validation_step_output[0].shape[-1])
            target = torch.concatenate(self.validation_step_target, dim=0)[:,0,:].view(-1, self.validation_step_target[0].shape[-1])
            fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']

            fig, ax = plt.subplots(fingers.__len__(), 1, figsize=(20,10))
            #  sample 200 non repeting values randomly or use all
            t = min(100, pred.shape[0])
            idx_x = torch.randperm(pred.shape[0])[:t]
            #  compute average for each finger
            for i, c in enumerate(fingers):
                idx = [j for j in range(len(self.loss_fn.keypoints)) if c in self.loss_fn.keypoints[j].lower()]
                #  sample 200 non repeting values randomly
                ax[i].plot(pred[idx_x,:][:,idx].mean(dim=1))
                ax[i].plot(target[idx_x,:][:,idx].mean(dim=1))
                ax[i].set_title(c)
                #  show legend only for the first plot
                if i == 0:
                    ax[i].legend(['pred', 'target'])
            self.logger.experiment.add_figure('validation sample', fig, self.current_epoch)

        self.validation_step_target = []
        self.validation_step_output = []
        
    def makegrid(output,numrows):
        outer=(torch.Tensor.cpu(output).detach())
        plt.figure(figsize=(20,5))
        b=np.array([]).reshape(0,outer.shape[2])
        c=np.array([]).reshape(numrows*outer.shape[2],0)
        i=0
        j=0
        while(i < outer.shape[1]):
            img=outer[0][i]
            b=np.concatenate((img,b),axis=0)
            j+=1
            if(j==numrows):
                c=np.concatenate((c,b),axis=1)
                b=np.array([]).reshape(0,outer.shape[2])
                j=0
                 
            i+=1
        return c
    
    def configure_optimizers(self):
        opt = self.cfg.SOLVER.OPTIMIZER.lower()
        if opt == 'adam':
            optimizer = optim.Adam(self.backbone.parameters(), lr=self.cfg.SOLVER.LR, weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)
        elif opt == 'sgd':
            optimizer = optim.SGD(self.backbone.parameters(), lr=self.cfg.SOLVER.LR, momentum=0.9, weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)
        else:
            raise NotImplementedError(f'Optimizer {opt} not implemented')
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.cfg.SOLVER.PATIENCE),
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
    
class EmgNetClassifier(EmgNet):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg)
        
        self.cfg = cfg
        dataloaders = build_dataloaders(cfg)
        cfg.DATA.LABEL_COLUMNS = list(dataloaders['train'].dataset.dataset.gesture_names_mapping_class.values())
        
        self.train_loader = dataloaders['train']
        self.val_loader = dataloaders['val']
        self.test_loader = dataloaders['test']

        self.backbone = build_backbone(cfg).to(self.device)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, x, target=None):
        x = self.backbone(x)
        #  compute the softmax
        x = torch.nn.functional.softmax(x, dim=1)
        pred = torch.argmax(x, dim=1)
        if target is not None:
            loss = self.loss_fn(x, target)
            return pred, loss
        return pred, x
    
    def training_step(self, batch, batch_idx):
        inputs, labels, gestures = batch
        inputs = inputs.to(self.device)
        gestures = gestures[0].to(self.device)

        pred, loss = self.forward(inputs, gestures)

        acc = torch.sum(pred == gestures).item() / len(gestures)

        self.log_dict({'train_loss': loss, 'train_acc': acc})

        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels, gestures = batch
        inputs = inputs.to(self.device)
        gestures = gestures[0].to(self.device)

        pred, losses = self.forward(inputs, gestures)
        acc = torch.sum(pred == gestures).item() / len(gestures) 

        self.log_dict({'val_loss': losses, 'val_acc': acc*100})

        return losses
    
    def test_step(self, batch, batch_idx):
        inputs, labels, gestures = batch
        inputs = inputs.to(self.device)
        gestures = gestures[0].to(self.device)

        pred, losses = self.forward(inputs, gestures)
        acc = torch.sum(pred == gestures).item() / len(gestures)
        self.log_dict({'test_loss': losses, 'test_acc': acc*100})

        return losses
    
    def configure_optimizers(self):
        return super().configure_optimizers()
    
class EmgNetPretrain(pl.LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__()
        
        cfg = kwargs['cfg']
        self.cfg = cfg

        #  setup dataloaders
        self.stage = cfg.STAGE
        dataloaders = build_dataloaders(cfg)
        self.pretrain_loader = dataloaders['pretrain']
        self.train_loader = dataloaders['train']
        self.val_loader = dataloaders['val']
        self.test_loader = dataloaders['test']
        self.test_2_loader = dataloaders['test_2']

        #  setup model
        # self.backbone_t = build_backbone(cfg).to(self.device)
        # self.backbone_f = build_backbone(cfg).to(self.device)
        self.backbone_t = make_test(cfg).to(self.device)
        self.backbone_f = make_test(cfg).to(self.device)
        infeat_t =  (self.backbone_f.d_model)
        infeat_f = (self.backbone_t.d_model)
        self.mlp = MLP(infeatures=infeat_t+infeat_f, outfeatures=len(self.cfg.DATA.LABEL_COLUMNS)).to(self.device)

        # self.backbone_f.mlp_head = nn.Identity()
        # self.backbone_t.mlp_head = nn.Identity()
            

        self.projector_t = nn.Sequential(
            nn.Linear(infeat_t, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        ).to(self.device)

        self.projector_f = nn.Sequential(
            nn.Linear(infeat_f, 256),
            nn.ReLU(),
            nn.Linear(256, 128)

        ).to(self.device)

        # setup loss
        self.loss_fn = make_loss(cfg)

        self.plot_output = 10
        self.validation_step_output = []
        self.validation_step_target = []

        self.save_hyperparameters()

    def forward(self, x_t, x_f):

        x, o_t = self.backbone_t(x_t)
        h_t = x.view(x.size(0), -1)

        z_t = self.projector_t(h_t)

        f, o_f = self.backbone_f(x_f)
        h_f = f.view(f.size(0), -1)

        z_f = self.projector_f(h_f)
        
        return h_t, o_t, z_t, h_f, o_f, z_f
    
    def pretrain_step(self, batch, batch_idx):

        data, aug1, data_f, aug1_f, _, _ = batch
        data = data.to(self.device)
        aug1 = aug1.to(self.device)
        data_f = data_f.to(self.device)
        aug1_f = aug1_f.to(self.device)

        h_t, o_t, z_t, h_f, o_f, z_f = self.forward(data, data_f)
        h_t_aug, _ , z_t_aug, h_f_aug, _, z_f_aug = self.forward(aug1, aug1_f)

        nt_xent_criterion = NTXentLoss_poly(batch_size=self.cfg.SOLVER.BATCH_SIZE, temperature=self.cfg.SOLVER.TEMPERATURE, device=self.device, use_cosine_similarity=True)
        
        l_t = nt_xent_criterion(h_t, h_f_aug)
        l_f = nt_xent_criterion(h_f, h_t_aug)
        l_TF = nt_xent_criterion(z_t, z_f)

        l_1, l_2, l_3 = nt_xent_criterion(z_t, z_f_aug), nt_xent_criterion(z_t_aug, z_f), nt_xent_criterion(z_t_aug, z_f_aug)

        loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)

        lam = 0.2
        loss = lam*(l_t + l_f) + loss_c

        self.log_dict({'pretrain_loss': loss, 'pretrain_loss_t': l_t, 'pretrain_loss_f': l_f, 'pretrain_loss_TF': l_TF, 'pretrain_loss_c': loss_c} )
        return loss
    
    def finetune_step(self, batch, batch_idx, stage='train'):
        data, _, data_f, _, label, _ = batch
        data = data.to(self.device)
        data_f = data_f.to(self.device)
        label = label.to(self.device)

        t, o_t = self.backbone_t(data)
        h_t = t.view(t.size(0), -1)
        f, o_f = self.backbone_f(data_f)
        h_f = f.view(f.size(0), -1)

        #  compute the simCLR loss
        nt_xent_criterion = NTXentLoss_poly(batch_size=self.cfg.SOLVER.BATCH_SIZE, temperature=self.cfg.SOLVER.TEMPERATURE, device=self.device, use_cosine_similarity=True)
        l_TF = nt_xent_criterion(h_t, h_f)

        #compute the time and freq loss
        # l_t = self.loss_fn(o_t, label)
        # l_f = self.loss_fn(o_f, label)

        # concatinate the features
        h = torch.cat((h_t, h_f), dim=1)
        pred = self.mlp(h)
        loss_c = self.loss_fn(pred, label)

        #  total loss
        l_T = l_TF  + loss_c[1]

        if self.current_epoch % self.plot_output == 0 and stage == 'val':
            self.validation_step_output.append(pred.detach().cpu())
            self.validation_step_target.append(label.detach().cpu())

        loss_dict = {i: v for i, v in zip(self.loss_fn.keypoints, loss_c[0])}
        if stage == 'val':
            self.log_dict({f'{stage}_loss_c': loss_c[1], f'{stage}_loss_TF':l_TF, **loss_dict})
        else:
            self.log_dict({f'{stage}_loss_c': loss_c[1], f'{stage}_loss_TF':l_TF})
        return l_T

    
    def training_step(self, batch, batch_idx):
        if self.stage == 'pretrain':
            return self.pretrain_step(batch, batch_idx)
        
        else:
            return self.finetune_step(batch, batch_idx, stage='train')

    def validation_step(self, batch, batch_idx):
        if self.stage == 'pretrain':
            # skip validation
            return
        else:
            return self.finetune_step(batch, batch_idx, stage='val')
    
    def test_step(self, batch, batch_idx, dataloader_idx):
        if self.stage == 'pretrain':
            # skip test
            return
        else:
            return self.finetune_step(batch, batch_idx, stage=f'test_{dataloader_idx}')
    
    def on_validation_epoch_end(self) -> None:
        # plot a sample output
        from matplotlib import pyplot as plt 
        
        if len(self.validation_step_output)!=0 and len(self.validation_step_output[0].shape) < 3 and self.current_epoch % self.plot_output == 0:
            pred = torch.concatenate(self.validation_step_output, dim=0).view(-1, self.validation_step_output[0].shape[-1])
            target = torch.concatenate(self.validation_step_target, dim=0)[:,0,:].view(-1, self.validation_step_target[0].shape[-1])
            fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']

            fig, ax = plt.subplots(fingers.__len__(), 1, figsize=(20,10))
            #  sample 200 non repeting values randomly or use all
            t = min(100, pred.shape[0])
            idx_x = torch.randperm(pred.shape[0])[:t]
            #  compute average for each finger
            for i, c in enumerate(fingers):
                idx = [j for j in range(len(self.loss_fn.keypoints)) if c in self.loss_fn.keypoints[j].lower()]
                ax[i].plot(pred[idx_x,:][:,idx].mean(dim=1))
                ax[i].plot(target[idx_x,:][:,idx].mean(dim=1))
                ax[i].set_title(c)
                #  show legend only for the first plot
                if i == 0:
                    ax[i].legend(['pred', 'target'])
            self.logger.experiment.add_figure('validation sample', fig, self.current_epoch)

        self.validation_step_target = []
        self.validation_step_output = []
    
    def configure_optimizers(self):
        if self.stage== 'pretrain':
            optimizer = torch.optim.Adam(self.backbone_t.parameters(), lr=self.cfg.SOLVER.LR)
            optimizer.add_param_group({'params': self.projector_t.parameters()})
            optimizer.add_param_group({'params': self.backbone_f.parameters()})
            optimizer.add_param_group({'params': self.projector_f.parameters()})
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.cfg.SOLVER.PATIENCE),
                    'monitor': 'pretrain_loss',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        else:
            optimizer = torch.optim.Adam(self.backbone_t.parameters(), lr=self.cfg.SOLVER.LR)
            optimizer.add_param_group({'params': self.mlp.parameters()})
            optimizer.add_param_group({'params': self.backbone_f.parameters()})
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.cfg.SOLVER.PATIENCE),
                    'monitor': 'val_loss_c',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }


    def train_dataloader(self):
        if self.stage == 'pretrain':
            return self.pretrain_loader
        return self.train_loader
    
    def val_dataloader(self):
        return self.val_loader
    
    def test_dataloader(self):
        return [self.test_loader, self.test_2_loader]