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

from hpe.models import build_model, build_backbone
from hpe.data import build_dataloader
from hpe.loss import make_loss  


class PlotSample(callbacks.Callback):
    def __init__(self, cfg, dataloaders, device):
        super().__init__()
        self.cfg = cfg
    
    def on_validation_epoch_end(self, trainer, pl_module):
        pass


class EmgNet(pl.LightningModule):

    def __init__(self, cfg, *args, **kwargs):
        super().__init__()
        
        self.cfg = cfg
        dataloaders = build_dataloader(cfg)
        self.train_loader = dataloaders['train']
        self.val_loader = dataloaders['val']
        self.test_loader = dataloaders['test']

        self.backbone = build_backbone(cfg).to(self.device)
        self.loss_fn = make_loss(cfg)
        self.criterion = torch.nn.MSELoss()

        self.plot_output = 10
        self.train_step_output = []
        self.validation_step_output = []
        self.validation_step_taregt = []
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
            self.validation_step_taregt.append(labels.detach().cpu())
        # loss = self.criterion(outputs, labels[:,-1,:])
        loss_dict = {i: v for i, v in zip(self.loss_fn.keypoints, losses[0])}
        self.log_dict({'val_loss': losses[1], **loss_dict})
        return losses[1]
    
    def test_step(self, batch, batch_idx):
        inputs, labels, gestures = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        outputs, losses = self.forward(inputs, labels)
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
        #  bar plot the values
        ax.bar(range(len(out)), out)
        #  set xticks to be the cfg.label_columns
        ax.set_xticks(range(len(out)))
        ax.set_xticklabels(self.cfg.DATA.LABEL_COLUMNS, rotation=45)

        self.logger.experiment.add_figure('final results', fig, self.current_epoch)


    def on_validation_epoch_end(self) -> None:
        # plot a sample output
        from matplotlib import pyplot as plt 
        
        if len(self.validation_step_output)!=0 and len(self.validation_step_output[0].shape) < 3 and self.current_epoch % self.plot_output == 0:
            pred = torch.concatenate(self.validation_step_output, dim=0).view(-1, self.validation_step_output[0].shape[-1])
            target = torch.concatenate(self.validation_step_taregt, dim=0)[:,0,:].view(-1, self.validation_step_taregt[0].shape[-1])
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

        self.validation_step_taregt = []
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
    
    
    # def on_train_epoch_end(self) -> None:
    #     if(self.current_epoch==1):
    #         sampleImg=torch.rand(1,self.cfg.DATA.SEGMENT_LENGTH,16, device=self.device)
    #         self.logger.experiment.add_graph(self.backbone,sampleImg)
    #     return super().on_train_epoch_end()

    def configure_optimizers(self):
        opt = self.cfg.SOLVER.OPTIMIZER.lower()
        if opt == 'adam':
            optimizer = optim.Adam(self.backbone.parameters(), lr=self.cfg.SOLVER.LR)
        elif opt == 'sgd':
            optimizer = optim.SGD(self.backbone.parameters(), lr=self.cfg.SOLVER.LR, momentum=0.9)
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
        dataloaders = build_dataloader(cfg)
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

    def __init__(self, cfg, *args, **kwargs):
        super().__init__()
        
        self.cfg = cfg
        cfg.DATA.ICA = True

        dataloaders = build_dataloader(cfg)
        self.train_loader = dataloaders['train']
        self.val_loader = dataloaders['val']
        self.test_loader = dataloaders['test']

        self.backbone = build_backbone(cfg)
        in_features = self.backbone.decoder[0].in_features
        self.projection_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.loss = make_loss(cfg)

        self.optimizer = torch.optim.Adam(self.backbone.parameters(), lr=cfg.SOLVER.LR)
        self.optimizer.add_param_group({'params': self.projection_head.parameters()})

    def forward(self, x, target=None):
        x = self.backbone(x)
        if target is not None:
            loss = self.loss(x, target)
            return x, loss
        return x, None

    def training_step(self, batch, batch_idx):

        inputs, labels, gestures = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        inputs_x, inputs_ica = inputs

        _, loss = self.forward(inputs, labels)
        # loss = self.criterion(outputs, labels)

        self.log_dict({'train_loss': loss[1]})
        return loss[1]

    def validation_step(self, batch, batch_idx):
        inputs, labels, gestures = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        outputs, losses = self.forward(inputs, labels)
        # loss = self.criterion(outputs, labels[:,-1,:])
        loss_dict = {i: v for i, v in zip(self.cfg.DATA.LABEL_COLUMNS,losses[0])}
        self.log_dict({'loss': losses[1], **loss_dict}, prefix='val')
        return losses[1]
    
    def test_step(self, batch, batch_idx):
        inputs, labels, gestures = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        outputs, losses = self.forward(inputs, labels)
        # loss = self.criterion(outputs, labels[:,-1,:])
        loss_dict = {i: v for i, v in zip(self.cfg.DATA.LABEL_COLUMNS,losses[0])}
        self.log_dict({'loss': losses[1], **loss_dict}, prefix='test')
        return losses[1]

    def configure_optimizers(self):
        return self.optimizer

    def train_dataloader(self):
        return self.train_loader