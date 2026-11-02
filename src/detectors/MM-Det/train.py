import torch
import os
import json
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from options.train_options import TrainOption
from utils.trainer import Trainer
from utils.utils import get_logger, get_train_dataset_config, set_random_seed
from dataset import get_train_dataloader
from builder import get_model
from models import MMDet


def create_optimizer(config, model):
    lr = config['lr']
    param_dict_list = []
    for component in [model.backbone, model.clip_proj, model.mm_proj, model.final_fusion, model.head]:    # only add customized modules
        param_dict_list.append({'params': component.parameters(), 'lr': lr})
    print('Finish assigning lr for modules.')
    optimizer = Adam(param_dict_list, weight_decay=config['weight_decay'])
    return optimizer
    
    
def create_scheduler(config, optimizer):
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['step_factor'], min_lr=1e-08, patience=config['patience'], cooldown=config['cooldown'], verbose=True)
    return lr_scheduler
    
    
if __name__ == '__main__':
    args = TrainOption().parse()
    config = args.__dict__
    logger = get_logger(__name__, config)
    logger.info(config)
    set_random_seed(config['seed'])
    config["datasets"] = get_train_dataset_config(config)
    train_dataloader, val_dataloader = get_train_dataloader(config)
    model = MMDet(config)
    optimizer = create_optimizer(config, model)
    scheduler = create_scheduler(config, optimizer)
    trainer = Trainer(
        config=config, 
        model=model, 
        logger=logger,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    trainer.train()
    