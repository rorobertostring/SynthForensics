import torch
import os
from copy import deepcopy
from tqdm import tqdm

from options.test_options import TestOption
from utils.trainer import ValTrainer
from utils.utils import get_logger, get_test_dataset_configs, set_random_seed
from dataset import get_test_dataloader
from builder import get_model
from models import MMDet
    
if __name__ == '__main__':
    args = TestOption().parse()
    config = args.__dict__
    logger = get_logger(__name__, config)
    logger.info(config)
    set_random_seed(config['seed'])
    dataset_classes = config['classes']
    logger.info(f'Validation on {dataset_classes}.')
    test_dataset_configs = get_test_dataset_configs(config)
    config['st_pretrained'] = False
    config['st_ckpt'] = None   # disable initialization
    model = MMDet(config)
    model.eval()
    path = None
    if os.path.exists(config['ckpt']):
        logger.info(f'Load checkpoint from {config["ckpt"]}')
        path = config['ckpt']
    elif os.path.exists(os.path.join('expts', config['expt'], 'checkpoints')):
        if os.path.exists(os.path.join('expts', config['expt'], 'checkpoints', 'current_model_best.pth')):
            logger.info(f'Load best checkpoint from {config["ckpt"]}')
            path = os.path.join('expts', config['expt'], 'checkpoints', 'current_model_best.pth')
        elif os.path.exists(os.path.join('expts', config['expt'], 'checkpoints', 'current_model_latest.pth')):
            logger.info(f'Load latest checkpoint from {config["ckpt"]}')
            path = os.path.join('expts', config['expt'], 'checkpoints', 'current_model_latest.pth')
    if path is None:
        raise ValueError(f'Checkpoint not found: {config["ckpt"]}')
    state_dict = torch.load(path)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if 'temporal_token' in k:
            k = k.replace('temporal_token', 'fc_token')
        new_state_dict[k.replace('module.', '')] = v
        
    state_dict = new_state_dict
    model.load_state_dict(state_dict, strict=config['cache_mm'])
    real_probs = []
    count = 0
    for dataset_class, test_dataset_config in zip(dataset_classes, test_dataset_configs):
        test_config = deepcopy(config)
        test_config['datasets'] = test_dataset_config
        trainer = ValTrainer(
            config=test_config, 
            model=model, 
            logger=logger,
        )
        if len(real_probs) != 0:
            trainer.reals_probs = real_probs
        trainer.val_dataloader = get_test_dataloader(test_dataset_config)
        if 'sample_size' in config:    # evaluation on sampled data to save time
            stop_count = config['sample_size']
        else:
            stop_count = -1
        results = trainer.nonover_validation_video(stop_count=stop_count, dataset_name=dataset_class)    # evaluate real videos for once, and fake videos for each
        logger.info(f'{dataset_class}')
        for metric, value in results['metrics'].items():
            logger.info(f'{metric}: {value}')
        if count == 0:
            real_probs = trainer.reals_probs
        count += 1