import os
import os
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from tensorboardX import SummaryWriter
from tqdm import tqdm

from .utils import construct_cached_mm_representations, get_nearest_mm_index


class Trainer:
    def __init__(self, config, model, logger, train_dataloader=None, val_dataloader=None, optimizer=None, scheduler=None):
        self.config = config
        self.model = model
        self.logger = logger
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.metrics = {'AUC': roc_auc_score}
        self.min_val_loss = 10    # record the lowest val loss
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cache_mm = config['cache_mm']
        if self.cache_mm:
            self.cached_mm_representations, self.cached_mm_representations_index = construct_cached_mm_representations(config)
        self.step = 1
        self.epoch = 0
        if config['mode'] == 'train':
            self.writer = SummaryWriter(log_dir=os.path.join('expts', config['expt'], config['log_dir']))
        
    def training_step(self, batch):
        fns, data, labels = batch
        original, recons = data
        original, recons, labels = original.float().to(self.device), recons.float().to(self.device), labels.to(self.device)
        input_batch = original, recons
        if self.cache_mm:
            visual_representations = []
            textual_representations = []
            for fn in fns:
                *_, dname, label, frame_id = fn.rsplit('/')
                prefix, index = frame_id.rsplit('__', maxsplit=1)
                cached_index = get_nearest_mm_index(self.cached_mm_representations_index[dname][label][prefix], index)
                visual_representations.append(self.cached_mm_representations[dname][label][f'{prefix}_{cached_index}.jpg']['visual'])
                textual_representations.append(self.cached_mm_representations[dname][label][f'{prefix}_{cached_index}.jpg']['textual']['-1'])
            visual_representations = torch.stack(visual_representations, dim=0).to(self.device)
            textual_representations = torch.stack(textual_representations, dim=0).to(self.device)
            pred = self.model(input_batch, cached_features={
                'visual': visual_representations,
                'textual': textual_representations
            })
        else:
            pred = self.model(input_batch)
        pred = torch.clamp(pred, -10, 10)

        self.step += 1
        loss = self.criterion(pred, labels)
        
        # Know issue: After the model converges for several epochs, there may be NaN in the output.
        # Stop the training phase here to ensure the parameters are not ruined.
        if torch.isnan(loss).any():
            print('Detect Nan in training. Stop.')
            exit(0)
        return loss
    
    def validation_step(self, batch):
        fns, data, labels = batch
        original, recons = data
        original, recons = original.float().to(self.device), recons.float().to(self.device)
        B, L, C, H, W = original.shape
        results = {
            'gt': {},
            'pred': {},
            'proba': {},
        }
        for i in range(0, L - self.config['window_size'] + 1, self.config['window_size']):
            video_clip = original[:, i: i + self.config['window_size'], :, :, :], recons[:, i: i + self.config['window_size'], :, :, :]
            if self.cache_mm:
                visual_representations = []
                textual_representations = []
                for fn in fns:
                    *_, dname, label, frame_id = fn.rsplit('/')
                    prefix, index = frame_id.rsplit('__', maxsplit=1)
                    cached_index = get_nearest_mm_index(self.cached_mm_representations_index[dname][label][prefix], int(index) + i)
                    visual_representations.append(self.cached_mm_representations[dname][label][f'{prefix}_{cached_index}.jpg']['visual'])
                    textual_representations.append(self.cached_mm_representations[dname][label][f'{prefix}_{cached_index}.jpg']['textual']['-1'])
                visual_representations = torch.stack(visual_representations, dim=0).to(self.device)
                textual_representations = torch.stack(textual_representations, dim=0).to(self.device)
                pred = self.model(video_clip, cached_features={
                    'visual': visual_representations,
                    'textual': textual_representations
                })
            else:
                pred = self.model(video_clip)
            log_probs = F.softmax(pred, dim=-1)
            for idx, fn in enumerate(fns):
                if fn not in results['gt']:
                    results['gt'][fn] = labels[idx].item()
                    results['pred'][fn] = [pred[idx].cpu().tolist()]
                    results['proba'][fn] = [log_probs[idx, 1].cpu().item()]
                else:
                    results['pred'][fn].append(pred[idx].cpu().tolist())
                    results['proba'][fn].append(log_probs[idx, 1].cpu().item())
        return results
        
    def validation_video(self, stop_count=-1):
        video_validation = {}
        frame_gts = []
        gts = []
        preds = []
        probas = []
        count = 0
        self.model.to(self.device)
        for batch in tqdm(self.val_dataloader, desc='Validation'):
            results = self.validation_step(batch)
            for fn in results['gt']:
                gts.append(results['gt'][fn])
                frame_gts.extend([results['gt'][fn]] * len(results['pred'][fn]))
                preds.extend(results['pred'][fn])
                probas.append(sum(results['proba'][fn]) / len(results['proba'][fn]))
            count += 1
            if stop_count != -1:
                if count >= stop_count:
                    break
        frame_gts = torch.tensor(frame_gts, dtype=torch.long)
        preds = torch.tensor(preds)
        loss = self.criterion(preds, frame_gts)
        video_validation['loss'] = loss
        video_validation['metrics'] = {}
        for name, metric in self.metrics.items():
            value = metric(gts, probas)
            video_validation['metrics'][name] = value
        return video_validation
    
    def train(self):
        train_loss = 0
        display_loss = 0
        log_loss = 0
        self.model.to(self.device)
        for idx, epoch in enumerate(range(self.epoch, self.config['epoch'])):
            self.epoch = epoch
            self.logger.info(f'Epoch {epoch + 1}/{self.config["epoch"]}')
            for batch in self.train_dataloader:
                self.model.train()
                self.optimizer.zero_grad()
                loss = self.training_step(batch)
                train_loss += loss.item()
                display_loss += loss.item()
                log_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                if self.step % self.config['display_step'] == 0:
                    self.logger.info(f'[Epoch {epoch + 1}, Step {self.step}] Training Loss: {display_loss / self.config["display_step"]}')
                    display_loss = 0
                if self.step % self.config['log_step'] == 0:
                    self.writer.add_scalar('loss/train_loss', log_loss / self.config['log_step'], global_step=self.step)
                    log_loss = 0
                if self.step % self.config['val_step'] == 0:
                    self.model.eval()
                    val_results = self.validation_video()
                    self.writer.add_scalar('loss/val_loss', val_results['loss'].item(), global_step=self.step)
                    self.logger.info(f'[Epoch {epoch + 1}, Step {self.step}] Val Loss: {val_results["loss"].item()}')
                    for metric, value in val_results['metrics'].items():
                        self.writer.add_scalar(f'val_metrics/{metric}', value, global_step=self.step)
                        self.logger.info(f'[Epoch {epoch + 1}, Step {self.step}] Val {metric}: {value}')
                    if self.scheduler is not None and isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_results['loss'].item())
                        self.logger.info(f'Patience: {self.scheduler.num_bad_epochs} / {self.config["patience"]}')
                    if val_results['loss'].item() < self.min_val_loss:
                        self.min_val_loss = val_results['loss'].item()
                        self.save_model('best')
                        self.save_model('latest')
                    # if val_results['loss'].item() < self.config['val_bound']:
                    #     self.logger.info('Finished.')
                    #     self.save_model('latest')
            if self.epoch % self.config['save_epoch'] == 0:
                self.save_model(str(epoch + 1))
                self.save_model('latest')
        self.save_model(str(epoch + 1))
        self.save_model('latest')

    
    def save_model(self, tag='latest'):
        os.makedirs(os.path.join('expts', self.config['expt'], 'checkpoints'), exist_ok=True)
        model_state_dict = self.model.state_dict()
        for k in model_state_dict:
            if 'mm_encoder.model' in k:    # deprecate frozen parameters from LMM branch
                del(model_state_dict[k])
        state_dict = {
            'model_state_dict': model_state_dict,
            'optimizer': self.optimizer.state_dict() if self.optimizer is not None else None,
            'scheduler': self.scheduler,
            'epoch': self.epoch,
            'step': self.step,
        }
        torch.save(state_dict, os.path.join('expts', self.config['expt'], 'checkpoints', f'current_model_{tag}.pth'))
        
    def load_model(self, path, strict=True):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict['model_state_dict'], strict=strict)
        self.epoch = state_dict['epoch']
        self.step = state_dict['step']
        if self.optimizer is not None:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        if self.scheduler is not None:
            self.scheduler = state_dict['scheduler']


class ValTrainer:
    def __init__(self, config, model, logger, train_dataloader=None, val_dataloader=None, optimizer=None, scheduler=None):
        self.config = config
        self.model = model
        self.logger = logger
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.metrics = {'AUC': roc_auc_score}
        self.min_val_loss = 10    # record the lowest val loss
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cache_mm = config['cache_mm']
        if self.cache_mm:
            self.cached_mm_representations, self.cached_mm_representations_index = construct_cached_mm_representations(config)
        self.step = 1
        self.epoch = 0
        if config['mode'] == 'train':
            self.writer = SummaryWriter(log_dir=os.path.join('expts', config['expt'], config['log_dir']))
        self.reals_probs = []
        self.reals_judge = True
        
    def training_step(self, batch):
        fns, data, labels = batch
        original, recons = data
        original, recons, labels = original.float().to(self.device), recons.float().to(self.device), labels.to(self.device)
        input_batch = original, recons
        if self.cache_mm:
            visual_representations = []
            textual_representations = []
            for fn in fns:
                *_, dname, label, frame_id = fn.rsplit('/')
                prefix, index = frame_id.rsplit('__', maxsplit=1)
                cached_index = get_nearest_mm_index(self.cached_mm_representations_index[dname][label][prefix], index)
                visual_representations.append(self.cached_mm_representations[dname][label][f'{prefix}_{cached_index}.jpg']['visual'])
                textual_representations.append(self.cached_mm_representations[dname][label][f'{prefix}_{cached_index}.jpg']['textual']['-1'])
            visual_representations = torch.stack(visual_representations, dim=0).to(self.device)
            textual_representations = torch.stack(textual_representations, dim=0).to(self.device)
            pred = self.model(input_batch, cached_features={
                'visual': visual_representations,
                'textual': textual_representations
            })
        else:
            pred = self.model(input_batch)
        self.step += 1
        loss = self.criterion(pred, labels)
        return loss
    
    def validation_step(self, batch):
        fns, data, labels = batch
        original, recons = data
        original, recons = original.float().to(self.device), recons.float().to(self.device)
        B, L, C, H, W = original.shape
        results = {
            'gt': {},
            'pred': {},
            'proba': {},
        }
        for i in range(0, L - self.config['window_size'] + 1, self.config['window_size']):
            video_clip = original[:, i: i + self.config['window_size'], :, :, :], recons[:, i: i + self.config['window_size'], :, :, :]
            if self.cache_mm:
                visual_representations = []
                textual_representations = []
                for fn in fns:
                    *_, dname, label, frame_id = fn.rsplit('/')
                    prefix, index = frame_id.rsplit('__', maxsplit=1)
                    cached_index = get_nearest_mm_index(self.cached_mm_representations_index[dname][label][prefix], int(index) + i)
                    visual_representations.append(self.cached_mm_representations[dname][label][f'{prefix}_{cached_index}.jpg']['visual'])
                    textual_representations.append(self.cached_mm_representations[dname][label][f'{prefix}_{cached_index}.jpg']['textual']['-1'])
                visual_representations = torch.stack(visual_representations, dim=0).to(self.device)
                textual_representations = torch.stack(textual_representations, dim=0).to(self.device)
                pred = self.model(video_clip, cached_features={
                    'visual': visual_representations,
                    'textual': textual_representations
                })
            else:
                pred = self.model(video_clip)
            log_probs = F.softmax(pred, dim=-1)
            for idx, fn in enumerate(fns):
                if fn not in results['gt']:
                    results['gt'][fn] = labels[idx].item()
                    results['pred'][fn] = [pred[idx].cpu().tolist()]
                    results['proba'][fn] = [log_probs[idx, 1].cpu().item()]
                else:
                    results['pred'][fn].append(pred[idx].cpu().tolist())
                    results['proba'][fn].append(log_probs[idx, 1].cpu().item())
        return results
        
    def validation_video(self, stop_count=-1):
        video_validation = {}
        frame_gts = []
        gts = []
        preds = []
        probas = []
        count = 0
        has_real = False
        self.model.to(self.device)
        for batch in tqdm(self.val_dataloader, desc='Validation'):
            results = self.validation_step(batch)
            for fn in results['gt']:
                gts.append(results['gt'][fn])
                frame_gts.extend([results['gt'][fn]] * len(results['pred'][fn]))
                preds.extend(results['pred'][fn])
                probas.append(sum(results['proba'][fn]) / len(results['proba'][fn]))
            count += 1
            if stop_count != -1:
                if count >= stop_count:
                    break
        frame_gts = torch.tensor(frame_gts, dtype=torch.long)
        preds = torch.tensor(preds)
        loss = self.criterion(preds, frame_gts)
        video_validation['loss'] = loss
        video_validation['metrics'] = {}
        for name, metric in self.metrics.items():
            value = metric(gts, probas)
            video_validation['metrics'][name] = value
        return video_validation
    
    def nonover_validation_video(self, stop_count=-1, dataset_name=''):
        video_validation = {}
        frame_gts = []
        gts = []
        preds = []
        probas = []
        count = 0
        real_probs = []
        self.model.to(self.device)
        os.makedirs(os.path.join('expts', self.config['expt'], 'csv'), exist_ok=True)

        with open(os.path.join('expts', self.config['expt'], 'csv', f'test_{dataset_name}.csv'), 'w') as f:
            f.write('fn,gt,preds\n')

        for batch in tqdm(self.val_dataloader, desc='Validation'):
            fns = batch[0]
            skip_val = False
            if len(self.reals_probs) != 0: 
                for fn in fns:
                    if '0_real' in fn:
                        skip_val = True
            if skip_val:
                continue
            results = self.validation_step(batch)
            with open(os.path.join('expts', self.config['expt'], 'csv', f'test_{dataset_name}.csv'), 'a') as f:

                for idx, fn in enumerate(results['gt']):
                    gts.append(results['gt'][fn])
                    frame_gts.extend([results['gt'][fn]] * len(results['pred'][fn]))
                    preds.extend(results['pred'][fn])
                    probas.append(sum(results['proba'][fn]) / len(results['proba'][fn]))
                    if '0_real' in fn:
                        real_probs.append(sum(results['proba'][fn]) / len(results['proba'][fn]))
                    f.write(f'{fn},{results["gt"][fn]},{sum(results["proba"][fn]) / len(results["proba"][fn])}\n')
            count += 1
            if stop_count != -1:
                if count >= stop_count:
                    break
        # frame_gts = torch.tensor(frame_gts, dtype=torch.long)
        # preds = torch.tensor(preds)
        # loss = self.criterion(preds, frame_gts)
        # video_validation['loss'] = loss
        video_validation['metrics'] = {}
        if len(self.reals_probs) != 0:
            probas.extend(self.reals_probs)
            gts.extend([0] * len(self.reals_probs))
        for name, metric in self.metrics.items():
            value = metric(gts, probas)
            video_validation['metrics'][name] = value
        if len(self.reals_probs) == 0:
            self.reals_probs = real_probs
        return video_validation

    def train(self):
        train_loss = 0
        display_loss = 0
        log_loss = 0
        self.model.to(self.device)
        for idx, epoch in enumerate(range(self.epoch, self.config['epoch'])):
            self.epoch = epoch
            self.logger.info(f'Epoch {epoch + 1}/{self.config["epoch"]}')
            for batch in self.train_dataloader:
                self.model.train()
                self.optimizer.zero_grad()
                loss = self.training_step(batch)
                train_loss += loss.item()
                display_loss += loss.item()
                log_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                if self.step % self.config['display_step'] == 0:
                    self.logger.info(f'[Epoch {epoch + 1}, Step {self.step}] Training Loss: {display_loss / self.config["display_step"]}')
                    display_loss = 0
                if self.step % self.config['log_step'] == 0:
                    self.writer.add_scalar('loss/train_loss', log_loss / self.config['log_step'], global_step=self.step)
                    log_loss = 0
                if self.step % self.config['val_step'] == 0:
                    self.model.eval()
                    val_results = self.validation_video()
                    self.writer.add_scalar('loss/val_loss', val_results['loss'].item(), global_step=self.step)
                    self.logger.info(f'[Epoch {epoch + 1}, Step {self.step}] Val Loss: {val_results["loss"].item()}')
                    for metric, value in val_results['metrics'].items():
                        self.writer.add_scalar(f'val_metrics/{metric}', value, global_step=self.step)
                        self.logger.info(f'[Epoch {epoch + 1}, Step {self.step}] Val {metric}: {value}')
                    if self.scheduler is not None and isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_results['loss'].item())
                        self.logger.info(f'Patience: {self.scheduler.num_bad_epochs} / {self.config["patience"]}')
                    if val_results['loss'].item() < self.min_val_loss:
                        self.min_val_loss = val_results['loss'].item()
                        self.save_model('best')
                        self.save_model('latest')
                    if val_results['loss'].item() < self.config['val_bound']:
                        self.logger('Finished.')
                        self.save_model('latest')
            if self.epoch % self.config['save_epoch'] == 0:
                self.save_model(str(epoch + 1))
                self.save_model('latest')

    
    def save_model(self, tag='latest'):
        os.makedirs(os.path.join('expts', self.config['expt'], 'checkpoints'), exist_ok=True)
        model_state_dict = self.model.state_dict()
        for k in model_state_dict:
            if 'mm_encoder.model' in k:    # deprecate frozen parameters from LMM branch
                del(model_state_dict[k])
        state_dict = {
            'model_state_dict': model_state_dict,
            'optimizer': self.optimizer.state_dict() if self.optimizer is not None else None,
            'scheduler': self.scheduler,
            'epoch': self.epoch,
            'step': self.step,
        }
        torch.save(state_dict, os.path.join('expts', self.config['expt'], 'checkpoints', f'current_model_{tag}.pth'))
        
    def load_model(self, path, strict=True):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict['model_state_dict'], strict=strict)
        self.epoch = state_dict['epoch']
        self.step = state_dict['step']
        if self.optimizer is not None:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        if self.scheduler is not None:
            self.scheduler = state_dict['scheduler']
