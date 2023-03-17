import time
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from common import read_yaml, init_log
from datasets import CaptionTagDataset
from datasets import collate_fn
from models import CaptionTagger
from models.utils import model_size
from models.training import step_lr
from models.training import batch_weights
from tokenizer import load_tokenizer

from metrics import roc_auc, average_precision, batch_precision_recall_f1
from metrics.accumulator import Accumulator
from metrics.lwlrap import lwlrap_accumulator
from metrics import scores

from typing import Optional


#device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_epoch(model, loader, optimizer, scheduler=None, log_interval=None, p_token_mask: float = 0.0, mask_token: int = 4):
    model.train()
    batch_t0 = time.time()
    train_loss = 0.0
    scaler = torch.cuda.amp.GradScaler()

    for batch, (x, xm, y_true) in enumerate(loader):
        x = x.to(device)
        xm = xm.to(device)
        y_true = y_true.to(device)

        if p_token_mask > 0:
            idx = torch.rand(x.shape) < p_token_mask
            idx = idx.to(device)

            # do not mask special tokens -> start from 5
            x[idx & (x > 4)] = mask_token


        with torch.cuda.amp.autocast():
            weights = batch_weights(y_true)
            y_pred = model(x, mask=xm)
            loss = F.binary_cross_entropy_with_logits(y_pred, y_true, weight=weights)

        train_loss += loss.item()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        if log_interval is not None and batch % log_interval == 0:
            t_batch = (time.time() - batch_t0) * 1000 / log_interval
            lr_str = ''
            if scheduler is not None:
                current_lr = optimizer.param_groups[0]['lr']
                lr_str = f' current lr {current_lr:.4g} |'

            pre, rec, f1 = batch_precision_recall_f1(y_pred, y_true)
            auc = roc_auc(y_pred, y_true)

            print(f'batch {batch:5d}/{len(loader)} | {int(t_batch):4d} ms/batch |{lr_str} training loss {loss.item():.4f} | precision {pre:.4f}, recall {rec:.4f}, f1 {f1:.4f} | auc {auc:.4f}')

            batch_t0 = time.time()

    return train_loss / len(loader)


@torch.no_grad()
def validate_epoch(model, loader):
    model.eval()
    val_loss = 0.0
    lwlraps = lwlrap_accumulator()

    yp = []
    yt = []

    for x, xm, y_true in loader:
        x = x.to(device)
        xm = xm.to(device)
        y_true = y_true.to(device)

        weights = batch_weights(y_true)
        y_pred = model(x, mask=xm)
        loss = F.binary_cross_entropy_with_logits(y_pred, y_true, weight=weights)

        val_loss += loss.item()

        yp.append(y_pred.cpu().numpy())
        yt.append(y_true.cpu().numpy())

        lwlraps.accumulate_samples(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())

    yp = np.concatenate(yp, axis=0)
    yt = np.concatenate(yt, axis=0)

    ap = scores.average_precision(yt, yp)
    auc = scores.roc_auc(yt, yp)

    metrics = {
        'map': ap,
        'auc': auc,
        'd_prime': scores.d_prime(auc),
        'lwlrap': lwlraps.overall_lwlrap(),
    }

    return val_loss / len(loader), metrics


def main(config_fn='settings.yaml'):
    cfg = read_yaml(config_fn)
    logger = init_log('train', level=cfg.get('log_level', 'info'))

    cache_dir = Path(cfg.get('cache_dir', 'cache'))
    tokenizer = load_tokenizer(cfg.get('tokenizer_path', 'tokenizer'))

    logger.info('constructing datasets')
    ds_train = CaptionTagDataset(data_dir=cache_dir / 'train', tokenizer=tokenizer)
    ds_val = CaptionTagDataset(data_dir=cache_dir / 'val', tokenizer=tokenizer, labels=ds_train.labels)

    # store labels for evaluation
    labels_path = Path(cfg.get('labels_path', 'labels.txt'))
    labels_path.parent.mkdir(exist_ok=True, parents=True)
    labels_path.write_text('\n'.join(ds_train.labels))

    batch_size = cfg.get('batch_size')
    num_workers = cfg.get('num_dataloader_workers', 2)

    train_loader = torch.utils.data.DataLoader(
        dataset=ds_train, 
        shuffle=True, 
        batch_size=batch_size, 
        num_workers=num_workers,
        collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(
        dataset=ds_val, 
        batch_size=batch_size, 
        num_workers=num_workers,
        collate_fn=collate_fn)

    d_model = cfg.get('d_model')
    d_ff = cfg.get('d_ff')
    n_heads = cfg.get('n_heads')
    n_layers = cfg.get('n_layers')
    dropout = cfg.get('dropout', 0.0)

    model = CaptionTagger(
        n_labels=len(ds_train.labels),
        vocab_size=tokenizer.get_vocab_size(),
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout)

    logger.info(f'model size {model_size(model)/1e6:.1f}M')

    model = model.to(device)

    learning_rate = cfg.get('learning_rate', 1e-4)
    epochs = cfg.get('epochs', 10)
    warmup_steps = cfg.get('warmup_steps', 4000)
    max_patience = cfg.get('patience')
    patience = max_patience
    log_interval = cfg.get('log_interval')
    p_token_mask = cfg.get('p_token_mask', 0.0)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step_lr(step, warmup_steps=warmup_steps))

    model_path = Path(cfg.get('model_path', 'model.pt'))
    model_path.parent.mkdir(exist_ok=True, parents=True)

    best_map = 0.0

    logger.info(f'start training for {epochs} epochs')
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, log_interval=log_interval, p_token_mask=p_token_mask)
        val_loss, metrics = validate_epoch(model, val_loader)

        val_map = metrics['map']
        val_auc = metrics['auc']
        val_dprime = metrics['d_prime']
        val_lwlrap = metrics['lwlrap']

        logger.info(f'epoch {epoch + 1} train loss {train_loss:.4f}, validation loss {val_loss:.4f}, map {val_map:.4f}, lwlrap {val_lwlrap:.4f}, auc {val_auc:.4f}')
        
        if val_map > best_map:
            best_map = val_map
            patience = max_patience
            torch.save(model.state_dict(), model_path)

        elif patience is not None:
            patience -= 1
            if patience <= 0:
                logger.info('results not improving, stopping...')
                break

if __name__ == '__main__':
    main()
