
from pathlib import Path

import torch
import torch.nn.functional as F

from train import validate_epoch

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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(config_fn='settings.yaml'):
    cfg = read_yaml(config_fn)
    logger = init_log('eval', level=cfg.get('log_level', 'info'))

    cache_dir = Path(cfg.get('cache_dir', 'cache'))
    tokenizer = load_tokenizer(cfg.get('tokenizer_path', 'tokenizer'))
    batch_size = cfg.get('batch_size', 32)
    num_workers = cfg.get('num_dataloader_workers', 2)

    labels = Path(cfg.get('labels_path', 'labels.txt')).read_text().splitlines()

    logger.info('constructing dataset')
    ds_eval = CaptionTagDataset(data_dir=cache_dir / 'eval', tokenizer=tokenizer, labels=labels)
    logger.info(f'{len(ds_eval)} caption/tag pairs to eval, {len(ds_eval.labels)} labels')

    loader = torch.utils.data.DataLoader(ds_eval, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)

    d_model = cfg.get('d_model')
    d_ff = cfg.get('d_ff')
    n_heads = cfg.get('n_heads')
    n_layers = cfg.get('n_layers')

    model = CaptionTagger(
        n_labels=len(labels),
        vocab_size=tokenizer.get_vocab_size(),
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        n_layers=n_layers)

    logger.info(f'model size {model_size(model)/1e6:.1f}M')
    model_path = cfg.get('model_path')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    eval_loss, metrics = validate_epoch(model, loader)

    for key, val in metrics.items():
        print(f'{key:15s}: {val:.4f}')

if __name__ == '__main__':
    main()
