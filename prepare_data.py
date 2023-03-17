
from pathlib import Path
from tqdm import tqdm

import csv
import torch

from common import read_yaml, init_log

from typing import Dict, List, Optional, Tuple, Union


def read_annotations(tsv_fns: List[Union[str, Path]], id_len: int = 11, ids: Optional[List[str]] = None):
    annotation = {}
    labels = set()

    for tsv_fn in tsv_fns:
        with Path(tsv_fn).open('rt') as tsv_file:
            reader = csv.DictReader(tsv_file, delimiter='\t')

            for row in reader:
                fn_id = row['filename'][:id_len]
                label = row['event_label']

                if ids is not None and fn_id not in ids:
                    continue

                labels.add(label)

                if fn_id not in annotation:
                    annotation[fn_id] = [label]
                else:
                    annotation[fn_id].append(label)

    return annotation, sorted(labels)


def read_captions(csv_fn: Union[str, Path]):
    captions = {}

    with Path(csv_fn).open('rt') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=',')

        for row in reader:
            yt_id = row['youtube_id']
            caption = row['caption']

            if yt_id not in captions:
                captions[yt_id] = [caption]
            else:
                captions[yt_id].append(caption)

    return captions


def prepare_data(captions, annotation, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    for yt_id in tqdm(captions):
        data = {
            'id': yt_id,
            'captions': captions[yt_id],
            'labels': annotation[yt_id]
        }

        torch.save(data, output_dir / f'{yt_id}.pt')


def main(config_fn='settings.yaml'):
    cfg = read_yaml(config_fn)
    logger = init_log('prepare-data', level=cfg.get('log_level', 'info'))

    annotation_fns = cfg.get('train_annotation_tsv_fn')
    if isinstance(annotation_fns, str):
        annotation_fns = [annotation_fns]

    logger.info('reading annotations')
    annotation, labels = read_annotations(annotation_fns)
    logger.info(f'got {len(labels)} labels for {len(annotation)} files')

    cache_dir = Path(cfg.get('cache_dir', 'cache'))
    cache_dir.mkdir(exist_ok=True, parents=True)

    # 1) make train split
    audiocaps_dir = Path(cfg.get('audiocaps_dir'))
    caption_tsv_fn = audiocaps_dir / 'dataset' / 'train.csv'
    
    logger.info('reading captions')
    captions = read_captions(caption_tsv_fn)
    logger.info(f'[train] got captions for {len(captions)} files')

    prepare_data(captions, annotation, cache_dir / 'train')

    # collect all the training captions in case we are fitting a tokenizer
    training_captions = []
    for caption_list in captions.values():
        training_captions.extend(caption_list)

    (cache_dir / 'training_captions.txt').write_text('\n'.join(training_captions))
    
    # 2) make validation
    caption_tsv_fn = audiocaps_dir / 'dataset' / 'val.csv'
    captions = read_captions(caption_tsv_fn)
    logger.info(f'[val] got captions for {len(captions)} files')

    prepare_data(captions, annotation, cache_dir / 'val')

    # 3) make eval
    annotation_fns = cfg.get('eval_annotation_tsv_fn')
    if isinstance(annotation_fns, str):
        annotation_fns = [annotation_fns]
    
    #logger.info('reading annotations')
    #annotation, labels = read_annotations(annotation_fns)
    #logger.info(f'got {len(labels)} labels for {len(annotation)} files')

    caption_tsv_fn = audiocaps_dir / 'dataset' / 'test.csv'
    captions = read_captions(caption_tsv_fn)
    logger.info(f'[eval] got captions for {len(captions)} files')

    prepare_data(captions, annotation, cache_dir / 'eval')

if __name__ == '__main__':
    main()
