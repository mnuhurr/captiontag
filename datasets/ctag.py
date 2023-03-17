
from pathlib import Path

import torch

from typing import Any, List, Optional, Union, Tuple


def collate_fn(batch):
    tokens = []
    tags = []

    for item in batch:
        tokens.append(item[0])
        tags.append(item[1])

    batch_size = len(batch)
    maxlen = max(map(len, tokens))

    x = torch.zeros(batch_size, maxlen, dtype=torch.int64)
    mask = torch.ones(batch_size, maxlen)

    for k, tok in enumerate(tokens):
        x[k, :len(tok)] = tok
        mask[k, :len(tok)] = 0

    return x, mask, torch.stack(tags)


class CaptionTagDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: Union[str, Path], tokenizer: Any, labels: Optional[List[str]] = None):

        self.labels = labels if labels is not None else self.collect_labels(data_dir)
        self.label_ind = {label: k for k, label in enumerate(self.labels)}

        # read captions
        self.data = []

        for fn in Path(data_dir).glob('*.pt'):
            x = torch.load(fn)

            # k-hot labels
            v = self.get_label_vec(x['labels'])

            for caption in x['captions']:
                tokens = torch.tensor(tokenizer.encode(caption).ids, dtype=torch.int64)
                self.data.append((tokens, v))
                
    @staticmethod
    def collect_labels(data_dir: Union[str, Path]) -> List[str]:
        labels = set()
        for fn in Path(data_dir).glob('*.pt'):
            x = torch.load(fn)
            for label in x['labels']:
                labels.add(label)

        return sorted(labels)

    def get_label_vec(self, labels: List[str]) -> torch.Tensor:
        v = torch.zeros(len(self.labels))

        for label in labels:
            if label not in self.label_ind:
                continue

            v[self.label_ind[label]] = 1

        return v

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[item]

