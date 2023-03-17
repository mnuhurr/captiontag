
from pathlib import Path
import json

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

from typing import Optional, Union, List


def load_tokenizer(tokenizer_path: Union[str, Path], max_length: Optional[int] = None) -> ByteLevelBPETokenizer:
    tokenizer_path = Path(tokenizer_path)
    vocab_fn = str(tokenizer_path / 'tokenizer-vocab.json')
    merges_fn = str(tokenizer_path / 'tokenizer-merges.txt')
    tokenizer = ByteLevelBPETokenizer(vocab_fn, merges_fn)
    tokenizer._tokenizer.post_processor = BertProcessing(
        ('</s>', tokenizer.token_to_id('</s>')),
        ('<s>', tokenizer.token_to_id('<s>')),
    )

    if max_length is not None:
        tokenizer.enable_truncation(max_length=max_length)

    return tokenizer


def fit_tokenizer(texts: List[str], tokenizer_path: Union[str, Path], vocab_size: int = 10000, min_frequency: int = 2, special_tokens: Optional[List[str]] = None):
    if special_tokens is None:
        special_tokens = ['<s>', '</s>', '<pad>', '<unk>', '<mask>']

    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train_from_iterator(texts, vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=special_tokens)

    # make sure that the directory exists
    Path(tokenizer_path).mkdir(exist_ok=True, parents=True)
    tokenizer.save_model(tokenizer_path, 'tokenizer')


def fit_tokenizer_to_files(files: List[Union[str, Path]], tokenizer_path: Union[str, Path], vocab_size: int = 10000, min_frequency: int = 2, special_tokens: Optional[List[str]] = None):
    if special_tokens is None:
        special_tokens = ['<s>', '</s>', '<pad>', '<unk>', '<mask>']

    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(files, vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=special_tokens)

    # make sure that the directory exists
    Path(tokenizer_path).mkdir(exist_ok=True, parents=True)
    tokenizer.save_model(tokenizer_path, 'tokenizer')


