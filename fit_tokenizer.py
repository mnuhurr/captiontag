
from pathlib import Path

from common import read_yaml, init_log
from tokenizer import fit_tokenizer_to_files


def main(config_fn='settings.yaml'):
    cfg = read_yaml(config_fn)

    cache_dir = Path(cfg.get('cache_dir'))
    text_fns = [str(cache_dir / 'training_captions.txt')]

    tokenizer_path = cfg.get('tokenizer_path', '.')
    vocab_size = cfg.get('vocab_size', 10000)
    min_frequency = cfg.get('min_frequency', 2)

    fit_tokenizer_to_files(text_fns, tokenizer_path, vocab_size=vocab_size, min_frequency=min_frequency)

if __name__ == '__main__':
    main()
