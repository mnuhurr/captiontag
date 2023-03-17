
# Caption tagger
...

### Example configuration

```
audiocaps_dir: data/audiocaps
train_annotation_tsv_fn: 
  - GoogleAudioSetReformatted/audioset_weak_train_unbalanced.tsv
  - GoogleAudioSetReformatted/audioset_weak_train_unbalanced2.tsv

eval_annotation_tsv_fn:
  - GoogleAudioSetReformatted/audioset_weak_eval.tsv

cache_dir: data/cache
labels_path: data/labels.txt

vocab_size: 5000
tokenizer_path: data/tokenizer

batch_size: 256
num_dataloader_workers: 8

d_model: 512
n_heads: 8
n_layers: 4
p_token_mask: 0.25
dropout: 0.25

epochs: 200
patience:  5
learning_rate: 1.0e-4
warmup_steps: 1000
#log_interval: 100

model_path: data/model.pt
```
