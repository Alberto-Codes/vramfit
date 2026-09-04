# Nemotron 3.5 Lightning 30B-A3B config

`config.json` is the unmodified model config of
[nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16),
fetched 2026-09-04 from the local Hugging Face cache (snapshot
`ce38b6ab8b252b4b8ee7165b4605e93191cafd73`). It is the second target's
real geometry: a hybrid `layers_block_type` stack of 52 layers, 23
`mamba`, 23 `moe`, and 6 `attention`, with 2 KV heads and head_dim 128.

The worked-example unit test (`tests/unit/test_worked_example.py`)
checks that the budget reader prices the 6 attention layers alone
(#427). Re-fetch only if NVIDIA revises the checkpoint, and update the
test in the same change.
