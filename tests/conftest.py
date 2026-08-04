from __future__ import annotations

import os
from pathlib import Path

import pytest
from hypothesis import settings

# Two hypothesis profiles per ADR-0009: "fast" keeps pre-commit quick,
# "thorough" runs on pre-push and CI via HYPOTHESIS_PROFILE=thorough.
settings.register_profile("fast", max_examples=25, deadline=None)
settings.register_profile("thorough", max_examples=200, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "fast"))

# Typer renders CLI errors through rich, which force-enables ANSI color
# when it sees GITHUB_ACTIONS or FORCE_COLOR at import time — the
# styling even splits option names mid-token, breaking output
# assertions. Neutralize the test process env before typer imports so
# assertions see the same plain text everywhere.
os.environ["NO_COLOR"] = "1"
os.environ.pop("FORCE_COLOR", None)
os.environ.pop("GITHUB_ACTIONS", None)
# Rich wraps error panels at 80 columns, splitting messages that embed
# long tmp paths across lines and breaking substring assertions. Widen
# the panel so a message stays on one line regardless of path length.
os.environ["TERMINAL_WIDTH"] = "400"

# Enough distinct text to train a tiny byte-level BPE and to fill a few
# calibration batches for the torch-tier tests.
CALIBRATION_TEXT = (
    "The scan measures how much each layer group can be squeezed before "
    "the model's output drifts. Sensitive groups keep their bits and "
    "tolerant groups give theirs up, which is the entire trick. "
) * 40


# GPU cap that forces `auto` sharding to offload most of the
# offload-scale model while keeping its first layers on the card.
OFFLOAD_GPU_CAP = 120 * 2**20


@pytest.fixture(scope="session")
def offload_model_dir(tmp_path_factory) -> Path:
    """A synthetic Llama checkpoint big enough to engage accelerate dispatch.

    transformers collapses small models under a GPU cap to plain CPU
    tensors — no hooks, no meta parameters (verified 2026-07-28 on
    transformers 5.14). The offload path only exists at scale, so this
    checkpoint carries ~310 MB of bf16 weights: under
    ``OFFLOAD_GPU_CAP`` the overflow layers offload for real. Skips
    when the scan extra is not installed.
    """
    torch = pytest.importorskip("torch", reason="scan extra not installed")
    pytest.importorskip("transformers", reason="scan extra not installed")
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import (
        LlamaConfig,
        LlamaForCausalLM,
        PreTrainedTokenizerFast,
    )

    directory = tmp_path_factory.mktemp("offload-model")
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=384, special_tokens=["<unk>"])
    tokenizer.train_from_iterator([CALIBRATION_TEXT], trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>")
    fast.save_pretrained(directory)
    config = LlamaConfig(
        vocab_size=512,
        hidden_size=1024,
        intermediate_size=2816,
        num_hidden_layers=12,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=4096,
    )
    torch.manual_seed(0)
    LlamaForCausalLM(config).to(torch.bfloat16).save_pretrained(directory)
    return directory


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory) -> Path:
    """A tiny random Llama checkpoint with a trained tokenizer, built offline.

    Skips when the scan extra (torch, transformers) is not installed.
    """
    torch = pytest.importorskip("torch", reason="scan extra not installed")
    pytest.importorskip("transformers", reason="scan extra not installed")
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import (
        LlamaConfig,
        LlamaForCausalLM,
        PreTrainedTokenizerFast,
    )

    directory = tmp_path_factory.mktemp("tiny-model")
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=384, special_tokens=["<unk>"])
    tokenizer.train_from_iterator([CALIBRATION_TEXT], trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>")
    fast.save_pretrained(directory)
    config = LlamaConfig(
        vocab_size=512,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
    )
    torch.manual_seed(0)
    LlamaForCausalLM(config).save_pretrained(directory)
    return directory


@pytest.fixture(scope="session")
def aligned_model_dir(tmp_path_factory) -> Path:
    """A tiny Llama checkpoint whose rows divide the K-quant super-block.

    Assisted pricing refuses rows that straddle 256-element
    super-blocks (ADR-0020), so its meter tests need aligned
    dimensions — ``tiny_model_dir`` is deliberately misaligned.
    Skips when the scan extra is not installed.
    """
    torch = pytest.importorskip("torch", reason="scan extra not installed")
    pytest.importorskip("transformers", reason="scan extra not installed")
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import (
        LlamaConfig,
        LlamaForCausalLM,
        PreTrainedTokenizerFast,
    )

    directory = tmp_path_factory.mktemp("aligned-model")
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=384, special_tokens=["<unk>"])
    tokenizer.train_from_iterator([CALIBRATION_TEXT], trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>")
    fast.save_pretrained(directory)
    config = LlamaConfig(
        vocab_size=512,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
    )
    torch.manual_seed(0)
    LlamaForCausalLM(config).save_pretrained(directory)
    return directory
