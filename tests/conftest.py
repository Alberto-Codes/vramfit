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

# Enough distinct text to train a tiny byte-level BPE and to fill a few
# calibration batches for the torch-tier tests.
CALIBRATION_TEXT = (
    "The scan measures how much each layer group can be squeezed before "
    "the model's output drifts. Sensitive groups keep their bits and "
    "tolerant groups give theirs up, which is the entire trick. "
) * 40


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
