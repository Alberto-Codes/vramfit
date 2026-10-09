"""Embed the checkpoint's chat template in the packed GGUF (#617).

``convert_hf_to_gguf.py`` embeds the chat template when it reads a
local checkpoint. A base GGUF converted another way can lack it, and
``llama-quantize`` copies the base's metadata as it finds it. Every
pack from such a base then lacks ``tokenizer.chat_template``, and
llama.cpp falls back to its ChatML template without a warning. The
pack step therefore reads the checkpoint's template and holds the
packed file against it after the quantizer exits 0.

The source is ``chat_template.jinja`` first, then the
``chat_template`` key of ``tokenizer_config.json``. A list there
takes the entry named ``default``, which is gguf-py's own rule.

The stamp rewrites the packed file through a temporary beside it.
gguf-py's reader cannot serve that rewrite, because it refuses the
``Q2_0`` tensors an expert-stack pack holds (ADR-0028). The owned
parse in [vramfit.adapters.outbound.gguf.header][] serves it. The
key-value bytes copy verbatim, one string pair follows them, and the
tensor infos follow that pair. Each info's offset is relative to the
data section, so no offset changes. The data section copies
verbatim after the alignment padding.

Examples:
    Embed the template a checkpoint ships:

    ```python
    source = read_source_chat_template(Path("model"))
    if source is not None:
        stamp_chat_template(Path("packed.gguf"), source.text)
    ```

See Also:
    - [vramfit.adapters.outbound.gguf.pack][]: The caller, right
      after the file-type relabel.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from vramfit.adapters.outbound.gguf.header import (
    CHAT_TEMPLATE_KEY as CHAT_TEMPLATE_KEY,  # noqa: PLC0414 - re-export: the stamp's key reads from this module
)
from vramfit.adapters.outbound.gguf.header import read_header, write_tensor_infos
from vramfit.adapters.outbound.gguf.types import PackError

JINJA_FILE: Final[str] = "chat_template.jinja"
TOKENIZER_CONFIG_FILE: Final[str] = "tokenizer_config.json"

# Magic, version, tensor count, key-value count (gguf.h, v2 and v3).
_PRELUDE: Final[struct.Struct] = struct.Struct("<4sIQQ")
_STRING_TYPE: Final[int] = 8
_COPY_CHUNK: Final[int] = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class SourceTemplate:
    """The chat template one checkpoint ships.

    Attributes:
        text (str): The template, verbatim.
        source (str): The checkpoint file it came from,
            ``chat_template.jinja`` or ``tokenizer_config.json``.

    Examples:
        Name where a template came from:

        ```python
        print(f"{source.source}: {len(source.text)} characters")
        ```
    """

    text: str
    source: str


def _config_template(value: object) -> str | None:
    """Pick the default template from a ``chat_template`` value.

    Args:
        value: The decoded ``chat_template`` value.

    Returns:
        The string itself, the ``default`` entry's template from a
        list, or None for any other shape.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        for entry in value:
            if isinstance(entry, dict) and entry.get("name") == "default":
                template = entry.get("template")
                return template if isinstance(template, str) else None
    return None


def read_source_chat_template(model_dir: Path) -> SourceTemplate | None:
    """Read the chat template a checkpoint ships.

    Args:
        model_dir: The checkpoint directory.

    Returns:
        The template and its source file, or None when the
        checkpoint ships none.

    Raises:
        PackError: If a source file exists but cannot be read, or
            ``tokenizer_config.json`` is not valid JSON.

    Examples:
        Read a checkpoint's template:

        ```python
        source = read_source_chat_template(Path("model"))
        ```
    """
    jinja = model_dir / JINJA_FILE
    config = model_dir / TOKENIZER_CONFIG_FILE
    path = jinja if jinja.is_file() else config
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            text = handle.read()
        if path == jinja:
            return SourceTemplate(text, JINJA_FILE)
        data = json.loads(text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackError(f"cannot read the chat template source {path}: {exc}") from exc
    if not isinstance(data, dict):
        return None
    template = _config_template(data.get("chat_template"))
    return None if template is None else SourceTemplate(template, TOKENIZER_CONFIG_FILE)


def _gguf_string(text: str) -> bytes:
    """Encode one GGUF string: a uint64 length, then UTF-8 bytes.

    Args:
        text: The string.

    Returns:
        The encoded bytes.
    """
    data = text.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _write_stamped(path: Path, temporary: Path, template: str) -> None:
    """Write ``path`` plus the template pair into ``temporary``.

    Args:
        path: The packed GGUF.
        temporary: The file to write.
        template: The chat template.

    Raises:
        PackError: If the header cannot be parsed or the verification
            of the written file fails.
        OSError: If a file cannot be read or written.
    """
    header = read_header(path)
    with path.open("rb") as source, temporary.open("xb") as out:
        magic, version, n_tensors, n_kv = _PRELUDE.unpack(source.read(_PRELUDE.size))
        out.write(_PRELUDE.pack(magic, version, n_tensors, n_kv + 1))
        out.write(source.read(header.infos_offset - _PRELUDE.size))
        out.write(_gguf_string(CHAT_TEMPLATE_KEY))
        out.write(struct.pack("<I", _STRING_TYPE))
        out.write(_gguf_string(template))
        write_tensor_infos(out, header.tensors)
        out.write(b"\0" * (-out.tell() % header.alignment))
        source.seek(header.data_start)
        shutil.copyfileobj(source, out, _COPY_CHUNK)
    stamped = read_header(temporary)
    data_bytes = path.stat().st_size - header.data_start
    if (
        stamped.tensors != header.tensors
        or stamped.chat_template != template
        or temporary.stat().st_size - stamped.data_start != data_bytes
    ):
        raise PackError(
            f"cannot verify the rewritten {temporary}: its tensors, data "
            f"section, or {CHAT_TEMPLATE_KEY} differ from what the stamp wrote"
        )


def stamp_chat_template(path: Path, template: str) -> None:
    """Add ``tokenizer.chat_template`` to a packed GGUF.

    The caller holds that the file declares no template yet. The
    rewrite goes through a temporary beside ``path``, and only a
    verified temporary replaces it.

    Args:
        path: The packed GGUF.
        template: The chat template to embed.

    Raises:
        PackError: If a file already holds the temporary's name, or
            the file cannot be read, written, or verified. The stamp
            removes only a temporary it created, and ``path`` is
            untouched.

    Examples:
        Stamp the checkpoint's template:

        ```python
        stamp_chat_template(Path("packed.gguf"), source.text)
        ```
    """
    temporary = path.with_name(path.stem + ".chat-template.gguf")
    if temporary.exists():
        raise PackError(
            f"cannot write {CHAT_TEMPLATE_KEY} into {path}: the temporary "
            f"{temporary} already exists, and the stamp would overwrite it"
        )
    try:
        _write_stamped(path, temporary, template)
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise PackError(f"cannot write {CHAT_TEMPLATE_KEY} into {path}: {exc}") from exc
    except PackError:
        temporary.unlink(missing_ok=True)
        raise


def embed_chat_template(model_dir: Path, path: Path) -> tuple[str | None, bool]:
    """Hold the packed file's chat template against the checkpoint's.

    A checkpoint with no template leaves the file as it is. A file
    with no template takes the checkpoint's. A file whose template
    differs refuses, because a silent drift is the defect #617 fixes.

    Args:
        model_dir: The checkpoint directory.
        path: The packed GGUF.

    Returns:
        The source file name or None, and whether the packed file
        carries ``tokenizer.chat_template`` afterwards.

    Raises:
        PackError: If a source cannot be read, the packed file's
            template differs from the checkpoint's, or the stamp
            fails.

    Examples:
        Record the outcome in the pack result:

        ```python
        source, embedded = embed_chat_template(Path("model"), out_path)
        ```
    """
    source = read_source_chat_template(model_dir)
    existing = read_header(path).chat_template
    if source is None:
        return None, existing is not None
    if existing is None:
        stamp_chat_template(path, source.text)
    elif existing != source.text:
        raise PackError(
            f"the packed file {path} carries a {CHAT_TEMPLATE_KEY} of "
            f"{len(existing)} characters, and the checkpoint's "
            f"{model_dir / source.source} holds a different one of "
            f"{len(source.text)} characters. The pack does not overwrite "
            "it. Check which template the base GGUF was converted with"
        )
    return source.source, True
