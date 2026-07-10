from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen

from llm_refinery.core.config import ConfigError

DEFAULT_MTP_HEAD_DIR = Path("~/.local/share/llm-refinery/mtp")


@dataclass(frozen=True)
class MtpHeadSpec:
    path: Path
    url: str | None = None


def resolve_mtp_head(value: Any) -> MtpHeadSpec:
    """Resolve a YAML mtp_head value to a local draft model path and optional URL."""
    if isinstance(value, str):
        return MtpHeadSpec(path=expand_user_path(value))

    if not isinstance(value, dict):
        raise ConfigError("mtp_head must be a path string or mapping")

    raw = {str(k): v for k, v in value.items()}
    url = str(raw["url"]) if raw.get("url") else None
    hf = str(raw["hf"]) if raw.get("hf") else None
    file = str(raw["file"]) if raw.get("file") else None
    revision = str(raw.get("revision") or "main")

    if url is None and hf and file:
        url = huggingface_resolve_url(hf, file, revision=revision)

    raw_path = raw.get("path")
    if raw_path:
        path = expand_user_path(str(raw_path))
    else:
        filename = mtp_head_filename(url=url, file=file)
        path = expand_user_path(DEFAULT_MTP_HEAD_DIR) / filename

    if url is None and not path.exists():
        raise ConfigError(
            "mtp_head path does not exist and no download source was provided; "
            "set mtp_head.url or mtp_head.hf + mtp_head.file"
        )

    return MtpHeadSpec(path=path, url=url)


def ensure_mtp_head(value: Any) -> MtpHeadSpec:
    spec = resolve_mtp_head(value)
    if spec.path.exists() and spec.path.stat().st_size > 0:
        return spec
    if spec.url is None:
        raise ConfigError(f"MTP head not found: {spec.path}")

    spec.path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = spec.path.with_name(f".{spec.path.name}.{uuid.uuid4().hex}.tmp")
    try:
        request = Request(spec.url, headers={"User-Agent": "llm-refinery"})
        with urlopen(request, timeout=60) as response, temp_path.open("wb") as output:
            shutil.copyfileobj(response, output)
        temp_path.replace(spec.path)
    finally:
        temp_path.unlink(missing_ok=True)
    return spec


def huggingface_resolve_url(repo: str, file: str, *, revision: str = "main") -> str:
    return (
        "https://huggingface.co/"
        f"{quote(repo, safe='/')}/resolve/{quote(revision, safe='')}/{quote(file, safe='/')}"
    )


def mtp_head_filename(*, url: str | None, file: str | None) -> str:
    if file:
        return Path(file).name
    if url:
        parsed = urlparse(url)
        filename = Path(unquote(parsed.path)).name
        if filename:
            return filename
    raise ConfigError("mtp_head requires path, file, or URL with a filename")


def expand_user_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(value))))
