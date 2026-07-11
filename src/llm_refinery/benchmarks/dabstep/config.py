from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_refinery.core.config import (
    ConfigError,
    coerce_command,
    coerce_list,
    load_yaml_mapping,
    reject_unknown_keys,
)
from llm_refinery.core.endpoints import OPENAI_CHAT, Endpoint

OFFICIAL_DABSTEP_DATASET_REPO = "adyen/DABstep"
# Last commit that changed the official task manifests as of 2026-07-10.
OFFICIAL_DABSTEP_DATASET_REVISION = "e68a4553c079601b09131851f4b7c6be9680d560"
_COMMIT_REVISION_RE = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class DabstepSettings:
    workspace: Path
    command: tuple[str, ...] = ("python3", "baseline/run.py")
    tasks_file: Path | None = None
    tasks_file_arg: str | None = None
    dataset_repo: str = OFFICIAL_DABSTEP_DATASET_REPO
    dataset_revision: str = OFFICIAL_DABSTEP_DATASET_REVISION
    split: str = "dev"
    task_ids: tuple[int, ...] = ()
    limit: int | None = 10
    concurrency: int = 1
    max_steps: int = 10
    timeout_s: float | None = 900.0
    retries: int = 0
    keep_going: bool = False
    model_id: str | None = None
    pass_env: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.command or any(not item.strip() for item in self.command):
            raise ConfigError("dabstep.command must contain non-empty arguments")
        if self.split not in {"dev", "default"}:
            raise ConfigError("dabstep.split must be either 'dev' or 'default'")
        if self.limit is not None and self.limit <= 0:
            raise ConfigError("dabstep.limit must be a positive integer or 'all'")
        if self.concurrency <= 0:
            raise ConfigError("dabstep.concurrency must be positive")
        if self.max_steps <= 0:
            raise ConfigError("dabstep.max_steps must be positive")
        if self.timeout_s is not None and self.timeout_s <= 0:
            raise ConfigError("dabstep.timeout_s must be positive or null")
        if self.retries < 0:
            raise ConfigError("dabstep.retries cannot be negative")
        if any(task_id < 0 for task_id in self.task_ids):
            raise ConfigError("dabstep.task_ids cannot contain negative values")
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ConfigError("dabstep.task_ids must be unique")
        dataset_repo = self.dataset_repo.strip()
        dataset_revision = self.dataset_revision.strip()
        if not dataset_repo or not dataset_revision:
            raise ConfigError("dabstep dataset_repo and dataset_revision cannot be empty")
        if dataset_repo != OFFICIAL_DABSTEP_DATASET_REPO:
            raise ConfigError(
                "the official DABStep baseline is hard-coded to dataset_repo "
                f"{OFFICIAL_DABSTEP_DATASET_REPO!r}"
            )
        tasks_file_arg = self.tasks_file_arg.strip() if self.tasks_file_arg is not None else None
        if self.tasks_file is not None and not tasks_file_arg:
            raise ConfigError(
                "dabstep.tasks_file requires tasks_file_arg for a compatible wrapper; "
                "the official baseline cannot consume a custom task manifest"
            )
        if self.tasks_file is None and tasks_file_arg:
            raise ConfigError("dabstep.tasks_file_arg requires tasks_file")
        if tasks_file_arg and not tasks_file_arg.startswith("-"):
            raise ConfigError("dabstep.tasks_file_arg must be a command-line flag")
        if self.tasks_file is None and not _COMMIT_REVISION_RE.fullmatch(dataset_revision):
            raise ConfigError(
                "dabstep.dataset_revision must be a pinned 40-character commit for the "
                "official baseline"
            )
        model_id = self.model_id.strip() if self.model_id is not None else None
        if self.model_id is not None and not model_id:
            raise ConfigError("dabstep.model_id cannot be empty")
        pass_env = tuple(name.strip() for name in self.pass_env)
        if any(not name for name in pass_env):
            raise ConfigError("dabstep.pass_env must contain non-empty variable names")
        if len(set(pass_env)) != len(pass_env):
            raise ConfigError("dabstep.pass_env variable names must be unique")
        object.__setattr__(self, "dataset_repo", dataset_repo)
        object.__setattr__(self, "dataset_revision", dataset_revision)
        object.__setattr__(self, "tasks_file_arg", tasks_file_arg)
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "pass_env", pass_env)

    @classmethod
    def from_mapping(
        cls,
        raw: dict[str, Any],
        *,
        source_path: Path | None,
    ) -> DabstepSettings:
        reject_unknown_keys(
            raw,
            {
                "workspace",
                "command",
                "tasks_file",
                "tasks_file_arg",
                "dataset_repo",
                "dataset_revision",
                "split",
                "task_ids",
                "limit",
                "concurrency",
                "max_steps",
                "timeout_s",
                "retries",
                "keep_going",
                "model_id",
                "pass_env",
            },
            context="dabstep settings",
        )
        workspace_value = raw.get("workspace")
        if not workspace_value:
            raise ConfigError("dabstep.workspace is required")
        base_dir = source_path.parent if source_path is not None else Path.cwd()
        workspace = _resolve_path(workspace_value, base_dir=base_dir)
        tasks_file = (
            _resolve_path(raw["tasks_file"], base_dir=base_dir) if raw.get("tasks_file") else None
        )
        task_ids = tuple(int(value) for value in coerce_list(raw.get("task_ids")))
        if "limit" in raw:
            limit_raw = raw["limit"]
            limit = (
                None
                if limit_raw is None or str(limit_raw).strip().lower() == "all"
                else int(limit_raw)
            )
        else:
            limit = None if task_ids else 10
        timeout_raw = raw.get("timeout_s", 900.0)
        timeout_s = None if timeout_raw is None else float(timeout_raw)
        command = tuple(coerce_command(raw.get("command", ["python3", "baseline/run.py"])))
        return cls(
            workspace=workspace,
            command=command,
            tasks_file=tasks_file,
            tasks_file_arg=str(raw["tasks_file_arg"]) if raw.get("tasks_file_arg") else None,
            dataset_repo=str(raw.get("dataset_repo") or OFFICIAL_DABSTEP_DATASET_REPO),
            dataset_revision=str(raw.get("dataset_revision") or OFFICIAL_DABSTEP_DATASET_REVISION),
            split=str(raw.get("split") or "dev").lower(),
            task_ids=task_ids,
            limit=limit,
            concurrency=int(raw.get("concurrency", 1)),
            max_steps=int(raw.get("max_steps", 10)),
            timeout_s=timeout_s,
            retries=int(raw.get("retries", 0)),
            keep_going=bool(raw.get("keep_going", False)),
            model_id=str(raw["model_id"]) if raw.get("model_id") else None,
            pass_env=tuple(str(value) for value in coerce_list(raw.get("pass_env"))),
        )

    def safe_json(self) -> dict[str, Any]:
        return {
            "workspace": str(self.workspace),
            "command": list(self.command),
            "tasks_file": str(self.tasks_file) if self.tasks_file else None,
            "tasks_file_arg": self.tasks_file_arg,
            "dataset_repo": self.dataset_repo,
            "dataset_revision": self.dataset_revision,
            "split": self.split,
            "task_ids": list(self.task_ids),
            "limit": self.limit,
            "concurrency": self.concurrency,
            "max_steps": self.max_steps,
            "timeout_s": self.timeout_s,
            "retries": self.retries,
            "keep_going": self.keep_going,
            "model_id": self.model_id,
            "pass_env": list(self.pass_env),
        }


@dataclass(frozen=True)
class DabstepConfig:
    name: str
    database: Path
    endpoint: Endpoint
    dabstep: DabstepSettings
    source_path: Path | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: dict[str, Any],
        source_path: Path | None = None,
    ) -> DabstepConfig:
        reject_unknown_keys(
            raw,
            {"name", "database", "endpoint", "dabstep"},
            context="DABStep configuration",
        )
        endpoint_raw = raw.get("endpoint")
        if not isinstance(endpoint_raw, dict):
            raise ConfigError("DABStep config requires an endpoint mapping")
        settings_raw = raw.get("dabstep")
        if not isinstance(settings_raw, dict):
            raise ConfigError("DABStep config requires a dabstep mapping")
        endpoint = Endpoint.from_mapping(
            endpoint_raw,
            context="DABStep endpoint",
            allowed_protocols=frozenset({OPENAI_CHAT}),
        )
        if endpoint.headers:
            raise ConfigError("DABStep's official baseline does not support endpoint headers")
        name = str(raw.get("name") or (source_path.stem if source_path else "dabstep"))
        if not name.strip():
            raise ConfigError("DABStep name cannot be empty")
        return cls(
            name=name.strip(),
            database=Path(str(raw.get("database") or "results/llm_refinery.duckdb")),
            endpoint=endpoint,
            dabstep=DabstepSettings.from_mapping(settings_raw, source_path=source_path),
            source_path=source_path,
        )

    def safe_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "database": str(self.database),
            "endpoint": self.endpoint.safe_json(),
            "dabstep": self.dabstep.safe_json(),
        }


def load_dabstep_config(path: str | Path) -> DabstepConfig:
    config_path, raw = load_yaml_mapping(path)
    return DabstepConfig.from_mapping(raw, source_path=config_path)


def _resolve_path(value: Any, *, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()
