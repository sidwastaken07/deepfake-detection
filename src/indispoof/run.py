"""Run context: one YAML config in, one append-only ``results/<run_id>.jsonl`` out.

Every run:
  1. resolves its YAML config (plus any ``--set key.path=value`` overrides),
  2. sets python, numpy and (if installed) torch seeds,
  3. creates ``results/<run_id>.jsonl`` exclusively, and writes a ``run_header`` line holding
     the resolved config, the git commit, the environment and any known non-determinism,
  4. appends one JSON line per logged record, and a final ``run_end`` line.

No metric exists unless it is in a results file.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import random
import re
import secrets
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
import yaml

RESULTS_SCHEMA_VERSION = 1
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")
HEADER_KEYS = (
    "type",
    "schema_version",
    "run_id",
    "created_utc",
    "seed",
    "config_path",
    "config_sha256",
    "config",
    "overrides",
    "git",
    "env",
    "determinism",
)


class ConfigError(ValueError):
    pass


class ResultsError(ValueError):
    pass


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def parse_override(spec: str) -> tuple[list[str], Any]:
    """``a.b.c=value`` -> (['a', 'b', 'c'], yaml-parsed value)."""
    if "=" not in spec:
        raise ConfigError(f"override {spec!r} must look like key.path=value")
    key, _, value = spec.partition("=")
    parts = key.strip().split(".")
    if not all(parts):
        raise ConfigError(f"override {spec!r} has an empty key component")
    return parts, yaml.safe_load(value)


def apply_overrides(config: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    for spec in overrides:
        parts, value = parse_override(spec)
        node = config
        for p in parts[:-1]:
            child = node.setdefault(p, {})
            if not isinstance(child, dict):
                raise ConfigError(f"override {spec!r}: {p!r} is not a mapping")
            node = child
        node[parts[-1]] = value
    return config


def load_config(
    path: str | os.PathLike[str], overrides: Iterable[str] = (), seed: int | None = None
) -> dict[str, Any]:
    """Load and resolve a run config. ``seed`` (e.g. from ``--seed``) wins over the file."""
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    config = apply_overrides(config, overrides)
    if seed is not None:
        config["seed"] = seed

    name = config.get("name")
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise ConfigError(f"{path}: 'name' must match {NAME_RE.pattern}")
    s = config.get("seed")
    if not isinstance(s, int) or isinstance(s, bool) or s < 0:
        raise ConfigError(f"{path}: 'seed' must be a non-negative integer")
    config.setdefault("results_dir", "results")
    try:
        json.dumps(config, allow_nan=False)
    except (TypeError, ValueError) as e:
        raise ConfigError(f"{path}: config is not JSON-serialisable ({e})") from e
    return config


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def git_state(cwd: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def seed_everything(seed: int) -> dict[str, Any]:
    """Seed every RNG we know about; return what was done and what remains non-deterministic."""
    random.seed(seed)
    # Legacy global RNG on purpose: third-party code (e.g. augmentation libs) still draws from it.
    np.random.seed(seed % 2**32)  # noqa: NPY002
    report: dict[str, Any] = {"python_random": True, "numpy_global": True, "notes": []}

    if os.environ.get("PYTHONHASHSEED") is None:
        report["notes"].append(
            "PYTHONHASHSEED unset at interpreter start; set/dict iteration order of str may vary"
        )

    try:
        import torch
    except ImportError:
        report["torch"] = None
        report["notes"].append("torch not installed; no torch seeding")
        return report

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    report["torch"] = {"version": torch.__version__, "deterministic_algorithms": "warn_only"}
    if torch.cuda.is_available() and not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
        report["notes"].append("CUBLAS_WORKSPACE_CONFIG unset; some cuBLAS ops non-deterministic")
    return report


def environment() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "argv": sys.argv,
    }


# ---------------------------------------------------------------------------
# Run context
# ---------------------------------------------------------------------------


class RunContext:
    """Owns one run's results file. Use as a context manager."""

    def __init__(
        self,
        config_path: str | os.PathLike[str],
        overrides: Iterable[str] = (),
        seed: int | None = None,
    ) -> None:
        overrides = list(overrides)
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path, overrides, seed)
        self.seed: int = self.config["seed"]
        stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        # The suffix comes from the OS, not a seeded RNG, so reruns never collide.
        self.run_id = f"{self.config['name']}_{stamp}_s{self.seed}_{secrets.token_hex(2)}"
        self.results_dir = Path(self.config["results_dir"])
        self.results_path = self.results_dir / f"{self.run_id}.jsonl"

        determinism = seed_everything(self.seed)
        self.rng = np.random.default_rng(self.seed)

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.results_path, "x", encoding="utf-8", newline="\n")
        self._closed = False
        header = {
            "type": "run_header",
            "schema_version": RESULTS_SCHEMA_VERSION,
            "run_id": self.run_id,
            "created_utc": _utcnow(),
            "seed": self.seed,
            "config_path": str(self.config_path),
            "config_sha256": hashlib.sha256(self.config_path.read_bytes()).hexdigest(),
            "config": self.config,
            "overrides": overrides,
            "git": git_state(),
            "env": environment(),
            "determinism": determinism,
        }
        self._write(header)
        print(f"[{self.run_id}] started -> {self.results_path}")
        if header["git"]["dirty"]:
            print(f"[{self.run_id}] warning: working tree has uncommitted changes")
        for note in determinism["notes"]:
            print(f"[{self.run_id}] non-determinism: {note}")

    def _write(self, record: dict[str, Any]) -> None:
        if self._closed:
            raise ResultsError(f"{self.results_path} is closed")
        self._fh.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self._fh.flush()

    def log(self, type: str, **payload: Any) -> None:
        """Append one record. ``type`` names the record kind (e.g. 'metrics')."""
        if type in ("run_header", "run_end"):
            raise ResultsError(f"record type {type!r} is reserved")
        self._write({"type": type, "run_id": self.run_id, "t_utc": _utcnow(), **payload})
        summary = ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in payload.items()
            if isinstance(v, (int, float, str)) and not isinstance(v, bool)
        )
        print(f"[{self.run_id}] {type}: {summary}")

    def close(self, status: str = "ok", error: str | None = None) -> None:
        if self._closed:
            return
        self._write(
            {"type": "run_end", "run_id": self.run_id, "t_utc": _utcnow(), "status": status}
            | ({"error": error} if error else {})
        )
        self._fh.close()
        self._closed = True
        print(f"[{self.run_id}] finished: {status}")

    def __enter__(self) -> RunContext:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc is None:
            self.close("ok")
        else:
            self.close("failed", f"{exc_type.__name__}: {exc}" if exc_type else str(exc))


def read_results(path: str | os.PathLike[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read and structurally validate a results file. Returns (header, records)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ResultsError(f"{path}: empty results file")
    try:
        parsed = [json.loads(line) for line in lines]
    except json.JSONDecodeError as e:
        raise ResultsError(f"{path}: invalid JSON line ({e.msg})") from e

    header, records = parsed[0], parsed[1:]
    if not isinstance(header, dict) or header.get("type") != "run_header":
        raise ResultsError(f"{path}: first line must be a run_header")
    missing = [k for k in HEADER_KEYS if k not in header]
    if missing:
        raise ResultsError(f"{path}: run_header missing {missing}")
    for i, r in enumerate(records, start=2):
        if not isinstance(r, dict) or "type" not in r:
            raise ResultsError(f"{path}:{i}: record without a type")
        if r.get("run_id") != header["run_id"]:
            raise ResultsError(f"{path}:{i}: run_id does not match header")
        if r["type"] == "run_header":
            raise ResultsError(f"{path}:{i}: second run_header")
    return header, records
