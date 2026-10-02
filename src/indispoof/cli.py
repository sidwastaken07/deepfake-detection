"""``indispoof`` command line. Thin dispatch only; logic lives in the package modules.

Subcommands mirror the CLI surface in docs/SPEC.md. Commands for phases not yet built
exit with status 2 so that no script can mistake a stub for a successful step.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from indispoof import __version__

NOT_IMPLEMENTED = 2

# command -> (phase that implements it, help text)
STUBS: dict[str, tuple[int, str]] = {
    "plan-text": (2, "sample the synthesis text plan"),
    "generate": (3, "synthesise spoof audio with one generator"),
    "qc": (3, "run the ASR quality gate over a spoof manifest"),
    "normalize": (4, "normalise audio to 16 kHz mono 16-bit"),
    "controls": (5, "run the confound probes"),
    "protocol": (6, "build and freeze the split manifests"),
    "train": (7, "train a detector"),
    "eval": (7, "score an evaluation manifest with a checkpoint"),
    "codec": (8, "apply a codec/channel condition to an evaluation manifest"),
    "report": (9, "generate LaTeX tables and figures from results/"),
}


def _add_run_args(p: argparse.ArgumentParser, config_required: bool) -> None:
    p.add_argument("--config", required=config_required, help="YAML run config")
    p.add_argument("--seed", type=int, help="override the config seed")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value (dotted key), repeatable",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="indispoof", description="IndiSpoof benchmark pipeline")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def stub(name: str, add_args: Callable[[argparse.ArgumentParser], None]) -> None:
        phase, help_text = STUBS[name]
        p = sub.add_parser(name, help=f"{help_text} [Phase {phase}, not implemented]")
        add_args(p)
        p.set_defaults(handler=_not_implemented, phase=phase)

    ingest = sub.add_parser("ingest", help="ingest a bonafide corpus into a manifest [Phase 1]")
    _add_run_args(ingest, True)
    ingest.add_argument("--corpus", help="corpus key from the config")
    ingest.add_argument("--lang", choices=["ta", "hi"])
    mode = ingest.add_mutually_exclusive_group()
    mode.add_argument("--merge", action="store_true", help="merge part manifests, run checks")
    mode.add_argument("--download", action="store_true", help="fetch shards from Hugging Face")
    mode.add_argument("--inspect", action="store_true", help="print schema of the first shard")
    ingest.add_argument("--dry-run", action="store_true", help="with --download: list only")
    ingest.add_argument("--allow-large", action="store_true", help="with --download: exceed max_gb")
    ingest.set_defaults(handler=_ingest)
    stub(
        "plan-text",
        lambda p: (
            p.add_argument("--lang"),
            p.add_argument("--n", type=int),
            p.add_argument("--seed", type=int),
        ),
    )
    stub(
        "generate",
        lambda p: (
            p.add_argument("--generator"),
            p.add_argument("--lang"),
            p.add_argument("--resume", action="store_true"),
        ),
    )
    stub("qc", lambda p: p.add_argument("--manifest"))
    stub("normalize", lambda p: p.add_argument("--manifest"))
    stub(
        "controls",
        lambda p: (
            p.add_argument("--manifest"),
            p.add_argument("--probe", choices=["C1", "C2", "C3", "C4", "all"]),
        ),
    )
    stub(
        "protocol",
        lambda p: (_add_run_args(p, False), p.add_argument("--freeze", action="store_true")),
    )
    stub("train", lambda p: _add_run_args(p, False))
    stub("eval", lambda p: (p.add_argument("--ckpt"), p.add_argument("--protocol")))
    stub("codec", lambda p: (p.add_argument("--manifest"), p.add_argument("--condition")))
    stub("report", lambda p: p.add_argument("--out"))

    smoke = sub.add_parser("smoke", help="Phase 0 harness check: random scores -> results file")
    _add_run_args(smoke, True)
    smoke.set_defaults(handler=_smoke)
    return parser


def _not_implemented(args: argparse.Namespace) -> int:
    print(
        f"indispoof {args.command}: not implemented yet (Phase {args.phase}).",
        file=sys.stderr,
    )
    return NOT_IMPLEMENTED


def _ingest(args: argparse.Namespace) -> int:
    from pathlib import Path

    from indispoof.data import ingest
    from indispoof.run import RunContext, load_config

    if args.merge:
        with RunContext(args.config, args.overrides, args.seed) as ctx:
            ingest.merge_parts(ctx)
        return 0
    if not (args.corpus and args.lang):
        print("indispoof ingest: --corpus and --lang are required", file=sys.stderr)
        return 1

    if args.download or args.inspect:
        config = load_config(args.config, args.overrides, args.seed)
        ccfg, lcfg = ingest.corpus_config(config, args.corpus, args.lang)
        if args.inspect:
            files = sorted(Path(lcfg["root"]).glob(lcfg.get("files", "**/*.parquet")))
            if not files:
                print(f"no local files match {lcfg['root']}/{lcfg.get('files')}", file=sys.stderr)
                return 1
            print(ingest.inspect_parquet(files[0]))
            return 0
        from indispoof.data.fetch import fetch

        hf = lcfg.get("hf")
        if not hf:
            print(
                f"{args.corpus}/{args.lang} has no 'hf' source; see docs/kaggle.md", file=sys.stderr
            )
            return 1
        fetch(
            hf["repo_id"],
            hf["pattern"],
            int(hf["n_files"]),
            Path(lcfg["root"]),
            seed=config["seed"],
            max_gb=float(hf.get("max_gb", 20)),
            allow_large=args.allow_large,
            dry_run=args.dry_run,
        )
        return 0

    with RunContext(args.config, args.overrides, args.seed) as ctx:
        ingest.ingest_corpus(ctx, args.corpus, args.lang)
    return 0


def _smoke(args: argparse.Namespace) -> int:
    from indispoof.run import RunContext
    from indispoof.smoke import run_smoke

    with RunContext(args.config, args.overrides, args.seed) as ctx:
        run_smoke(ctx)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
