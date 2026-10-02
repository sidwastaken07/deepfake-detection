# Working on IndiSpoof

Read `docs/SPEC.md` before doing any work. It is the build specification and it takes precedence
over convenience. The rules below apply to every session.

- Work strictly in phase order. Do not start phase N+1 until phase N's Definition of Done is met
  and recorded in `RESULTS.md` with a commit hash. Phases 5, 6 and 7 are hard gates; Phase 5 also
  needs explicit human sign-off.
- Ask the human before: any download over ~20 GB, any paid API call, deleting anything under
  `data/raw/`.
- Never invent numbers. Nothing goes into `RESULTS.md` or `paper/` unless a run in `results/`
  produced it. Empty cells stay empty.
- No audio is read from a directory scan. All data access goes through a manifest
  (`indispoof.data.manifest`). Manifests are immutable: corrections are new versions.
- The manifest schema (spec section 3) may gain fields, but existing fields must never be
  renamed or repurposed.
- Logic lives in `src/indispoof`; `scripts/` holds thin wrappers only. No hard-coded paths in src.
- Run `make check` (ruff + pytest) before every commit.
- Git commits are authored by the repo owner (`sidwastaken07 <sp.siddharth2006@gmail.com>`), with
  Claude as `Co-Authored-By`. Check `git config user.email` before committing.
