# awscreen for agents

Read this if you are an agent (or a human) editing this package. Short on
purpose: the commands, the traps that cost a session, and where the rest lives.
Nothing here is read at runtime — it is for you.

## What this is

PyPI distribution **`awscreen`** (version in `pyproject.toml`), import package
`awscreen`, Python >= 3.10. See this machine — what is on screen, and where to
click it. A screenshot in, a clickable element out.

This repository is a **synced mirror** of the AitherOS monorepo (lane
`.github/workflows/sync-awscreen.yml`). Hand edits made here are overwritten on
the next sync — change the source and let the lane publish.

## Build, test, verify

```bash
python -m pytest tests -q        # the suite: 37 tests, green at v1.0.0
pip install -e .                 # editable install for developing against it
```

The suite was run from a source checkout with no prior install. The publish
lane (`publish-brick.yml`) additionally builds the wheel, installs it and
imports it — a tree that tests green can still ship a broken wheel.

## Rules that keep this useful

- **The finder is the product.** `test_finder.py` pins plain-language →
  element resolution; `test_awscreen.py` pins capture. A finder that returns
  a plausible-looking point when it is not sure is the defect that makes a GUI
  agent dangerous — coordinates carry confidence, or they carry an error.
- **Wrong click vs no click.** These are different outcomes and must be
  different return values; the caller decides what to do with each, and it
  cannot decide anything if the tool smoothed them into "done".
- **Screen content is user content.** Never log pixels, window titles or
  typed text into diagnostics; describe the shape of what was found, not what
  it said.
- **The registry drives the public surface.** This repo's README header,
  `llms.txt` and `aither-manifest.json` are generated from the ecosystem
  registry (one yaml in the AitherOS monorepo) and rewritten on every sync.
  Change the registry; do not hand-edit the generated blocks.

## Read next

- `llms.txt` — the install/use card written for an agent to execute
- `README.md` — the human front door
- `docs/` — the generated docs site source
