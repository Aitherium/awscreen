# Changelog

All notable changes to `awscreen` are recorded here.

## [1.2.0] - 2026-11-20

### Added

- **awscreen**: `compare_states` / `compare_images` — a deterministic element-level
  diff of two screens: `added`, `removed`, `moved` and `unchanged`, with per-element
  pixel deltas when the frames are supplied. Matching is greedy IoU then a same-label /
  same-role identity pass, so a far jump reads as `moved`, not removal plus addition.
- **awscreen**: `paint` — a JSON paint manifest (`clear`, `rect`, `ellipse`, `line`,
  `polygon`, `pixels`, `text`) rendered to a PNG. Geometry ops are byte-stable, so the
  same manifest always paints the same pixels: the deterministic pixel-by-pixel primitive.
- **awscreen**: exact text, with provenance. `Element.text_source` names the reader and
  `Element.sources` carries `vlm-ocr`/`ocr` alongside the geometry sources. The
  crop-inspection reply already answers the crop's exact text, so every inspected
  element now stores it with **zero extra model calls**.
- **awscreen**: an `awscreen.ocr.OcrProvider` seam with `TesseractTsvProvider` (one frame
  pass, line boxes matched to the tightest containing region, so `blobs`/`grid` boxes
  still carry strings) and `NullOcrProvider`. A missing engine is a reported skip, never
  a guess; `state --no-ocr` disables the pass.
- **awscreen**: `awscreen state --text-only [--regions FILE]` answers just the exact
  strings for a region list, OCR first and one vision text call per region OCR could not
  read. Exit `0` text found, `1` a lane ran and found none, `2` no lane could run.
- **awscreen**: `awscreen._synthetic.draw_mixed_screen()` draws ten labelled controls
  with exact text truth for the offline tests; the state self-test now covers the OCR
  assignment, the graceful skip and the `vlm-ocr` path (12 checks).

### Changed

- **awscreen**: crop inspection now defaults to `bonsai2-27b-5090` on the 5090 through
  MicroScheduler (resolved once and pinned by `awvision.resolve_pinned`, as before).
  Measured 2026-11-20 on the same three UI crops: the 27B labelled roles correctly
  (text/button) where `bonsai-4b-5090` got them wrong, for 1-2 s more per crop
  (1.3-2.6 s vs 0.5-1.5 s). Override with `--model` on `awscreen state` or
  `AWSCREEN_VLM_MODEL`; a CLI flag beats the env, which beats awvision's own settings.
- **awscreen**: the 4B lane is the intended local OCR lane once an OCR model is
  installed. The OCR proposer already exists and skips honestly when no engine is
  present, so nothing routes to the 4B for vision inspection by default.

### Fixed

- **awscreen**: the OCR proposer no longer spawns tesseract without `CREATE_NO_WINDOW` on
  Windows (PQ040), and it uses AitherVision's tesseract TSV lane when the monorepo is on
  the path, so the fleet has one OCR implementation instead of two.
- **awscreen**: `state --screen` reports the captured monitor's real origin (via `mss`
  when installed) instead of a hardcoded `(0,0)`, so elements on a second monitor carry
  desktop pixels, not that monitor's pixels.
- **awscreen**: the awrun `screen` kind exits 2 when the vision endpoint answers none of
  the crops, the same "could not judge" contract `awscreen state` uses.

## [1.1.0] - 2026-11-19

### Added

- **awscreen**: `awscreen state` turns one screenshot into a structured `ScreenState`
  (absolute screen-pixel bbox, label, text, role, confidence and source per element)
  through frame -> proposer chain -> crop -> PIL neurons -> bounded parallel inspection
  -> IoU merge -> optional DOM join.
- **awscreen**: `awscreen.env:ScreenEnv` exposes a `ScreenState` to planners as an
  `awpredict` environment with the MCTS members (`get_state_hash`, `evaluate`, `clone`);
  `step()` acts only through an injected `act_fn` and `clone()` replays recorded
  transitions only.
- **awscreen**: an awrun `screen` kind (`python -m awscreen.awrun_kind`) queues one local
  crop-inspection batch as a durable job; a swarm or remote target is refused.
- **awscreen**: the gateway's `screen_state` tool wraps the pipeline; it accepts base64
  pixels, captures nothing and wakes no GPU.

### Fixed

- **awscreen**: bbox localization, crop transform and origin reuse were corrected after
  measurement on a real 2175x1360 screenshot (the save button within 1 px on every edge).
