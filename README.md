# awscreen

<!-- aither-header:start GENERATED from the ecosystem registry. Edits here are overwritten; change the registry instead. -->

**[Docs](https://aitherium.github.io/awscreen/)**  ·  [Source](https://github.com/Aitherium/awscreen)  ·  `pip install awscreen`  ·  [The Aither World](https://aitherium.github.io/)

> **The Aither World** is an operating system for agents — a Linux you can hand to one, the runtimes it works in, and the tools it works with. [awnix](https://github.com/Aitherium/awnix) is the Linux underneath it; **awscreen** is one of its 72 bricks — each installs on its own, runs offline, and needs no account. 6 more are in development and 23 are planned.
>
> **Start here:** Find the thing you want to click by describing what it looks like.

<!-- aither-header:end -->

Find clickable elements by describing what they look like.

Part of the [Aither World (`aw*`)](https://aitherium.github.io/) family of CLI tools.

## Install

```bash
pip install awscreen
```

## Usage

Find elements in a screenshot by natural language description:

```bash
# Point at your local vision model
export AWSCREEN_URL=http://localhost:8150   # a LOCAL vision model
export AWSCREEN_MODEL=your-vision-model

# Find the save button in a screenshot
awscreen screenshot.png "the blue save button in the top toolbar"
```

Or use in Python:

```python
from awscreen import Finder

finder = Finder()  # reads AWSCREEN_URL / AWVISION_URL
screenshot = finder.load_image("screenshot.png")
results = finder.find(screenshot, "the blue save button")

for match in results:
    print(f"Found at ({match.x}, {match.y}): {match.description}")
```

## Screen state: everything on screen, and where

`awscreen state` turns one screenshot into one structured record: every element with
its absolute pixel box, label, text, role, confidence and the sources that saw it.

```bash
awscreen state shot.png --json                 # propose -> crop -> inspect in parallel -> merge
awscreen state shot.png --proposer blobs --no-inspect   # pixels only, zero model calls
awscreen state --screen --out last.json        # capture the primary monitor
awscreen state --screen --monitor 2 --out right.json  # capture monitor 2, real origin
awscreen state right.png --monitor 2 --json    # verify the file IS monitor 2's pixels
awscreen state shot.png --prev last.json       # unchanged frame: reuse, zero model calls
awscreen state shot.png --dom dom.json         # join browser DOM rects onto the pixels
awscreen state shot.png --budget-ms 5000       # deadline: calls stop when a whole crop no longer fits
awscreen state --screen --uia                  # Windows: join the focused window's UIA tree
awscreen state shot.png --uia uia.json         # join a recorded UIA tree (any platform)
awscreen state shot.png --text-only --regions regions.json   # just the exact strings
awscreen state --self-test                     # offline; never dials
```

**Inspection model.** Crops are inspected by `bonsai2-27b-5090` through MicroScheduler,
resolved once for the whole frame and pinned. On the same three UI crops the 27B labelled
roles correctly (text/button) where the 4B got them wrong, for 1-2 s more per crop. The
`--model` flag beats `AWSCREEN_VLM_MODEL`, which beats awvision's own `AWVISION_MODEL` /
host config. The `bonsai-4b-5090` lane is the intended OCR-model lane once one is
installed. Exact strings do not depend on it: a local tesseract pass reads them, and the
crop-inspection call returns them too (see **Exact text** below).

**Exact text.** Two readers fill `element.text`, and `element.text_source` says which:

- the **OCR provider** (`awscreen.ocr.TesseractTsvProvider`) runs once over the whole
  frame and its line boxes are matched to the tightest containing region, so boxes from
  `blobs`/`grid` still carry readable strings. No engine installed is a reported skip,
  never a guess. `--no-ocr` turns it off.
- the **crop inspection** call already reads the text: every crop the model inspects
  returns its exact string in the same reply, stored with `text_source: "vlm-ocr"` and
  tag `vlm-ocr` in `sources`. That costs **zero extra model calls**. `--text-only` skips
  roles and geometry and answers just the strings for a region list (`--regions`), OCR
  first and one vision call per region OCR could not read.

How it sees, fast and cheap:

1. **Propose** regions. The chain is `omniparser` (only when its `/health` answers in 2 s
   with models already loaded; a cold unit is never woken), then `ocr` (local tesseract
   TSV line boxes), then `blobs` (pure Pillow connected components), then `grid`. Every
   attempt is reported in `proposer.tried` with its reason and time.
2. **Crop** each region with a margin, downscaled to 448 px at most. A `CropTransform`
   maps any crop pixel back to image and screen pixels exactly (`--origin` places a
   monitor that does not start at 0,0).

   `--monitor N` (1-based, `mss` order) makes that origin automatic and checked: with
   `--screen` it captures that monitor with `mss` and verifies the grabbed size; with an
   image file it verifies the file's dimensions EQUAL that monitor's, then takes its real
   desktop origin. A frame that does not match the claimed monitor exits 2 — a Pillow
   grab of the primary display labelled "monitor 2" would put every click 1920 px off,
   and silence is the bug. Without `mss`, monitor 1 still falls back to Pillow; any other
   monitor is refused instead of being silently replaced by the primary.
3. **Neurons** (dominant colour, edge density, ink ratio, luminance, template match)
   run on every crop. Blank or already-decided crops never reach a model.
4. **Read text** with one OCR pass over the frame; each line goes to the tightest
   containing region and lands in `element.text` with `text_source: "tesseract-tsv"`.
   An absent engine is reported in `model.ocr` and skipped.
5. **Inspect** the rest in parallel with a small vision model: **at most 4 in flight**,
   whatever `--concurrency` says, a per-crop timeout, and at most `--max-regions` calls
   (default 24). Its reply also carries the exact text, so those elements get
   `text_source: "vlm-ocr"` from the same call. A crop that times out or fails becomes an
   element with `status: timeout|error`, never dropped and never guessed. With
   `--budget-ms` (or `AWSCREEN_BUDGET_MS`) the call cap is shrunk to the whole crops the
   remaining time covers; the clock is re-read after every wave, and every crop it left
   uninspected is returned with `status: budget` — still present, with no model answer
   invented.
6. **Merge** by IoU. Geometry goes to the stronger source
   (`uia > dom > omniparser > vlm > ocr > cv > grid`); text is unioned, the reader is kept
   in `text_source`, and `sources` lists every contributor.
7. **Join UI Automation** (`--uia`). On Windows the focused window's UIA tree maps real
   control types (a `MenuItem` stays `menuitem`) and enabled state onto the merged
   elements by IoU, and adds nodes the pixels missed. Off Windows, or when no bridge is
   present, it is skipped with a printed reason — no element is invented.

The vision endpoint and model are resolved once per process by awvision's pinned
resolver (your `AWVISION_URL` or config endpoint, else the local scheduler; never a
plain-http fallback). **Crops never leave the host** except to that endpoint, and
`--executor awrun` queues one local `screen` item; a swarm or remote runner is refused.

Exit codes: `0` at least one element, `1` none, `2` could not judge (unreadable
image, a requested proposer unavailable, the vision endpoint answered no crop, or a
frame that is not the monitor it was claimed to be).

### ScreenState (`awscreen.screen_state/1`)

```json
{"schema": "awscreen.screen_state/1",
 "frame": {"path": "shot.png", "width": 1280, "height": 800, "origin": [0, 0]},
 "proposer": {"ran": "blobs", "tried": [{"name": "omniparser", "ok": false, "reason": "...", "ms": 2001}]},
 "elements": [{"id": "e0", "bbox": [100, 120, 260, 168], "label": "Save", "text": "SAVE",
               "text_source": "vlm-ocr", "role": "button", "confidence": 0.7,
               "source": "vlm", "sources": ["vlm", "vlm-ocr", "cv"],
               "interactable": true, "status": "ok"}],
 "timing": {"propose_ms": 27, "ocr_ms": 90, "inspect_ms": 6900, "uia_join_ms": 12, "total_ms": 7100},
 "model": {"endpoint_host": "127.0.0.1", "model": "...", "calls": 4, "ok": 4,
           "ocr": {"provider": "tesseract-tsv", "ok": true, "lines": 12, "reason": "..."}},
 "uia": {"attempted": true, "ok": true, "source": "powershell", "nodes": 34,
         "matched": 3, "added": 2, "skipped": 5}}
```

### DOM join input

```json
{"viewport": {"dpr": 2, "vw": 1280, "vh": 800, "sx": 0, "sy": 0},
 "origin_px": [0, 0], "image_scale": "fit",
 "elements": [{"ref": "#save", "tag": "button", "label": "Save", "rect": [100, 120, 160, 48]}]}
```

`rect` is CSS viewport px (`page: true` for page coordinates, scroll subtracted). Scale
is `image_scale`, else `image_width / vw` for `"fit"`, else `dpr`.
`awscreen.dom.AITHERBROWSER_RECT_JS` returns exactly this shape from a page eval.

### UI Automation join input (Windows)

```json
{"schema": "awscreen.uia/1", "source": "powershell", "image_scale": 1.0,
 "window": {"title": "Untitled - Notepad", "process_id": 4242, "class": "Notepad"},
 "nodes": [{"name": "Save", "control_type": "MenuItem", "rect": [100, 120, 160, 48],
            "is_enabled": true, "automation_id": "SaveItem", "depth": 2}]}
```

`rect` is `[x, y, w, h]` in desktop screen px (UIA's `BoundingRectangle`); `image_scale`
maps screen px to image px when the screenshot was resized. `--uia` with no value (or
`--uia auto`) walks the foreground window's tree: the `uiautomation` package when
importable, else PowerShell's `UIAutomationClient` in a `CREATE_NO_WINDOW` child. Off
Windows or with no bridge, the join is skipped and the reason is printed; a recorded
tree JSON still joins anywhere. Control types map to roles (`Button` → `button`, `Edit`
→ `input`, `MenuItem` → `menuitem`, ...), container chrome (Window/Pane/Group/List/
MenuBar/...) is skipped, and a disabled control is never `interactable`. A matched DOM
CSS rect is not overwritten by a UIA rect; UIA still supplies the role.

### Planning over the screen

`awscreen.env.ScreenEnv` is an environment a planner can search: `observe`, `actions`
(click / type / scroll with absolute xy), `step`, plus `get_state_hash`, `evaluate`,
`clone`, `token_grid`. `step` never acts without an injected `act_fn`; `clone()` replays
only recorded transitions, and predicts "no change" for anything unseen.

## How it works

`awscreen` analyzes screenshots with a local vision model to find interactive elements matching your description. No pixel-perfect coordinates needed — just describe what you're looking for in natural language.

### Features

- **Visual matching**: Describe elements like "the red button in the bottom right"
- **Confidence scores**: Each match includes a confidence rating
- **Bounding boxes**: Get precise pixel coordinates for click targets
- **Text output or JSON**: Choose human-readable or machine-parseable output

## Exit codes

- `0`: Elements found; details printed
- `1`: No elements found matching the description
- `2`: Cannot proceed (unreachable vision endpoint, invalid image, API error)

## Endpoint

No API key. awscreen sends a picture of your screen, so it sends it to a
loopback endpoint you run -- nothing leaves the machine. It speaks the
OpenAI `/v1/chat/completions` shape, the same one awvision uses, so one
local vision model serves both. Configure it via:

1. Environment variable: `AWSCREEN_URL` (falls back to `AWVISION_URL`)
2. CLI argument: `awscreen image.png description --endpoint http://localhost:8150`

## Related

- `awiam` — identify who the caller is
- `awbac` — control what they may do
- `awseal` — prove it

## License

Apache 2.0

<!-- aither-ecosystem:start GENERATED from the ecosystem registry. Edits here are overwritten; change the registry instead. -->

## The aw family

Standalone tools that share one idea: **replace something you would otherwise have to _trust_ with something you can _check_.**

Each installs on its own, works offline, and needs no account.

| | instead of trusting | you check |
|---|---|---|
| [awdk](https://github.com/Aitherium/awdk) | a framework's idea of how your agents should run | one loop you can read, pointed at a backend you already pay for |
| [awskills](https://github.com/Aitherium/awskills) | that an agent knows your procedure | the procedure written down, versioned, and loadable by any agent |
| [awpack](https://github.com/Aitherium/awpack) | that the pack you want shipped inside somebody's SDK, under whatever licence that SDK happens to carry | the pack as its own versioned artifact, with its own licence, that any agent runtime can install |
| [awm](https://github.com/Aitherium/awm) | that memory stayed in its lane | tenant:user:project scopes, so a write cannot cross a boundary |
| [awdesk](https://github.com/Aitherium/awdesk) | that the agent is somewhere behind a browser tab | a tray icon, a face on your desktop, and the decision card that pops when it needs you |
| [awnode](https://github.com/Aitherium/awnode) | a vendor's cloud with every prompt | a local gateway routing to backends you chose |
| [awgraph](https://github.com/Aitherium/awgraph) | that grep found everything | an AST + tree-sitter call graph an agent can traverse |
| [awgit](https://github.com/Aitherium/awgit) | that no one else is editing this file | a lease, refused at commit time if you do not hold it |
| [awdelphi](https://github.com/Aitherium/awdelphi) | one agent's confident take on a decision | the round trace, the anonymity, and who dissents |
| [awclassify](https://github.com/Aitherium/awclassify) | a filename, a folder, or whoever last touched it | doc_type, visibility, audience and topics, with the evidence lines that decided each |
| [awdecide](https://github.com/Aitherium/awdecide) | a hosted classifier's probability that never learns whether it was right | the decision, its probability, and the calibration curve from your own resolved outcomes |
| [awtoll](https://github.com/Aitherium/awtoll) | that your tooling is saving you context | the measured token cost of each tool call, and what the alternative cost |
| [awseal](https://github.com/Aitherium/awseal) | that the artifact came from who you think | an Ed25519 seal — the key that verifies is not the key that forges |
| [awshare](https://github.com/Aitherium/awshare) | that the download is intact | content-addressed bundles, verified on fetch |
| [awsuite](https://github.com/Aitherium/awsuite) | that an agent holding your mailbox will not send on its own | every send, draft, upload and create returns a dry-run until confirm is true |
| [awpermit](https://github.com/Aitherium/awpermit) | that an admitted agent will only do what its owner meant | an Ed25519 permit naming the agent, the principal, the door, the actions and the cap |
| [awnest](https://github.com/Aitherium/awnest) | that there is a person on the other end | a verdict with evidence, where "we could not tell" is not "yes" |
| [awrena](https://github.com/Aitherium/awrena) | a leaderboard someone can edit, and votes nobody counted | a scored duel with both answers kept, and a result bound to them |
| [awnboard](https://github.com/Aitherium/awnboard) | a share link anyone who sees it can use | an invitation addressed to one person, for one gate, revocable |
| [awnix](https://github.com/Aitherium/awnix) | that the box is what you left it as | an immutable image you built, with atomic rollback |
| [awrecover](https://github.com/Aitherium/awrecover) | that the restore worked | a restore that fully lands or does not land at all |
| [awstorage](https://github.com/Aitherium/awstorage) | a du you ran last month, and a peers file that says 3 TB free | an inventory snapshot per node with a diff since the last one, and each tree classified re-fetchable or not |
| [awrelay](https://github.com/Aitherium/awrelay) | a SaaS in the middle of your agents | findings, alerts and coordination over your own transport |
| [awask](https://github.com/Aitherium/awask) | that anyone read the paragraph where you asked | the ask itself, with a button that steers the session that raised it |
| [awmail](https://github.com/Aitherium/awmail) | a mailbox somebody else can read | mail your agents send and receive over your own server |
| [awdiscord](https://github.com/Aitherium/awdiscord) | a self-bot on someone's account, or a bot that acts without asking | slash commands, rate limits, and approval buttons only your operators can press |
| [awswarm](https://github.com/Aitherium/awswarm) | that a model either fits your GPU or it doesn't run at all | a placement plan and an acquisition-probability estimate before you spend on a run |
| [awfind](https://github.com/Aitherium/awfind) | one vendor's idea of the web | results from whichever providers you configured |
| [awbrowse](https://github.com/Aitherium/awbrowse) | that the page said what you were told | the render, the DOM and the requests it made |
| [awvoice](https://github.com/Aitherium/awvoice) | that a cloud vendor may hold your audio | a transcript and a wav from a service you host |
| [awvision](https://github.com/Aitherium/awvision) | a filename and a caption somebody wrote | what a model actually reports about the pixels |
| **awscreen** _(you are here)_ | a selector that was true when the page was written | the elements actually rendered, by what they look like |
| [awbeads](https://github.com/Aitherium/awbeads) | that a layout your users built survives the next deploy | the arrangement as data you can read back, diff, and hand to another surface |
| [awsprite](https://github.com/Aitherium/awsprite) | a chatbot with a face that resets every session | a knowledge base you can list, edit, and watch shape what it becomes |
| [awbonsai](https://github.com/Aitherium/awbonsai) | that inference always means a request left the machine | a WebGPU model answering on the tab's own GPU, with a consent record logged before it ever loaded |
| [gawbbonet](https://github.com/Aitherium/gawbbonet) | the model to keep a 300-message campaign coherent by itself | campaign facts recalled from scoped memory you can list and edit |
| [aitherkvcache](https://github.com/Aitherium/aitherkvcache) | a vendor's quantisation defaults | sub-byte KV cache kernels you can benchmark yourself |
| [awrtifact](https://github.com/Aitherium/awrtifact) | a hand-rolled split script and a hand-edited worker manifest | byte-verified parts in a release, served with Range + CORS, sizes asserted by a live gate |
| [AitherZero](https://github.com/Aitherium/AitherZero) | a pile of scripts nobody has numbered | numbered, discoverable automation with declarative playbooks |
| [AitherConnect](https://github.com/Aitherium/AitherConnect) | what a page tells your browser to do | a federated search and desktop bridge you host |
| [awreason](https://github.com/Aitherium/awreason) | a confident paragraph | the phases it went through, and every tool call it made to get there |
| [awrecurse](https://github.com/Aitherium/awrecurse) | that everything you pasted in was actually read | which slices it opened, and what it concluded from each |
| [awprism](https://github.com/Aitherium/awprism) | the first explanation that fits | the ranked alternatives, and the observation that separates them |
| [awrepl](https://github.com/Aitherium/awrepl) | what the agent believes the value is | the value, printed from the live session |
| [awreport](https://github.com/Aitherium/awreport) | that the report you pasted carried no token in it | a redacted report, and the duplicate it merged into instead of filing twice |
| [awresearch](https://github.com/Aitherium/awresearch) | a summary of pages nobody opened | every claim against the source it came from |
| [awfocus](https://github.com/Aitherium/awfocus) | twelve terminal tabs and a bad memory | one command that names every session, finds any transcript, and opens or steers the one you want |
| [awgym](https://github.com/Aitherium/awgym) | that a world model learned anything from the games it saw | transitions captured from real play, fed back, and the retrodiction score falling on grids it never saw |
| [awpredict](https://github.com/Aitherium/awpredict) | a model because it trained without erroring | its prediction against a self-updating lookup, on the rows that are actually novel |
| [awevolve](https://github.com/Aitherium/awevolve) | that your optimisation loop is finding anything | every version it kept, the score that version earned, and the edit that produced it |
| [awsh](https://github.com/Aitherium/awsh) | that you already know the name of the command | what it decided your line meant, before it acts on it |
| [awmine](https://github.com/Aitherium/awmine) | that a session's lesson survived the session | a row per outcome, a candidate per lesson, and the transcript line each one came from |
| [awrise](https://github.com/Aitherium/awrise) | that a scheduled agent ran at all, and ran exactly once | a durable record of every wake -- fired, skipped, overlapped or timed out -- each with its reason |
| [awkno](https://github.com/Aitherium/awkno) | that the docs site is up, or that you remember the family | the whole ecosystem in your terminal, with no network at all |
| [awwall](https://github.com/Aitherium/awwall) | that a service only talks to the hosts you think it talks to | an explicit egress allowlist, where a denial names the rule that denied it |
| [awembed](https://github.com/Aitherium/awembed) | a general-purpose embedder that has never seen your code | a held-out split of whole directories, scored teacher vs student vs int8 |
| [awtax](https://github.com/Aitherium/awtax) | a closed tax app's sealed file you can never read again | a plain, provider-neutral schema of every figure, with the page it came from |
| [awsettings](https://github.com/Aitherium/awsettings) | that you will remember to re-approve the same thing on every box you work from | one profile, unioned rather than overwritten, with the credentials left behind |
| [awavatar](https://github.com/Aitherium/awavatar) | a cloud 3D vendor's opaque task id | a manifest with a sha256, a licence and a rig-audit verdict per file |

[**awnix**](https://github.com/Aitherium/awnix) is the ground floor — A Linux you can hand to an agent — immutable x86_64 & ARM64 (aarch64) base, capabilities included.

## The Aitherium ecosystem

Every repository here is public. Each publishes an `aither-manifest.json` beside its page, so any surface can read every sibling's — the network is browsable from any node in it.

| repo | what it is | pages |
|---|---|---|
| [awdk](https://github.com/Aitherium/awdk) | Build AI agent fleets — 3 lines, any backend, local or cloud | [docs](https://aitherium.github.io/awdk/) |
| [awskills](https://github.com/Aitherium/awskills) | Portable agent skills — self-contained procedures an agent loads on demand | [docs](https://aitherium.github.io/awskills/) |
| [awpack](https://github.com/Aitherium/awpack) | First-party agent packs — the ones we build, versioned and installable on their own | [docs](https://aitherium.github.io/awpack/) |
| [awm](https://github.com/Aitherium/awm) | A portable, scoped agent memory | [docs](https://aitherium.github.io/awm/) |
| [awdesk](https://github.com/Aitherium/awdesk) | Aither World Desk -- the desktop body of AitherOS Online: tray, avatars, decision cards, the Living Desktop as an overlay | [docs](https://aitherium.github.io/awdesk/) |
| [awnode](https://github.com/Aitherium/awnode) | A lightweight local gateway — bridges your apps to the AI backends you chose | [docs](https://aitherium.github.io/awnode/) |
| [awrun](https://github.com/Aitherium/awrun) | A priority-aware queue and dispatcher for agentic runs and ad-hoc CI builds. It also judges whether the runner pool is big enough for the queue it is draining, and can ask a host to grow it -- reserving capacity is zero-sum, so a saturated pool needs more of it, not a different share of it | [docs](https://aitherium.github.io/awrun/) |
| [awpool](https://github.com/Aitherium/awpool) | The free half of the elastic work pool: dispatches public-safe work onto the free GitHub-hosted runners of the public aw* mirrors, so their idle CI minutes become compute. awrun remains the paid/self-hosted lane; this is its free twin | [docs](https://aitherium.github.io/awpool/) |
| [awedgerunner](https://github.com/Aitherium/awedgerunner) | Run agent and CI work on edge runners nobody has to rack -- a volunteer's idle GPU, a phone on your own mesh, an edge worker, a free or spot CI runner -- under a lease with a hard budget cap, a named payer and an expiry | [docs](https://aitherium.github.io/awedgerunner/) |
| [awgraph](https://github.com/Aitherium/awgraph) | A semantic code graph for agents — AST + tree-sitter, call graphs | [docs](https://aitherium.github.io/awgraph/) |
| [awgit](https://github.com/Aitherium/awgit) | Semantic version control on top of git — edit-ops and leases | [docs](https://aitherium.github.io/awgit/) |
| [awdelphi](https://github.com/Aitherium/awdelphi) | Anonymous multi-round expert panels — a converged answer with a trace | [docs](https://aitherium.github.io/awdelphi/) |
| [awclassify](https://github.com/Aitherium/awclassify) | Classify any document -- what it is, who may read it, who it is for, what it is about | [docs](https://aitherium.github.io/awclassify/) |
| [awdecide](https://github.com/Aitherium/awdecide) | One typed-decision contract -- choice / score / bool with a probability -- over a ladder of backends you already run (rules, tiny local models, an LLM's logprobs), fail-closed, with a Brier ledger that resolves every decision against its outcome | [docs](https://aitherium.github.io/awdecide/) |
| [awtoll](https://github.com/Aitherium/awtoll) | What every tool call costs you in context, measured from your own transcripts | [docs](https://aitherium.github.io/awtoll/) |
| [awseal](https://github.com/Aitherium/awseal) | Sign an artifact so a stranger can verify it | [docs](https://aitherium.github.io/awseal/) |
| [awshare](https://github.com/Aitherium/awshare) | Publish an artifact and fetch it back verified | [docs](https://aitherium.github.io/awshare/) |
| [awsuite](https://github.com/Aitherium/awsuite) | Your Google Workspace as agent tools, and no write happens without a yes | [docs](https://aitherium.github.io/awsuite/) |
| [awdit](https://github.com/Aitherium/awdit) | An append-only audit trail whose gaps are DETECTABLE | [docs](https://aitherium.github.io/awdit/) |
| [awbac](https://github.com/Aitherium/awbac) | Role-based access control that fails closed and explains itself | [docs](https://aitherium.github.io/awbac/) |
| [awiam](https://github.com/Aitherium/awiam) | Who is this caller? A directory and session store that fails honestly | [docs](https://aitherium.github.io/awiam/) |
| [awtunnel](https://github.com/Aitherium/awtunnel) | Reach a service that has no public address | [docs](https://aitherium.github.io/awtunnel/) |
| [awpermit](https://github.com/Aitherium/awpermit) | Let an agent through your door with a scope, a spend cap, and someone accountable | [docs](https://aitherium.github.io/awpermit/) |
| [awnest](https://github.com/Aitherium/awnest) | Prove there is a human before you let them into the nest | [docs](https://aitherium.github.io/awnest/) |
| [awrena](https://github.com/Aitherium/awrena) | Put two agents head to head and get a verdict you can check | [docs](https://aitherium.github.io/awrena/) |
| [awnboard](https://github.com/Aitherium/awnboard) | A front gate you can put in front of anything, and hand someone the key to | [docs](https://aitherium.github.io/awnboard/) |
| [awnix](https://github.com/Aitherium/awnix) | A Linux you can hand to an agent — immutable x86_64 & ARM64 (aarch64) base, capabilities included | [docs](https://aitherium.github.io/awnix/) |
| [awrecover](https://github.com/Aitherium/awrecover) | Labelled snapshots with an all-or-nothing restore | [docs](https://aitherium.github.io/awrecover/) |
| [awstorage](https://github.com/Aitherium/awstorage) | Every drive on every node, indexed, classified and diffed -- so you can see what you own before you delete it | [docs](https://aitherium.github.io/awstorage/) |
| [awrelay](https://github.com/Aitherium/awrelay) | Portable agent messaging — findings, alerts, coordination | [docs](https://aitherium.github.io/awrelay/) |
| [awask](https://github.com/Aitherium/awask) | Your agent asks you a question — and acts on your answer | [docs](https://aitherium.github.io/awask/) |
| [awmail](https://github.com/Aitherium/awmail) | Give an agent an email address — send, and actually receive | [docs](https://aitherium.github.io/awmail/) |
| [awdiscord](https://github.com/Aitherium/awdiscord) | Put any agent on Discord as a real bot, with an operator in the loop | [docs](https://aitherium.github.io/awdiscord/) |
| [awnet](https://github.com/Aitherium/awnet) | The agentic web — agents host a mesh, and agents join one | [docs](https://aitherium.github.io/awnet/) |
| [awswarm](https://github.com/Aitherium/awswarm) | Run one model too big for any single GPU across a pool of small ones | [docs](https://aitherium.github.io/awswarm/) |
| [awfind](https://github.com/Aitherium/awfind) | A portable search client — query, results, ranking | [docs](https://aitherium.github.io/awfind/) |
| [awbrowse](https://github.com/Aitherium/awbrowse) | A portable browser client — navigate, console, network, DOM, screenshot | [docs](https://aitherium.github.io/awbrowse/) |
| [awvoice](https://github.com/Aitherium/awvoice) | Hear and speak — transcribe audio, synthesize a voice | [docs](https://aitherium.github.io/awvoice/) |
| [awvision](https://github.com/Aitherium/awvision) | See an image — describe it, ask it a question, compare two | [docs](https://aitherium.github.io/awvision/) |
| **awscreen** _(you are here)_ | See this machine — what is on screen, and where to click it | [docs](https://aitherium.github.io/awscreen/) |
| [awkit](https://github.com/Aitherium/awkit) | Render an agent panel from a tool result — one component, any React app | [docs](https://aitherium.github.io/awkit/) |
| [awbeads](https://github.com/Aitherium/awbeads) | A spatial canvas for a page — arrange things, connect them, and keep the arrangement | [docs](https://aitherium.github.io/awbeads/) |
| [awsprite](https://github.com/Aitherium/awsprite) | Hatch a companion that grows only from what you teach it, then take it home | [docs](https://aitherium.github.io/awsprite/) |
| [awbonsai](https://github.com/Aitherium/awbonsai) | Run a real model in the visitor's own browser — no server round trip, no upload | [docs](https://aitherium.github.io/awbonsai/) |
| [awknowledge](https://github.com/Aitherium/awknowledge) | How to run a coding agent so the result survives — the laws, with evidence | [docs](https://aitherium.github.io/awknowledge/) |
| [awbrain](https://github.com/Aitherium/awbrain) | Your history as a wiki of linked markdown — claims pinned to the evidence | [docs](https://aitherium.github.io/awbrain/) |
| [gawbbonet](https://github.com/Aitherium/gawbbonet) | GobboNet campaigns with a real agent brain — scoped memory, graph recall | [docs](https://aitherium.github.io/gawbbonet/) |
| [aitherkvcache](https://github.com/Aitherium/aitherkvcache) | Near-optimal KV cache quantization for LLM inference — sub-byte compression | [docs](https://aitherium.github.io/aitherkvcache/) |
| [awrtifact](https://github.com/Aitherium/awrtifact) | Deliberately chunk artifacts into GitHub release assets — the productized aitherkvcache mirror lane | [docs](https://aitherium.github.io/awrtifact/) |
| [AitherZero](https://github.com/Aitherium/AitherZero) | PowerShell 7+ automation framework — numbered, self-describing scripts | [docs](https://aitherium.github.io/AitherZero/) |
| [AitherConnect](https://github.com/Aitherium/AitherConnect) | Browser extension — federated AI search, page context, and the Living OS overlay | [docs](https://aitherium.github.io/AitherConnect/) |
| [awreason](https://github.com/Aitherium/awreason) | A portable reasoning client — sessions, phases, thoughts, and the chain that produced the answer | [docs](https://aitherium.github.io/awreason/) |
| [awrecurse](https://github.com/Aitherium/awrecurse) | Answer a question over a context far larger than the window — recursively, with the trace kept | [docs](https://aitherium.github.io/awrecurse/) |
| [awprism](https://github.com/Aitherium/awprism) | Turn a failure into ranked hypotheses — and say what would confirm each one | [docs](https://aitherium.github.io/awprism/) |
| [awrepl](https://github.com/Aitherium/awrepl) | A REPL an agent can actually use — state that survives between turns | [docs](https://aitherium.github.io/awrepl/) |
| [awreport](https://github.com/Aitherium/awreport) | File a bug report that has already scrubbed your secrets and collapsed the duplicate | [docs](https://aitherium.github.io/awreport/) |
| [awresearch](https://github.com/Aitherium/awresearch) | Ask a research question, get a cited report you can check | [docs](https://aitherium.github.io/awresearch/) |
| [awfocus](https://github.com/Aitherium/awfocus) | See, search and steer every Claude session from one command | [docs](https://aitherium.github.io/awfocus/) |
| [awgym](https://github.com/Aitherium/awgym) | An ARC training gym — a game a world model can watch, and six roles that play through it | [docs](https://aitherium.github.io/awgym/) |
| [awpredict](https://github.com/Aitherium/awpredict) | Predict what your environment does next, and how surprised you were | [docs](https://aitherium.github.io/awpredict/) |
| [awevolve](https://github.com/Aitherium/awevolve) | Point an agent at a file and a command that scores it, and let it improve | [docs](https://aitherium.github.io/awevolve/) |
| [awsh](https://github.com/Aitherium/awsh) | Your terminal answers you -- type a question where a command would go | [docs](https://aitherium.github.io/awsh/) |
| [awmine](https://github.com/Aitherium/awmine) | Mine what your agents did -- outcomes, lessons and procedures out of the transcripts they left behind | [docs](https://aitherium.github.io/awmine/) |
| [awrise](https://github.com/Aitherium/awrise) | Wake an agent on a schedule, let it do one thing, and put it back to sleep | [docs](https://aitherium.github.io/awrise/) |
| [awkno](https://github.com/Aitherium/awkno) | The man page for the Aither World — every brick, stack and law, offline | [docs](https://aitherium.github.io/awkno/) |
| [awwall](https://github.com/Aitherium/awwall) | Say what a workload may reach, and watch everything else fail closed | [docs](https://aitherium.github.io/awwall/) |
| [awrouter](https://github.com/Aitherium/awrouter) | OpenRouter for your own fleet: pick a model backend by cost/latency/ capability, fail over, fit the context window, stream. Standalone, OpenAI-compatible, no Aither-specifics required to be valuable | [docs](https://aitherium.github.io/awrouter/) |
| [awembed](https://github.com/Aitherium/awembed) | Train an embedding model that knows your corpus, and prove it beats the big one | [docs](https://aitherium.github.io/awembed/) |
| [awtax](https://github.com/Aitherium/awtax) | Turn any tax PDF -- returns, W-2, 1099, statements, even scans -- into structured data you can check | [docs](https://aitherium.github.io/awtax/) |
| [awflow](https://github.com/Aitherium/awflow) | A deterministic workflow runtime — chain agent calls with journal replay and budget control | [docs](https://aitherium.github.io/awflow/) |
| [awsettings](https://github.com/Aitherium/awsettings) | Your agent's permissions and config, following you to the next machine | [docs](https://aitherium.github.io/awsettings/) |
| [awavatar](https://github.com/Aitherium/awavatar) | One character spec in, a rigged, animated, multi-style avatar pack out | [docs](https://aitherium.github.io/awavatar/) |

**Built on** [REA (Reverse Engineer Anything)](https://github.com/morluto/rea) · [llama.cpp](https://github.com/ggml-org/llama.cpp) · [vLLM](https://github.com/vllm-project/vllm) · [ComfyUI](https://github.com/comfyanonymous/ComfyUI) · [CentOS Stream](https://www.centos.org/centos-stream/) · [Podman](https://github.com/containers/podman) · [Docker](https://github.com/moby/moby) · [LanceDB](https://github.com/lancedb/lancedb) · [WireGuard](https://www.wireguard.com/) · [FFmpeg](https://ffmpeg.org/) · [Blender + Rigify](https://www.blender.org/) · [headroom](https://github.com/headroomlabs-ai/headroom) · [SANA](https://github.com/NVlabs/Sana) · [Hunyuan3D](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) · [repowise](https://github.com/repowise-dev/repowise) · [Playwright](https://github.com/microsoft/playwright) · [Chromium](https://www.chromium.org/) · [Next.js](https://github.com/vercel/next.js) · [React](https://github.com/facebook/react) · [PhotoCraft](https://github.com/storytold/photocraft).

<div id="aither-constellation" data-self="awscreen"></div>
<script src="aither-constellation.js"></script>

<!-- aither-ecosystem:end -->
