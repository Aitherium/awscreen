# awscreen

Find clickable elements by describing what they look like.

Part of the [Aither World (`aw*`)](https://github.com/Aitherium/awscreens) family of CLI tools.

## Install

```bash
pip install awscreen
```

## Usage

Find elements in a screenshot by natural language description:

```bash
# Set your API key
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

## How it works

`awscreen` analyzes screenshots using Claude's vision capabilities to find interactive elements matching your description. No pixel-perfect coordinates needed — just describe what you're looking for in natural language.

### Features

- **Visual matching**: Describe elements like "the red button in the bottom right"
- **Confidence scores**: Each match includes a confidence rating
- **Bounding boxes**: Get precise pixel coordinates for click targets
- **Text output or JSON**: Choose human-readable or machine-parseable output

## Exit codes

- `0`: Elements found; details printed
- `1`: No elements found matching the description
- `2`: Cannot proceed (missing API key, invalid image, API error)

## API key

Requires an Anthropic API key for vision analysis. Provide it via:

1. Environment variable: `AWSCREEN_URL` (falls back to `AWVISION_URL`)
2. CLI argument: `awscreen image.png description --api-key sk-ant-...`

No API key. awscreen sends a picture of your screen, so it sends it to a
loopback endpoint you run -- nothing leaves the machine. It speaks the
OpenAI /v1/chat/completions shape, the same one awvision uses, so one
local vision model serves both.

## Related

- `awiam` — identify who the caller is
- `awbac` — control what they may do
- `awseal` — prove it

## License

Apache 2.0
