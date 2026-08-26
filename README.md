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
export ANTHROPIC_API_KEY="sk-ant-..."

# Find the save button in a screenshot
awscreen screenshot.png "the blue save button in the top toolbar"
```

Or use in Python:

```python
from awscreen import Finder

finder = Finder()  # reads ANTHROPIC_API_KEY env var
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

1. Environment variable: `ANTHROPIC_API_KEY=sk-ant-...`
2. CLI argument: `awscreen image.png description --api-key sk-ant-...`

Get your API key at https://console.anthropic.com.

## Related

- `awiam` — identify who the caller is
- `awbac` — control what they may do
- `awseal` — prove it

## License

Apache 2.0
