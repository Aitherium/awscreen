"""Core visual element finder using Claude vision analysis."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

try:
    from PIL import Image
except ImportError:
    Image = None


@dataclass
class Element:
    """A clickable element found on screen."""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    description: str


@dataclass
class Screenshot:
    """A captured or loaded screenshot."""
    path: Path
    width: int
    height: int
    format: str

    @property
    def data_uri(self) -> str:
        """Encode image as data URI for API submission."""
        mime = "image/png" if self.format.lower() == "png" else "image/jpeg"
        with open(self.path, "rb") as f:
            b64 = base64.standard_b64encode(f.read()).decode("ascii")
        return f"data:{mime};base64,{b64}"


class Finder:
    """Find clickable elements by visual description.

    Requires an Anthropic API key (passed at init or via ANTHROPIC_API_KEY env).
    """

    def __init__(self, api_key: str | None = None):
        """Initialize the finder.

        Args:
            api_key: Anthropic API key. If None, reads ANTHROPIC_API_KEY env var.
                    Raises ValueError if neither is available.
        """
        if not api_key:
            import os
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "API key required: pass api_key= or set ANTHROPIC_API_KEY env var"
            )
        if Anthropic is None:
            raise ImportError(
                "awscreen requires 'anthropic' package: pip install anthropic"
            )
        self.client = Anthropic(api_key=api_key)

    def load_image(self, path: str) -> Screenshot:
        """Load an image from disk.

        Args:
            path: Path to image file (PNG or JPEG).

        Returns:
            Screenshot object ready for find().

        Raises:
            FileNotFoundError: If path doesn't exist.
            ValueError: If Image is not available or file is not a valid image.
        """
        if Image is None:
            raise ImportError(
                "awscreen requires 'Pillow' package: pip install Pillow"
            )
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        try:
            img = Image.open(p)
            fmt = img.format or "PNG"
            return Screenshot(
                path=p,
                width=img.width,
                height=img.height,
                format=fmt,
            )
        except Exception as exc:
            raise ValueError(f"Cannot load image {p}: {exc}")

    def capture(self) -> Screenshot:
        """Capture current screen. Currently not implemented — use load_image."""
        raise NotImplementedError(
            "Screen capture not yet implemented. Use load_image(path) instead."
        )

    def find(self, screenshot: Screenshot, description: str) -> list[Element]:
        """Find clickable elements matching a description.

        Args:
            screenshot: Screenshot object from load_image() or capture().
            description: Natural language description of what to find.
                        E.g., "the blue save button in the top toolbar"

        Returns:
            List of Element objects with positions and confidence scores.
            Empty list if no matches found.

        Raises:
            ValueError: If API key is invalid or API call fails.
        """
        if not description or not description.strip():
            return []

        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": (
                                        "image/png"
                                        if screenshot.format.lower() == "png"
                                        else "image/jpeg"
                                    ),
                                    "data": self._load_image_base64(
                                        screenshot.path
                                    ),
                                },
                            },
                            {
                                "type": "text",
                                "text": self._build_prompt(
                                    description,
                                    screenshot.width,
                                    screenshot.height,
                                ),
                            },
                        ],
                    }
                ],
            )
        except Exception as exc:
            raise ValueError(
                f"API call failed (check key and network): {exc}"
            )

        return self._parse_response(message.content[0].text)

    def _load_image_base64(self, path: Path) -> str:
        """Load image and encode as base64."""
        with open(path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("ascii")

    def _build_prompt(self, description: str, width: int, height: int) -> str:
        """Build the vision prompt for Claude."""
        return f"""Find clickable elements matching this description: "{description}"

Respond with a JSON array of objects, each with:
- x: left edge coordinate (0-{width})
- y: top edge coordinate (0-{height})
- width: element width in pixels
- height: element height in pixels
- confidence: 0.0-1.0 (1.0 = certain, 0.5 = possible, <0.5 = unlikely)
- description: what you found

Return ONLY valid JSON, no other text. If no matches found, return [].
Example response:
[{{"x":10,"y":20,"width":100,"height":40,"confidence":0.95,"description":"blue button"}}]
"""

    def _parse_response(self, response: str) -> list[Element]:
        """Parse API response into Element objects."""
        response = response.strip()
        if not response or response == "[]":
            return []

        # Extract JSON from response (might have surrounding text)
        try:
            # Try direct parse first
            data = json.loads(response)
        except json.JSONDecodeError:
            # Try to find JSON array in response
            start = response.find("[")
            end = response.rfind("]")
            if start < 0 or end <= start:
                return []
            try:
                data = json.loads(response[start : end + 1])
            except json.JSONDecodeError:
                return []

        if not isinstance(data, list):
            return []

        elements = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                elem = Element(
                    x=int(item.get("x", 0)),
                    y=int(item.get("y", 0)),
                    width=int(item.get("width", 0)),
                    height=int(item.get("height", 0)),
                    confidence=float(item.get("confidence", 0.0)),
                    description=str(item.get("description", "")),
                )
                if elem.width > 0 and elem.height > 0 and elem.confidence > 0:
                    elements.append(elem)
            except (TypeError, ValueError):
                # Skip malformed items
                continue

        return elements
