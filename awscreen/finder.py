"""Core visual element finder, against a LOCAL vision endpoint.

This brick sends a picture of YOUR SCREEN somewhere. That makes the choice of
"somewhere" the most consequential decision in the file, so it is stated here:
it goes to a loopback endpoint you run, in the OpenAI /v1/chat/completions
shape -- the same one awvision uses, deliberately, so one local model serves
both and there is one thing to stand up rather than two.

It was first written against a HOSTED API with an api-key, which would have
shipped a brick that uploads your desktop to a third party and bills you for
it. Nothing in the aw* family does that, and a screen recorder is the worst
possible place to start.

Endpoint and model: AWSCREEN_URL / AWSCREEN_MODEL, falling back to
AWVISION_URL / AWVISION_MODEL so configuring the vision lane once configures
both.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

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
    """Find clickable elements by visual description, via a local vision model.

    No API key. Nothing leaves the machine.
    """

    def __init__(self, endpoint: str | None = None, model: str | None = None,
                 api_key: str | None = None):
        """Point it at a local vision endpoint.

        Args:
            endpoint: base URL, default AWSCREEN_URL / AWVISION_URL /
                      http://localhost:8150
            model:    model id, default AWSCREEN_MODEL / AWVISION_MODEL
            api_key:  accepted and ignored. Kept so an older call site does not
                      break, but this talks to loopback and needs no credential
                      -- silently accepting one and sending it somewhere would
                      be worse than refusing it.
        """
        self.endpoint = (endpoint
                         or os.getenv("AWSCREEN_URL")
                         or os.getenv("AWVISION_URL")
                         or "http://localhost:8150").rstrip("/")
        self.model = (model
                      or os.getenv("AWSCREEN_MODEL")
                      or os.getenv("AWVISION_MODEL")
                      or "gpt-4-vision")

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

        b64 = self._load_image_base64(screenshot.path)
        media = "image/png" if screenshot.format.lower() == "png" else "image/jpeg"
        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": self._build_prompt(description,
                                                screenshot.width,
                                                screenshot.height)},
                    {"type": "image_url",
                     "image_url": {"url": "data:" + media + ";base64," + b64}},
                ],
            }],
        }
        url = self.endpoint + "/v1/chat/completions"
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ValueError(
                "Vision endpoint returned HTTP " + str(exc.code) + " at " + url
                + " -- start a vision-capable model there, or set AWSCREEN_URL.")
        except urllib.error.URLError as exc:
            raise ValueError(
                "Cannot reach a vision endpoint at " + self.endpoint
                + " (" + str(exc.reason) + "). awscreen needs a LOCAL vision"
                + " model; nothing is sent off this machine."
                + " Set AWSCREEN_URL or AWVISION_URL.")

        choices = body.get("choices") or []
        if not choices:
            raise ValueError(
                "Model '" + self.model + "' returned no choices. It is most"
                " likely not vision-capable.")
        content = ((choices[0].get("message") or {}).get("content") or "")
        if not content or content.isspace():
            # A TEXT model handed an image answers 200 with nothing. Refusing
            # here is the whole point: a blank answer would be read as "the
            # element is not on screen", which is a different and wrong fact.
            raise ValueError(
                "Model '" + self.model + "' at " + self.endpoint + " returned"
                " EMPTY content for an image. That is what a text-only model"
                " does when handed a picture -- it did not fail to find your"
                " element, it cannot see. Point AWSCREEN_MODEL at a"
                " vision-capable model.")
        return self._parse_response(content)

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
