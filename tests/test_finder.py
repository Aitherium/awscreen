"""awscreen finder tests.

Tests both positive (elements found) and negative (not found, errors) cases.
A suite that only tests positive cases passes trivially on a broken implementation
that returns everything, and one that only tests negatives passes on one that
returns nothing — both directions required.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from awscreen import Element, Finder, Screenshot

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import anthropic as _anthropic  # noqa: F401

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

HAS_PILLOW = Image is not None


class TestElementDataclass:
    """Element structure."""

    def test_element_creation(self):
        """Element can be created with all fields."""
        elem = Element(
            x=10, y=20, width=100, height=50, confidence=0.95, description="button"
        )
        assert elem.x == 10
        assert elem.y == 20
        assert elem.width == 100
        assert elem.height == 50
        assert elem.confidence == 0.95
        assert elem.description == "button"

    def test_element_zero_dimensions_are_valid(self):
        """Element can have zero width/height (edge case)."""
        elem = Element(x=0, y=0, width=0, height=0, confidence=0.5, description="")
        assert elem.width == 0


class TestScreenshot:
    """Screenshot structure and data URIs."""

    def test_screenshot_creation(self):
        """Screenshot can be created with image info."""
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            p = Path(tmp.name)
            ss = Screenshot(path=p, width=1920, height=1080, format="PNG")
            assert ss.width == 1920
            assert ss.height == 1080
            assert ss.format == "PNG"

    def test_screenshot_data_uri_property(self):
        """data_uri property encodes image as base64 data URI."""
        # Create a minimal PNG (1x1 pixel)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            p = Path(tmp.name)
            # Minimal valid PNG header
            png_bytes = (
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
                b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
                b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x055"
                b"\xcd\xfb\xe7\x00\x00\x00\x00IEND\xaeB`\x82"
            )
            tmp.write(png_bytes)
            tmp.flush()
            tmp.close()

            ss = Screenshot(path=p, width=1, height=1, format="PNG")
            uri = ss.data_uri
            assert uri.startswith("data:image/png;base64,")
            assert len(uri) > 25

            p.unlink()


class TestFinderInit:
    """Finder initialization."""

    def test_finder_requires_api_key(self):
        """Finder raises ValueError if no API key provided."""
        with pytest.raises(ValueError, match="API key"):
            Finder(api_key=None)

    @pytest.mark.skipif(not HAS_ANTHROPIC, reason="Anthropic not available")
    def test_finder_accepts_api_key_argument(self):
        """Finder accepts api_key argument."""
        finder = Finder(api_key="test-key-12345")
        # OK: initialized
        assert finder.client is not None

    def test_finder_init_with_env_var_not_set(self, monkeypatch):
        """Finder reads ANTHROPIC_API_KEY env var if no argument."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key"):
            Finder()


class TestImageLoading:
    """Image file handling."""

    @pytest.mark.skipif(not HAS_ANTHROPIC, reason="Anthropic not available")
    def test_load_image_file_not_found(self):
        """load_image raises FileNotFoundError for missing file."""
        finder = Finder(api_key="test")
        with pytest.raises(FileNotFoundError):
            finder.load_image("/nonexistent/path/image.png")

    @pytest.mark.skipif(
        not HAS_ANTHROPIC or not HAS_PILLOW,
        reason="Anthropic and Pillow required"
    )
    def test_load_image_creates_screenshot(self):
        """load_image returns Screenshot with correct dimensions."""
        finder = Finder(api_key="test")

        # Create a minimal valid PNG using PIL
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            p = Path(tmp.name)
            tmp.close()

            # Use PIL to create a valid PNG
            img = Image.new('RGB', (2, 3), color='red')
            img.save(str(p), 'PNG')

            try:
                screenshot = finder.load_image(str(p))
                assert screenshot.width == 2
                assert screenshot.height == 3
                assert screenshot.format.upper() == "PNG"
                assert screenshot.path == p
            finally:
                p.unlink()


class TestResponseParsing:
    """JSON response parsing."""

    @pytest.mark.skipif(not HAS_ANTHROPIC, reason="Anthropic not available")
    def test_parse_valid_json_response(self):
        """Parse response containing valid element JSON."""
        finder = Finder(api_key="test")

        response = (
            '[{"x":10,"y":20,"width":100,"height":50,'
            '"confidence":0.95,"description":"button"}]'
        )
        elements = finder._parse_response(response)

        assert len(elements) == 1
        assert elements[0].x == 10
        assert elements[0].y == 20
        assert elements[0].width == 100
        assert elements[0].height == 50
        assert elements[0].confidence == 0.95
        assert elements[0].description == "button"

    @pytest.mark.skipif(not HAS_ANTHROPIC, reason="Anthropic not available")
    def test_parse_empty_array(self):
        """Parse empty JSON array returns empty list."""
        finder = Finder(api_key="test")

        elements = finder._parse_response("[]")
        assert len(elements) == 0

    @pytest.mark.skipif(not HAS_ANTHROPIC, reason="Anthropic not available")
    def test_parse_multiple_elements(self):
        """Parse response with multiple elements."""
        finder = Finder(api_key="test")

        response = (
            '['
            '{"x":0,"y":0,"width":50,"height":30,"confidence":0.9,"description":"first"},'
            '{"x":100,"y":100,"width":60,"height":40,"confidence":0.85,"description":"second"}'
            "]"
        )
        elements = finder._parse_response(response)

        assert len(elements) == 2
        assert elements[0].description == "first"
        assert elements[1].description == "second"

    @pytest.mark.skipif(not HAS_ANTHROPIC, reason="Anthropic not available")
    def test_parse_json_with_surrounding_text(self):
        """Parse response with JSON embedded in text."""
        finder = Finder(api_key="test")

        response = (
            "I found the button. Here it is:\n"
            '[{"x":50,"y":60,"width":80,"height":40,"confidence":0.92,"description":"save"}]\n'
            "That's it."
        )
        elements = finder._parse_response(response)

        assert len(elements) == 1
        assert elements[0].description == "save"

    @pytest.mark.skipif(not HAS_ANTHROPIC, reason="Anthropic not available")
    def test_parse_rejects_zero_or_negative_dimensions(self):
        """Elements with width/height ≤ 0 are skipped."""
        finder = Finder(api_key="test")

        response = (
            '['
            '{"x":10,"y":10,"width":100,"height":50,'
            '"confidence":0.9,"description":"good"},'
            '{"x":20,"y":20,"width":0,"height":50,'
            '"confidence":0.8,"description":"zero width"},'
            '{"x":30,"y":30,"width":100,"height":-5,'
            '"confidence":0.7,"description":"negative height"}'
            "]"
        )
        elements = finder._parse_response(response)

        assert len(elements) == 1
        assert elements[0].description == "good"

    @pytest.mark.skipif(not HAS_ANTHROPIC, reason="Anthropic not available")
    def test_parse_rejects_zero_or_negative_confidence(self):
        """Elements with confidence ≤ 0 are skipped."""
        finder = Finder(api_key="test")

        response = (
            '['
            '{"x":10,"y":10,"width":100,"height":50,'
            '"confidence":0.5,"description":"ok"},'
            '{"x":20,"y":20,"width":100,"height":50,'
            '"confidence":0,"description":"zero conf"},'
            '{"x":30,"y":30,"width":100,"height":50,'
            '"confidence":-0.1,"description":"neg conf"}'
            "]"
        )
        elements = finder._parse_response(response)

        assert len(elements) == 1
        assert elements[0].description == "ok"

    @pytest.mark.skipif(not HAS_ANTHROPIC, reason="Anthropic not available")
    def test_parse_invalid_json_returns_empty(self):
        """Invalid JSON returns empty list, not exception."""
        finder = Finder(api_key="test")

        elements = finder._parse_response("not json at all")
        assert len(elements) == 0

    @pytest.mark.skipif(not HAS_ANTHROPIC, reason="Anthropic not available")
    def test_parse_truncated_json_returns_empty(self):
        """Truncated JSON returns empty list."""
        finder = Finder(api_key="test")

        elements = finder._parse_response('[{"x":10')
        assert len(elements) == 0

    @pytest.mark.skipif(not HAS_ANTHROPIC, reason="Anthropic not available")
    def test_parse_malformed_element_skipped(self):
        """Malformed elements in array are skipped, valid ones kept."""
        finder = Finder(api_key="test")

        response = (
            '['
            '{"x":10,"y":20,"width":100,"height":50,"confidence":0.95,"description":"good"},'
            '{"x":"not a number"},'
            '{"y":99,"width":null,"height":50,"confidence":0.8,"description":"malformed"}'
            "]"
        )
        elements = finder._parse_response(response)

        # Only the good element
        assert len(elements) == 1
        assert elements[0].description == "good"

    @pytest.mark.skipif(not HAS_ANTHROPIC, reason="Anthropic not available")
    def test_parse_non_array_returns_empty(self):
        """Non-array JSON is rejected."""
        finder = Finder(api_key="test")

        response = '{"result": "not an array"}'
        elements = finder._parse_response(response)
        assert len(elements) == 0


class TestFindDescription:
    """Description validation."""

    @pytest.mark.skipif(not HAS_ANTHROPIC, reason="Anthropic not available")
    def test_find_empty_description_returns_empty(self):
        """find() with empty description returns empty list."""
        finder = Finder(api_key="test")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            p = Path(tmp.name)
            png_bytes = (
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
                b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
                b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x055"
                b"\xcd\xfb\xe7\x00\x00\x00\x00IEND\xaeB`\x82"
            )
            tmp.write(png_bytes)
            tmp.flush()
            tmp.close()

            try:
                screenshot = finder.load_image(str(p))
                elements = finder.find(screenshot, "")
                assert len(elements) == 0
            finally:
                p.unlink()

    @pytest.mark.skipif(not HAS_ANTHROPIC, reason="Anthropic not available")
    def test_find_whitespace_only_description_returns_empty(self):
        """find() with whitespace-only description returns empty."""
        finder = Finder(api_key="test")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            p = Path(tmp.name)
            png_bytes = (
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
                b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
                b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x055"
                b"\xcd\xfb\xe7\x00\x00\x00\x00IEND\xaeB`\x82"
            )
            tmp.write(png_bytes)
            tmp.flush()
            tmp.close()

            try:
                screenshot = finder.load_image(str(p))
                elements = finder.find(screenshot, "   ")
                assert len(elements) == 0
            finally:
                p.unlink()
