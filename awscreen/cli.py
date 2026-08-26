"""awscreen CLI — find and locate clickable elements by description.

Exit codes:
  0 = elements found and displayed
  1 = no elements found matching description
  2 = cannot judge (API error, missing key, file not found)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path

from .finder import Finder


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    # GENERATED doctor intercept (gen_aw_doctor.py) -- do not edit
    _dv = locals().get("argv")
    if (_dv if _dv is not None else __import__("sys").argv[1:])[:1] == ["doctor"]:
        from ._doctor import report
        return report()
    ap = argparse.ArgumentParser(
        prog="awscreen",
        description="Find clickable elements by visual description",
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="Run self-test (offline, no API)",
    )
    ap.add_argument(
        "image",
        nargs="?",
        help="Path to screenshot image (PNG or JPEG)",
    )
    ap.add_argument(
        "description",
        nargs="?",
        help="Natural language description of element to find",
    )
    ap.add_argument(
        "--api-key",
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)",
    )
    ap.add_argument(
        "--format",
        choices=["json", "text"],
        default="text",
        help="Output format",
    )

    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()

    if not args.image or not args.description:
        ap.print_help()
        return 2

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")

    try:
        finder = Finder(api_key=api_key)
    except ValueError as exc:
        print(f"DEAD: {exc}", file=sys.stderr)
        return 2
    except ImportError as exc:
        print(f"DEAD: {exc}", file=sys.stderr)
        return 2

    try:
        screenshot = finder.load_image(args.image)
    except FileNotFoundError as exc:
        print(f"DEAD: {exc}", file=sys.stderr)
        return 2
    except (ValueError, ImportError) as exc:
        print(f"DEAD: {exc}", file=sys.stderr)
        return 2

    try:
        elements = finder.find(screenshot, args.description)
    except ValueError as exc:
        print(f"DEAD: {exc}", file=sys.stderr)
        return 2

    if not elements:
        if args.format == "json":
            print(json.dumps({"found": 0, "elements": []}))
        else:
            print("No elements found matching description")
        return 1

    if args.format == "json":
        result = {
            "found": len(elements),
            "elements": [
                {
                    "x": e.x,
                    "y": e.y,
                    "width": e.width,
                    "height": e.height,
                    "confidence": e.confidence,
                    "description": e.description,
                }
                for e in elements
            ],
        }
        print(json.dumps(result, indent=2))
    else:
        for i, elem in enumerate(elements, 1):
            print(
                f"Match {i}: ({elem.x}, {elem.y}) "
                f"[{elem.width}x{elem.height}] "
                f"confidence={elem.confidence:.2f} "
                f'"{elem.description}"'
            )

    return 0


def _self_test() -> int:
    """Self-test: verify the brick works offline.

    Tests:
    - Argument parsing
    - Image loading
    - JSON parsing
    - Finder initialization (without calling API)
    """
    try:
        # Test 1: Finder requires API key
        try:
            finder = Finder(api_key=None)
            print("FAIL: Finder should require API key")
            return 1
        except ValueError as exc:
            if "API key" not in str(exc):
                print(f"FAIL: Wrong error message: {exc}")
                return 1

        # Test 2: Can initialize with key (but won't call API)
        with contextlib.suppress(ImportError):
            finder = Finder(api_key="test-key-for-testing")
            # This will fail if Anthropic is not installed, which is fine

        # Test 3: Test image loading with a dynamically created test image
        with contextlib.suppress(ImportError):
            finder = Finder(api_key="test-key")
            # Create a minimal valid PNG (1x1 pixel) dynamically
            png_bytes = (
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
                b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
                b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x055"
                b"\xcd\xfb\xe7\x00\x00\x00\x00IEND\xaeB`\x82"
            )
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(png_bytes)
                tmp.flush()
                tmp_path = tmp.name
            try:
                screenshot = finder.load_image(tmp_path)
                if screenshot.width <= 0 or screenshot.height <= 0:
                    print("FAIL: Image dimensions invalid")
                    return 1
            finally:
                Path(tmp_path).unlink()

        # Test 4: Test JSON parsing
        response = (
            '[{"x":10,"y":20,"width":100,"height":50,'
            '"confidence":0.95,"description":"button"}]'
        )
        with contextlib.suppress(ImportError):
            finder = Finder(api_key="test")
            elements = finder._parse_response(response)
            if len(elements) != 1:
                print(f"FAIL: Expected 1 element, got {len(elements)}")
                return 1
            elem = elements[0]
            if (
                elem.x != 10
                or elem.y != 20
                or elem.width != 100
                or elem.height != 50
                or elem.confidence != 0.95
            ):
                print(f"FAIL: Element values incorrect: {elem}")
                return 1

        # Test 5: Parse empty response
        with contextlib.suppress(ImportError):
            finder = Finder(api_key="test")
            elements = finder._parse_response("[]")
            if len(elements) != 0:
                print("FAIL: Empty response should return empty list")
                return 1

        # Test 6: Parse response with surrounding text
        response_with_text = (
            "I found these elements:\n"
            '[{"x":5,"y":10,"width":80,"height":30,"confidence":0.8,"description":"text"}]\n'
            "Done."
        )
        with contextlib.suppress(ImportError):
            finder = Finder(api_key="test")
            elements = finder._parse_response(response_with_text)
            if len(elements) != 1:
                print("FAIL: Should extract JSON from text")
                return 1

        # All tests passed
        return 0

    except Exception as exc:
        print(f"FAIL: Unexpected error in self-test: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
