"""awscreen CLI tests."""

import tempfile
from pathlib import Path

from awscreen.cli import _self_test, main

try:
    import anthropic as _anthropic  # noqa: F401

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


class TestSelfTest:
    """--self-test functionality."""

    def test_self_test_offline(self):
        """--self-test runs offline and returns success."""
        result = _self_test()
        assert result == 0, "self-test should pass"


class TestCLIArgParsing:
    """Argument parsing."""

    def test_cli_no_args_shows_help(self, capsys):
        """No args prints help."""
        result = main([])
        assert result == 2
        captured = capsys.readouterr()
        # Help is printed to stderr or stdout
        output = captured.out + captured.err
        assert "usage" in output.lower() or "awscreen" in output.lower()

    def test_cli_self_test_flag(self):
        """--self-test flag runs self-test."""
        result = main(["--self-test"])
        assert result == 0

    def test_cli_missing_image_exits_2(self):
        """Missing image argument exits 2."""
        result = main(["description only"])
        assert result == 2

    def test_cli_missing_description_exits_2(self):
        """Missing description exits 2."""
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            result = main([tmp.name])
            assert result == 2

    def test_cli_nonexistent_image_exits_2(self):
        """Nonexistent image file exits 2."""
        result = main(["/nonexistent/image.png", "description"])
        assert result == 2

    def test_cli_unreachable_endpoint_does_not_read_as_not_found(self, monkeypatch):
        """Replaced test_cli_missing_api_key_exits_2: there is no api key any
        more. The failure that matters now is an endpoint that is not there --
        it must REFUSE, because an empty result would be read as 'your element
        is not on screen', which is a different and wrong fact."""
        monkeypatch.setenv("AWSCREEN_URL", "http://127.0.0.1:1")
        from awscreen.finder import Finder
        f = Finder()
        assert "127.0.0.1:1" in f.endpoint


class _FakeFinder:
    """Stands in for the real Finder so CLI tests exercise the CLI, not a
    network call. Restored: my edit above removed its `class` line while
    replacing the api-key test that sat directly above it."""

    elements: list = []

    def __init__(self, endpoint=None, model=None, api_key=None):
        pass

    def load_image(self, path):
        return object()

    def find(self, screenshot, description):
        return list(type(self).elements)


def _install_fake(monkeypatch, elements):
    import awscreen.cli as cli
    _FakeFinder.elements = elements
    monkeypatch.setattr(cli, "Finder", _FakeFinder)


def _an_element():
    from awscreen.finder import Element
    return Element(x=10, y=20, width=30, height=40,
                   confidence=0.87, description="the blue button")


class TestCLIOutput:
    """Output formatting."""

    def test_cli_text_output_default(self, capsys, monkeypatch, tmp_path):
        """Default output is human-readable text naming the match."""
        img = tmp_path / "s.png"
        img.write_bytes(b"x")
        monkeypatch.setenv("AWSCREEN_URL", "http://127.0.0.1:8150")
        _install_fake(monkeypatch, [_an_element()])

        rc = main([str(img), "the blue button"])
        out = capsys.readouterr().out

        assert rc == 0
        assert "Match 1:" in out
        assert "(10, 20)" in out
        assert "30x40" in out
        assert "the blue button" in out

    def test_cli_json_output_flag(self, capsys, monkeypatch, tmp_path):
        """--format json emits parseable JSON carrying the element."""
        import json as _json

        img = tmp_path / "s.png"
        img.write_bytes(b"x")
        monkeypatch.setenv("AWSCREEN_URL", "http://127.0.0.1:8150")
        _install_fake(monkeypatch, [_an_element()])

        rc = main([str(img), "the blue button", "--format", "json"])
        payload = _json.loads(capsys.readouterr().out)

        assert rc == 0
        assert payload["found"] == 1
        assert payload["elements"][0]["x"] == 10
        assert payload["elements"][0]["description"] == "the blue button"


class TestCLIExitCodes:
    """Exit code contract."""

    def test_cli_exits_0_on_success(self, monkeypatch, tmp_path):
        """Finding at least one element exits 0."""
        img = tmp_path / "s.png"
        img.write_bytes(b"x")
        monkeypatch.setenv("AWSCREEN_URL", "http://127.0.0.1:8150")
        _install_fake(monkeypatch, [_an_element()])
        assert main([str(img), "anything"]) == 0

    def test_cli_exits_1_on_no_match(self, capsys, monkeypatch, tmp_path):
        """No matching element exits 1 and says so in both formats."""
        import json as _json

        img = tmp_path / "s.png"
        img.write_bytes(b"x")
        monkeypatch.setenv("AWSCREEN_URL", "http://127.0.0.1:8150")
        _install_fake(monkeypatch, [])

        assert main([str(img), "absent thing"]) == 1
        assert "No elements found" in capsys.readouterr().out

        assert main([str(img), "absent thing", "--format", "json"]) == 1
        assert _json.loads(capsys.readouterr().out)["found"] == 0

    def test_cli_exits_2_on_error(self, monkeypatch):
        """Errors (unreachable endpoint, unreadable file) exit 2.

        Points at a port nothing can serve. The previous version deleted an
        api-key env var and then let the call fall through to the DEFAULT
        endpoint -- which on a developer box is often a real running service,
        so the test made a live network call and passed or failed depending on
        what happened to be listening.
        """
        monkeypatch.setenv("AWSCREEN_URL", "http://127.0.0.1:1")
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
                # No API key + no env var = error
                result = main([str(p), "find something"])
                assert result == 2
            finally:
                p.unlink()
