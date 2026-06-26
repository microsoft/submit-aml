"""Tests for environment helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from submit_aml.environment import _check_has_patch
from submit_aml.environment import _render_dockerfile
from submit_aml.environment import generate_build_context
from submit_aml.environment import parse_key_value_pairs

# ---------------------------------------------------------------------------
# parse_key_value_pairs
# ---------------------------------------------------------------------------


def test_parse_key_value_pairs_none() -> None:
    """None input returns an empty dict."""
    assert parse_key_value_pairs(None) == {}


def test_parse_key_value_pairs_valid() -> None:
    """A well-formed list produces the expected mapping."""
    result = parse_key_value_pairs(["FOO=bar", "BAZ=qux"])
    assert result == {"FOO": "bar", "BAZ": "qux"}


def test_parse_key_value_pairs_invalid_format_raises() -> None:
    """Items without exactly one '=' raise ValueError."""
    with pytest.raises(ValueError, match="Invalid format"):
        parse_key_value_pairs(["NO_EQUALS_SIGN"])


def test_parse_key_value_pairs_too_many_equals_raises() -> None:
    """Items with more than one '=' raise ValueError."""
    with pytest.raises(ValueError, match="Invalid format"):
        parse_key_value_pairs(["A=B=C"])


# ---------------------------------------------------------------------------
# _check_has_patch
# ---------------------------------------------------------------------------


def test_check_has_patch_with_patch(tmp_path: Path) -> None:
    """No warning is emitted when the file has a full version like '3.12.10'."""
    pv = tmp_path / ".python-version"
    pv.write_text("3.12.10\n")
    # Should not raise; we just verify it completes.
    _check_has_patch(pv)


def test_check_has_patch_without_patch(
    tmp_path: Path, capfd: pytest.CaptureFixture
) -> None:
    """A warning is logged when the patch component is missing."""
    pv = tmp_path / ".python-version"
    pv.write_text("3.12\n")
    # _check_has_patch logs a warning; it should not raise.
    _check_has_patch(pv)


# ---------------------------------------------------------------------------
# _render_dockerfile
# ---------------------------------------------------------------------------


def test_render_dockerfile_substitutes_present_placeholders() -> None:
    """Placeholders present in the template are substituted."""
    template = "FROM {base_docker_image}\n{docker_run}RUN {uv_sync_command}\n"
    rendered = _render_dockerfile(
        template,
        {
            "base_docker_image": "my-image",
            "uv_sync_command": "uv sync",
            "docker_run": "\nRUN apt-get update\n",
        },
    )
    assert rendered == "FROM my-image\n\nRUN apt-get update\nRUN uv sync\n"


def test_render_dockerfile_without_placeholders_is_verbatim() -> None:
    """A Dockerfile without placeholders is returned unchanged."""
    template = 'FROM my-image\nRUN echo hello\nCMD ["bash"]\n'
    rendered = _render_dockerfile(
        template,
        {
            "base_docker_image": "other-image",
            "uv_sync_command": "uv sync",
            "docker_run": "\nRUN apt-get update\n",
        },
    )
    assert rendered == template


def test_render_dockerfile_leaves_unrelated_braces_untouched() -> None:
    """Unrelated braces (e.g. shell `${VAR}`) are not altered."""
    template = "FROM {base_docker_image}\nRUN echo ${HOME} && echo {custom}\n"
    rendered = _render_dockerfile(
        template,
        {
            "base_docker_image": "my-image",
            "uv_sync_command": "uv sync",
            "docker_run": "",
        },
    )
    assert rendered == "FROM my-image\nRUN echo ${HOME} && echo {custom}\n"


def test_render_dockerfile_does_not_rescan_substituted_values() -> None:
    """Placeholders are filled in a single pass; inserted values are not re-rendered.

    Each placeholder is replaced exactly once, scanning only the original
    template. Text introduced by a substitution is never searched for more
    placeholders, so a `{base_docker_image}` token that appears inside the
    docker_run value is left untouched even though `{base_docker_image}` is
    itself a placeholder. This behavior is identical to `str.format()`.
    """
    template = "FROM {base_docker_image}\n{docker_run}RUN {uv_sync_command}\n"
    rendered = _render_dockerfile(
        template,
        {
            "base_docker_image": "ubuntu:22.04",
            "uv_sync_command": "uv sync --frozen",
            # The `{base_docker_image}` token must survive verbatim
            # instead of being expanded to ubuntu.
            "docker_run": "\nRUN echo 'FROM {base_docker_image}' > /Dockerfile.tmpl\n",
        },
    )
    assert rendered == (
        "FROM ubuntu:22.04\n"
        "\nRUN echo 'FROM {base_docker_image}' > /Dockerfile.tmpl\n"
        "RUN uv sync --frozen\n"
    )


# ---------------------------------------------------------------------------
# generate_build_context (custom Dockerfile)
# ---------------------------------------------------------------------------


def _make_project(project_dir: Path) -> Path:
    """Create the minimal env files required by generate_build_context."""
    (project_dir / "pyproject.toml").write_text(
        "[project]\nname = 'demo'\nversion = '0.0.0'\n"
    )
    (project_dir / "uv.lock").write_text("")
    (project_dir / ".python-version").write_text("3.12.10\n")
    return project_dir


def test_generate_build_context_uses_custom_docker_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A custom Dockerfile is rendered and written into the build context."""
    monkeypatch.setattr(
        "submit_aml.environment._check_lock_file_up_to_date", lambda _p: None
    )
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _make_project(project_dir)

    custom = tmp_path / "Custom.Dockerfile"
    custom.write_text("FROM {base_docker_image}\nRUN echo ${HOME}\n")

    context = generate_build_context(
        project_dir,
        base_docker_image="my-image",
        docker_file=custom,
    )

    written = (Path(context.path) / "Dockerfile").read_text()
    assert written == "FROM my-image\nRUN echo ${HOME}\n"


def test_generate_build_context_missing_docker_file_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A nonexistent custom Dockerfile raises FileNotFoundError."""
    monkeypatch.setattr(
        "submit_aml.environment._check_lock_file_up_to_date", lambda _p: None
    )
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _make_project(project_dir)

    missing = tmp_path / "does-not-exist.Dockerfile"
    with pytest.raises(FileNotFoundError, match="Custom Dockerfile not found"):
        generate_build_context(
            project_dir,
            base_docker_image="my-image",
            docker_file=missing,
        )


def test_generate_build_context_uses_default_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without docker_file the bundled template is rendered into the context."""
    monkeypatch.setattr(
        "submit_aml.environment._check_lock_file_up_to_date", lambda _p: None
    )
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _make_project(project_dir)

    context = generate_build_context(project_dir, base_docker_image="my-image")

    written = (Path(context.path) / "Dockerfile").read_text()
    assert "FROM my-image" in written
    assert "{base_docker_image}" not in written
