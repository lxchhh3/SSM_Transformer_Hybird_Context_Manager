"""Packaging contract — the repo must install as a real package.

The deployment story (README "Run it") starts the server on the dev machine;
packaging it means `pip install` + a `ctx-mcp-server` console command, no
PYTHONPATH juggling. These tests pin the contract:
  - pyproject declares a build backend, the `mcp` runtime dep, and the
    console entry point, and ships ONLY the `ctx` package (scripts/ and
    research/ are the archived experiment record, not the product);
  - the entry point target (`ctx.mcp_server:main`) actually exists.
"""

import importlib
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_declares_build_backend(pyproject):
    assert "build-system" in pyproject, "no [build-system] -> not pip-installable"
    assert pyproject["build-system"]["build-backend"]


def test_declares_mcp_runtime_dependency(pyproject):
    deps = pyproject["project"].get("dependencies", [])
    assert any(d.split()[0].startswith("mcp") for d in deps), (
        "the live product is the MCP server; `mcp` must be a runtime dep")


def test_console_entry_points_declared(pyproject):
    scripts = pyproject["project"].get("scripts", {})
    assert scripts.get("ctx-mcp-server") == "ctx.mcp_server:main"
    assert scripts.get("ctx-cc-hook") == "ctx.hooks:main"


def test_wheel_ships_only_ctx_package(pyproject):
    include = (pyproject.get("tool", {}).get("setuptools", {})
               .get("packages", {}).get("find", {}).get("include"))
    assert include == ["ctx"], (
        "wheel must ship exactly the ctx package (scripts/tests/research stay in-repo)")


def test_version_single_source(pyproject):
    import ctx
    assert pyproject["project"]["version"] == ctx.__version__


def test_entry_point_target_exists(monkeypatch, tmp_path):
    pytest.importorskip("mcp")
    # the module builds its ContextService at import time — point it at a temp DB
    monkeypatch.setenv("CTX_DB", str(tmp_path / "pkgtest.db"))
    import ctx.mcp_server as m
    importlib.reload(m)
    assert callable(m.main)
