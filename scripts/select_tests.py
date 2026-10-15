#!/usr/bin/env python3
"""Run only the tests affected by the current changes.

Compares the working tree (staged, unstaged and untracked files) against a
git ref -- HEAD by default -- and selects:

  * library tests (every tests/test_*.py except test_cli.py), as whole
    files, when they import a changed hycline module or any hycline module
    that depends on one (imports are traced statically, including
    imports inside functions, and through package __init__ files);
  * tests/test_cli.py tests one by one, by the commands they run. The CLI
    runs as a subprocess, so its dependencies are traced through cli.py
    itself: a command is affected when its handler -- or a cli.py function
    the handler calls -- imports an affected module, when a changed line of
    cli.py falls in one of those functions, or when its section of
    _build_parser changed. A test is selected when it names an affected
    command (as a string, in its body or the helpers and module constants
    it uses);
  * any changed test file in full;
  * for a changed data or script file outside hycline/ (examples/,
    benchmarking/), the tests whose source mentions its file name.

Documentation (*.md, *.pdf, *.png, references/) selects nothing. Anything
it can't trace -- run_hycline.py, pyproject.toml, requirements files, a
changed line of cli.py outside any function, other unknown files -- selects
the full suite.

Usage:
    python scripts/select_tests.py              # print the selection and why
    python scripts/select_tests.py --run        # run it (in parallel with
                                                # pytest-xdist if installed)
    python scripts/select_tests.py --base origin/main --run -- -x -q
Arguments after "--" are passed to pytest.

This is a fast check while working. It traces Python imports and names,
not behaviour, so it can miss a dependency it can't see statically (e.g.
a file read by a path built at run time). Run the full suite before
pushing to main.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PKG = "hycline"
CLI_FILE = "hycline/cli.py"
CLI_TESTS = "tests/test_cli.py"
DOC_SUFFIXES = {".md", ".pdf", ".png", ".jpg", ".jpeg", ".svg", ".txt"}
DOC_DIRS = ("references/",)
# Changes here can affect any test; no finer tracing is attempted.
FULL_SUITE_FILES = {"run_hycline.py", "pyproject.toml", "requirements.txt",
                    "requirements-bayes.txt", "tests/__init__.py", "tests/conftest.py",
                    "conftest.py", "pytest.ini", "setup.cfg", "tox.ini"}


# ---------------------------------------------------------------------------
# git
# ---------------------------------------------------------------------------

def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True,
                          text=True).stdout


def changed_files(base: str) -> list[str]:
    tracked = git("diff", "--name-only", base).split()
    untracked = git("ls-files", "--others", "--exclude-standard").split()
    return sorted(set(tracked) | set(untracked))


def changed_lines(base: str, path: str) -> set[int] | None:
    """Line numbers (in the working-tree file) touched relative to `base`.
    A pure deletion marks the line it happened at. None means the whole
    file is new."""
    if path in git("ls-files", "--others", "--exclude-standard").split():
        return None
    lines = set()
    for m in re.finditer(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@",
                         git("diff", "-U0", base, "--", path), re.M):
        start, count = int(m.group(1)), int(m.group(2) or 1)
        lines.update(range(start, start + count) if count else [max(start, 1)])
    return lines


# ---------------------------------------------------------------------------
# hycline import graph
# ---------------------------------------------------------------------------

def module_name(path: Path) -> str:
    parts = list(path.relative_to(REPO).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def package_modules() -> dict[str, Path]:
    return {module_name(p): p for p in (REPO / PKG).rglob("*.py")}


def imported_modules(tree: ast.AST, current: str, is_pkg: bool, known: set[str]) -> set[str]:
    """hycline modules imported anywhere in `tree` (including inside
    functions), resolving relative imports and `from pkg import submodule`."""
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base_parts = current.split(".")
                if not is_pkg:
                    base_parts = base_parts[:-1]
                base_parts = base_parts[:len(base_parts) - (node.level - 1)]
                base = ".".join(base_parts + ([node.module] if node.module else []))
            else:
                base = node.module or ""
            names = [base] + [f"{base}.{a.name}" for a in node.names]
        else:
            continue
        for name in names:
            # The longest known module prefix of the dotted name.
            parts = name.split(".")
            for i in range(len(parts), 0, -1):
                cand = ".".join(parts[:i])
                if cand in known:
                    found.add(cand)
                    break
    return found


def dependency_graph(mods: dict[str, Path]) -> dict[str, set[str]]:
    known = set(mods)
    graph = {}
    for name, path in mods.items():
        tree = ast.parse(path.read_text())
        deps = imported_modules(tree, name, path.name == "__init__.py", known)
        graph[name] = deps - {name}
    return graph


def affected_modules(changed: set[str], graph: dict[str, set[str]]) -> set[str]:
    """`changed` plus every module that (transitively) imports one of them."""
    affected = set(changed)
    grew = True
    while grew:
        grew = False
        for mod, deps in graph.items():
            if mod not in affected and deps & affected:
                affected.add(mod)
                grew = True
    return affected


# ---------------------------------------------------------------------------
# cli.py: command -> functions -> modules
# ---------------------------------------------------------------------------

def top_level_functions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def called_names(node: ast.AST) -> set[str]:
    return {n.func.id for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}


def function_span(fn: ast.FunctionDef) -> tuple[int, int]:
    start = min([fn.lineno] + [d.lineno for d in fn.decorator_list])
    return start, fn.end_lineno


def cli_command_handlers(funcs: dict[str, ast.FunctionDef]) -> dict[str, str]:
    """command name -> handler function, from main()'s dispatch chain."""
    handlers = {}
    for node in ast.walk(funcs["main"]):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.comparators[0], ast.Constant)):
            calls = [c for c in called_names(ast.Module(body=node.body, type_ignores=[]))
                     if c in funcs]
            if calls:
                handlers[node.test.comparators[0].value] = calls[0]
    return handlers


def parser_sections(fn: ast.FunctionDef) -> list[tuple[int, str]]:
    """(start line, command) for each subparsers.add_parser("cmd", ...) call
    in _build_parser, in order."""
    starts = []
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_parser" and node.args
                and isinstance(node.args[0], ast.Constant)):
            starts.append((node.lineno, node.args[0].value))
    return sorted(starts)


def affected_commands(base: str, cli_changed: bool, affected: set[str],
                      known: set[str], reasons: list[str]) -> set[str] | str:
    """Commands affected by the change, or "all"."""
    tree = ast.parse((REPO / CLI_FILE).read_text())
    funcs = top_level_functions(tree)
    handlers = cli_command_handlers(funcs)
    all_cmds = set(handlers)

    # Module-level imports in cli.py run for every command.
    module_level = ast.Module(body=[n for n in tree.body if n not in funcs.values()],
                              type_ignores=[])
    if imported_modules(module_level, "hycline.cli", False, known) & affected:
        reasons.append("cli.py imports an affected module at module level -> every command")
        return "all"

    def closure(fname: str) -> set[str]:
        seen, todo = set(), [fname]
        while todo:
            f = todo.pop()
            if f in seen or f not in funcs:
                continue
            seen.add(f)
            todo.extend(called_names(funcs[f]) & set(funcs))
        return seen

    cmd_funcs = {cmd: closure(h) for cmd, h in handlers.items()}
    func_mods = {name: imported_modules(fn, "hycline.cli", False, known)
                 for name, fn in funcs.items()}
    cmds = {cmd for cmd, fs in cmd_funcs.items()
            if any(func_mods[f] & affected for f in fs)}

    if cli_changed:
        lines = changed_lines(base, CLI_FILE)
        if lines is None:
            reasons.append("cli.py is new -> every command")
            return "all"
        sections = parser_sections(funcs["_build_parser"])
        for line in sorted(lines):
            owner = next((name for name, fn in funcs.items()
                          if function_span(fn)[0] <= line <= function_span(fn)[1]), None)
            if owner is None or owner == "main":
                reasons.append(f"cli.py line {line} is outside any command's code -> every command")
                return "all"
            if owner == "_build_parser":
                before = [cmd for start, cmd in sections if start <= line]
                if not before:
                    reasons.append(f"cli.py line {line} is in the shared parser setup -> every command")
                    return "all"
                cmds.add(before[-1])
                continue
            cmds |= {cmd for cmd, fs in cmd_funcs.items() if owner in fs}
    return cmds if cmds != all_cmds else "all"


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def cli_tests_for(commands: set[str] | str, mentions: set[str]) -> tuple[list[str], int]:
    """test_cli.py test functions that name one of `commands` or mention one
    of the `mentions` file names, counting the helpers and module-level
    constants each test uses. Returns (selected names, total tests)."""
    source = (REPO / CLI_TESTS).read_text()
    tree = ast.parse(source)
    funcs = top_level_functions(tree)
    constants = {t.id: node for node in tree.body if isinstance(node, ast.Assign)
                 for t in node.targets if isinstance(t, ast.Name)}

    def reachable_source(name: str) -> str:
        seen, todo, parts = set(), [name], []
        while todo:
            n = todo.pop()
            if n in seen:
                continue
            seen.add(n)
            node = funcs.get(n) or constants.get(n)
            if node is None:
                continue
            parts.append(ast.get_source_segment(source, node) or "")
            todo.extend(x.id for x in ast.walk(node) if isinstance(x, ast.Name))
        return "\n".join(parts)

    tests = [n for n in funcs if n.startswith("test_")]
    if commands == "all":
        return tests, len(tests)
    selected = []
    for name in tests:
        text = reachable_source(name)
        strings = {x.value for x in ast.walk(ast.parse(text)) if isinstance(x, ast.Constant)
                   and isinstance(x.value, str)}
        if strings & commands or any(m in text for m in mentions):
            selected.append(name)
    return selected, len(tests)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default="HEAD",
                    help="Git ref to compare the working tree against (default: HEAD, i.e. "
                         "uncommitted changes; e.g. origin/main to include local commits).")
    ap.add_argument("--run", action="store_true", help="Run the selected tests.")
    ap.add_argument("pytest_args", nargs="*", help="Arguments passed to pytest (after --).")
    args = ap.parse_args(argv)

    files = changed_files(args.base)
    reasons: list[str] = []
    full = False
    mods = package_modules()
    known = set(mods)
    changed_mods, test_files, mentions = set(), set(), set()
    cli_changed = False

    for f in files:
        p = Path(f)
        if f in FULL_SUITE_FILES:
            full = True
            reasons.append(f"{f} -> full suite")
        elif f.startswith(f"{PKG}/") and p.suffix == ".py":
            if (REPO / f).exists():
                changed_mods.add(module_name(REPO / p))
                cli_changed |= f == CLI_FILE
            else:
                full = True
                reasons.append(f"{f} was deleted -> full suite")
        elif f.startswith("tests/") and p.name.startswith("test_") and p.suffix == ".py":
            if (REPO / f).exists():
                test_files.add(f)
                reasons.append(f"{f} changed -> whole file")
        elif p.suffix.lower() in DOC_SUFFIXES or f.startswith(DOC_DIRS):
            continue
        elif f.startswith(("examples/", "benchmarking/", "scripts/")):
            mentions.update({p.name, p.stem})
        else:
            full = True
            reasons.append(f"{f}: can't trace -> full suite")

    all_test_files = sorted(str(p.relative_to(REPO)) for p in (REPO / "tests").glob("test_*.py"))
    if full:
        selection = ["tests"]
    else:
        graph = dependency_graph(mods)
        affected = affected_modules(changed_mods, graph)
        if changed_mods:
            reasons.append("changed modules: " + ", ".join(sorted(changed_mods)))
            reasons.append(f"affected (they or their importers): {len(affected)} modules")
        if mentions:
            reasons.append("tests mentioning: " + ", ".join(sorted(mentions)))
        selection = []
        for tf in all_test_files:
            if tf == CLI_TESTS:
                continue
            text = (REPO / tf).read_text()
            imports = imported_modules(ast.parse(text), "tests", False, known)
            if tf in test_files or imports & affected or any(m in text for m in mentions):
                selection.append(tf)
        if CLI_TESTS in test_files:
            selection.append(CLI_TESTS)
        elif changed_mods or mentions:
            cmds = affected_commands(args.base, cli_changed, affected, known, reasons)
            if cmds and cmds != "all":
                reasons.append("affected commands: " + ", ".join(sorted(cmds)))
            names, total = cli_tests_for(cmds or set(), mentions)
            if len(names) == total:
                selection.append(CLI_TESTS)
            else:
                selection += [f"{CLI_TESTS}::{n}" for n in names]
            reasons.append(f"{CLI_TESTS}: {len(names)} of {total} tests")

    print(f"Changed files vs {args.base}: {len(files)}")
    for r in reasons:
        print(f"  {r}")
    if not selection:
        print("No tests selected.")
        return 0
    files_selected = sorted({s.split("::")[0] for s in selection})
    print("Selected: " + ("full suite" if selection == ["tests"] else
                          f"{len(selection)} item(s) in {len(files_selected)} file(s)"))
    for s in selection:
        print(f"  {s}")
    if not args.run:
        return 0

    cmd = [sys.executable, "-m", "pytest", *selection]
    try:
        import xdist  # noqa: F401
        if not any(a == "-n" or a.startswith("-n") for a in args.pytest_args):
            cmd += ["-n", "auto"]
    except ImportError:
        pass
    cmd += args.pytest_args
    print("Running: pytest <selection> " + " ".join(cmd[3 + len(selection):]), flush=True)
    return subprocess.run(cmd, cwd=REPO).returncode


if __name__ == "__main__":
    sys.exit(main())
