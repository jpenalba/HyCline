"""scripts/select_tests.py: the static tracing it selects tests with."""
import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import select_tests as st  # noqa: E402

CLI_COMMANDS = {
    "fit-geographic", "compare-tails", "fit-trait", "fit-trait-multi", "fit-genomic",
    "fit-hierarchical-geographic", "fit-hierarchical-genomic", "coupling-geographic",
    "coupling-genomic", "estimate-hybrid-index", "generate-geographic-input",
    "generate-genomic-input", "filter-aims",
}


def _cli_funcs():
    return st.top_level_functions(ast.parse((REPO_ROOT / st.CLI_FILE).read_text()))


def test_every_cli_command_has_a_handler_and_a_parser_section():
    funcs = _cli_funcs()
    assert set(st.cli_command_handlers(funcs)) == CLI_COMMANDS
    assert {cmd for _, cmd in st.parser_sections(funcs["_build_parser"])} == CLI_COMMANDS


def test_import_graph_follows_function_level_imports():
    mods = st.package_modules()
    graph = st.dependency_graph(mods)
    # filter_aims imports beagle only inside filter_aims_beagle.
    assert "hycline.data.beagle" in graph["hycline.data.filter_aims"]
    affected = st.affected_modules({"hycline.data.beagle"}, graph)
    assert {"hycline.data.filter_aims", "hycline.cli"} <= affected
    assert "hycline.models.cline" not in affected


def test_relative_imports_resolve():
    tree = ast.parse("from .beagle import read_beagle\nfrom .. import cli\n")
    known = {"hycline", "hycline.cli", "hycline.data", "hycline.data.beagle"}
    assert st.imported_modules(tree, "hycline.data.vcf", False, known) == {
        "hycline.data.beagle", "hycline.cli", "hycline"}


def test_cli_tests_selected_by_command_name():
    names, total = st.cli_tests_for({"filter-aims"}, set())
    assert 0 < len(names) < total
    assert "test_cli_filter_aims_beagle_auto_detect_and_filter" in names
    assert "test_cli_help_runs_without_error" not in names
    # A file name reached only through a module-level constant
    # (MANAKIN_CSV = REPO_ROOT / "examples" / "manakin_ada.csv").
    names, _ = st.cli_tests_for(set(), {"manakin_ada"})
    source = (REPO_ROOT / st.CLI_TESTS).read_text()
    assert names
    assert all("MANAKIN_CSV" in source.split(f"def {n}(")[1].split("\ndef ")[0]
               for n in names)
