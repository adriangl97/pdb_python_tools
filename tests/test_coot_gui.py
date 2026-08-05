"""
Tests for the Coot GUI extension and its installer.

The dialog itself needs the GTK that Coot embeds -- PyGTK in Coot 0.9,
PyGObject in Coot 1 -- so what is covered here is everything around it: that
the options the GUI offers are really accepted by the tools, that the extension
file stays loadable and inert outside Coot, that it picks the right GTK and the
right kind of menu for the Coot it is in, and that the installer puts it where
each Coot looks for it.
"""
import ast
import json
import os
import re
import subprocess
import sys
import types

import pytest

from conftest import REPO_ROOT, pdb_atom_line, write_pdb

from pdb_python_tools import coot_extension as extension
from pdb_python_tools import coot_setup

HAS_SCIPY = True
try:
    import scipy  # noqa: F401
except ImportError:
    HAS_SCIPY = False

needs_scipy = pytest.mark.skipif(not HAS_SCIPY, reason="requires scipy")

# The flags the GUI adds to every command line
SHARED_FLAGS = ("--precision", "--format", "--output", "--force", "--coot")

_HELP_CACHE = {}


def tool_help(module):
    """The --help output of one tool, fetched once per module."""
    if module not in _HELP_CACHE:
        result = subprocess.run(
            [sys.executable, "-m", "pdb_python_tools." + module, "--help"],
            cwd=REPO_ROOT, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        _HELP_CACHE[module] = result.stdout
    return _HELP_CACHE[module]


def option_flags(tool):
    """Every flag a tool's options can put on the command line."""
    flags = []
    for option in tool.options:
        if option.kind == "choice":
            flags.extend(flag for _label, flag in option.choices if flag)
        elif option.flag:
            flags.append(option.flag)
    return flags


class TestToolSpecs:
    """The GUI's idea of each tool has to match the tool itself."""

    @pytest.mark.parametrize("tool", extension.TOOLS, ids=lambda t: t.module)
    def test_every_flag_is_accepted_by_the_tool(self, tool):
        help_text = tool_help(tool.module)
        for flag in option_flags(tool):
            assert flag in help_text

    @pytest.mark.parametrize("tool", extension.TOOLS, ids=lambda t: t.module)
    def test_shared_flags_are_accepted_by_the_tool(self, tool):
        help_text = tool_help(tool.module)
        for flag in SHARED_FLAGS:
            assert flag in help_text

    @pytest.mark.parametrize("tool", extension.TOOLS, ids=lambda t: t.module)
    def test_input_count_matches_the_tool(self, tool):
        # The usage line names one positional per input structure
        usage = tool_help(tool.module).split("\n\n")[0]
        expected = 2 if tool.module in ("atom_tracker", "CA_difference") else 1
        assert len(tool.models) == expected
        assert usage.count("pdb") >= expected

    def test_every_tool_has_a_label_and_a_model(self):
        for tool in extension.TOOLS:
            assert tool.label
            assert tool.models

    def test_required_options_are_the_required_ones(self):
        required = {(tool.module, option.flag)
                    for tool in extension.TOOLS for option in tool.options
                    if option.required}
        assert required == {("find_contacts", "-c"), ("find_contacts", "-d")}


class TestExtensionFile:
    """The file runs under Coot 0.9's Python 2, and must be inert outside Coot."""

    def test_importing_it_does_nothing_outside_coot(self, capsys):
        extension._install_menu_quietly()
        assert capsys.readouterr().out == ""

    def test_coot_functions_are_absent_outside_coot(self):
        assert extension._coot_function("set_rotation_centre") is None
        assert extension._coot_function("no_such_coot_function") is None

    def test_no_open_models_outside_coot(self):
        assert extension.open_models() == []

    def test_stays_python_2_compatible(self):
        with open(extension.__file__.replace(".pyc", ".py")) as handle:
            tree = ast.parse(handle.read())
        # f-strings are the easiest Python 3 only syntax to introduce here
        assert not [node for node in ast.walk(tree)
                    if isinstance(node, ast.JoinedStr)]

    def test_environment_that_coot_sets_is_dropped(self):
        for name in ("PYTHONHOME", "PYTHONPATH"):
            assert name in extension._ENV_TO_DROP

    def test_a_module_that_raises_on_import_is_not_fatal(self, tmp_path, monkeypatch):
        # Coot 0.9 without graphics has a coot_utils that raises a NameError on
        # import; looking a function up must survive that
        (tmp_path / "coot_utils.py").write_text("raise NameError('is_protein_chain_p')\n")
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.delitem(sys.modules, "coot_utils", raising=False)
        assert extension._coot_function("molecule_number_list") is None


class TestToolkit:
    """Which GTK the extension talks to depends on which Coot it is in."""

    def test_pygtk_is_used_when_coot_09_embeds_it(self, monkeypatch):
        gtk = types.ModuleType("gtk")
        gtk.pygtk_version = (2, 24, 0)
        monkeypatch.setitem(sys.modules, "gtk", gtk)
        toolkit = extension._make_toolkit()
        assert toolkit.version == "gtk2"
        assert toolkit.gtk is gtk

    def test_pygobjects_stand_in_gtk_is_not_pygtk(self, monkeypatch):
        # PyGObject installs a "gtk" module that imports and then raises on every
        # attribute; taking it for PyGTK is how a Coot 1 run used to break
        monkeypatch.setitem(sys.modules, "gtk", types.ModuleType("gtk"))
        monkeypatch.setitem(sys.modules, "gi", None)
        assert extension._make_toolkit() is None

    def test_no_gtk_at_all(self, monkeypatch):
        for name in ("gtk", "gi"):
            monkeypatch.setitem(sys.modules, name, None)
        assert extension._make_toolkit() is None

    def test_the_toolkit_is_looked_for_once(self, monkeypatch):
        monkeypatch.setattr(extension, "_TOOLKIT", [])
        calls = []
        monkeypatch.setattr(extension, "_make_toolkit",
                            lambda: calls.append(1) or "toolkit")
        assert extension._toolkit() == "toolkit"
        assert extension._toolkit() == "toolkit"
        assert len(calls) == 1


def tool_by_module(module):
    """The GUI's spec for one tool."""
    return next(tool for tool in extension.TOOLS if tool.module == module)


class TestOptionArguments:
    """What the option widgets are turned into on the command line."""

    def test_unticked_check_adds_nothing(self):
        options = [extension.Option("check", "-HET", "Include HETATMs")]
        assert extension.option_arguments(options, [False]) == []

    def test_ticked_check_adds_its_flag(self):
        options = [extension.Option("check", "-HET", "Include HETATMs")]
        assert extension.option_arguments(options, [True]) == ["-HET"]

    def test_number_is_passed_with_its_flag(self):
        options = [extension.Option("float", "-d", "Distance", 4.0)]
        assert extension.option_arguments(options, ["4.5"]) == ["-d", "4.5"]

    def test_empty_optional_number_is_left_out(self):
        options = [extension.Option("float", "-m", "Margin", 0.0)]
        assert extension.option_arguments(options, ["  "]) == []

    def test_empty_required_option_is_reported(self):
        options = [extension.Option("chain", "-c", "Chain", required=True)]
        with pytest.raises(ValueError, match="Chain is required"):
            extension.option_arguments(options, [""])

    def test_unparsable_number_is_reported(self):
        options = [extension.Option("float", "-d", "Distance")]
        with pytest.raises(ValueError, match="Distance"):
            extension.option_arguments(options, ["wide"])

    def test_first_choice_adds_nothing(self):
        options = [tool_by_module("nucleotide_conformation").options[0]]
        assert extension.option_arguments(options, [0]) == []

    def test_later_choice_adds_its_flag(self):
        options = [tool_by_module("nucleotide_conformation").options[0]]
        assert extension.option_arguments(options, [1]) == ["-s"]
        assert extension.option_arguments(options, [2]) == ["-a"]

    def test_chain_combo_value_is_passed(self):
        options = [extension.Option("chain", "-c", "Chain", required=True)]
        assert extension.option_arguments(options, ["4"]) == ["-c", "4"]

    def test_options_keep_their_order(self):
        tool = tool_by_module("find_contacts")
        values = ["A", "4.5", False, True, True, False]
        assert extension.option_arguments(tool.options, values) == [
            "-c", "A", "-d", "4.5", "-a", "-HET"]


class TestBuildCommand:
    def test_runs_the_tool_as_a_module(self):
        argv = extension.build_command(
            tool_by_module("find_contacts"), "/opt/py3", ["/tmp/a.cif"],
            ["-c", "A"], 2, "tsv", "/tmp/t.tsv", "/tmp/s.py")
        assert argv[:3] == ["/opt/py3", "-m", "pdb_python_tools.find_contacts"]

    def test_inputs_come_before_the_options(self):
        argv = extension.build_command(
            tool_by_module("atom_tracker"), "python3", ["/tmp/a.cif", "/tmp/b.cif"],
            ["-HET"], 2, "tsv", "/tmp/t.tsv", "/tmp/s.py")
        assert argv.index("/tmp/a.cif") < argv.index("/tmp/b.cif") < argv.index("-HET")

    def test_writes_both_the_table_and_the_coot_script(self):
        argv = extension.build_command(
            tool_by_module("CA_difference"), "python3", ["/tmp/a.cif", "/tmp/b.cif"],
            [], 3, "csv", "/tmp/t.csv", "/tmp/s.py")
        assert argv[argv.index("-o") + 1] == "/tmp/t.csv"
        assert argv[argv.index("--coot") + 1] == "/tmp/s.py"
        assert argv[argv.index("--precision") + 1] == "3"
        assert argv[argv.index("--format") + 1] == "csv"
        # Re-running must not trip over the previous run's files
        assert "--force" in argv


class TestSubprocessEnvironment:
    def test_coots_python_settings_are_dropped(self, monkeypatch):
        monkeypatch.setenv("PYTHONHOME", "/coot/python2")
        monkeypatch.setenv("PYTHONPATH", "/coot/lib")
        environment = extension.subprocess_environment()
        assert "PYTHONHOME" not in environment
        assert "PYTHONPATH" not in environment

    def test_the_rest_of_the_environment_is_kept(self, monkeypatch):
        monkeypatch.setenv("PATH", "/usr/bin")
        assert extension.subprocess_environment()["PATH"] == "/usr/bin"


@pytest.fixture
def fake_coot(monkeypatch):
    """
    Install fake Coot scripting functions.

    Coot execs the extension in its own namespace, so its functions are found in
    the module's globals; that is what these tests fill in.
    """
    def install(name, function):
        monkeypatch.setitem(extension.__dict__, name, function)
    return install


class TestOpenModels:
    def test_lists_the_open_models(self, fake_coot):
        fake_coot("molecule_number_list", lambda: [0, 1])
        fake_coot("is_valid_model_molecule", lambda imol: True)
        fake_coot("molecule_name", lambda imol: "/data/%d.cif" % imol)
        assert extension.open_models() == [(0, "/data/0.cif"), (1, "/data/1.cif")]

    def test_maps_are_left_out(self, fake_coot):
        fake_coot("molecule_number_list", lambda: [0, 1, 2])
        fake_coot("is_valid_model_molecule", lambda imol: imol != 1)
        fake_coot("molecule_name", lambda imol: "molecule %d" % imol)
        assert [imol for imol, _name in extension.open_models()] == [0, 2]

    def test_falls_back_to_the_molecule_count(self, fake_coot):
        fake_coot("graphics_n_molecules", lambda: 2)
        fake_coot("is_valid_model_molecule", lambda imol: True)
        fake_coot("molecule_name", lambda imol: "")
        assert extension.open_models() == [(0, "molecule 0"), (1, "molecule 1")]


class TestChainIds:
    def test_uses_chain_ids_when_coot_has_it(self, fake_coot):
        fake_coot("chain_ids", lambda imol: ["A", "4", "t"])
        assert extension.chain_ids_of(0) == ["A", "4", "t"]

    def test_falls_back_to_chain_id_per_chain(self, fake_coot):
        fake_coot("n_chains", lambda imol: 2)
        fake_coot("chain_id", lambda imol, i: ["A", "B"][i])
        assert extension.chain_ids_of(0) == ["A", "B"]

    def test_no_chains_without_coot(self):
        assert extension.chain_ids_of(0) == []


class TestExportModel:
    def test_prefers_mmcif(self, fake_coot, tmp_path):
        def write_cif(imol, path):
            with open(path, "w") as handle:
                handle.write("data_test\n")
        fake_coot("write_cif_file", write_cif)
        path = extension.export_model(3, str(tmp_path))
        assert path.endswith("imol_3.cif")
        assert os.path.exists(path)

    def test_falls_back_to_pdb(self, fake_coot, tmp_path):
        def write_nothing(imol, path):
            # Coot versions without a working cif writer leave an empty file
            open(path, "w").close()

        def write_pdb(imol, path):
            with open(path, "w") as handle:
                handle.write("END\n")
        fake_coot("write_cif_file", write_nothing)
        fake_coot("write_pdb_file", write_pdb)
        assert extension.export_model(0, str(tmp_path)).endswith("imol_0.pdb")

    def test_reports_when_coot_cannot_write(self, tmp_path):
        with pytest.raises(ValueError, match="could not write molecule"):
            extension.export_model(0, str(tmp_path))


class TestRunCootScript:
    def test_uses_coots_own_run_script(self, fake_coot, tmp_path):
        opened = []
        fake_coot("run_script", opened.append)
        extension.run_coot_script(str(tmp_path / "coot.py"))
        assert opened == [str(tmp_path / "coot.py")]

    def test_fallback_executes_it_in_coots_namespace(self, tmp_path, monkeypatch):
        """
        Without run_script the file is executed against Coot's main namespace,
        which is where a generated script finds set_rotation_centre.
        """
        import __main__
        centred = []
        monkeypatch.setattr(__main__, "set_rotation_centre",
                            lambda x, y, z: centred.append((x, y, z)), raising=False)
        script = tmp_path / "coot.py"
        script.write_text("set_rotation_centre(1.0, 2.0, 3.0)\nRAN = True\n")
        extension.run_coot_script(str(script))
        assert centred == [(1.0, 2.0, 3.0)]
        # The script's own names must not leak into Coot's namespace
        assert not hasattr(__main__, "RAN")


class TestErrorText:
    def test_short_message_is_kept_whole(self):
        assert extension._tail("error: no such file: a.cif") == \
            "error: no such file: a.cif"

    def test_long_message_keeps_its_end(self):
        text = "x" * 100 + "the actual error"
        tail = extension._tail(text, limit=20)
        assert tail.endswith("the actual error")
        assert len(tail) < len(text)


class TestCountRows:
    def test_counts_data_rows_only(self, tmp_path):
        target = tmp_path / "table.tsv"
        target.write_text("# Syn pyrimidines: 1/2 (50.00%)\nChain\tResidue\nA\t1\nA\t2\n")
        assert extension._count_rows(str(target)) == 2

    def test_header_only_table_has_no_rows(self, tmp_path):
        target = tmp_path / "table.tsv"
        target.write_text("Chain\tResidue\n")
        assert extension._count_rows(str(target)) == 0

    def test_missing_file_has_no_rows(self, tmp_path):
        assert extension._count_rows(str(tmp_path / "nope.tsv")) == 0


class TestConfig:
    @pytest.fixture(autouse=True)
    def config_in_tmp_path(self, tmp_path, monkeypatch):
        """Keep the tests off the real settings file."""
        path = str(tmp_path / "prefs" / extension.CONFIG_NAME)
        monkeypatch.setattr(extension, "CONFIG_PATH", path)
        monkeypatch.delenv("PDB_PYTHON_TOOLS_PYTHON", raising=False)
        return path

    def test_missing_config_is_empty(self):
        assert extension.load_config() == {}

    def test_roundtrip(self):
        extension.save_config({"python": "/opt/py3"})
        assert extension.load_config() == {"python": "/opt/py3"}

    def test_saved_next_to_the_extension(self, config_in_tmp_path):
        # Coot's startup directory need not exist yet
        assert extension.save_config({"python": "/opt/py3"}) == config_in_tmp_path
        assert os.path.exists(config_in_tmp_path)

    def test_corrupt_config_is_ignored(self, config_in_tmp_path):
        os.makedirs(os.path.dirname(config_in_tmp_path))
        with open(config_in_tmp_path, "w") as handle:
            handle.write("not json")
        assert extension.load_config() == {}

    def test_an_unwritable_directory_does_not_raise(self, monkeypatch, tmp_path):
        # Losing the settings must not take a finished run down with it
        blocked = tmp_path / "not-a-dir"
        blocked.write_text("")
        monkeypatch.setattr(extension, "CONFIG_PATH",
                            str(blocked / extension.CONFIG_NAME))
        assert extension.save_config({"python": "/opt/py3"}) is None

    def test_default_python_falls_back_to_python3(self, monkeypatch):
        monkeypatch.setattr(extension, "embedded_interpreter_has_the_tools",
                            lambda: False)
        assert extension.default_python() == "python3"

    def test_default_python_comes_from_the_config(self):
        extension.save_config({"python": "/opt/py3"})
        assert extension.default_python() == "/opt/py3"

    def test_environment_wins_over_the_config(self, monkeypatch):
        extension.save_config({"python": "/opt/py3"})
        monkeypatch.setenv("PDB_PYTHON_TOOLS_PYTHON", "/usr/bin/python3")
        assert extension.default_python() == "/usr/bin/python3"

    def test_coot_1s_own_python_is_offered_when_it_has_the_tools(self, monkeypatch):
        # Coot 1 embeds a Python 3: when the tools are installed in it, it is
        # the obvious default and nothing has to be set up at all
        monkeypatch.setattr(extension, "embedded_interpreter_has_the_tools",
                            lambda: True)
        assert extension.default_python() == sys.executable

    def test_a_recorded_interpreter_wins_over_coots_own(self, monkeypatch):
        monkeypatch.setattr(extension, "embedded_interpreter_has_the_tools",
                            lambda: True)
        extension.save_config({"python": "/opt/py3"})
        assert extension.default_python() == "/opt/py3"


class TestEmbeddedInterpreter:
    @needs_scipy
    def test_this_interpreter_has_the_tools(self):
        # the tests run in exactly the kind of interpreter the dialog looks for
        assert extension.embedded_interpreter_has_the_tools() is True

    def test_a_missing_dependency_rules_it_out(self, monkeypatch):
        import importlib.util
        monkeypatch.setattr(importlib.util, "find_spec",
                            lambda name: None if name == "scipy" else object())
        assert extension.embedded_interpreter_has_the_tools() is False

    def test_the_modules_are_looked_up_not_imported(self, monkeypatch):
        # pulling numpy and scipy into a running Coot to answer a question about
        # the dialog's default would be a poor trade
        import importlib.util
        looked_up = []
        monkeypatch.setattr(importlib.util, "find_spec",
                            lambda name: looked_up.append(name) or object())
        assert extension.embedded_interpreter_has_the_tools() is True
        assert looked_up == ["pdb_python_tools", "numpy", "scipy"]


class TestInstaller:
    def test_installs_under_the_name_coot_loads(self, tmp_path):
        target = coot_setup.install(str(tmp_path / "prefs"))
        assert os.path.basename(target) == "pdb_python_tools.py"
        assert os.path.exists(target)

    def test_installed_copy_matches_the_extension(self, tmp_path):
        target = coot_setup.install(str(tmp_path / "prefs"))
        with open(target) as installed, open(coot_setup.EXTENSION_PATH) as source:
            assert installed.read() == source.read()

    def test_refuses_to_overwrite(self, tmp_path):
        directory = str(tmp_path / "prefs")
        coot_setup.install(directory)
        with pytest.raises(FileExistsError, match="use --force"):
            coot_setup.install(directory)

    def test_force_overwrites(self, tmp_path):
        directory = str(tmp_path / "prefs")
        target = coot_setup.install(directory)
        with open(target, "w") as handle:
            handle.write("stale\n")
        coot_setup.install(directory, force=True)
        with open(target) as handle:
            assert "stale" not in handle.read()

    def test_symlink_points_at_the_extension(self, tmp_path):
        target = coot_setup.install(str(tmp_path / "prefs"), symlink=True)
        assert os.path.realpath(target) == os.path.realpath(coot_setup.EXTENSION_PATH)

    def test_the_settings_go_where_the_extension_reads_them(self):
        # The two files run under different interpreters and cannot share the
        # constant, so they have to agree on it
        assert coot_setup.CONFIG_PATH == extension.CONFIG_PATH
        assert coot_setup.CONFIG_NAME == extension.CONFIG_NAME

    def test_the_settings_stay_out_of_coots_own_directories(self):
        # One settings file has to serve both Coots, which read different
        # directories -- and Coot runs every *.py it finds in them
        for directory in (coot_setup.COOT_09_DIR, coot_setup.COOT_1_DIR):
            assert os.path.dirname(coot_setup.CONFIG_PATH) != directory
        assert not coot_setup.CONFIG_PATH.endswith(".py")

    def test_records_this_interpreter(self, tmp_path):
        config_path = str(tmp_path / "config.json")
        assert coot_setup.record_interpreter(config_path=config_path) == sys.executable
        with open(config_path) as handle:
            assert json.load(handle)["python"] == sys.executable

    def test_creates_the_directory_it_records_into(self, tmp_path):
        config_path = str(tmp_path / "prefs" / coot_setup.CONFIG_NAME)
        coot_setup.record_interpreter("/opt/py3", config_path=config_path)
        assert os.path.exists(config_path)

    def test_recording_keeps_other_settings(self, tmp_path):
        config_path = str(tmp_path / "config.json")
        with open(config_path, "w") as handle:
            json.dump({"other": 1}, handle)
        coot_setup.record_interpreter("/opt/py3", config_path=config_path)
        with open(config_path) as handle:
            assert json.load(handle) == {"other": 1, "python": "/opt/py3"}

    def test_the_extension_ships_with_the_package(self):
        assert os.path.exists(coot_setup.EXTENSION_PATH)

    def test_cli_prints_the_path(self):
        result = subprocess.run(
            [sys.executable, "-m", "pdb_python_tools.coot_setup", "--path"],
            cwd=REPO_ROOT, capture_output=True, text=True)
        assert result.stdout.strip() == coot_setup.EXTENSION_PATH

    def test_cli_needs_something_to_do(self):
        result = subprocess.run(
            [sys.executable, "-m", "pdb_python_tools.coot_setup"],
            cwd=REPO_ROOT, capture_output=True, text=True)
        assert result.returncode != 0
        assert "--install" in result.stderr

    def test_cli_installs_and_records_where_it_is_told(self, tmp_path):
        # XDG_CONFIG_HOME is where the settings go, and where Coot 1 would read
        # the extension from, so this run touches nothing outside tmp_path
        environment = dict(os.environ, XDG_CONFIG_HOME=str(tmp_path / "config"))
        result = subprocess.run(
            [sys.executable, "-m", "pdb_python_tools.coot_setup", "--install",
             "--coot", "1"],
            cwd=REPO_ROOT, capture_output=True, text=True, env=environment)
        assert result.returncode == 0, result.stderr
        # Coot 1 runs the scripts in XDG_CONFIG_HOME itself
        installed = tmp_path / "config" / coot_setup.INSTALLED_NAME
        settings = tmp_path / "config" / "pdb_python_tools" / coot_setup.CONFIG_NAME
        assert installed.exists()
        assert json.loads(settings.read_text())["python"] == sys.executable


class TestInstallDirectories:
    """Which Coot's startup directory the extension is installed into."""

    def test_a_named_coot_gets_its_own_directory(self):
        assert coot_setup.install_directories("0.9") == [coot_setup.COOT_09_DIR]
        assert coot_setup.install_directories("1") == [coot_setup.COOT_1_DIR]
        assert coot_setup.install_directories("both") == [coot_setup.COOT_09_DIR,
                                                          coot_setup.COOT_1_DIR]

    def test_the_directories_are_the_ones_coot_reads(self):
        # Coot 0.9 runs ~/.coot-preferences/*.py, Coot 1 the *.py in its XDG
        # configuration directory
        assert os.path.basename(coot_setup.COOT_09_DIR) == ".coot-preferences"
        assert coot_setup.COOT_1_DIR == coot_setup.coot_1_dir()

    def test_coot_1_falls_back_to_config_coot(self, monkeypatch):
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        assert coot_setup.coot_1_dir() == os.path.expanduser(
            os.path.join("~", ".config", "Coot"))

    def test_coot_1_takes_xdg_config_home_as_it_stands(self, monkeypatch, tmp_path):
        # Coot 1 reads that directory itself: it adds no directory of its own
        # under it, whatever the XDG convention says
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        assert coot_setup.coot_1_dir() == str(tmp_path)

    def test_the_settings_follow_xdg_config_home(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        assert coot_setup._config_home() == str(tmp_path)

    def test_auto_picks_the_coots_that_are_there(self, tmp_path, monkeypatch):
        here = tmp_path / "coot1"
        here.mkdir()
        monkeypatch.setattr(coot_setup, "COOT_09_DIR", str(tmp_path / "nowhere"))
        monkeypatch.setattr(coot_setup, "COOT_1_DIR", str(here))
        assert coot_setup.install_directories() == [str(here)]

    def test_auto_installs_for_both_when_neither_is_there(self, tmp_path, monkeypatch):
        # A directory can be made before the Coot that reads it, so a machine
        # without either is not a machine without Coot
        missing = [str(tmp_path / "none-0.9"), str(tmp_path / "none-1")]
        monkeypatch.setattr(coot_setup, "COOT_09_DIR", missing[0])
        monkeypatch.setattr(coot_setup, "COOT_1_DIR", missing[1])
        assert coot_setup.install_directories() == missing


class TestMenu:
    """
    The menu is built differently in each Coot: 0.9 takes menu items in its menu
    bar, Coot 1 takes a menu button on the toolbar, driven by actions.
    """

    def test_coot_09_gets_a_menu_in_the_menu_bar(self, fake_coot):
        added = []
        fake_coot("coot_menubar_menu", lambda label: "menu:" + label)
        fake_coot("add_simple_coot_menu_menuitem",
                  lambda menu, label, callback: added.append((menu, label)))
        assert extension.add_pdb_python_tools_menu() is True
        assert [label for _menu, label in added] == [t.label for t in extension.TOOLS]
        assert {menu for menu, _label in added} == {"menu:" + extension.MENU_NAME}

    def test_coot_1_gets_a_menu_button_on_the_toolbar(self, fake_coot):
        # Coot 1 has coot_menubar_menu but not add_simple_coot_menu_menuitem, so
        # the menu bar path has to decline rather than half-build a menu
        added = []
        fake_coot("coot_menubar_menu", lambda label: "menu:" + label)
        fake_coot("attach_module_menu_button", lambda name: "gio-menu:" + name)
        fake_coot("add_simple_action_to_menu",
                  lambda menu, label, action, callback: added.append((label, action)))
        assert extension.add_pdb_python_tools_menu() is True
        assert added == [(tool.label, extension._action_name(tool))
                         for tool in extension.TOOLS]

    def test_no_menu_without_a_coot_to_add_it_to(self):
        assert extension.add_pdb_python_tools_menu() is False

    def test_action_names_are_names_gio_accepts(self):
        # Gio only allows alphanumerics, '-' and '.', which rules out the
        # underscores in the module names
        for tool in extension.TOOLS:
            assert re.match(r"^[A-Za-z0-9.-]+$", extension._action_name(tool))
        names = [extension._action_name(tool) for tool in extension.TOOLS]
        assert len(set(names)) == len(names)

    def test_a_menu_entry_opens_its_dialog(self, monkeypatch):
        opened = []
        monkeypatch.setattr(extension, "ToolDialog",
                            lambda tool: types.SimpleNamespace(
                                show=lambda: opened.append(tool)))
        tool = extension.TOOLS[0]
        # Coot 1 calls it with the action and its parameter, Coot 0.9 with none
        extension._tool_callback(tool)("action", None)
        extension._tool_callback(tool)()
        assert opened == [tool, tool]


@pytest.fixture
def pair(tmp_path):
    """Two small aligned structures, as the GUI would export them from Coot."""
    first = write_pdb(tmp_path / "imol_0.pdb", [
        pdb_atom_line(1, "N", "SER", "A", 1, 0.0, 0.0, 0.0),
        pdb_atom_line(2, "CA", "SER", "A", 1, 1.0, 0.0, 0.0),
        pdb_atom_line(3, "OG", "SER", "A", 1, 2.0, 0.0, 0.0),
        pdb_atom_line(4, "CA", "GLY", "B", 1, 0.0, 0.0, 8.0),
    ])
    second = write_pdb(tmp_path / "imol_1.pdb", [
        pdb_atom_line(1, "N", "SER", "A", 1, 0.0, 0.0, 0.0),
        pdb_atom_line(2, "CA", "SER", "A", 1, 1.0, 0.0, 0.0),
        pdb_atom_line(3, "OG", "SER", "A", 1, 5.0, 0.0, 0.0),
        pdb_atom_line(4, "CA", "GLY", "B", 1, 0.0, 0.0, 8.0),
    ])
    return first, second


@needs_scipy
class TestGeneratedCommandLine:
    """The command line the dialog assembles has to work as written."""

    def run_as_the_gui_does(self, module, inputs, extra, tmp_path):
        table = tmp_path / ("%s.tsv" % module)
        script = tmp_path / ("%s_coot.py" % module)
        argv = [sys.executable, "-m", "pdb_python_tools." + module]
        argv.extend(str(path) for path in inputs)
        argv.extend(extra)
        argv.extend(["--precision", "2", "--format", "tsv",
                     "-o", str(table), "--coot", str(script), "--force"])
        result = subprocess.run(argv, cwd=REPO_ROOT, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        return table, script

    def test_atom_tracker(self, pair, tmp_path):
        table, script = self.run_as_the_gui_does(
            "atom_tracker", pair, ["-HET", "--min-change", "0.01"], tmp_path)
        assert "SER" in table.read_text()
        compile(script.read_text(), str(script), "exec")

    def test_find_contacts(self, pair, tmp_path):
        table, script = self.run_as_the_gui_does(
            "find_contacts", pair[:1], ["-c", "A", "-d", "9.0"], tmp_path)
        assert "GLY" in table.read_text()
        compile(script.read_text(), str(script), "exec")

    def test_CA_difference(self, pair, tmp_path):
        table, script = self.run_as_the_gui_does("CA_difference", pair, [], tmp_path)
        assert "SER" in table.read_text()
        compile(script.read_text(), str(script), "exec")

    def test_nucleotide_conformation(self, tmp_path):
        rna = write_pdb(tmp_path / "imol_0.pdb", [
            pdb_atom_line(1, "O4'", "U", "A", 1, 1.0, 1.0, 0.0),
            pdb_atom_line(2, "C1'", "U", "A", 1, 0.0, 0.0, 0.0),
            pdb_atom_line(3, "N1", "U", "A", 1, 1.4, -0.5, 0.0),
            pdb_atom_line(4, "C2", "U", "A", 1, 2.0, 0.4, 0.8),
        ])
        table, script = self.run_as_the_gui_does(
            "nucleotide_conformation", [rna], ["-a", "-m", "5.0"], tmp_path)
        assert "syn" in table.read_text() or "anti" in table.read_text()
        compile(script.read_text(), str(script), "exec")

    def test_rerunning_overwrites_the_previous_results(self, pair, tmp_path):
        # The dialog always passes --force, so a second Run must not fail
        self.run_as_the_gui_does("CA_difference", pair, [], tmp_path)
        self.run_as_the_gui_does("CA_difference", pair, [], tmp_path)

    def test_generated_script_recenters_on_click(self, pair, tmp_path):
        """
        The script Coot opens is what makes a row clickable: without PyGTK it
        falls back to printing, and set_rotation_centre is what a click calls.
        """
        _table, script = self.run_as_the_gui_does("CA_difference", pair, [], tmp_path)
        content = script.read_text()
        assert "set_rotation_centre" in content
        namespace = {"__name__": "not_main"}
        exec(compile(content, str(script), "exec"), namespace)
        assert namespace["MARKERS"]
