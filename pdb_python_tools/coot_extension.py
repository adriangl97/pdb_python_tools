#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pdb_python_tools inside the Coot GUI.

Loading this file into Coot (0.9) adds a "pdb_python_tools" menu to the menu
bar with one entry per tool. Each entry opens a dialog where the input
structures are picked from the models already open in Coot, the tool's options
are filled in, and "Run" starts the tool. The results come back as the
clickable list the tools already write with --coot, opened automatically once
the run finishes: clicking a row recenters the view on that residue.

The tools are Python 3 and need numpy/scipy, while Coot 0.9 embeds Python 2, so
they are run as a subprocess under an external Python 3 interpreter. That
interpreter is taken from, in order: the entry in the dialog, the
PDB_PYTHON_TOOLS_PYTHON environment variable, the one recorded next to this
file by "pdb_python_tools.coot_setup --install", and finally plain "python3"
from PATH.

Install with "pdb_python_tools.coot_setup --install", which copies this file
into ~/.coot-preferences where Coot loads it at startup, or load it once from
Coot with Calculate -> Run Script...

"""
from __future__ import print_function

import json
import os
import subprocess
import tempfile

MENU_NAME = "pdb_python_tools"

# Written by "pdb_python_tools.coot_setup --install". It sits next to this file
# in Coot's startup directory: Coot only runs the *.py and *.scm in there, so a
# settings file is left alone.
COOT_PREFERENCES_DIR = os.path.expanduser(os.path.join("~", ".coot-preferences"))
CONFIG_NAME = "pdb_python_tools_coot.json"
CONFIG_PATH = os.path.join(COOT_PREFERENCES_DIR, CONFIG_NAME)

# Coot points these at its own bundled Python 2 and libraries. Leaving them in
# place would make the external Python 3 load Coot's standard library instead of
# its own, so they are dropped from the subprocess environment.
_ENV_TO_DROP = ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONEXECUTABLE",
                "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH")

# Where Coot keeps its scripting functions. 
_COOT_MODULES = ("coot", "coot_utils", "coot_gui")


# ---------------------------------------------------------------------------
# Talking to Coot
# ---------------------------------------------------------------------------

def _coot_function(name):
    """
    Return a Coot scripting function by name, or None when Coot does not have it
    """
    function = globals().get(name)
    if function is not None:
        return function
    for module_name in _COOT_MODULES:
        try:
            module = __import__(module_name)
        except ImportError:
            continue
        function = getattr(module, name, None)
        if function is not None:
            return function
    import __main__
    return getattr(__main__, name, None)


def open_models():
    """
    Every model molecule currently open in Coot, as a list of (imol, name).

    Maps and other non-model molecules are left out.
    """
    is_model = _coot_function("is_valid_model_molecule")
    molecule_name = _coot_function("molecule_name")
    number_list = _coot_function("molecule_number_list")
    if number_list is not None:
        candidates = number_list()
    else:
        n_molecules = _coot_function("graphics_n_molecules")
        candidates = list(range(n_molecules())) if n_molecules is not None else []
    models = []
    for imol in candidates:
        if is_model is not None and not is_model(imol):
            continue
        name = molecule_name(imol) if molecule_name is not None else ""
        models.append((imol, name or "molecule %d" % imol))
    return models


def chain_ids_of(imol):
    """The chain ids of a model molecule, or [] when they cannot be read."""
    chain_ids = _coot_function("chain_ids")
    if chain_ids is not None:
        try:
            return list(chain_ids(imol))
        except Exception:
            pass
    n_chains = _coot_function("n_chains")
    chain_id = _coot_function("chain_id")
    if n_chains is None or chain_id is None:
        return []
    return [chain_id(imol, i) for i in range(n_chains(imol))]


def export_model(imol, directory):
    """
    Write the current coordinates of molecule `imol` into `directory` and
    return the path.
    """
    attempts = [("write_cif_file", "imol_%d.cif" % imol),
                ("write_pdb_file", "imol_%d.pdb" % imol)]
    for function_name, file_name in attempts:
        write = _coot_function(function_name)
        if write is None:
            continue
        path = os.path.join(directory, file_name)
        try:
            write(imol, path)
        except Exception:
            continue
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
    raise ValueError("Coot could not write molecule %d to a file" % imol)


def run_coot_script(path):
    """
    Run a generated Coot script inside this Coot session.

    Coot's own run_script() is used when available. The fallback executes the
    file against a copy of Coot's main namespace, so the script still finds
    set_rotation_centre without leaking its own names into that namespace.
    """
    run_script = _coot_function("run_script")
    if run_script is not None:
        run_script(path)
        return
    import __main__
    namespace = dict(__main__.__dict__)
    namespace["__file__"] = path
    handle = open(path)
    try:
        source = handle.read()
    finally:
        handle.close()
    exec(compile(source, path, "exec"), namespace)


# ---------------------------------------------------------------------------
# Stored settings
# ---------------------------------------------------------------------------

def load_config():
    """The saved settings, or {} when there is no readable config file."""
    try:
        handle = open(CONFIG_PATH)
    except (IOError, OSError):
        return {}
    try:
        config = json.load(handle)
    except ValueError:
        return {}
    finally:
        handle.close()
    return config if isinstance(config, dict) else {}


def save_config(config):
    """
    Write the settings into Coot's startup directory and return the path.

    An unwritable directory is not worth interrupting a run for, so it returns
    None instead of raising.
    """
    directory = os.path.dirname(CONFIG_PATH)
    if not os.path.isdir(directory):
        try:
            os.makedirs(directory)
        except OSError:
            return None
    try:
        handle = open(CONFIG_PATH, "w")
    except (IOError, OSError):
        return None
    try:
        json.dump(config, handle, indent=1, sort_keys=True)
    finally:
        handle.close()
    return CONFIG_PATH


def default_python():
    """The Python 3 interpreter to run the tools with."""
    from_env = os.environ.get("PDB_PYTHON_TOOLS_PYTHON")
    if from_env:
        return from_env
    return load_config().get("python") or "python3"


# ---------------------------------------------------------------------------
# The tools and their options
# ---------------------------------------------------------------------------

class Option(object):
    """
    One command-line option of a tool, and the widget that fills it in.

    kind is one of:
      "check"  a flag passed on its own when the box is ticked
      "float"  an entry passed as "flag value" when it is not empty
      "int"    the same, but the value has to be a whole number
      "chain"  a combo of the chain ids of the selected model
      "choice" a combo of (label, flag) pairs; the flag is passed when non-empty
    """

    def __init__(self, kind, flag, label, default=None, choices=None,
                 required=False, tooltip=""):
        self.kind = kind
        self.flag = flag
        self.label = label
        self.default = default
        self.choices = choices or []
        self.required = required
        self.tooltip = tooltip


class Tool(object):
    """One of the command-line tools, as it is presented in the GUI."""

    def __init__(self, module, label, models, options, tooltip=""):
        self.module = module
        self.label = label
        # One entry per input structure: the label shown next to its combo
        self.models = models
        self.options = options
        self.tooltip = tooltip


TOOLS = [
    Tool("atom_tracker", "Atom tracker",
         ["Model", "Compared with"],
         [Option("check", "-HET", "Include HETATMs"),
          Option("check", "-hy", "Include hydrogens"),
          Option("float", "--min-change", "Minimum displacement", 0.01,
                 tooltip="only report residues that moved more than this")],
         tooltip="Per-residue coordinate change between two equivalent, "
                 "pre-aligned models"),
    Tool("find_contacts", "Find contacts",
         ["Model"],
         [Option("chain", "-c", "Chain", required=True),
          Option("float", "-d", "Distance", 4.0, required=True,
                 tooltip="contact cutoff in Angstrom"),
          Option("check", "-p", "Polar atoms only"),
          Option("check", "-a", "All atom pairs",
                 tooltip="off: only the shortest contact per residue pair"),
          Option("check", "-HET", "Include HETATMs"),
          Option("check", "-hy", "Include hydrogens")],
         tooltip="Contacts of one chain with every other chain, within a cutoff"),
    Tool("CA_difference", "CA difference",
         ["Model", "Compared with"],
         [Option("check", "-HET", "Include HETATMs"),
          Option("check", "-hy", "Include hydrogens")],
         tooltip="Nearest CA/C1' distance in a second model for every residue; "
                 "the models need not be equivalent"),
    Tool("nucleotide_conformation", "Nucleotide conformation",
         ["Model"],
         [Option("choice", "", "Report",
                 choices=[("Syn pyrimidines", ""),
                          ("All syn nucleotides", "-s"),
                          ("Every nucleotide", "-a")]),
          Option("float", "-m", "Borderline margin", 0.0,
                 tooltip="degrees around the +/-90 syn/anti boundary to flag "
                         "as borderline (0 = off)")],
         tooltip="Glycosidic syn/anti conformation of RNA and DNA nucleotides"),
]


# ---------------------------------------------------------------------------
# Building the command line
# ---------------------------------------------------------------------------

def option_arguments(options, values):
    """
    Turn the values collected from the dialog into command-line arguments.

    `values` is parallel to `options`: a bool for a "check", the index of the
    chosen row for a "choice", and the text typed or picked for the rest.

    Raises ValueError, with a message meant for the dialog, when a required
    option was left empty or a number does not parse.
    """
    args = []
    for option, value in zip(options, values):
        if option.kind == "check":
            if value:
                args.append(option.flag)
            continue
        if option.kind == "choice":
            flag = option.choices[value][1] if 0 <= value < len(option.choices) else ""
            if flag:
                args.append(flag)
            continue
        text = (value or "").strip()
        if not text:
            if option.required:
                raise ValueError("%s is required." % option.label)
            continue
        if option.kind in ("float", "int"):
            converter = float if option.kind == "float" else int
            try:
                converter(text)
            except ValueError:
                raise ValueError("%s: '%s' is not a number." % (option.label, text))
        args.extend([option.flag, text])
    return args


def build_command(tool, python, inputs, option_args, precision, fmt, table, script):
    """
    The full command line for one run.

    The table goes to `table` and the clickable Coot script to `script`, both
    with --force: the script lives in a fresh temporary directory, and an
    existing table has already been confirmed by the dialog.
    """
    argv = [python, "-m", "pdb_python_tools." + tool.module]
    argv.extend(inputs)
    argv.extend(option_args)
    argv.extend(["--precision", str(precision), "--format", fmt,
                 "-o", table, "--coot", script, "--force"])
    return argv


def subprocess_environment():
    """The environment to run the tools in, without Coot's Python 2 settings."""
    environment = dict(os.environ)
    for name in _ENV_TO_DROP:
        environment.pop(name, None)
    return environment


# ---------------------------------------------------------------------------
# GTK helpers
# ---------------------------------------------------------------------------

def _gtk():
    """Coot 0.9's PyGTK module, or None when it cannot be imported."""
    try:
        import gtk
    except ImportError:
        return None
    return gtk


def _timeout_add(milliseconds, function):
    """Call `function` every `milliseconds` from the GTK main loop."""
    try:
        import gobject
        return gobject.timeout_add(milliseconds, function)
    except ImportError:
        pass
    gtk = _gtk()
    return gtk.timeout_add(milliseconds, function)


def _combo_text(combo):
    """The text of the active row of a text combo, or "" when nothing is active."""
    active = combo.get_active()
    if active < 0:
        return ""
    return combo.get_model()[active][0]


def _set_tooltip(widget, text):
    """Set a tooltip when the GTK version in use supports it."""
    if text and hasattr(widget, "set_tooltip_text"):
        widget.set_tooltip_text(text)


def _labelled_row(gtk, label_text, *widgets):
    """An hbox of a fixed-width label followed by widgets."""
    box = gtk.HBox(False, 6)
    label = gtk.Label(label_text)
    label.set_alignment(0, 0.5)
    label.set_size_request(140, -1)
    box.pack_start(label, False, False, 0)
    for index, widget in enumerate(widgets):
        expand = index == 0
        box.pack_start(widget, expand, expand, 0)
    return box


def _message(parent, kind, text):
    """Show a modal message dialog and wait for it to be dismissed."""
    gtk = _gtk()
    types = {"error": gtk.MESSAGE_ERROR, "info": gtk.MESSAGE_INFO,
             "question": gtk.MESSAGE_QUESTION}
    buttons = gtk.BUTTONS_OK_CANCEL if kind == "question" else gtk.BUTTONS_CLOSE
    dialog = gtk.MessageDialog(parent, gtk.DIALOG_MODAL, types[kind], buttons)
    # Set as a property rather than through the constructor: the message is a
    # path or a tool's error output, not a format string
    dialog.set_property("text", text)
    response = dialog.run()
    dialog.destroy()
    return response == gtk.RESPONSE_OK


# ---------------------------------------------------------------------------
# The tool dialog
# ---------------------------------------------------------------------------

class ToolDialog(object):
    """The dialog for one tool: inputs from the open models, options, Run."""

    def __init__(self, tool):
        self.tool = tool
        self.gtk = _gtk()
        self.models = []
        self.model_combos = []
        self.option_widgets = []
        self.chain_combos = []
        self.process = None
        self.run_state = None

    # -- construction ------------------------------------------------------

    def show(self):
        gtk = self.gtk
        if gtk is None:
            print("pdb_python_tools: no PyGTK available, cannot open the dialog")
            return
        self.window = gtk.Window(gtk.WINDOW_TOPLEVEL)
        self.window.set_title("pdb_python_tools: %s" % self.tool.label)
        self.window.set_default_size(460, -1)
        self.window.connect("destroy", self._on_destroy)

        outer = gtk.VBox(False, 4)
        outer.set_border_width(8)

        if self.tool.tooltip:
            heading = gtk.Label(self.tool.tooltip)
            heading.set_alignment(0, 0.5)
            heading.set_line_wrap(True)
            outer.pack_start(heading, False, False, 2)

        for index, label in enumerate(self.tool.models):
            combo = gtk.combo_box_new_text()
            self.model_combos.append(combo)
            if index == 0:
                refresh = gtk.Button("Refresh")
                _set_tooltip(refresh, "re-read the list of open models")
                refresh.connect("clicked", self._on_refresh)
                outer.pack_start(_labelled_row(gtk, label, combo, refresh),
                                 False, False, 1)
            else:
                outer.pack_start(_labelled_row(gtk, label, combo), False, False, 1)
        self.model_combos[0].connect("changed", self._on_model_changed)

        outer.pack_start(gtk.HSeparator(), False, False, 4)
        for option in self.tool.options:
            outer.pack_start(self._option_row(option), False, False, 1)

        outer.pack_start(gtk.HSeparator(), False, False, 4)
        self.precision_entry = gtk.Entry()
        self.precision_entry.set_width_chars(4)
        self.precision_entry.set_text("2")
        _set_tooltip(self.precision_entry,
                     "decimal places for the reported distances or angles")
        self.format_combo = gtk.combo_box_new_text()
        for fmt in ("tsv", "csv"):
            self.format_combo.append_text(fmt)
        self.format_combo.set_active(0)
        outer.pack_start(_labelled_row(gtk, "Precision / format",
                                       self.precision_entry, self.format_combo),
                         False, False, 1)

        self.output_entry = gtk.Entry()
        _set_tooltip(self.output_entry,
                     "where to keep the table; left empty it goes to a "
                     "temporary file")
        browse = gtk.Button("Browse...")
        browse.connect("clicked", self._on_browse)
        outer.pack_start(_labelled_row(gtk, "Save table to",
                                       self.output_entry, browse),
                         False, False, 1)

        self.open_check = gtk.CheckButton("Open the results in Coot when the run finishes")
        self.open_check.set_active(True)
        outer.pack_start(self.open_check, False, False, 1)

        self.python_entry = gtk.Entry()
        self.python_entry.set_text(default_python())
        _set_tooltip(self.python_entry,
                     "Python 3 interpreter that has pdb_python_tools installed")
        outer.pack_start(_labelled_row(gtk, "Python 3", self.python_entry),
                         False, False, 1)

        outer.pack_start(gtk.HSeparator(), False, False, 4)
        self.status_label = gtk.Label("")
        self.status_label.set_alignment(0, 0.5)
        self.status_label.set_line_wrap(True)
        outer.pack_start(self.status_label, False, False, 2)

        buttons = gtk.HBox(True, 6)
        self.run_button = gtk.Button("Run")
        self.run_button.connect("clicked", self._on_run)
        close_button = gtk.Button("Close")
        close_button.connect("clicked", lambda *args: self.window.destroy())
        buttons.pack_start(self.run_button, True, True, 0)
        buttons.pack_start(close_button, True, True, 0)
        outer.pack_start(buttons, False, False, 2)

        self.window.add(outer)
        self._on_refresh()
        self.window.show_all()

    def _option_row(self, option):
        """The widget row for one option, remembering the widget it holds."""
        gtk = self.gtk
        if option.kind == "check":
            widget = gtk.CheckButton(option.label)
            widget.set_active(bool(option.default))
            row = widget
        elif option.kind == "choice":
            widget = gtk.combo_box_new_text()
            for label, _flag in option.choices:
                widget.append_text(label)
            widget.set_active(0)
            row = _labelled_row(gtk, option.label, widget)
        elif option.kind == "chain":
            widget = gtk.combo_box_new_text()
            self.chain_combos.append(widget)
            row = _labelled_row(gtk, option.label, widget)
        else:
            widget = gtk.Entry()
            widget.set_width_chars(8)
            if option.default is not None:
                widget.set_text(str(option.default))
            row = _labelled_row(gtk, option.label, widget)
        _set_tooltip(widget, option.tooltip)
        self.option_widgets.append((option, widget))
        return row

    # -- keeping the model lists current -----------------------------------

    def _on_refresh(self, *args):
        """Re-read the open models and repopulate the combos."""
        self.models = open_models()
        for index, combo in enumerate(self.model_combos):
            previous = combo.get_active()
            combo.get_model().clear()
            for imol, name in self.models:
                combo.append_text("%d: %s" % (imol, os.path.basename(name)))
            if not self.models:
                continue
            # Default the second input to a different model than the first
            if previous < 0 or previous >= len(self.models):
                previous = min(index, len(self.models) - 1)
            combo.set_active(previous)
        self._on_model_changed()
        if not self.models:
            self._set_status("No model open in Coot.")

    def _on_model_changed(self, *args):
        """Refill the chain combo from the first selected model."""
        if not self.chain_combos:
            return
        imol = self._selected_imol(0)
        chains = chain_ids_of(imol) if imol is not None else []
        for combo in self.chain_combos:
            previous = _combo_text(combo)
            combo.get_model().clear()
            for chain in chains:
                combo.append_text(chain)
            if previous in chains:
                combo.set_active(chains.index(previous))
            elif chains:
                combo.set_active(0)

    def _selected_imol(self, index):
        """The molecule number chosen in combo `index`, or None."""
        active = self.model_combos[index].get_active()
        if active < 0 or active >= len(self.models):
            return None
        return self.models[active][0]

    # -- assembling the command line ---------------------------------------

    def _widget_value(self, option, widget):
        """What the user put into one option's widget."""
        if option.kind in ("check", "choice"):
            return widget.get_active()
        if option.kind == "chain":
            return _combo_text(widget)
        return widget.get_text()

    def _option_args(self):
        """The command-line arguments for the option widgets."""
        values = [self._widget_value(option, widget)
                  for option, widget in self.option_widgets]
        return option_arguments(self.tool.options, values)

    def _prepare(self):
        """
        Everything the run needs: the argv, the working directory and the
        paths the tool will write to.
        """
        python = self.python_entry.get_text().strip() or "python3"
        precision = self.precision_entry.get_text().strip() or "2"
        try:
            int(precision)
        except ValueError:
            raise ValueError("Precision: '%s' is not a whole number." % precision)
        imols = []
        for index, label in enumerate(self.tool.models):
            imol = self._selected_imol(index)
            if imol is None:
                raise ValueError("Pick a model for '%s'." % label)
            imols.append(imol)
        option_args = self._option_args()

        fmt = _combo_text(self.format_combo) or "tsv"
        output = self.output_entry.get_text().strip()
        # --force is passed for the sake of the temporary script, so an existing
        # table the user named is confirmed here instead
        if output and os.path.exists(output):
            if not _message(self.window, "question",
                            "%s already exists. Overwrite it?" % output):
                raise ValueError("Cancelled.")

        work_dir = tempfile.mkdtemp(prefix="pdb_python_tools_")
        table_path = output or os.path.join(work_dir, "%s.%s" % (self.tool.module, fmt))
        script_path = os.path.join(work_dir, "%s_coot.py" % self.tool.module)
        inputs = [export_model(imol, work_dir) for imol in imols]
        argv = build_command(self.tool, python, inputs, option_args, precision,
                             fmt, table_path, script_path)
        return {"argv": argv, "work_dir": work_dir, "inputs": inputs,
                "table": table_path, "script": script_path, "python": python}

    # -- running -----------------------------------------------------------

    def _on_run(self, *args):
        if self.process is not None:
            return
        try:
            state = self._prepare()
        except ValueError as exc:
            self._set_status(str(exc))
            return
        except Exception as exc:
            self._error("Could not start the run:\n\n%s" % exc)
            return

        state["stdout"] = open(os.path.join(state["work_dir"], "stdout.txt"), "w+")
        state["stderr"] = open(os.path.join(state["work_dir"], "stderr.txt"), "w+")
        environment = subprocess_environment()
        try:
            state["process"] = subprocess.Popen(
                state["argv"], stdout=state["stdout"], stderr=state["stderr"],
                cwd=state["work_dir"], env=environment)
        except OSError as exc:
            self._error("Could not run %s:\n\n%s\n\nSet the Python 3 interpreter "
                        "that has pdb_python_tools installed." % (state["python"], exc))
            return
        self.run_state = state
        self.process = state["process"]
        self.run_button.set_sensitive(False)
        self._set_status("Running %s..." % self.tool.module)
        _timeout_add(200, self._poll)

    def _poll(self):
        """Check on the subprocess from the GTK main loop."""
        if self.process is None:
            return False
        if self.process.poll() is None:
            return True
        self._finished()
        return False

    def _finished(self):
        """Report the result and, when asked to, open the generated script."""
        state = self.run_state
        returncode = self.process.returncode
        self.process = None
        self.run_state = None
        self.run_button.set_sensitive(True)
        errors = self._read_back(state["stderr"])
        output = self._read_back(state["stdout"])
        # The exported copies are only needed while the tool reads them
        for path in state["inputs"]:
            try:
                os.remove(path)
            except OSError:
                pass

        if returncode != 0:
            # The tools report a bad file or a bad argument on stderr
            message = errors.strip() or output.strip() or "No error message."
            self._error("%s failed (exit status %d).\n\n%s"
                        % (self.tool.module, returncode, _tail(message)))
            return

        rows = _count_rows(state["table"])
        self._set_status("%s: %d row(s). Table: %s"
                         % (self.tool.module, rows, state["table"]))
        # Remember an interpreter that worked
        config = load_config()
        if config.get("python") != state["python"]:
            config["python"] = state["python"]
            save_config(config)

        if self.open_check.get_active():
            try:
                run_coot_script(state["script"])
            except Exception as exc:
                self._error("The results were written to %s but Coot could not "
                            "open them:\n\n%s" % (state["script"], exc))

    def _read_back(self, handle):
        """Read a subprocess output file back from the start and close it."""
        try:
            handle.seek(0)
            text = handle.read()
        except (IOError, OSError, ValueError):
            text = ""
        try:
            handle.close()
        except (IOError, OSError):
            pass
        return text

    # -- small helpers -----------------------------------------------------

    def _on_browse(self, *args):
        gtk = self.gtk
        chooser = gtk.FileChooserDialog(
            "Save the table as", self.window, gtk.FILE_CHOOSER_ACTION_SAVE,
            (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, gtk.STOCK_SAVE, gtk.RESPONSE_OK))
        chooser.set_do_overwrite_confirmation(True)
        chooser.set_current_name("%s.%s" % (self.tool.module,
                                            _combo_text(self.format_combo) or "tsv"))
        if chooser.run() == gtk.RESPONSE_OK:
            self.output_entry.set_text(chooser.get_filename() or "")
        chooser.destroy()

    def _on_destroy(self, *args):
        """Stop a run that is still going when the dialog is closed."""
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.terminate()
            except OSError:
                pass
        self.process = None

    def _set_status(self, text):
        self.status_label.set_text(text)

    def _error(self, text):
        self._set_status(text.splitlines()[0])
        _message(self.window, "error", text)


def _tail(text, limit=2000):
    """The end of a message, so a long traceback still fits in a dialog."""
    if len(text) <= limit:
        return text
    return "..." + text[-limit:]


def _count_rows(path):
    """How many data rows a written table has, ignoring comments and the header."""
    try:
        handle = open(path)
    except (IOError, OSError):
        return 0
    try:
        rows = 0
        header_seen = False
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            if not header_seen:
                header_seen = True
                continue
            rows += 1
        return rows
    finally:
        handle.close()


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def pdb_python_tools_gui(module=None):
    """
    Open the pdb_python_tools GUI.

    Without an argument this shows the list of tools; with a module name
    ("find_contacts", ...) it opens that tool's dialog directly. Also callable
    from Coot's Python scripting window.
    """
    if module is not None:
        for tool in TOOLS:
            if tool.module == module:
                ToolDialog(tool).show()
                return
        raise ValueError("Unknown tool: %s" % module)

    gtk = _gtk()
    if gtk is None:
        print("pdb_python_tools: no PyGTK available, cannot open the dialog")
        return
    window = gtk.Window(gtk.WINDOW_TOPLEVEL)
    window.set_title(MENU_NAME)
    window.set_default_size(300, -1)
    box = gtk.VBox(False, 4)
    box.set_border_width(8)
    for tool in TOOLS:
        button = gtk.Button(tool.label)
        _set_tooltip(button, tool.tooltip)
        button.connect("clicked", _tool_callback(tool))
        box.pack_start(button, False, False, 0)
    close_button = gtk.Button("Close")
    close_button.connect("clicked", lambda *args: window.destroy())
    box.pack_start(gtk.HSeparator(), False, False, 4)
    box.pack_start(close_button, False, False, 0)
    window.add(box)
    window.show_all()


def _tool_callback(tool):
    """A menu/button callback that opens `tool`'s dialog."""
    def callback(*args):
        ToolDialog(tool).show()
    return callback


def add_pdb_python_tools_menu():
    """
    Add the pdb_python_tools menu to Coot's menu bar.

    Returns True when the menu was added. A Coot without the menu helpers still
    has pdb_python_tools_gui() in the scripting window.
    """
    coot_menubar_menu = _coot_function("coot_menubar_menu")
    add_menu_item = _coot_function("add_simple_coot_menu_menuitem")
    if coot_menubar_menu is None or add_menu_item is None:
        return False
    menu = coot_menubar_menu(MENU_NAME)
    if menu is None:
        return False
    for tool in TOOLS:
        add_menu_item(menu, tool.label, _tool_callback(tool))
    return True


def _install_menu_quietly():
    """
    Add the menu when this file is loaded by Coot.

    Outside Coot the file is inert, so importing it from Python 3 does nothing.
    Inside Coot anything that goes wrong is reported and swallowed so a broken
    extension must not stop Coot from starting.
    """
    if _coot_function("set_rotation_centre") is None:
        return
    try:
        if not add_pdb_python_tools_menu():
            print("pdb_python_tools: could not add the menu; "
                  "run pdb_python_tools_gui() from the scripting window instead")
    except Exception as exc:
        print("pdb_python_tools: could not add the menu: %s" % exc)


_install_menu_quietly()
