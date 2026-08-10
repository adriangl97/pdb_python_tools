#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pdb_python_tools inside the Coot GUI.

Loading this file into Coot (0.9 or 1) adds a "pdb_python_tools" menu. Each
entry opens a dialog where the input structures are picked from the models
already open in Coot, the tool's options are filled in, and "Run" starts the
tool. The results come back as the clickable list the tools already write with
--coot, opened automatically once the run finishes: clicking a row recenters
the view on that residue. atom_tracker also opens a bar graph of the
displacement, one chart per chain, whose bars recenter the view in the same way.

The tools need numpy/scipy, so they are run as a subprocess under an external Python 3
interpreter. That interpreter is taken from, in order: the entry in the dialog,
the PDB_PYTHON_TOOLS_PYTHON environment variable, the one recorded by
"pdb_python_tools.coot_setup --install", the interpreter Coot itself embeds
when the tools are installed in it (Coot 1 only), and finally plain "python3"
from PATH.

Install with "pdb_python_tools.coot_setup --install", which copies this file
into the startup directory of every Coot it finds, or load it once from Coot
with Calculate -> Run Script...

"""
from __future__ import print_function

import json
import os
import subprocess
import sys
import tempfile

MENU_NAME = "pdb_python_tools"

CONFIG_NAME = "pdb_python_tools_coot.json"


def _config_home():
    """$XDG_CONFIG_HOME, or ~/.config when it is not set."""
    from_env = os.environ.get("XDG_CONFIG_HOME")
    if from_env:
        return from_env
    return os.path.expanduser(os.path.join("~", ".config"))


# Written by "pdb_python_tools.coot_setup --install"
CONFIG_PATH = os.path.join(_config_home(), "pdb_python_tools", CONFIG_NAME)

# Coot points these at its own bundled Python and libraries. Leaving them in
# place would make the external Python 3 load Coot's standard library instead of
# its own, so they are dropped from the subprocess environment.
_ENV_TO_DROP = ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONEXECUTABLE",
                "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH")

# Where Coot keeps its scripting functions.
_COOT_MODULES = ("coot", "coot_utils", "coot_gui", "coot_gui_api")


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
        except Exception:
            # Not only ImportError: some of these modules raise on import when
            # Coot is started without graphics, and a lookup is never worth
            # taking the caller down with it
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
    Write the settings and return the path they went to.

    An unwritable directory is not worth interrupting a run for, so it returns
    None instead of raising.
    """
    directory = os.path.dirname(CONFIG_PATH)
    if directory and not os.path.isdir(directory):
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


def embedded_interpreter_has_the_tools():
    """
    Whether the Python that Coot embeds could run the tools itself.

    Coot 0.9's Python 2 never can. Coot 1's Python 3 can when pdb_python_tools,
    numpy and scipy are installed in it. The modules are looked up rather than
    imported, so nothing heavy is pulled into the running Coot.
    """
    if sys.version_info[0] < 3:
        return False
    try:
        from importlib import util
    except ImportError:
        return False
    try:
        for name in ("pdb_python_tools", "numpy", "scipy"):
            if util.find_spec(name) is None:
                return False
    except (ImportError, ValueError, AttributeError):
        return False
    return True


def default_python():
    """The Python 3 interpreter to run the tools with."""
    from_env = os.environ.get("PDB_PYTHON_TOOLS_PYTHON")
    if from_env:
        return from_env
    recorded = load_config().get("python")
    if recorded:
        return recorded
    if embedded_interpreter_has_the_tools():
        return sys.executable
    return "python3"


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
                 "pre-aligned models; the results come back as a clickable "
                 "list and a bar graph of the displacement per chain"),
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

    A `table` of None leaves -o out, so the tool writes the table to stdout and
    no table file is created.
    """
    argv = [python, "-m", "pdb_python_tools." + tool.module]
    argv.extend(inputs)
    argv.extend(option_args)
    argv.extend(["--precision", str(precision), "--format", fmt])
    if table:
        argv.extend(["-o", table])
    argv.extend(["--coot", script, "--force"])
    return argv


def subprocess_environment():
    """The environment to run the tools in, without Coot's own Python settings."""
    environment = dict(os.environ)
    for name in _ENV_TO_DROP:
        environment.pop(name, None)
    return environment


# ---------------------------------------------------------------------------
# GTK, in whichever version Coot embeds
# ---------------------------------------------------------------------------
#
# Coot 0.9 embeds PyGTK (GTK 2) and Coot 1 embeds PyGObject (GTK 4). 
# The two are different enough that the dialog never touches
# either directly: it goes through one of the small toolkits below, which spell
# out the same handful of operations once per GTK version. Everything the two
# GTKs agree on -- get_text/set_text, get_active/set_active, set_sensitive,
# set_tooltip_text -- is left to the widgets themselves.

class _Gtk2Toolkit(object):
    """PyGTK, the GTK 2 binding Coot 0.9 embeds."""

    version = "gtk2"

    def __init__(self, gtk, gobject=None):
        self.gtk = gtk
        self.gobject = gobject

    # -- windows and boxes -------------------------------------------------

    def window(self, title, width=-1, height=-1, on_destroy=None):
        window = self.gtk.Window(self.gtk.WINDOW_TOPLEVEL)
        window.set_title(title)
        window.set_default_size(width, height)
        if on_destroy is not None:
            window.connect("destroy", lambda *args: on_destroy())
        return window

    def show(self, window):
        window.show_all()

    def close(self, window):
        window.destroy()

    def vbox(self, spacing=4, border=0):
        box = self.gtk.VBox(False, spacing)
        box.set_border_width(border)
        return box

    def hbox(self, spacing=6):
        return self.gtk.HBox(False, spacing)

    def pack(self, box, child, expand=False):
        box.pack_start(child, expand, expand, 0)

    def window_content(self, window, child):
        window.add(child)

    def scrolled(self):
        scrolled = self.gtk.ScrolledWindow()
        scrolled.set_policy(self.gtk.POLICY_AUTOMATIC, self.gtk.POLICY_AUTOMATIC)
        return scrolled

    def scrolled_content(self, scrolled, child):
        scrolled.add_with_viewport(child)

    def separator(self):
        return self.gtk.HSeparator()

    # -- widgets -----------------------------------------------------------

    def label(self, text="", wrap=False, width=-1):
        label = self.gtk.Label(text)
        label.set_alignment(0, 0.5)
        if wrap:
            label.set_line_wrap(True)
        if width > 0:
            label.set_size_request(width, -1)
        return label

    def button(self, text, on_click):
        button = self.gtk.Button(text)
        button.connect("clicked", lambda *args: on_click())
        return button

    def entry(self, text="", width_chars=-1):
        entry = self.gtk.Entry()
        if width_chars > 0:
            entry.set_width_chars(width_chars)
        entry.set_text(text)
        return entry

    def check(self, text, active=False):
        check = self.gtk.CheckButton(text)
        check.set_active(active)
        return check

    def combo(self):
        return self.gtk.combo_box_new_text()

    def combo_clear(self, combo):
        combo.get_model().clear()

    def combo_append(self, combo, text):
        combo.append_text(text)

    def combo_text(self, combo):
        active = combo.get_active()
        if active < 0:
            return ""
        return combo.get_model()[active][0]

    # -- the main loop and its dialogs -------------------------------------

    def timeout_add(self, milliseconds, function):
        if self.gobject is not None:
            return self.gobject.timeout_add(milliseconds, function)
        return self.gtk.timeout_add(milliseconds, function)

    def _message(self, parent, kind, buttons, text):
        dialog = self.gtk.MessageDialog(parent, self.gtk.DIALOG_MODAL, kind, buttons)
        # Set as a property rather than through the constructor: the message is
        # a path or a tool's error output, not a format string
        dialog.set_property("text", text)
        response = dialog.run()
        dialog.destroy()
        return response

    def confirm(self, parent, text, on_yes):
        if self._message(parent, self.gtk.MESSAGE_QUESTION,
                         self.gtk.BUTTONS_OK_CANCEL, text) == self.gtk.RESPONSE_OK:
            on_yes()

    def error(self, parent, text):
        self._message(parent, self.gtk.MESSAGE_ERROR, self.gtk.BUTTONS_CLOSE, text)

    def save_as(self, parent, name, on_chosen):
        chooser = self.gtk.FileChooserDialog(
            "Save the table as", parent, self.gtk.FILE_CHOOSER_ACTION_SAVE,
            (self.gtk.STOCK_CANCEL, self.gtk.RESPONSE_CANCEL,
             self.gtk.STOCK_SAVE, self.gtk.RESPONSE_OK))
        chooser.set_do_overwrite_confirmation(True)
        chooser.set_current_name(name)
        chosen = chooser.get_filename() if chooser.run() == self.gtk.RESPONSE_OK else None
        chooser.destroy()
        if chosen:
            on_chosen(chosen)


class _GiToolkit(object):
    """PyGObject with GTK 4."""

    def __init__(self, gtk, gio, glib, gtk_version):
        self.gtk = gtk
        self.gio = gio
        self.glib = glib
        self.version = "gtk" + gtk_version.split(".")[0]
        self.gtk4 = gtk_version.startswith("4")
        # GTK 4 dialogs are answered through a callback, so they have to be
        # kept alive until that callback runs
        self._pending = []

    # -- windows and boxes -------------------------------------------------

    def window(self, title, width=-1, height=-1, on_destroy=None):
        window = self.gtk.Window()
        window.set_title(title)
        window.set_default_size(width, height)
        if on_destroy is not None:
            window.connect("destroy", lambda *args: on_destroy())
        return window

    def show(self, window):
        if self.gtk4:
            window.present()
        else:
            window.show_all()

    def close(self, window):
        window.destroy()

    def vbox(self, spacing=4, border=0):
        box = self.gtk.Box(orientation=self.gtk.Orientation.VERTICAL, spacing=spacing)
        self._set_border(box, border)
        return box

    def hbox(self, spacing=6):
        return self.gtk.Box(orientation=self.gtk.Orientation.HORIZONTAL,
                            spacing=spacing)

    def _set_border(self, box, border):
        if not border:
            return
        if self.gtk4:
            for side in ("start", "end", "top", "bottom"):
                getattr(box, "set_margin_" + side)(border)
        else:
            box.set_border_width(border)

    def pack(self, box, child, expand=False):
        if not self.gtk4:
            box.pack_start(child, expand, expand, 0)
            return
        # GTK 4 has no packing arguments: a child grows because it says so, and
        # only along the axis of the box it sits in
        if expand:
            child.set_hexpand(True)
            if box.get_orientation() == self.gtk.Orientation.VERTICAL:
                child.set_vexpand(True)
        box.append(child)

    def window_content(self, window, child):
        if self.gtk4:
            window.set_child(child)
        else:
            window.add(child)

    def scrolled(self):
        scrolled = self.gtk.ScrolledWindow()
        scrolled.set_policy(self.gtk.PolicyType.AUTOMATIC,
                            self.gtk.PolicyType.AUTOMATIC)
        return scrolled

    def scrolled_content(self, scrolled, child):
        if self.gtk4:
            scrolled.set_child(child)
        else:
            scrolled.add(child)

    def separator(self):
        return self.gtk.Separator(orientation=self.gtk.Orientation.HORIZONTAL)

    # -- widgets -----------------------------------------------------------

    def label(self, text="", wrap=False, width=-1):
        label = self.gtk.Label(label=text)
        label.set_xalign(0)
        if wrap:
            if hasattr(label, "set_wrap"):
                label.set_wrap(True)
            else:
                label.set_line_wrap(True)
        if width > 0:
            label.set_size_request(width, -1)
        return label

    def button(self, text, on_click):
        button = self.gtk.Button(label=text)
        button.connect("clicked", lambda *args: on_click())
        return button

    def entry(self, text="", width_chars=-1):
        entry = self.gtk.Entry()
        if width_chars > 0:
            entry.set_width_chars(width_chars)
        entry.set_text(text)
        return entry

    def check(self, text, active=False):
        check = self.gtk.CheckButton(label=text)
        check.set_active(active)
        return check

    def combo(self):
        return self.gtk.ComboBoxText()

    def combo_clear(self, combo):
        combo.remove_all()

    def combo_append(self, combo, text):
        combo.append_text(text)

    def combo_text(self, combo):
        return combo.get_active_text() or ""

    # -- the main loop and its dialogs -------------------------------------

    def timeout_add(self, milliseconds, function):
        return self.glib.timeout_add(milliseconds, function)

    def _keep(self, dialog):
        self._pending.append(dialog)

    def _release(self, dialog):
        if dialog in self._pending:
            self._pending.remove(dialog)

    def confirm(self, parent, text, on_yes):
        """
        Ask, and call `on_yes` when the answer is yes.

        A GTK 4 dialog cannot be run modally from Python, so the answer arrives
        in a callback instead of as a return value. The GTK 2 toolkit calls
        `on_yes` from inside confirm(); either way the caller does the same
        thing with it.
        """
        if hasattr(self.gtk, "AlertDialog"):
            dialog = self.gtk.AlertDialog()
            dialog.set_message(text)
            dialog.set_buttons(["Cancel", "Overwrite"])
            dialog.set_cancel_button(0)
            dialog.set_default_button(1)
            self._keep(dialog)

            def chosen(source, result, *args):
                self._release(dialog)
                try:
                    answer = source.choose_finish(result)
                except Exception:
                    return  # dismissed without answering
                if answer == 1:
                    on_yes()
            dialog.choose(parent, None, chosen)
            return
        dialog = self.gtk.MessageDialog(
            transient_for=parent, modal=True,
            message_type=self.gtk.MessageType.QUESTION,
            buttons=self.gtk.ButtonsType.OK_CANCEL, text=text)

        def responded(source, response):
            source.destroy()
            if response == self.gtk.ResponseType.OK:
                on_yes()
        dialog.connect("response", responded)
        dialog.show()

    def error(self, parent, text):
        if hasattr(self.gtk, "AlertDialog"):
            dialog = self.gtk.AlertDialog()
            dialog.set_message(text)
            dialog.show(parent)
            return
        dialog = self.gtk.MessageDialog(
            transient_for=parent, modal=True,
            message_type=self.gtk.MessageType.ERROR,
            buttons=self.gtk.ButtonsType.CLOSE, text=text)
        dialog.connect("response", lambda source, response: source.destroy())
        dialog.show()

    def save_as(self, parent, name, on_chosen):
        if hasattr(self.gtk, "FileDialog"):
            dialog = self.gtk.FileDialog()
            dialog.set_initial_name(name)
            self._keep(dialog)

            def chosen(source, result, *args):
                self._release(dialog)
                try:
                    picked = source.save_finish(result)
                except Exception:
                    return  # cancelled
                if picked is not None and picked.get_path():
                    on_chosen(picked.get_path())
            dialog.save(parent, None, chosen)
            return
        chooser = self.gtk.FileChooserNative(
            title="Save the table as", transient_for=parent,
            action=self.gtk.FileChooserAction.SAVE)
        chooser.set_current_name(name)
        self._keep(chooser)

        def responded(source, response):
            self._release(chooser)
            if response == self.gtk.ResponseType.ACCEPT:
                picked = source.get_file()
                if picked is not None and picked.get_path():
                    on_chosen(picked.get_path())
        chooser.connect("response", responded)
        chooser.show()


def _make_toolkit():
    """Build the toolkit for the GTK in this process, or None when there is none."""
    try:
        import gtk  # Coot 0.9
    except ImportError:
        pass
    else:
        # PyGObject ships a stand-in "gtk" module that imports and then raises
        # on every attribute, so PyGTK has to be recognised by one of its own
        if getattr(gtk, "pygtk_version", None) is not None:
            try:
                import gobject
            except ImportError:
                gobject = None
            if not hasattr(gobject, "timeout_add"):
                gobject = None
            return _Gtk2Toolkit(gtk, gobject)
    try:
        import gi  # Coot 1
    except ImportError:
        return None
    # Coot has already settled on a version by the time this runs; asking for a
    # different one would fail, so the one it picked is the one to use
    try:
        gtk_version = gi.get_required_version("Gtk")
    except Exception:
        gtk_version = None
    if gtk_version is None:
        for candidate in ("4.0", "3.0"):
            try:
                gi.require_version("Gtk", candidate)
            except (ValueError, AttributeError):
                continue
            gtk_version = candidate
            break
    if gtk_version is None:
        return None
    try:
        from gi.repository import Gtk, Gio, GLib
    except ImportError:
        return None
    return _GiToolkit(Gtk, Gio, GLib, gtk_version)


# One toolkit per session, in a list so that "not looked yet" and "looked, no
# GTK" stay apart
_TOOLKIT = []


def _toolkit():
    """The GTK toolkit Coot brought, or None when there is no GTK at all."""
    if not _TOOLKIT:
        _TOOLKIT.append(_make_toolkit())
    return _TOOLKIT[0]


def _set_tooltip(widget, text):
    """Set a tooltip when the GTK version in use supports it."""
    if text and hasattr(widget, "set_tooltip_text"):
        widget.set_tooltip_text(text)


def _labelled_row(tk, label_text, *widgets):
    """A row of a fixed-width label followed by widgets."""
    box = tk.hbox()
    tk.pack(box, tk.label(label_text, width=140))
    for index, widget in enumerate(widgets):
        tk.pack(box, widget, expand=index == 0)
    return box


# ---------------------------------------------------------------------------
# The tool dialog
# ---------------------------------------------------------------------------

class ToolDialog(object):
    """The dialog for one tool: inputs from the open models, options, Run."""

    def __init__(self, tool):
        self.tool = tool
        self.tk = _toolkit()
        self.models = []
        self.model_combos = []
        self.option_widgets = []
        self.chain_combos = []
        self.process = None
        self.run_state = None

    # -- construction ------------------------------------------------------

    def show(self):
        tk = self.tk
        if tk is None:
            print("pdb_python_tools: no GTK available, cannot open the dialog")
            return
        self.window = tk.window("pdb_python_tools: %s" % self.tool.label,
                                width=460, on_destroy=self._on_destroy)

        outer = tk.vbox(spacing=4, border=8)

        if self.tool.tooltip:
            tk.pack(outer, tk.label(self.tool.tooltip, wrap=True))

        for index, label in enumerate(self.tool.models):
            combo = tk.combo()
            self.model_combos.append(combo)
            if index == 0:
                refresh = tk.button("Refresh", self._on_refresh)
                _set_tooltip(refresh, "re-read the list of open models")
                tk.pack(outer, _labelled_row(tk, label, combo, refresh))
            else:
                tk.pack(outer, _labelled_row(tk, label, combo))
        self.model_combos[0].connect("changed", self._on_model_changed)

        tk.pack(outer, tk.separator())
        for option in self.tool.options:
            tk.pack(outer, self._option_row(option))

        tk.pack(outer, tk.separator())
        self.precision_entry = tk.entry("2", width_chars=4)
        _set_tooltip(self.precision_entry,
                     "decimal places for the reported distances or angles")
        self.format_combo = tk.combo()
        for fmt in ("tsv", "csv"):
            tk.combo_append(self.format_combo, fmt)
        self.format_combo.set_active(0)
        tk.pack(outer, _labelled_row(tk, "Precision / format",
                                     self.precision_entry, self.format_combo))

        self.output_entry = tk.entry()
        _set_tooltip(self.output_entry,
                     "where to keep the table; left empty no table file is "
                     "written")
        browse = tk.button("Browse...", self._on_browse)
        tk.pack(outer, _labelled_row(tk, "Save table to",
                                     self.output_entry, browse))

        self.open_check = tk.check(
            "Open the results in Coot when the run finishes", active=True)
        tk.pack(outer, self.open_check)

        self.python_entry = tk.entry(default_python())
        _set_tooltip(self.python_entry,
                     "Python 3 interpreter that has pdb_python_tools installed")
        tk.pack(outer, _labelled_row(tk, "Python 3", self.python_entry))

        tk.pack(outer, tk.separator())
        self.status_label = tk.label("", wrap=True)
        tk.pack(outer, self.status_label)

        buttons = tk.hbox()
        self.run_button = tk.button("Run", self._on_run)
        close_button = tk.button("Close", lambda: tk.close(self.window))
        tk.pack(buttons, self.run_button, expand=True)
        tk.pack(buttons, close_button, expand=True)
        tk.pack(outer, buttons)

        tk.window_content(self.window, outer)
        self._on_refresh()
        tk.show(self.window)

    def _option_row(self, option):
        """The widget row for one option, remembering the widget it holds."""
        tk = self.tk
        if option.kind == "check":
            widget = tk.check(option.label, active=bool(option.default))
            row = widget
        elif option.kind == "choice":
            widget = tk.combo()
            for label, _flag in option.choices:
                tk.combo_append(widget, label)
            widget.set_active(0)
            row = _labelled_row(tk, option.label, widget)
        elif option.kind == "chain":
            widget = tk.combo()
            self.chain_combos.append(widget)
            row = _labelled_row(tk, option.label, widget)
        else:
            text = "" if option.default is None else str(option.default)
            widget = tk.entry(text, width_chars=8)
            row = _labelled_row(tk, option.label, widget)
        _set_tooltip(widget, option.tooltip)
        self.option_widgets.append((option, widget))
        return row

    # -- keeping the model lists current -----------------------------------

    def _on_refresh(self, *args):
        """Re-read the open models and repopulate the combos."""
        tk = self.tk
        self.models = open_models()
        for index, combo in enumerate(self.model_combos):
            previous = combo.get_active()
            tk.combo_clear(combo)
            for imol, name in self.models:
                tk.combo_append(combo, "%d: %s" % (imol, os.path.basename(name)))
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
        tk = self.tk
        imol = self._selected_imol(0)
        chains = chain_ids_of(imol) if imol is not None else []
        for combo in self.chain_combos:
            previous = tk.combo_text(combo)
            tk.combo_clear(combo)
            for chain in chains:
                tk.combo_append(combo, chain)
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
            return self.tk.combo_text(widget)
        return widget.get_text()

    def _option_args(self):
        """The command-line arguments for the option widgets."""
        values = [self._widget_value(option, widget)
                  for option, widget in self.option_widgets]
        return option_arguments(self.tool.options, values)

    def _checked_settings(self):
        """
        Everything the dialog says, checked but not acted on yet.

        Nothing is written here, so the run can still be called off by a bad
        entry, or by the answer to the overwrite question.
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
        return {"python": python, "precision": precision, "imols": imols,
                "option_args": self._option_args(),
                "fmt": self.tk.combo_text(self.format_combo) or "tsv",
                "output": self.output_entry.get_text().strip()}

    def _prepare(self, settings):
        """
        Everything the run needs: the argv, the working directory and the
        paths the tool will write to.
        """
        work_dir = tempfile.mkdtemp(prefix="pdb_python_tools_")
        # No name given means no table file at all: the tool prints it instead
        table_path = settings["output"] or None
        script_path = os.path.join(work_dir, "%s_coot.py" % self.tool.module)
        inputs = [export_model(imol, work_dir) for imol in settings["imols"]]
        argv = build_command(self.tool, settings["python"], inputs,
                             settings["option_args"], settings["precision"],
                             settings["fmt"], table_path, script_path)
        return {"argv": argv, "work_dir": work_dir, "inputs": inputs,
                "table": table_path, "script": script_path,
                "python": settings["python"]}

    # -- running -----------------------------------------------------------

    def _on_run(self):
        if self.process is not None:
            return
        try:
            settings = self._checked_settings()
        except ValueError as exc:
            self._set_status(str(exc))
            return
        except Exception as exc:
            self._error("Could not start the run:\n\n%s" % exc)
            return
        # --force is passed for the sake of the temporary script, so an existing
        # table the user named is confirmed here instead
        output = settings["output"]
        if output and os.path.exists(output):
            self.tk.confirm(self.window,
                            "%s already exists. Overwrite it?" % output,
                            lambda: self._start(settings))
            return
        self._start(settings)

    def _start(self, settings):
        """Export the models and launch the tool."""
        try:
            state = self._prepare(settings)
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
        self.tk.timeout_add(200, self._poll)

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

        if state["table"]:
            rows = _count_rows(state["table"])
            self._set_status("%s: %d row(s). Table: %s"
                             % (self.tool.module, rows, state["table"]))
        else:
            rows = _count_rows_in(output.splitlines())
            self._set_status("%s: %d row(s). The table was not saved."
                             % (self.tool.module, rows))
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

    def _on_browse(self):
        name = "%s.%s" % (self.tool.module,
                          self.tk.combo_text(self.format_combo) or "tsv")
        self.tk.save_as(self.window, name, self.output_entry.set_text)

    def _on_destroy(self):
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
        self.tk.error(self.window, text)


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
        return _count_rows_in(handle)
    finally:
        handle.close()


def _count_rows_in(lines):
    """The same count for a table that was printed instead of written."""
    rows = 0
    header_seen = False
    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        if not header_seen:
            header_seen = True
            continue
        rows += 1
    return rows


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

    tk = _toolkit()
    if tk is None:
        print("pdb_python_tools: no GTK available, cannot open the dialog")
        return
    window = tk.window(MENU_NAME, width=300)
    box = tk.vbox(spacing=4, border=8)
    for tool in TOOLS:
        button = tk.button(tool.label, _tool_callback(tool))
        _set_tooltip(button, tool.tooltip)
        tk.pack(box, button)
    tk.pack(box, tk.separator())
    tk.pack(box, tk.button("Close", lambda: tk.close(window)))
    tk.window_content(window, box)
    tk.show(window)


def _tool_callback(tool):
    """A menu/button callback that opens `tool`'s dialog."""
    def callback(*args):
        ToolDialog(tool).show()
    return callback


def _action_name(tool):
    """
    The name of the Coot 1 action that opens one tool's dialog.

    A Gio action name is only allowed alphanumerics, '-' and '.', so the
    module's underscores do not survive into it.
    """
    return "pdb-python-tools-" + tool.module.replace("_", "-").lower()


def _add_menu_to_menubar():
    """
    Coot 0.9: a pdb_python_tools menu in the menu bar.

    Returns False when this Coot does not have the two helpers that build one,
    which is how Coot 1 ends
    up on the toolbar instead.
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


def _add_menu_to_toolbar():
    """
    Coot 1: a pdb_python_tools menu button on the main toolbar.

    Coot 1 builds its menus from GMenu models driven by actions on the
    application, and coot_gui has a helper for each half. A Coot 1 without them
    gets the same menu built here, straight through coot_gui_api.
    """
    attach = _coot_function("attach_module_menu_button")
    add_action = _coot_function("add_simple_action_to_menu")
    if attach is not None and add_action is not None:
        menu = attach(MENU_NAME)
        if menu is not None:
            for tool in TOOLS:
                add_action(menu, tool.label, _action_name(tool),
                           _tool_callback(tool))
            return True
    return _add_menu_with_gio()


def _add_menu_with_gio():
    """Build the Coot 1 toolbar menu without coot_gui's helpers."""
    tk = _toolkit()
    if tk is None or getattr(tk, "gio", None) is None:
        return False
    try:
        import coot_gui_api
    except ImportError:
        return False
    application = getattr(coot_gui_api, "application", None)
    main_toolbar = getattr(coot_gui_api, "main_toolbar", None)
    if application is None or main_toolbar is None:
        return False
    app = application()
    toolbar = main_toolbar()
    if app is None or toolbar is None:
        return False
    menu = tk.gio.Menu.new()
    popover = tk.gtk.PopoverMenu()
    popover.set_menu_model(menu)
    button = tk.gtk.MenuButton(label=MENU_NAME)
    button.set_popover(popover)
    toolbar.append(button)
    for tool in TOOLS:
        name = _action_name(tool)
        action = tk.gio.SimpleAction.new(name, None)
        action.connect("activate", _tool_callback(tool))
        app.add_action(action)
        menu.append(tool.label, "app." + name)
    return True


def add_pdb_python_tools_menu():
    """
    Add the pdb_python_tools menu to Coot.

    In Coot 0.9 that is a menu in the menu bar, in Coot 1 a menu button on the
    main toolbar. Returns True when one of them was added. A Coot that takes
    neither still has pdb_python_tools_gui() in the scripting window.
    """
    return _add_menu_to_menubar() or _add_menu_to_toolbar()


def _install_menu_quietly():
    """
    Add the menu when this file is loaded by Coot.

    Outside Coot the file is inert, so importing it from Python 3 does nothing.
    Inside Coot anything that goes wrong is reported and swallowed so a broken
    extension must not stop Coot from starting.
    """
    if _coot_function("set_rotation_centre") is None:
        return
    advice = ("pdb_python_tools: could not add the menu; "
              "run pdb_python_tools_gui() from the scripting window instead")
    try:
        if add_pdb_python_tools_menu():
            return
    except Exception as exc:
        print("pdb_python_tools: could not add the menu: %s" % exc)
        return
    # Startup scripts can run before Coot has finished building its window, in
    # which case there is nothing to add the menu to yet: try once more later
    tk = _toolkit()
    if tk is None:
        print(advice)
        return

    def retry():
        try:
            if not add_pdb_python_tools_menu():
                print(advice)
        except Exception as exc:
            print("pdb_python_tools: could not add the menu: %s" % exc)
        return False

    try:
        tk.timeout_add(2000, retry)
    except Exception:
        print(advice)


_install_menu_quietly()
