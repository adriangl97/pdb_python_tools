#!/usr/bin/env python3
"""
coot_setup.py - install the Coot GUI extension.

Copies coot_extension.py into Coot's startup directory. Coot 0.9 and Coot 1 read different directories
(~/.coot-preferences and ~/.config/Coot, or $XDG_CONFIG_HOME when that is set),
so by default the extension goes into the one of every Coot that looks
installed.

It also records the interpreter it was run with, in a settings file the
extension reads. That is the interpreter the extension runs the tools with, so
installing with the same Python the package was installed into is what makes
the menu work without any further setup.
"""
import argparse
import json
import os
import shutil
import sys

from .core import add_version_arg

# Coot 0.9 runs every .py file in this directory at startup
COOT_09_DIR = os.path.expanduser(os.path.join("~", ".coot-preferences"))


def _config_home():
    """$XDG_CONFIG_HOME, or ~/.config when it is not set."""
    return os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser(
        os.path.join("~", ".config"))


def coot_1_dir():
    """
    The directory Coot 1 runs its startup scripts from.

    Coot 1 takes $XDG_CONFIG_HOME as it stands, and only falls back to
    ~/.config/Coot when it is unset, so this follows Coot rather than the XDG
    convention of a directory per application.
    """
    from_env = os.environ.get("XDG_CONFIG_HOME")
    if from_env:
        return from_env
    return os.path.expanduser(os.path.join("~", ".config", "Coot"))


COOT_1_DIR = coot_1_dir()

# The name the extension is installed under, and the file it comes from
INSTALLED_NAME = "pdb_python_tools.py"
EXTENSION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "coot_extension.py")
# Read by the extension to find this interpreter again (see coot_extension.py).
CONFIG_NAME = "pdb_python_tools_coot.json"
CONFIG_PATH = os.path.join(_config_home(), "pdb_python_tools", CONFIG_NAME)


def install_directories(coot="auto"):
    """
    The startup directories to install into.

    "auto" takes the Coots that look installed: a Coot that has been started
    once has its startup directory. When neither is there it installs for both,
    since the directory can be created before the Coot that reads it is.
    """
    named = {"0.9": [COOT_09_DIR], "1": [COOT_1_DIR],
             "both": [COOT_09_DIR, COOT_1_DIR]}
    if coot in named:
        return named[coot]
    existing = [directory for directory in (COOT_09_DIR, COOT_1_DIR)
                if os.path.isdir(directory)]
    return existing or [COOT_09_DIR, COOT_1_DIR]


def record_interpreter(python=None, config_path=CONFIG_PATH):
    """
    Store the Python 3 interpreter the extension should run the tools with.

    Defaults to the interpreter running this command, which is by definition
    one that has pdb_python_tools installed, and to the settings file the
    extension reads.
    """
    directory = os.path.dirname(config_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path) as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                config = loaded
        except (OSError, ValueError):
            config = {}
    config["python"] = python or sys.executable
    with open(config_path, "w") as handle:
        json.dump(config, handle, indent=1, sort_keys=True)
    return config["python"]


def install(directory=COOT_09_DIR, force=False, symlink=False):
    """
    Put the extension in `directory` and return the path it was written to.

    Refuses to replace an existing file unless `force` is given, so a modified
    copy is never overwritten by accident.
    """
    if not os.path.exists(EXTENSION_PATH):
        raise FileNotFoundError("Extension file is missing: %s" % EXTENSION_PATH)
    os.makedirs(directory, exist_ok=True)
    target = os.path.join(directory, INSTALLED_NAME)
    if os.path.lexists(target):
        if not force:
            raise FileExistsError(
                "Refusing to overwrite existing file: %s (use --force)" % target)
        os.remove(target)
    if symlink:
        os.symlink(EXTENSION_PATH, target)
    else:
        shutil.copyfile(EXTENSION_PATH, target)
    return target


def main():
    parser = argparse.ArgumentParser(
        prog='pdb_python_tools.coot_setup',
        description='Install the pdb_python_tools extension for the Coot GUI '
                    '(Coot 0.9 and Coot 1)',
        epilog='Usage: pdb_python_tools.coot_setup --install')
    parser.add_argument('--install', action='store_true',
                        help='copy the extension into Coot\'s startup directory')
    parser.add_argument('--coot', choices=('auto', '0.9', '1', 'both'),
                        default='auto',
                        help='which Coot to install for (default: auto, every '
                             'Coot whose startup directory is already there: '
                             '%s for 0.9, %s for 1)' % (COOT_09_DIR, COOT_1_DIR))
    parser.add_argument('--dir', default=None,
                        help='startup directory to install into, instead of the '
                             'one --coot would pick')
    parser.add_argument('--symlink', action='store_true',
                        help='link to the installed extension instead of copying '
                             'it, so it follows package upgrades')
    parser.add_argument('--force', action='store_true',
                        help='allow replacing an existing file')
    parser.add_argument('--python', default=None,
                        help='Python 3 interpreter the extension should run the '
                             'tools with (default: this one, %s)' % sys.executable)
    parser.add_argument('--path', action='store_true',
                        help='print the path of the extension file and exit')
    add_version_arg(parser)
    args = parser.parse_args()

    if args.path:
        print(EXTENSION_PATH)
        return
    if not args.install:
        parser.error("nothing to do: pass --install (or --path)")

    directories = [args.dir] if args.dir else install_directories(args.coot)
    targets = []
    for directory in directories:
        try:
            targets.append(install(directory, force=args.force,
                                   symlink=args.symlink))
        except (FileExistsError, FileNotFoundError) as exc:
            sys.exit("error: %s" % exc)
        except OSError as exc:
            sys.exit("error: could not install into %s: %s" % (directory, exc))
    interpreter = record_interpreter(args.python)

    for target in targets:
        print("Installed: %s" % target)
    print("Interpreter recorded in %s: %s" % (CONFIG_PATH, interpreter))
    print("Restart Coot: the tools are under the 'pdb_python_tools' menu "
          "(the menu bar in Coot 0.9, the toolbar in Coot 1).")


if __name__ == "__main__":
    main()
