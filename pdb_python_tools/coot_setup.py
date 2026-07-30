#!/usr/bin/env python3
"""
coot_setup.py - install the Coot GUI extension.

Copies coot_extension.py into Coot's startup directory (~/.coot-preferences by
default), where Coot 0.9 loads every .py file when it starts, so the
"pdb_python_tools" menu is there in every session.

It also records the interpreter it was run with, in a settings file next to the
extension. That is the interpreter the extension runs the tools with, so
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
DEFAULT_COOT_DIR = os.path.expanduser(os.path.join("~", ".coot-preferences"))
# The name the extension is installed under, and the file it comes from
INSTALLED_NAME = "pdb_python_tools.py"
EXTENSION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "coot_extension.py")
# Read by the extension to find this interpreter again (see coot_extension.py).
# It lives beside the extension; Coot only runs the *.py and *.scm it finds in
# there, so the settings file is left alone.
CONFIG_NAME = "pdb_python_tools_coot.json"
CONFIG_PATH = os.path.join(DEFAULT_COOT_DIR, CONFIG_NAME)


def record_interpreter(python=None, config_path=CONFIG_PATH):
    """
    Store the Python 3 interpreter the extension should run the tools with.

    Defaults to the interpreter running this command, which is by definition
    one that has pdb_python_tools installed, and to the settings file the
    extension reads, next to it in Coot's startup directory.
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


def install(directory=DEFAULT_COOT_DIR, force=False, symlink=False):
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
        description='Install the pdb_python_tools extension for the Coot (0.9) GUI',
        epilog='Usage: pdb_python_tools.coot_setup --install')
    parser.add_argument('--install', action='store_true',
                        help='copy the extension into Coot\'s startup directory')
    parser.add_argument('--dir', default=DEFAULT_COOT_DIR,
                        help='startup directory to install into '
                             '(default: %s)' % DEFAULT_COOT_DIR)
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

    try:
        target = install(args.dir, force=args.force, symlink=args.symlink)
    except (FileExistsError, FileNotFoundError) as exc:
        sys.exit("error: %s" % exc)
    except OSError as exc:
        sys.exit("error: could not install into %s: %s" % (args.dir, exc))
    # The settings stay in the default directory even for a custom --dir: that
    # is the one place the extension knows to look
    interpreter = record_interpreter(args.python)

    print("Installed: %s" % target)
    print("Interpreter recorded in %s: %s" % (CONFIG_PATH, interpreter))
    print("Restart Coot: the tools are under the 'pdb_python_tools' menu.")


if __name__ == "__main__":
    main()
