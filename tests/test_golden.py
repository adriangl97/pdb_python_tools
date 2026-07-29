"""
Golden-file tests: regenerate the reference tables in test_files/ and compare.

These run the tools over the two ribosome structures, so they are the
slowest tests in the suite (a couple of seconds each) and are marked `slow`:

    pytest -m "not slow"        # skip them
    pytest -m slow              # run only them

They are skipped when the structures are not present.
"""
import os
import subprocess
import sys

import pytest

from conftest import REPO_ROOT, TEST_FILES_DIR

pytestmark = pytest.mark.slow

STRUCTURE_1 = os.path.join(TEST_FILES_DIR, "6ot3.cif")
STRUCTURE_2 = os.path.join(TEST_FILES_DIR, "6ouo_aligned.cif")

HAS_SCIPY = True
try:
    import scipy  # noqa: F401
except ImportError:
    HAS_SCIPY = False

# The commands documented in the README's "Test files" section, paired with the
# reference output each one produced.
GOLDEN_CASES = [
    ("test_atom_tracker.tsv", "atom_tracker",
     ["-HET", STRUCTURE_1, STRUCTURE_2], False),
    ("test_find_contacts.tsv", "find_contacts",
     [STRUCTURE_2, "-c", "4", "-d", "4.5"], True),
    ("test_nucleotide_conformation.tsv", "nucleotide_conformation",
     [STRUCTURE_2], False),
]


def require_structures(*paths):
    for path in paths:
        if not os.path.exists(path):
            pytest.skip("large test structure not present: %s" % os.path.basename(path))


@pytest.mark.parametrize("reference,tool,args,scipy_needed", GOLDEN_CASES,
                         ids=[case[1] for case in GOLDEN_CASES])
def test_reference_output_is_reproduced(reference, tool, args, scipy_needed, tmp_path):
    """The checked-in .tsv must match exactly what the tool produces."""
    if scipy_needed and not HAS_SCIPY:
        pytest.skip("requires scipy")
    reference_path = os.path.join(TEST_FILES_DIR, reference)
    require_structures(*[a for a in args if a.endswith(".cif")])
    if not os.path.exists(reference_path):
        pytest.skip("reference output not present: %s" % reference)

    produced = tmp_path / reference
    result = subprocess.run(
        [sys.executable, "-m", "pdb_python_tools." + tool] + args
        + ["-o", str(produced)],
        cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    expected = open(reference_path).read().splitlines()
    actual = produced.read_text().splitlines()
    assert actual[0] == expected[0], "header changed"
    assert len(actual) == len(expected), (
        "row count changed: %d -> %d" % (len(expected), len(actual)))
    mismatches = [(i, e, a) for i, (e, a) in enumerate(zip(expected, actual), 1)
                  if e != a]
    assert not mismatches, "rows differ:\n" + "\n".join(
        "  line %d:\n    expected %s\n    actual   %s" % m for m in mismatches[:5])
