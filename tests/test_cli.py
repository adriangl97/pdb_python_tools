"""
End-to-end tests for the command-line tools.

"""
import gzip
import math
import shutil
import subprocess
import sys

import pytest

from conftest import REPO_ROOT, pdb_atom_line, write_pdb

HAS_SCIPY = True
try:
    import scipy  # noqa: F401
except ImportError:
    HAS_SCIPY = False

needs_scipy = pytest.mark.skipif(not HAS_SCIPY, reason="requires scipy")


def run_tool(tool, *args):
    """Run one CLI in a subprocess and return the CompletedProcess."""
    return subprocess.run(
        [sys.executable, "-m", "pdb_python_tools." + tool] + [str(a) for a in args],
        cwd=REPO_ROOT, capture_output=True, text=True)


@pytest.fixture
def pair(tmp_path):
    """
    Two small aligned structures: chain A SER 1 whose OG moved 3 A, and chain B
    U 1 (a nucleotide, for the CA/C1' column).
    """
    first = write_pdb(tmp_path / "a.pdb", [
        pdb_atom_line(1, "N", "SER", "A", 1, 0.0, 0.0, 0.0),
        pdb_atom_line(2, "CA", "SER", "A", 1, 1.0, 0.0, 0.0),
        pdb_atom_line(3, "OG", "SER", "A", 1, 2.0, 0.0, 0.0),
        pdb_atom_line(4, "C1'", "U", "B", 1, 0.0, 0.0, 10.0),
        pdb_atom_line(5, "O4'", "U", "B", 1, 1.0, 0.0, 10.0),
    ])
    second = write_pdb(tmp_path / "b.pdb", [
        pdb_atom_line(1, "N", "SER", "A", 1, 0.0, 0.0, 0.0),
        pdb_atom_line(2, "CA", "SER", "A", 1, 1.0, 0.0, 0.0),
        pdb_atom_line(3, "OG", "SER", "A", 1, 5.0, 0.0, 0.0),
        pdb_atom_line(4, "C1'", "U", "B", 1, 0.0, 0.0, 12.0),
        pdb_atom_line(5, "O4'", "U", "B", 1, 1.0, 0.0, 12.0),
    ])
    return first, second


@pytest.fixture
def partial(tmp_path):
    """
    A pair where the second structure models less than the first.

    SER A 1 loses its CB, so only N, CA and OG can be compared; the MG ligand has
    no CA/C1' at all; and PHE A 2 shares no atom name with its counterpart, so
    nothing about it can be measured.
    """
    first = write_pdb(tmp_path / "full.pdb", [
        pdb_atom_line(1, "N", "SER", "A", 1, 0.0, 0.0, 0.0),
        pdb_atom_line(2, "CA", "SER", "A", 1, 1.0, 0.0, 0.0),
        pdb_atom_line(3, "OG", "SER", "A", 1, 2.0, 0.0, 0.0),
        pdb_atom_line(4, "CB", "SER", "A", 1, 3.0, 0.0, 0.0),
        pdb_atom_line(5, "CD1", "PHE", "A", 2, 0.0, 8.0, 0.0),
        pdb_atom_line(6, "MG", "MG", "C", 1, 0.0, 0.0, 20.0, record="HETATM",
                      element="MG"),
    ])
    second = write_pdb(tmp_path / "partial.pdb", [
        pdb_atom_line(1, "N", "SER", "A", 1, 0.0, 0.0, 0.0),
        pdb_atom_line(2, "CA", "SER", "A", 1, 1.0, 0.0, 0.0),
        # OG moved 3 A; CB is not modelled at all
        pdb_atom_line(3, "OG", "SER", "A", 1, 5.0, 0.0, 0.0),
        pdb_atom_line(4, "CZ", "PHE", "A", 2, 0.0, 8.0, 0.0),
        pdb_atom_line(5, "MG", "MG", "C", 1, 0.0, 0.0, 24.0, record="HETATM",
                      element="MG"),
    ])
    return first, second


@pytest.fixture
def two_chains(tmp_path):
    """Two chains 3 A apart, for the contact search."""
    return write_pdb(tmp_path / "contacts.pdb", [
        pdb_atom_line(1, "OG", "SER", "A", 1, 0.0, 0.0, 0.0),
        pdb_atom_line(2, "CB", "SER", "A", 1, 0.0, 0.0, 1.0),
        pdb_atom_line(3, "N", "GLY", "B", 1, 3.0, 0.0, 0.0),
    ])


def nucleotide_lines(serial, restyp, chain, resseq, chi_deg, z_offset=0.0):
    """
    Four ATOM records for one nucleotide with an exact glycosidic chi.

    Purines are measured O4'-C1'-N9-C4 and pyrimidines O4'-C1'-N1-C2; placing
    the fourth atom at (1, cos t, sin t) makes chi exactly t degrees.
    """
    base = ("N9", "C4") if restyp in ("A", "G", "DA", "DG") else ("N1", "C2")
    t = math.radians(chi_deg)
    return [
        pdb_atom_line(serial, "O4'", restyp, chain, resseq, 0.0, 1.0, z_offset),
        pdb_atom_line(serial + 1, "C1'", restyp, chain, resseq, 0.0, 0.0, z_offset),
        pdb_atom_line(serial + 2, base[0], restyp, chain, resseq, 1.0, 0.0, z_offset),
        pdb_atom_line(serial + 3, base[1], restyp, chain, resseq,
                      1.0, math.cos(t), z_offset + math.sin(t)),
    ]


@pytest.fixture
def rna(tmp_path):
    """One syn U (chi = 0) and one anti C (chi = 180)."""
    return write_pdb(tmp_path / "rna.pdb", [
        pdb_atom_line(1, "O4'", "U", "A", 1, 0.0, 1.0, 0.0),
        pdb_atom_line(2, "C1'", "U", "A", 1, 0.0, 0.0, 0.0),
        pdb_atom_line(3, "N1", "U", "A", 1, 1.0, 0.0, 0.0),
        pdb_atom_line(4, "C2", "U", "A", 1, 1.0, 1.0, 0.0),
        pdb_atom_line(5, "O4'", "C", "A", 2, 0.0, 1.0, 20.0),
        pdb_atom_line(6, "C1'", "C", "A", 2, 0.0, 0.0, 20.0),
        pdb_atom_line(7, "N1", "C", "A", 2, 1.0, 0.0, 20.0),
        pdb_atom_line(8, "C2", "C", "A", 2, 1.0, -1.0, 20.0),
    ])


@pytest.fixture
def dna(tmp_path):
    """
    A DNA strand with one nucleotide of each base: syn DT and syn DC, plus anti DA and anti DG.
    """
    lines = []
    lines += nucleotide_lines(1, "DT", "A", 1, 0.0, z_offset=0.0)
    lines += nucleotide_lines(5, "DC", "A", 2, 45.0, z_offset=20.0)
    lines += nucleotide_lines(9, "DA", "A", 3, 180.0, z_offset=40.0)
    lines += nucleotide_lines(13, "DG", "A", 4, -120.0, z_offset=60.0)
    return write_pdb(tmp_path / "dna.pdb", lines)


@pytest.fixture
def hybrid(tmp_path):
    """
    A DNA/RNA hybrid: chain A is DNA (syn DT, anti DA), chain B is RNA (syn U,
    anti A). One syn pyrimidine per strand, so the default view must report both.
    """
    lines = []
    lines += nucleotide_lines(1, "DT", "A", 1, 30.0, z_offset=0.0)
    lines += nucleotide_lines(5, "DA", "A", 2, 175.0, z_offset=20.0)
    lines += nucleotide_lines(9, "U", "B", 1, 60.0, z_offset=40.0)
    lines += nucleotide_lines(13, "A", "B", 2, -175.0, z_offset=60.0)
    return write_pdb(tmp_path / "hybrid.pdb", lines)


ALL_TOOLS = ["atom_tracker", "find_contacts", "CA_difference",
             "nucleotide_conformation"]


def run_piped(tool, args, head_lines=2):
    """
    Run a CLI with its stdout piped into `head -n`, which closes the pipe early.

    Returns (tool_returncode, tool_stderr, lines_head_received).
    """
    head = subprocess.Popen(["head", "-n", str(head_lines)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True)
    tool_proc = subprocess.Popen(
        [sys.executable, "-m", "pdb_python_tools." + tool] + [str(a) for a in args],
        cwd=REPO_ROOT, stdout=head.stdin, stderr=subprocess.PIPE, text=True)
    head.stdin.close()
    out = head.stdout.read()
    head.wait()
    stderr = tool_proc.stderr.read()
    tool_proc.stderr.close()
    tool_proc.wait()
    return tool_proc.returncode, stderr, out.splitlines()


@pytest.fixture
def many_rows(tmp_path):
    """
    A structure with enough residues that the output cannot fit in the pipe
    buffer, so a reader closing early really does break the pipe mid-write.
    """
    lines = []
    for i in range(1, 4001):
        lines.append(pdb_atom_line(i, "CA", "GLY", "A", i, float(i), 0.0, 0.0))
    return write_pdb(tmp_path / "many.pdb", lines)


class TestBrokenPipe:
    """
    Piping into a reader that exits early (`| head`, quitting `less`) is normal
    use and must not produce a traceback.
    """

    def test_output_is_large_enough_to_break_the_pipe(self, many_rows):
        # Guards the fixture: a short output would fit in the pipe buffer and
        # the test would pass without ever exercising the broken-pipe path
        result = run_tool("nucleotide_conformation", many_rows, "-a")
        rows = run_tool("CA_difference", many_rows, many_rows).stdout
        assert len(rows) > 65536, "fixture too small to close the pipe mid-write"

    def test_ca_difference_pipes_cleanly(self, many_rows):
        code, stderr, lines = run_piped("CA_difference", [many_rows, many_rows])
        assert "BrokenPipeError" not in stderr
        assert "Traceback" not in stderr
        assert "Exception ignored" not in stderr
        assert len(lines) == 2

    def test_atom_tracker_pipes_cleanly(self, many_rows):
        code, stderr, lines = run_piped("atom_tracker", [many_rows, many_rows,
                                                         "--min-change", "-1"])
        assert "BrokenPipeError" not in stderr
        assert "Traceback" not in stderr
        assert len(lines) == 2

    def test_nucleotide_conformation_pipes_cleanly(self, rna):
        code, stderr, lines = run_piped("nucleotide_conformation", [rna, "-a"])
        assert "BrokenPipeError" not in stderr
        assert "Traceback" not in stderr

    @needs_scipy
    def test_find_contacts_pipes_cleanly(self, two_chains):
        code, stderr, lines = run_piped("find_contacts",
                                        [two_chains, "-c", "A", "-d", "4.0"])
        assert "BrokenPipeError" not in stderr
        assert "Traceback" not in stderr

    def test_the_rows_that_did_arrive_are_correct(self, many_rows):
        _, _, lines = run_piped("CA_difference", [many_rows, many_rows], head_lines=3)
        assert lines[0].split("\t")[0] == "Chain1"
        assert len(lines) == 3

    def test_writing_to_a_file_is_unaffected(self, many_rows, tmp_path):
        # The guard must not swallow anything when stdout is not a pipe
        target = tmp_path / "out.tsv"
        result = run_tool("CA_difference", many_rows, many_rows, "-o", target)
        assert result.returncode == 0, result.stderr
        assert len(target.read_text().splitlines()) == 4001


class TestVersion:
    @pytest.mark.parametrize("tool", ALL_TOOLS)
    def test_version_flag(self, tool):
        result = run_tool(tool, "--version")
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().startswith("pdb_python_tools ")

    def test_version_matches_the_package(self):
        from pdb_python_tools import __version__
        out = run_tool("atom_tracker", "--version").stdout.strip()
        assert out == "pdb_python_tools " + __version__

    @pytest.mark.parametrize("tool", ALL_TOOLS)
    def test_version_needs_no_input_file(self, tool):
        # --version must work before any positional argument is supplied
        assert run_tool(tool, "--version").returncode == 0


class TestErrorMessages:
    """User-triggerable errors give a short message, not a traceback."""

    @pytest.mark.parametrize("tool,args", [
        ("atom_tracker", ["MISSING", "MISSING"]),
        ("CA_difference", ["MISSING", "MISSING"]),
        ("find_contacts", ["MISSING", "-c", "A", "-d", "4"]),
        ("nucleotide_conformation", ["MISSING"]),
    ])
    def test_missing_file(self, tool, args, tmp_path):
        missing = str(tmp_path / "nope.pdb")
        result = run_tool(tool, *[missing if a == "MISSING" else a for a in args])
        assert result.returncode == 1
        assert "Traceback" not in result.stderr
        assert "no such file" in result.stderr
        assert missing in result.stderr

    @pytest.mark.parametrize("tool,args", [
        ("atom_tracker", ["BAD", "BAD"]),
        ("nucleotide_conformation", ["BAD"]),
    ])
    def test_unknown_extension(self, tool, args, tmp_path):
        bad = tmp_path / "structure.xyz"
        bad.write_text("")
        result = run_tool(tool, *[str(bad) if a == "BAD" else a for a in args])
        assert result.returncode == 1
        assert "Traceback" not in result.stderr
        assert "Unrecognized structure extension" in result.stderr

    def test_corrupt_gzip(self, tmp_path):
        bad = tmp_path / "broken.cif.gz"
        bad.write_text("not actually gzipped\n")
        result = run_tool("nucleotide_conformation", bad)
        assert result.returncode == 1
        assert "Traceback" not in result.stderr
        assert "error:" in result.stderr


class TestGzipCli:
    def test_gzipped_input_matches_plain(self, rna, tmp_path):
        packed = tmp_path / "rna.pdb.gz"
        with open(rna, "rb") as src, gzip.open(packed, "wb") as dst:
            shutil.copyfileobj(src, dst)
        plain_out = run_tool("nucleotide_conformation", rna, "-a")
        packed_out = run_tool("nucleotide_conformation", packed, "-a")
        assert packed_out.returncode == 0, packed_out.stderr
        assert packed_out.stdout == plain_out.stdout
        assert len(packed_out.stdout.splitlines()) == 3


class TestHelp:
    @pytest.mark.parametrize("tool", ALL_TOOLS)
    def test_help_exits_zero(self, tool):
        result = run_tool(tool, "-h")
        assert result.returncode == 0
        assert "usage:" in result.stdout

    @pytest.mark.parametrize("tool", ALL_TOOLS)
    def test_help_documents_the_shared_flags(self, tool):
        out = run_tool(tool, "-h").stdout
        for flag in ("--format", "--output", "--force", "--precision", "--coot"):
            assert flag in out, "%s missing %s" % (tool, flag)

    @pytest.mark.parametrize("tool", ALL_TOOLS)
    def test_missing_arguments_exit_nonzero(self, tool):
        assert run_tool(tool).returncode != 0


class TestAtomTracker:
    def test_reports_the_displacement(self, pair):
        result = run_tool("atom_tracker", *pair)
        assert result.returncode == 0, result.stderr
        lines = result.stdout.splitlines()
        assert lines[0].split("\t") == ["Chain", "Residue", "Residue name",
                                        "Max_Distance", "Max_atom",
                                        "Average_distance", "CA/C1'_distance"]
        rows = [line.split("\t") for line in lines[1:]]
        ser = [r for r in rows if r[0] == "A"][0]
        assert ser[2] == "SER"
        assert ser[3] == "3.00"
        assert ser[4] == "OG"
        assert ser[6] == "0.00"

    def test_nucleotide_c1_prime_column(self, pair):
        rows = [line.split("\t") for line in run_tool("atom_tracker", *pair).stdout.splitlines()[1:]]
        nucleotide = [r for r in rows if r[0] == "B"][0]
        assert nucleotide[6] == "2.00"

    def test_sorted_by_largest_displacement(self, pair):
        rows = [line.split("\t") for line in run_tool("atom_tracker", *pair).stdout.splitlines()[1:]]
        distances = [float(r[3]) for r in rows]
        assert distances == sorted(distances, reverse=True)

    def test_min_change_filters_rows(self, pair):
        result = run_tool("atom_tracker", *pair, "--min-change", "2.5")
        rows = result.stdout.splitlines()[1:]
        assert len(rows) == 1
        assert rows[0].split("\t")[2] == "SER"

    def test_csv_output(self, pair):
        result = run_tool("atom_tracker", *pair, "-f", "csv")
        assert result.stdout.splitlines()[0].startswith("Chain,Residue,")

    def test_precision(self, pair):
        result = run_tool("atom_tracker", *pair, "--precision", "4")
        assert "3.0000" in result.stdout

    def test_writes_to_a_file(self, pair, tmp_path):
        target = tmp_path / "out.tsv"
        result = run_tool("atom_tracker", *pair, "-o", target)
        assert result.returncode == 0
        assert result.stdout == ""
        assert target.read_text().startswith("Chain\t")

    def test_refuses_to_overwrite(self, pair, tmp_path):
        target = tmp_path / "out.tsv"
        target.write_text("keep me\n")
        result = run_tool("atom_tracker", *pair, "-o", target)
        assert result.returncode != 0
        assert "--force" in result.stderr
        assert target.read_text() == "keep me\n"

    def test_force_overwrites(self, pair, tmp_path):
        target = tmp_path / "out.tsv"
        target.write_text("replace me\n")
        result = run_tool("atom_tracker", *pair, "-o", target, "--force")
        assert result.returncode == 0
        assert target.read_text().startswith("Chain\t")

    def test_coot_script_is_valid_python(self, pair, tmp_path):
        target = tmp_path / "coot.py"
        result = run_tool("atom_tracker", *pair, "--coot", target)
        assert result.returncode == 0
        compile(target.read_text(), str(target), "exec")
        assert "A 1 SER" in target.read_text()

    def test_hetatm_and_hydrogen_flags_accepted(self, pair):
        assert run_tool("atom_tracker", *pair, "-HET", "-hy").returncode == 0

    def test_missing_input_exits_nonzero(self, tmp_path):
        result = run_tool("atom_tracker", tmp_path / "nope.pdb", tmp_path / "nope.pdb")
        assert result.returncode != 0


class TestUnmatchedAtoms:
    """
    Atoms with no counterpart in the second structure are not measured
    """

    def rows(self, pair, *extra):
        out = run_tool("atom_tracker", *pair, *extra).stdout.splitlines()
        return [line.split("\t") for line in out[1:]]

    def test_average_uses_only_the_matched_atoms(self, partial):
        ser = [r for r in self.rows(partial, "-HET") if r[2] == "SER"][0]
        # N, CA and OG matched (0, 0 and 3 A); the unmodelled CB is left out
        assert ser[3] == "3.00"
        assert ser[5] == "1.00"

    def test_max_is_unaffected(self, partial):
        ser = [r for r in self.rows(partial, "-HET") if r[2] == "SER"][0]
        assert (ser[3], ser[4]) == ("3.00", "OG")

    def test_residue_with_nothing_matched_is_dropped(self, partial):
        # PHE 2's only atom has no counterpart, so there is nothing to report
        assert [r for r in self.rows(partial, "-HET", "--min-change", "-1")
                if r[2] == "PHE"] == []

    def test_residue_without_a_ca_reports_na(self, partial):
        mg = [r for r in self.rows(partial, "-HET") if r[2] == "MG"][0]
        assert mg[3] == "4.00"
        assert mg[6] == "NA"

    def test_ca_missing_from_the_second_structure_reports_na(self, tmp_path):
        first = write_pdb(tmp_path / "withca.pdb", [
            pdb_atom_line(1, "CA", "SER", "A", 1, 0.0, 0.0, 0.0),
            pdb_atom_line(2, "OG", "SER", "A", 1, 2.0, 0.0, 0.0),
        ])
        second = write_pdb(tmp_path / "noca.pdb", [
            pdb_atom_line(1, "OG", "SER", "A", 1, 5.0, 0.0, 0.0),
        ])
        row = self.rows((first, second))[0]
        assert row[3] == "3.00"
        assert row[6] == "NA"


@needs_scipy
class TestFindContacts:
    def test_finds_the_contact(self, two_chains):
        result = run_tool("find_contacts", two_chains, "-c", "A", "-d", "4.0")
        assert result.returncode == 0, result.stderr
        lines = result.stdout.splitlines()
        assert lines[0].split("\t") == ["Residue1", "Residue1 number", "Chain2",
                                        "Residue2", "Residue2 number", "Distance"]
        assert len(lines) == 2
        assert lines[1].split("\t")[5] == "3.00"

    def test_one_row_per_residue_pair_by_default(self, two_chains):
        result = run_tool("find_contacts", two_chains, "-c", "A", "-d", "5.0")
        assert len(result.stdout.splitlines()) == 2

    def test_all_lists_every_atom_pair(self, two_chains):
        result = run_tool("find_contacts", two_chains, "-c", "A", "-d", "5.0", "-a")
        lines = result.stdout.splitlines()
        assert lines[0].split("\t")[0] == "Chain1"
        assert len(lines) == 3

    def test_polar_only(self, two_chains):
        result = run_tool("find_contacts", two_chains, "-c", "A", "-d", "5.0", "-a", "-p")
        assert len(result.stdout.splitlines()) == 2

    def test_no_contacts_prints_only_the_header(self, two_chains):
        result = run_tool("find_contacts", two_chains, "-c", "A", "-d", "0.5")
        assert result.returncode == 0
        assert len(result.stdout.splitlines()) == 1

    def test_chain_and_distance_are_required(self, two_chains):
        assert run_tool("find_contacts", two_chains).returncode != 0
        assert run_tool("find_contacts", two_chains, "-c", "A").returncode != 0

    def test_coot_script_is_valid_python(self, two_chains, tmp_path):
        target = tmp_path / "coot.py"
        run_tool("find_contacts", two_chains, "-c", "A", "-d", "4.0", "--coot", target)
        compile(target.read_text(), str(target), "exec")


@needs_scipy
class TestCaDifference:
    def test_reports_nearest_ca(self, pair):
        result = run_tool("CA_difference", *pair)
        assert result.returncode == 0, result.stderr
        lines = result.stdout.splitlines()
        assert lines[0].split("\t") == ["Chain1", "Residue1", "Residue name1",
                                        "Chain2", "Residue2", "Residue name2",
                                        "CA/C1'_distance"]
        rows = [line.split("\t") for line in lines[1:]]
        assert len(rows) == 2
        nucleotide = [r for r in rows if r[0] == "B"][0]
        assert nucleotide[6] == "2.00"

    def test_sorted_by_distance(self, pair):
        rows = [line.split("\t") for line in run_tool("CA_difference", *pair).stdout.splitlines()[1:]]
        distances = [float(r[6]) for r in rows]
        assert distances == sorted(distances, reverse=True)

    def test_coot_script_is_valid_python(self, pair, tmp_path):
        target = tmp_path / "coot.py"
        run_tool("CA_difference", *pair, "--coot", target)
        compile(target.read_text(), str(target), "exec")


class TestNucleotideConformation:
    def test_default_view_lists_syn_pyrimidines(self, rna):
        result = run_tool("nucleotide_conformation", rna)
        assert result.returncode == 0, result.stderr
        lines = result.stdout.splitlines()
        assert lines[0].split("\t") == ["Chain", "Residue", "Residue name",
                                        "Chi", "Conformation"]
        assert len(lines) == 2
        row = lines[1].split("\t")
        assert row[2] == "U"
        assert row[3] == "0.00"
        assert row[4] == "syn"

    def test_all_view_lists_every_nucleotide(self, rna):
        result = run_tool("nucleotide_conformation", rna, "-a")
        assert len(result.stdout.splitlines()) == 3

    def test_syn_view(self, rna):
        result = run_tool("nucleotide_conformation", rna, "-s")
        assert len(result.stdout.splitlines()) == 2

    def test_dna_default_view_lists_syn_pyrimidines(self, dna):
        result = run_tool("nucleotide_conformation", dna)
        assert result.returncode == 0, result.stderr
        rows = [line.split("\t") for line in result.stdout.splitlines()[1:]]
        assert [(r[2], r[3], r[4]) for r in rows] == [("DT", "0.00", "syn"),
                                                      ("DC", "45.00", "syn")]

    def test_dna_all_view_lists_every_nucleotide(self, dna):
        result = run_tool("nucleotide_conformation", dna, "-a")
        rows = [line.split("\t") for line in result.stdout.splitlines()[1:]]
        assert [(r[2], r[4]) for r in rows] == [("DT", "syn"), ("DC", "syn"),
                                                ("DA", "anti"), ("DG", "anti")]

    def test_dna_syn_view_includes_purines(self, dna, tmp_path):
        syn_purine = write_pdb(tmp_path / "syndg.pdb",
                               nucleotide_lines(1, "DG", "A", 1, 60.0))
        rows = run_tool("nucleotide_conformation", syn_purine, "-s").stdout.splitlines()[1:]
        assert [r.split("\t")[2] for r in rows] == ["DG"]
        assert run_tool("nucleotide_conformation", syn_purine).stdout.splitlines()[1:] == []

    def test_hybrid_structure_reports_both_strands(self, hybrid):
        result = run_tool("nucleotide_conformation", hybrid, "-a")
        assert result.returncode == 0, result.stderr
        rows = [line.split("\t") for line in result.stdout.splitlines()[1:]]
        assert [(r[0], r[2]) for r in rows] == [("A", "DT"), ("A", "DA"),
                                                ("B", "U"), ("B", "A")]

    def test_hybrid_default_view_flags_both_pyrimidines(self, hybrid):
        rows = [line.split("\t") for line in
                run_tool("nucleotide_conformation", hybrid).stdout.splitlines()[1:]]
        assert [(r[0], r[2]) for r in rows] == [("A", "DT"), ("B", "U")]

    def test_all_and_syn_are_mutually_exclusive(self, rna):
        result = run_tool("nucleotide_conformation", rna, "-a", "-s")
        assert result.returncode != 0
        assert "not allowed with" in result.stderr

    def test_margin_adds_the_borderline_column(self, rna):
        result = run_tool("nucleotide_conformation", rna, "-a", "-m", "5")
        header = result.stdout.splitlines()[0].split("\t")
        assert header[-1] == "Borderline"
        assert all(row.split("\t")[-1] in ("yes", "no")
                   for row in result.stdout.splitlines()[1:])

    def test_no_margin_column_by_default(self, rna):
        header = run_tool("nucleotide_conformation", rna, "-a").stdout.splitlines()[0]
        assert "Borderline" not in header

    @pytest.mark.parametrize("restyp", ["U", "DC"])
    def test_margin_surfaces_borderline_anti(self, restyp, tmp_path):
        """
        A pyrimidine at chi = -92 is anti, so the default view hides it, but it is
        close enough to the -90 boundary that -m 5 should surface and flag it.
        """
        borderline = write_pdb(tmp_path / ("border_%s.pdb" % restyp),
                               nucleotide_lines(1, restyp, "A", 1, -92.0))
        plain = run_tool("nucleotide_conformation", borderline)
        assert plain.stdout.splitlines()[1:] == []
        rows = run_tool("nucleotide_conformation", borderline,
                        "-m", "5").stdout.splitlines()[1:]
        assert len(rows) == 1
        assert rows[0].split("\t")[4] == "anti"
        assert rows[0].split("\t")[-1] == "yes"

    def test_precision(self, rna):
        result = run_tool("nucleotide_conformation", rna, "-a", "--precision", "3")
        assert "0.000" in result.stdout

    def test_coot_script_is_valid_python(self, rna, tmp_path):
        target = tmp_path / "coot.py"
        run_tool("nucleotide_conformation", rna, "--coot", target)
        compile(target.read_text(), str(target), "exec")
        assert "A 1 U" in target.read_text()

    @pytest.fixture
    def alt_rna(self, tmp_path):
        """One U modelled in two conformations: A is syn, B is anti."""
        lines = [
            pdb_atom_line(1, "O4'", "U", "A", 1, 0.0, 1.0, 0.0),
            pdb_atom_line(2, "C1'", "U", "A", 1, 0.0, 0.0, 0.0),
            pdb_atom_line(3, "N1", "U", "A", 1, 1.0, 0.0, 0.0),
        ]
        for serial, (alt, chi) in enumerate([("A", 30.0), ("B", 150.0)], start=4):
            t = math.radians(chi)
            lines.append(pdb_atom_line(serial, "C2", "U", "A", 1,
                                       1.0, math.cos(t), math.sin(t), altloc=alt))
        return write_pdb(tmp_path / "altrna.pdb", lines)

    def test_each_conformation_gets_its_own_row(self, alt_rna):
        result = run_tool("nucleotide_conformation", alt_rna, "-a")
        assert result.returncode == 0, result.stderr
        lines = result.stdout.splitlines()
        assert lines[0].split("\t") == ["Chain", "Residue", "Residue name",
                                        "Chi", "Conformation", "Altloc"]
        rows = [line.split("\t") for line in lines[1:]]
        assert [(r[3], r[4], r[5]) for r in rows] == [("30.00", "syn", "A"),
                                                      ("150.00", "anti", "B")]

    def test_altloc_column_absent_without_alternates(self, rna):
        header = run_tool("nucleotide_conformation", rna, "-a").stdout.splitlines()[0]
        assert "Altloc" not in header

    def test_default_view_flags_only_the_syn_conformation(self, alt_rna):
        rows = run_tool("nucleotide_conformation", alt_rna).stdout.splitlines()[1:]
        assert [r.split("\t")[5] for r in rows] == ["A"]

    def test_altloc_column_coexists_with_the_margin_column(self, alt_rna):
        header = run_tool("nucleotide_conformation", alt_rna, "-a",
                          "-m", "5").stdout.splitlines()[0].split("\t")
        assert header[-2:] == ["Altloc", "Borderline"]

    def test_coot_script_labels_each_conformation(self, alt_rna, tmp_path):
        target = tmp_path / "coot.py"
        run_tool("nucleotide_conformation", alt_rna, "-a", "--coot", target)
        content = target.read_text()
        compile(content, str(target), "exec")
        assert "alt A" in content and "alt B" in content
