"""
End-to-end tests for the command-line tools.

"""
import math
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
