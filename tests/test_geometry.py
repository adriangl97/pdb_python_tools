"""
Tests for the dihedral maths and the RNA/DNA syn/anti classification.
"""
import math

import pytest

from pdb_python_tools.core import (_NUCLEOTIDES, _PURINES, _PYRIMIDINES,
                                   _dihedral, classify_nucleotide_conformation)

from conftest import make_atom, make_residue

# Atom names defining chi, per base type
PURINE_NAMES = ("O4'", "C1'", "N9", "C4")
PYRIMIDINE_NAMES = ("O4'", "C1'", "N1", "C2")
RNA_BASES = ("A", "G", "C", "U")
DNA_BASES = ("DA", "DG", "DC", "DT", "DU")
PURINE_RESIDUES = ("A", "G", "DA", "DG")


def fourth_point(chi_deg):
    """The fourth dihedral point that yields `chi_deg` for the fixed first three."""
    t = math.radians(chi_deg)
    return (1.0, math.cos(t), math.sin(t))


def nucleotide(restyp, chi_deg, chainid="A", seqid="1"):
    """Build an RNA or DNA residue whose glycosidic chi is exactly `chi_deg`."""
    names = PURINE_NAMES if restyp in PURINE_RESIDUES else PYRIMIDINE_NAMES
    x, y, z = fourth_point(chi_deg)
    atoms = [
        make_atom(names[0], 0.0, 1.0, 0.0),
        make_atom(names[1], 0.0, 0.0, 0.0),
        make_atom(names[2], 1.0, 0.0, 0.0),
        make_atom(names[3], x, y, z),
    ]
    return make_residue(restyp, atoms, chainid=chainid, seqid=seqid)


class TestDihedral:
    def test_eclipsed_is_zero(self):
        assert _dihedral((0, 1, 0), (0, 0, 0), (1, 0, 0), (1, 1, 0)) == pytest.approx(0.0)

    def test_anti_is_180(self):
        angle = _dihedral((0, 1, 0), (0, 0, 0), (1, 0, 0), (1, -1, 0))
        assert abs(angle) == pytest.approx(180.0)

    def test_positive_ninety(self):
        assert _dihedral((0, 1, 0), (0, 0, 0), (1, 0, 0), (1, 0, 1)) == pytest.approx(90.0)

    def test_negative_ninety(self):
        assert _dihedral((0, 1, 0), (0, 0, 0), (1, 0, 0), (1, 0, -1)) == pytest.approx(-90.0)

    @pytest.mark.parametrize("chi", [-179.0, -120.0, -90.0, -45.0, 0.0, 30.0,
                                     60.0, 90.0, 150.0, 179.0])
    def test_round_trip_through_the_construction(self, chi):
        angle = _dihedral((0, 1, 0), (0, 0, 0), (1, 0, 0), fourth_point(chi))
        assert angle == pytest.approx(chi, abs=1e-9)

    def test_sign_flips_with_mirrored_geometry(self):
        forward = _dihedral((0, 1, 0), (0, 0, 0), (1, 0, 0), (1, 0.5, 0.5))
        mirrored = _dihedral((0, 1, 0), (0, 0, 0), (1, 0, 0), (1, 0.5, -0.5))
        assert forward == pytest.approx(-mirrored)

    def test_central_bond_length_does_not_matter(self):
        near = _dihedral((0, 1, 0), (0, 0, 0), (1, 0, 0), (1, 0.5, 0.5))
        far = _dihedral((0, 1, 0), (0, 0, 0), (5, 0, 0), (5, 0.5, 0.5))
        assert near == pytest.approx(far)


class TestNucleotideTables:
    def test_purine_pyrimidine_split(self):
        # The split decides which base atoms chi is measured on
        assert _PURINES == {"A", "G", "DA", "DG"}
        assert _PYRIMIDINES == {"C", "U", "DC", "DT", "DU"}

    def test_the_two_groups_partition_the_standard_bases(self):
        # Every standard RNA/DNA base must land in exactly one group, otherwise
        # classify_nucleotide_conformation either skips it or measures chi on the
        # wrong atoms
        assert _PURINES & _PYRIMIDINES == set()
        assert _PURINES | _PYRIMIDINES == _NUCLEOTIDES
        assert _NUCLEOTIDES == set(RNA_BASES) | set(DNA_BASES)


class TestClassifyNucleotideConformation:
    @pytest.mark.parametrize("restyp", list(RNA_BASES) + list(DNA_BASES))
    def test_all_standard_bases_are_classified(self, restyp):
        results = classify_nucleotide_conformation([nucleotide(restyp, 0.0)])
        assert len(results) == 1
        assert results[0][0].restyp == restyp

    @pytest.mark.parametrize("restyp", ["U", "DT"])
    @pytest.mark.parametrize("chi", [0.0, 45.0, -45.0, 89.0, -89.0])
    def test_inside_the_window_is_syn(self, chi, restyp):
        _, measured, conformation, _ = classify_nucleotide_conformation(
            [nucleotide(restyp, chi)])[0]
        assert measured == pytest.approx(chi, abs=1e-9)
        assert conformation == "syn"

    @pytest.mark.parametrize("restyp", ["G", "DG"])
    @pytest.mark.parametrize("chi", [91.0, -91.0, 120.0, -120.0, 179.0])
    def test_outside_the_window_is_anti(self, chi, restyp):
        _, measured, conformation, _ = classify_nucleotide_conformation(
            [nucleotide(restyp, chi)])[0]
        assert measured == pytest.approx(chi, abs=1e-9)
        assert conformation == "anti"

    @pytest.mark.parametrize("restyp", ["C", "DC"])
    @pytest.mark.parametrize("chi", [90.0, -90.0])
    def test_boundary_is_inclusive_syn(self, chi, restyp):
        _, _, conformation, _ = classify_nucleotide_conformation(
            [nucleotide(restyp, chi)])[0]
        assert conformation == "syn"

    @pytest.mark.parametrize("rna,dna", [("A", "DA"), ("G", "DG"), ("C", "DC"),
                                         ("U", "DU")])
    @pytest.mark.parametrize("chi", [-150.0, -60.0, 0.0, 60.0, 150.0])
    def test_dna_and_rna_agree_on_identical_geometry(self, rna, dna, chi):
        _, rna_chi, rna_conf, _ = classify_nucleotide_conformation(
            [nucleotide(rna, chi)])[0]
        _, dna_chi, dna_conf, _ = classify_nucleotide_conformation(
            [nucleotide(dna, chi)])[0]
        assert rna_chi == pytest.approx(dna_chi)
        assert rna_conf == dna_conf

    @pytest.mark.parametrize("restyp", ["G", "DG"])
    def test_purines_use_n9_c4(self, restyp):
        """
        A purine also has N1 and C2, but its chi is measured on N9/C4
        """
        x, y, z = fourth_point(30.0)
        atoms = [
            make_atom("O4'", 0.0, 1.0, 0.0),
            make_atom("C1'", 0.0, 0.0, 0.0),
            make_atom("N9", 1.0, 0.0, 0.0),
            make_atom("C4", x, y, z),
            make_atom("N1", 50.0, 50.0, 50.0),
            make_atom("C2", 60.0, 60.0, 60.0),
        ]
        _, chi, _, _ = classify_nucleotide_conformation([make_residue(restyp, atoms)])[0]
        assert chi == pytest.approx(30.0)

    @pytest.mark.parametrize("restyp", ["U", "DT"])
    def test_pyrimidines_use_n1_c2(self, restyp):
        x, y, z = fourth_point(30.0)
        atoms = [
            make_atom("O4'", 0.0, 1.0, 0.0),
            make_atom("C1'", 0.0, 0.0, 0.0),
            make_atom("N1", 1.0, 0.0, 0.0),
            make_atom("C2", x, y, z),
            make_atom("N3", 50.0, 50.0, 50.0),
            make_atom("C4", 60.0, 60.0, 60.0),
        ]
        _, chi, _, _ = classify_nucleotide_conformation([make_residue(restyp, atoms)])[0]
        assert chi == pytest.approx(30.0)

    @pytest.mark.parametrize("restyp", ["G", "DG"])
    def test_purine_missing_n9_c4_is_skipped(self, restyp):
        # N1/C2 are present for a purine, but are not the chi atoms
        atoms = [make_atom("O4'", 0.0), make_atom("C1'", 1.0),
                 make_atom("N1", 2.0), make_atom("C2", 3.0)]
        assert classify_nucleotide_conformation([make_residue(restyp, atoms)]) == []

    @pytest.mark.parametrize("restyp,names", [
        ("SER", ["N", "CA", "CB", "OG"]),
        ("ALA", ["N", "CA", "CB"]),
        ("MG", ["MG"]),
        ("HOH", ["O"]),
    ])
    def test_non_nucleotide_residues_skipped(self, restyp, names):
        atoms = [make_atom(n, float(i)) for i, n in enumerate(names)]
        assert classify_nucleotide_conformation([make_residue(restyp, atoms)]) == []

    def test_no_altloc_reports_a_blank_id(self):
        assert classify_nucleotide_conformation([nucleotide("U", 30.0)])[0][3] == ""

    def test_each_conformation_is_measured_separately(self):
        """
        A nucleotide modelled twice gives one chi per conformation, each built
        only from that conformation's own atoms.
        """
        a = fourth_point(30.0)
        b = fourth_point(150.0)
        atoms = [
            make_atom("O4'", 0.0, 1.0, 0.0),
            make_atom("C1'", 0.0, 0.0, 0.0),
            make_atom("N1", 1.0, 0.0, 0.0),
            make_atom("C2", *a, altloc="A"),
            make_atom("C2", *b, altloc="B"),
        ]
        results = classify_nucleotide_conformation([make_residue("U", atoms)])
        assert [(round(chi, 6), conf, alt) for _, chi, conf, alt in results] == [
            (30.0, "syn", "A"), (150.0, "anti", "B")]

    def test_conformations_do_not_share_base_atoms(self):
        atoms = []
        for alt, chi in (("A", 20.0), ("B", 160.0)):
            x, y, z = fourth_point(chi)
            atoms += [make_atom("O4'", 0.0, 1.0, 0.0, altloc=alt),
                      make_atom("C1'", 0.0, 0.0, 0.0, altloc=alt),
                      make_atom("N9", 1.0, 0.0, 0.0, altloc=alt),
                      make_atom("C4", x, y, z, altloc=alt)]
        results = classify_nucleotide_conformation([make_residue("G", atoms)])
        assert [(round(chi, 6), alt) for _, chi, _, alt in results] == [
            (20.0, "A"), (160.0, "B")]

    def test_conformations_are_reported_in_id_order(self):
        atoms = [make_atom("O4'", 0.0, 1.0, 0.0), make_atom("C1'", 0.0, 0.0, 0.0),
                 make_atom("N1", 1.0, 0.0, 0.0)]
        for alt, chi in (("C", 10.0), ("A", 20.0), ("B", 30.0)):
            atoms.append(make_atom("C2", *fourth_point(chi), altloc=alt))
        results = classify_nucleotide_conformation([make_residue("U", atoms)])
        assert [alt for _, _, _, alt in results] == ["A", "B", "C"]

    def test_an_incomplete_conformation_is_skipped_on_its_own(self):
        atoms = [
            make_atom("O4'", 0.0, 1.0, 0.0),
            make_atom("C1'", 0.0, 0.0, 0.0),
            make_atom("N1", 1.0, 0.0, 0.0, altloc="A"),
            make_atom("C2", *fourth_point(30.0), altloc="A"),
            make_atom("N1", 1.0, 0.0, 0.0, altloc="B"),
        ]
        results = classify_nucleotide_conformation([make_residue("U", atoms)])
        assert [alt for _, _, _, alt in results] == ["A"]

    def test_residue_missing_a_chi_atom_is_skipped(self):
        resi = nucleotide("G", 0.0)
        resi.atom_list = resi.atom_list[:3]
        assert classify_nucleotide_conformation([resi]) == []

    def test_extra_atoms_are_ignored(self):
        resi = nucleotide("U", 30.0)
        resi.atom_list.append(make_atom("P", 50.0, 50.0, 50.0))
        _, chi, _, _ = classify_nucleotide_conformation([resi])[0]
        assert chi == pytest.approx(30.0)

    def test_duplicate_atom_names_use_the_first(self):
        resi = nucleotide("U", 30.0)
        shifted = make_atom(PYRIMIDINE_NAMES[3], *fourth_point(120.0))
        resi.atom_list.append(shifted)
        _, chi, _, _ = classify_nucleotide_conformation([resi])[0]
        assert chi == pytest.approx(30.0)

    def test_input_order_is_preserved(self):
        residues = [nucleotide("U", 0.0, seqid="1"),
                    nucleotide("G", 180.0, seqid="2"),
                    nucleotide("DC", 45.0, seqid="3"),
                    nucleotide("DA", -60.0, seqid="4")]
        results = classify_nucleotide_conformation(residues)
        assert [r[0].seqid for r in results] == ["1", "2", "3", "4"]

    def test_mixed_structure_only_returns_nucleotides(self):
        protein = make_residue("SER", [make_atom("CA", 0.0)], seqid="5")
        residues = [protein, nucleotide("U", 0.0, seqid="1"),
                    nucleotide("DT", 0.0, seqid="2")]
        results = classify_nucleotide_conformation(residues)
        assert [r[0].restyp for r in results] == ["U", "DT"]

    def test_hybrid_structure_classifies_both_strands(self):
        """A DNA/RNA hybrid: both chains must appear in the results."""
        residues = [nucleotide("DA", 10.0, chainid="A", seqid="1"),
                    nucleotide("DT", 20.0, chainid="A", seqid="2"),
                    nucleotide("A", 30.0, chainid="B", seqid="1"),
                    nucleotide("U", 40.0, chainid="B", seqid="2")]
        results = classify_nucleotide_conformation(residues)
        assert [(r[0].chainid, r[0].restyp) for r in results] == [
            ("A", "DA"), ("A", "DT"), ("B", "A"), ("B", "U")]

    def test_empty_input(self):
        assert classify_nucleotide_conformation([]) == []
