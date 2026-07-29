"""
Tests for the dihedral maths and the RNA/DNA syn/anti classification.
"""
import math

import pytest

from pdb_python_tools.core import (CONFORMATION_GROUPS, _NUCLEOTIDES, _PURINES,
                                   _PYRIMIDINES, _dihedral,
                                   classify_nucleotide_conformation,
                                   count_nucleotide_conformations,
                                   format_percentage, is_pyrimidine,
                                   nucleotide_chi_atoms)

from conftest import make_atom, make_residue

# Atom names defining chi, per base type
PURINE_NAMES = ("O4'", "C1'", "N9", "C4")
PYRIMIDINE_NAMES = ("O4'", "C1'", "N1", "C2")
# Pseudouridine and friends hang off C5 instead of N1
C_GLYCOSIDE_NAMES = ("O4'", "C1'", "C5", "C4")
RNA_BASES = ("A", "G", "C", "U")
DNA_BASES = ("DA", "DG", "DC", "DT", "DU")
PURINE_RESIDUES = ("A", "G", "DA", "DG")


def fourth_point(chi_deg):
    """The fourth dihedral point that yields `chi_deg` for the fixed first three."""
    t = math.radians(chi_deg)
    return (1.0, math.cos(t), math.sin(t))


def base_atoms(names, chi_deg, bond=1.0):
    """
    The four chi atoms, placed so chi is exactly `chi_deg` and the base sits
    `bond` A from the C1'
    """
    _, y, z = fourth_point(chi_deg)
    return [
        make_atom(names[0], 0.0, 1.0, 0.0),
        make_atom(names[1], 0.0, 0.0, 0.0),
        make_atom(names[2], bond, 0.0, 0.0),
        make_atom(names[3], bond, y, z),
    ]


def nucleotide(restyp, chi_deg, chainid="A", seqid="1"):
    """Build an RNA or DNA residue whose glycosidic chi is exactly `chi_deg`."""
    names = PURINE_NAMES if restyp in PURINE_RESIDUES else PYRIMIDINE_NAMES
    return make_residue(restyp, base_atoms(names, chi_deg), chainid=chainid,
                        seqid=seqid)


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


def modified(restyp, chi_deg, names, bond=1.0, extra=()):
    """
    A nucleotide carrying a non-standard residue name, i.e. what a modified
    base looks like when it is read from HETATM records.
    """
    atoms = base_atoms(names, chi_deg, bond=bond) + list(extra)
    return make_residue(restyp, atoms)


def c_glycoside(restyp, chi_deg, n1_distance=3.8, decoy_chi=None):
    """
    A pseudouridine-like residue: the base is joined to the sugar through C5,
    while the N1 a normal pyrimidine hangs off sits across the ring, ~3.8 A from
    the C1'. `decoy_chi` is the angle O4'-C1'-N1-C2 would give, so a test can
    tell the two torsions apart
    """
    if decoy_chi is None:
        decoy_chi = chi_deg + 120.0
    _, y, z = fourth_point(decoy_chi)
    atoms = base_atoms(C_GLYCOSIDE_NAMES, chi_deg) + [
        make_atom("N1", n1_distance, 0.0, 0.0),
        make_atom("C2", n1_distance, y, z),
        make_atom("N3", 50.0, 50.0, 50.0),
        make_atom("C6", 60.0, 60.0, 60.0),
    ]
    return make_residue(restyp, atoms)


class TestNucleotideChiAtoms:
    @pytest.mark.parametrize("restyp", list(RNA_BASES) + list(DNA_BASES))
    def test_standard_bases_come_from_the_name(self, restyp):
        expected = PURINE_NAMES if restyp in PURINE_RESIDUES else PYRIMIDINE_NAMES
        # No atoms at all: the residue name alone decides
        assert nucleotide_chi_atoms(make_residue(restyp, [])) == expected

    def test_modified_purine_is_read_off_the_atom_names(self):
        # 1MG has the atoms of a purine under a name the tool knows nothing about
        assert nucleotide_chi_atoms(modified("1MG", 0.0, PURINE_NAMES)) == PURINE_NAMES

    def test_modified_pyrimidine_is_read_off_the_atom_names(self):
        assert nucleotide_chi_atoms(
            modified("5MU", 0.0, PYRIMIDINE_NAMES)) == PYRIMIDINE_NAMES

    def test_a_base_without_a_sugar_is_not_a_nucleotide(self):
        # A free base or a ligand reusing the base atom names: no O4'/C1', so
        # there is no glycosidic torsion to measure
        atoms = [make_atom(n, float(i)) for i, n in enumerate(("N1", "C2", "N9", "C4"))]
        assert nucleotide_chi_atoms(make_residue("LIG", atoms)) is None

    def test_a_sugar_without_a_base_is_not_a_nucleotide(self):
        atoms = [make_atom(n, float(i)) for i, n in enumerate(("O4'", "C1'", "C2'"))]
        assert nucleotide_chi_atoms(make_residue("RIB", atoms)) is None

    def test_a_base_bonded_through_c5_is_a_c_glycoside(self):
        assert nucleotide_chi_atoms(c_glycoside("PSU", 0.0)) == C_GLYCOSIDE_NAMES

    def test_the_bonded_atom_decides_between_n1_and_c5(self):
        """
        Atom names alone cannot tell the two apart: a pseudouridine carries the
        N1 and C2 of a pyrimidine, and a pyrimidine carries a C5. Only the atom
        actually bonded to the C1' does.
        """
        pyrimidine = modified("5MU", 0.0, PYRIMIDINE_NAMES,
                              extra=[make_atom("C5", 3.4, 0.0, 0.0)])
        assert nucleotide_chi_atoms(pyrimidine) == PYRIMIDINE_NAMES
        assert nucleotide_chi_atoms(c_glycoside("PSU", 0.0)) == C_GLYCOSIDE_NAMES

    @pytest.mark.parametrize("restyp,names,expected", [
        ("U", PYRIMIDINE_NAMES, True),
        ("DC", PYRIMIDINE_NAMES, True),
        ("G", PURINE_NAMES, False),
        ("5MU", PYRIMIDINE_NAMES, True),
        ("1MG", PURINE_NAMES, False),
        ("PSU", C_GLYCOSIDE_NAMES, True),
        ("MG", ("MG",), False),
    ])
    def test_is_pyrimidine(self, restyp, names, expected):
        # A C-glycoside is a pyrimidine too, so it belongs in the default view
        atoms = [make_atom(n, float(i)) for i, n in enumerate(names)]
        assert is_pyrimidine(make_residue(restyp, atoms)) is expected


class TestModifiedNucleotides:
    """
    Residues that are not standard bases by name but carry the standard atom
    names, i.e. modified nucleotides read from HETATM records.
    """

    @pytest.mark.parametrize("restyp", ["1MG", "2MA", "G7M", "MA6"])
    def test_modified_purine_uses_n9_c4(self, restyp):
        """
        The purine/pyrimidine call has to come first: a purine also carries N1
        and C2, and measuring chi on those would give a meaningless angle.
        """
        decoys = [make_atom("N1", 50.0, 50.0, 50.0), make_atom("C2", 60.0, 60.0, 60.0)]
        results = classify_nucleotide_conformation(
            [modified(restyp, 30.0, PURINE_NAMES, extra=decoys)])
        assert [(round(chi, 6), conf) for _, chi, conf, _ in results] == [(30.0, "syn")]

    @pytest.mark.parametrize("restyp", ["5MU", "4SU", "H2U", "OMC"])
    def test_modified_pyrimidine_uses_n1_c2(self, restyp):
        decoys = [make_atom("N3", 50.0, 50.0, 50.0), make_atom("C4", 60.0, 60.0, 60.0)]
        results = classify_nucleotide_conformation(
            [modified(restyp, 120.0, PYRIMIDINE_NAMES, extra=decoys)])
        assert [(round(chi, 6), conf) for _, chi, conf, _ in results] == [(120.0, "anti")]

    @pytest.mark.parametrize("names", [PURINE_NAMES, PYRIMIDINE_NAMES])
    def test_a_base_with_nothing_bonded_to_the_sugar_is_skipped(self, names):
        """
        No atom within bonding distance of the C1' means no glycosidic bond, so
        there is no torsion to measure
        """
        assert classify_nucleotide_conformation(
            [modified("LIG", 30.0, names, bond=3.8)]) == []

    def test_standard_bases_are_measured_whatever_the_bond_length(self):
        # A stretched bond in a standard nucleotide is still a modelling error
        results = classify_nucleotide_conformation(
            [make_residue("U", base_atoms(PYRIMIDINE_NAMES, 30.0, bond=3.8))])
        assert [round(chi, 6) for _, chi, _, _ in results] == [30.0]

    def test_modified_base_missing_a_chi_atom_is_skipped(self):
        resi = modified("1MG", 0.0, PURINE_NAMES)
        resi.atom_list = resi.atom_list[:3]
        assert classify_nucleotide_conformation([resi]) == []

    def test_alternate_conformations_still_split(self):
        atoms = base_atoms(PURINE_NAMES, 0.0)[:3]
        for alt, chi in (("A", 20.0), ("B", 160.0)):
            _, y, z = fourth_point(chi)
            atoms.append(make_atom("C4", 1.0, y, z, altloc=alt))
        results = classify_nucleotide_conformation([make_residue("1MG", atoms)])
        assert [(round(chi, 6), alt) for _, chi, _, alt in results] == [
            (20.0, "A"), (160.0, "B")]

    def test_ligands_reusing_the_names_are_still_skipped(self):
        atoms = [make_atom("N1", 0.0), make_atom("C2", 1.0), make_atom("C4", 2.0)]
        assert classify_nucleotide_conformation([make_residue("LIG", atoms)]) == []

    def test_modified_and_standard_bases_mix(self):
        residues = [nucleotide("U", 0.0, seqid="1"),
                    modified("H2U", 20.0, PYRIMIDINE_NAMES),
                    nucleotide("G", 180.0, seqid="3")]
        results = classify_nucleotide_conformation(residues)
        assert [r[0].restyp for r in results] == ["U", "H2U", "G"]


class TestCGlycosides:
    """
    Pseudouridine and its derivatives (PSU, 3TD): the base is joined to the
    sugar through C5, so chi is O4'-C1'-C5-C4
    """

    @pytest.mark.parametrize("restyp", ["PSU", "3TD"])
    @pytest.mark.parametrize("chi", [-160.0, -90.0, 0.0, 40.0, 179.0])
    def test_chi_comes_from_c5_c4(self, restyp, chi):
        _, measured, conf, _ = classify_nucleotide_conformation(
            [c_glycoside(restyp, chi)])[0]
        assert measured == pytest.approx(chi, abs=1e-9)
        assert conf == ("syn" if -90 <= chi <= 90 else "anti")

    def test_the_n1_torsion_is_not_used(self):
        """
        The N1 is still there and still gives a dihedral, but it is not correct
        """
        resi = c_glycoside("PSU", 20.0, decoy_chi=140.0)
        _, measured, conf, _ = classify_nucleotide_conformation([resi])[0]
        assert measured == pytest.approx(20.0)
        assert conf == "syn"

    def test_c5_must_be_bonded_to_the_sugar(self):
        # C5 across the ring, as in a normal pyrimidine, is not a glycosidic bond
        atoms = base_atoms(C_GLYCOSIDE_NAMES, 30.0, bond=3.4)
        assert classify_nucleotide_conformation([make_residue("LIG", atoms)]) == []

    def test_counts_as_a_pyrimidine(self):
        assert is_pyrimidine(c_glycoside("PSU", 0.0))

    @pytest.mark.parametrize("chi,expected", [(45.0, "syn"), (-150.0, "anti")])
    def test_syn_anti_window_is_the_same(self, chi, expected):
        _, _, conf, _ = classify_nucleotide_conformation(
            [c_glycoside("PSU", chi)])[0]
        assert conf == expected

    def test_alternate_conformations_are_split(self):
        atoms = base_atoms(C_GLYCOSIDE_NAMES, 0.0)[:3]
        for alt, chi in (("A", 30.0), ("B", 150.0)):
            _, y, z = fourth_point(chi)
            atoms.append(make_atom("C4", 1.0, y, z, altloc=alt))
        results = classify_nucleotide_conformation([make_residue("PSU", atoms)])
        assert [(round(chi, 6), conf, alt) for _, chi, conf, alt in results] == [
            (30.0, "syn", "A"), (150.0, "anti", "B")]

    def test_missing_c4_is_skipped(self):
        resi = c_glycoside("PSU", 0.0)
        resi.atom_list = [a for a in resi.atom_list if a.altid != "C4"]
        assert classify_nucleotide_conformation([resi]) == []

    def test_mixes_with_the_other_base_types(self):
        residues = [nucleotide("U", 0.0, seqid="1"),
                    c_glycoside("PSU", 40.0),
                    modified("1MG", 170.0, PURINE_NAMES)]
        results = classify_nucleotide_conformation(residues)
        assert [(r[0].restyp, round(r[1], 6)) for r in results] == [
            ("U", 0.0), ("PSU", 40.0), ("1MG", 170.0)]


class TestCountNucleotideConformations:
    """The syn counts reported alongside the table, per base group."""

    @pytest.fixture
    def mixed(self):
        """Two syn and one anti pyrimidine, one syn and three anti purines."""
        residues = [nucleotide("U", 0.0, seqid="1"),
                    nucleotide("DC", 45.0, seqid="2"),
                    nucleotide("C", 170.0, seqid="3"),
                    nucleotide("G", 60.0, seqid="4"),
                    nucleotide("A", -175.0, seqid="5"),
                    nucleotide("DA", 120.0, seqid="6"),
                    nucleotide("DG", -100.0, seqid="7")]
        return classify_nucleotide_conformation(residues)

    def test_counts_per_group(self, mixed):
        counts = count_nucleotide_conformations(mixed)
        assert counts["pyrimidines"] == (2, 0, 3)
        assert counts["purines"] == (1, 0, 4)

    def test_the_total_is_the_sum_of_both_groups(self, mixed):
        counts = count_nucleotide_conformations(mixed)
        assert counts["nucleotides"] == (3, 0, 7)

    def test_c_glycosides_count_as_pyrimidines(self):
        results = classify_nucleotide_conformation([c_glycoside("PSU", 40.0)])
        counts = count_nucleotide_conformations(results)
        assert counts["pyrimidines"] == (1, 0, 1)
        assert counts["purines"] == (0, 0, 0)

    def test_modified_bases_are_counted(self):
        results = classify_nucleotide_conformation(
            [modified("5MU", 40.0, PYRIMIDINE_NAMES),
             modified("1MG", 170.0, PURINE_NAMES)])
        counts = count_nucleotide_conformations(results)
        assert (counts["pyrimidines"], counts["purines"]) == ((1, 0, 1), (0, 0, 1))

    def test_each_alternate_conformation_is_counted(self):
        atoms = base_atoms(PYRIMIDINE_NAMES, 0.0)[:3]
        for alt, chi in (("A", 30.0), ("B", 150.0)):
            _, y, z = fourth_point(chi)
            atoms.append(make_atom("C2", 1.0, y, z, altloc=alt))
        results = classify_nucleotide_conformation([make_residue("U", atoms)])
        assert count_nucleotide_conformations(results)["pyrimidines"] == (1, 0, 2)

    def test_borderline_counts_need_a_predicate(self, mixed):
        borderline = count_nucleotide_conformations(
            mixed, lambda chi: min(abs(chi - 90), abs(chi + 90)) <= 15)
        # chi = -100 (DG) and chi = 120 is not within 15 of either boundary
        assert borderline["purines"] == (1, 1, 4)
        assert borderline["pyrimidines"] == (2, 0, 3)

    def test_empty_input_counts_zero(self):
        counts = count_nucleotide_conformations([])
        assert all(counts[group] == (0, 0, 0) for group in CONFORMATION_GROUPS)


class TestFormatPercentage:
    def test_count_and_percentage(self):
        assert format_percentage(1, 4) == "1/4 (25.00%)"

    def test_precision_is_respected(self):
        assert format_percentage(1, 3, precision=1) == "1/3 (33.3%)"

    def test_full_precision(self):
        assert format_percentage(1, 4, full_precision=True) == "1/4 (25.0%)"

    def test_no_percentage_without_any_nucleotide(self):
        assert format_percentage(0, 0) == "0/0 (NA)"
