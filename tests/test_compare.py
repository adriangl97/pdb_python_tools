"""
Tests for compare_pdb_resi_xyz
"""
import pytest

from pdb_python_tools.core import _SWAP, _SYMMETRIC, compare_pdb_resi_xyz

from conftest import make_atom, make_residue

# Every (residue type, atom name, partner name) the symmetry table declares
SYMMETRIC_CASES = [
    (restyp, first, second)
    for restyp, pairs in _SYMMETRIC.items()
    for first, second in pairs
]


def displacements(restyp, atoms1, atoms2):
    """Compare two single-residue structures and return {atom name: displacement}."""
    resi1 = make_residue(restyp, atoms1)
    resi2 = make_residue(restyp, atoms2)
    compare_pdb_resi_xyz([resi1], [resi2])
    return {atom.altid: atom.xyz_change for atom in resi1.atom_list}


class TestPlainMatching:
    def test_matched_atom_gets_its_displacement(self):
        moved = displacements("SER", [make_atom("OG", 0.0)], [make_atom("OG", 3.0)])
        assert moved["OG"] == pytest.approx(3.0)

    def test_displacement_is_three_dimensional(self):
        moved = displacements("SER", [make_atom("OG", 0.0, 0.0, 0.0)],
                                     [make_atom("OG", 1.0, 2.0, 2.0)])
        assert moved["OG"] == pytest.approx(3.0)

    def test_unmatched_atom_keeps_zero(self):
        # An atom with no counterpart is left at the constructor's 0 sentinel
        moved = displacements("SER", [make_atom("OG", 10.0)], [make_atom("N", 0.0)])
        assert moved["OG"] == 0

    def test_residue_missing_from_second_structure_is_skipped(self):
        resi1 = make_residue("SER", [make_atom("OG", 10.0)], seqid="1")
        other = make_residue("SER", [make_atom("OG", 0.0)], seqid="99")
        compare_pdb_resi_xyz([resi1], [other])
        assert resi1.atom_list[0].xyz_change == 0

    def test_first_occurrence_of_a_duplicated_name_wins(self):
        # Alternate conformations share an atom name; the first is used
        moved = displacements("SER", [make_atom("OG", 0.0)],
                                     [make_atom("OG", 2.0), make_atom("OG", 9.0)])
        assert moved["OG"] == pytest.approx(2.0)

    def test_non_symmetric_residue_ignores_other_atoms(self):
        # A nearby unrelated atom must not affect a non-symmetric residue
        moved = displacements("SER", [make_atom("OG", 10.0)],
                                     [make_atom("OG", 13.0), make_atom("N", 10.0)])
        assert moved["OG"] == pytest.approx(3.0)


class TestSymmetricAtoms:
    def test_unrelated_atom_does_not_mask_real_movement(self):
        """
        Regression test: the displacement of a symmetric atom is measured against
        its own name and its declared partner only.
        """
        moved = displacements("TYR",
                              [make_atom("CE1", 10.0)],
                              [make_atom("CE1", 15.0), make_atom("N", 10.0)])
        assert moved["CE1"] == pytest.approx(5.0)

    @pytest.mark.parametrize("restyp,first,second", SYMMETRIC_CASES,
                             ids=["%s-%s/%s" % case for case in SYMMETRIC_CASES])
    def test_decoy_atom_ignored_for_every_declared_pair(self, restyp, first, second):
        moved = displacements(
            restyp,
            [make_atom(first, 0.0), make_atom(second, 10.0)],
            [make_atom(first, 6.0), make_atom(second, 16.0), make_atom("N", 0.0)])
        assert moved[first] == pytest.approx(6.0)

    def test_decoy_nearer_than_the_partner_is_still_ignored(self):
        # CB sits closest to the original CG1, but it is not interchangeable with
        # it, so the answer is the distance to the partner CG2 (3.0), not to CB
        moved = displacements("VAL",
                              [make_atom("CG1", 0.0)],
                              [make_atom("CG1", 8.0),  
                               make_atom("CG2", 3.0),  
                               make_atom("CB", 1.0)])  
        assert moved["CG1"] == pytest.approx(3.0)

    def test_partner_further_away_does_not_win(self):
        moved = displacements("ASP",
                              [make_atom("OD1", 0.0)],
                              [make_atom("OD1", 2.0), make_atom("OD2", 9.0)])
        assert moved["OD1"] == pytest.approx(2.0)

    def test_partner_closer_wins(self):
        # A pure swap of the two equivalent atoms is not a real movement
        moved = displacements("ASP",
                              [make_atom("OD1", 0.0)],
                              [make_atom("OD1", 9.0), make_atom("OD2", 2.0)])
        assert moved["OD1"] == pytest.approx(2.0)

    @pytest.mark.parametrize("restyp,first,second", SYMMETRIC_CASES,
                             ids=["%s-%s/%s" % case for case in SYMMETRIC_CASES])
    def test_exact_swap_reports_no_movement(self, restyp, first, second):
        """A clean 180 degree flip of an equivalent pair should report ~0 for both."""
        moved = displacements(restyp,
                              [make_atom(first, 0.0), make_atom(second, 4.0)],
                              [make_atom(first, 4.0), make_atom(second, 0.0)])
        assert moved[first] == pytest.approx(0.0)
        assert moved[second] == pytest.approx(0.0)

    @pytest.mark.parametrize("restyp,first,second", SYMMETRIC_CASES,
                             ids=["%s-%s/%s" % case for case in SYMMETRIC_CASES])
    def test_swap_plus_real_shift_reports_the_shift(self, restyp, first, second):
        """
        With the pair swapped and shifted by 1 A, the reported displacement is
        the shift, not the swap distance.
        """
        moved = displacements(restyp,
                              [make_atom(first, 0.0), make_atom(second, 4.0)],
                              [make_atom(first, 5.0), make_atom(second, 1.0)])
        assert moved[first] == pytest.approx(1.0)
        assert moved[second] == pytest.approx(1.0)

    def test_symmetry_only_applies_to_the_declared_residue_type(self):
        # ASP's carboxylate OD1/OD2 are interchangeable, but ASN's OD1/ND2 are not
        moved = displacements("ASN",
                              [make_atom("OD1", 0.0)],
                              [make_atom("OD1", 9.0), make_atom("ND2", 2.0)])
        assert moved["OD1"] == pytest.approx(9.0)


class TestSymmetryTable:
    def test_declared_pairs(self):
        """
        The residue types whose atom names are genuinely interchangeable
        """
        assert _SYMMETRIC == {
            "TYR": (("CD1", "CD2"), ("CE1", "CE2")),
            "PHE": (("CD1", "CD2"), ("CE1", "CE2")),
            "GLU": (("OE1", "OE2"),),
            "ASP": (("OD1", "OD2"),),
            "ARG": (("NH1", "NH2"),),
            "LEU": (("CD1", "CD2"),),
            "VAL": (("CG1", "CG2"),),
        }

    def test_pairs_are_mutual(self):
        for restyp, partners in _SWAP.items():
            for name, partner in partners.items():
                assert partners[partner] == name, "%s %s" % (restyp, name)

    def test_swap_lookup_covers_every_declared_pair(self):
        assert set(_SWAP) == set(_SYMMETRIC)
        for restyp, pairs in _SYMMETRIC.items():
            expected = {n for pair in pairs for n in pair}
            assert set(_SWAP[restyp]) == expected


class TestCaDisplacement:
    def test_ca_displacement_recorded(self):
        resi1 = make_residue("SER", [make_atom("CA", 0.0), make_atom("OG", 1.0)])
        resi2 = make_residue("SER", [make_atom("CA", 2.0), make_atom("OG", 3.0)])
        compare_pdb_resi_xyz([resi1], [resi2])
        assert resi1.CA.xyz_change == pytest.approx(2.0)

    def test_c1_prime_displacement_recorded(self):
        resi1 = make_residue("G", [make_atom("C1'", 0.0)])
        resi2 = make_residue("G", [make_atom("C1'", 4.0)])
        compare_pdb_resi_xyz([resi1], [resi2])
        assert resi1.CA.xyz_change == pytest.approx(4.0)

    def test_no_ca_leaves_the_placeholder_untouched(self):
        resi1 = make_residue("MG", [make_atom("MG", 0.0, element="MG")])
        resi2 = make_residue("MG", [make_atom("MG", 5.0, element="MG")])
        compare_pdb_resi_xyz([resi1], [resi2])
        # The placeholder CA keeps its sentinel altid, so callers report NA
        assert resi1.CA.altid not in ("CA", "C1'")
        assert resi1.CA.xyz_change == 0

    def test_ca_missing_from_one_side_is_not_computed(self):
        resi1 = make_residue("SER", [make_atom("CA", 0.0), make_atom("OG", 1.0)])
        resi2 = make_residue("SER", [make_atom("OG", 3.0)])
        compare_pdb_resi_xyz([resi1], [resi2])
        assert resi1.CA.xyz_change == 0
