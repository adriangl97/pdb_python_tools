"""
Tests for the two scipy cKDTree searches: inter-chain contacts and the
nearest CA/C1' lookup.
"""
import pytest

pytest.importorskip("scipy", reason="find_contacts/CA_difference need scipy")

from pdb_python_tools.core import (find_contacts_kdtree,  # noqa: E402
                                   find_nearest_ca)

from conftest import make_atom, make_residue  # noqa: E402


def contacts(residues, distance, chain, polar=False):
    """Run the contact search and return comparable (name1, name2, dist) tuples."""
    return sorted((a1.altid, a2.altid, round(dist, 6))
                  for a1, a2, dist in find_contacts_kdtree(residues, distance,
                                                           chain, polar))


class TestFindContacts:
    def test_finds_a_contact_across_chains(self):
        residues = [
            make_residue("SER", [make_atom("OG", 0.0)], chainid="A", seqid="1"),
            make_residue("SER", [make_atom("OG", 3.0)], chainid="B", seqid="1"),
        ]
        assert contacts(residues, 4.0, "A") == [("OG", "OG", 3.0)]

    def test_ignores_pairs_beyond_the_cutoff(self):
        residues = [
            make_residue("SER", [make_atom("OG", 0.0)], chainid="A", seqid="1"),
            make_residue("SER", [make_atom("OG", 5.0)], chainid="B", seqid="1"),
        ]
        assert contacts(residues, 4.0, "A") == []

    def test_cutoff_is_inclusive(self):
        residues = [
            make_residue("SER", [make_atom("OG", 0.0)], chainid="A", seqid="1"),
            make_residue("SER", [make_atom("OG", 4.0)], chainid="B", seqid="1"),
        ]
        assert contacts(residues, 4.0, "A") == [("OG", "OG", 4.0)]

    def test_intra_chain_contacts_excluded(self):
        residues = [
            make_residue("SER", [make_atom("OG", 0.0)], chainid="A", seqid="1"),
            make_residue("SER", [make_atom("OG", 1.0)], chainid="A", seqid="2"),
        ]
        assert contacts(residues, 4.0, "A") == []

    def test_query_atom_always_comes_first(self):
        residues = [
            make_residue("SER", [make_atom("OG", 0.0)], chainid="A", seqid="1"),
            make_residue("GLY", [make_atom("N", 1.0)], chainid="B", seqid="1"),
        ]
        pairs = find_contacts_kdtree(residues, 4.0, "B", False)
        assert [(a1.chainid, a2.chainid) for a1, a2, _ in pairs] == [("B", "A")]

    def test_all_other_chains_are_targets(self):
        residues = [
            make_residue("SER", [make_atom("OG", 0.0)], chainid="A", seqid="1"),
            make_residue("GLY", [make_atom("N", 1.0)], chainid="B", seqid="1"),
            make_residue("ALA", [make_atom("CB", 2.0)], chainid="C", seqid="1"),
        ]
        assert contacts(residues, 4.0, "A") == [("OG", "CB", 2.0), ("OG", "N", 1.0)]

    def test_every_atom_pair_is_reported(self):
        residues = [
            make_residue("SER", [make_atom("OG", 0.0), make_atom("CB", 1.0)],
                         chainid="A", seqid="1"),
            make_residue("GLY", [make_atom("N", 2.0), make_atom("CA", 3.0)],
                         chainid="B", seqid="1"),
        ]
        assert len(find_contacts_kdtree(residues, 10.0, "A", False)) == 4

    def test_polar_filter_keeps_only_n_o_p_s(self):
        residues = [
            make_residue("SER", [make_atom("OG", 0.0), make_atom("CB", 0.5)],
                         chainid="A", seqid="1"),
            make_residue("GLY", [make_atom("N", 1.0), make_atom("CA", 1.5)],
                         chainid="B", seqid="1"),
        ]
        assert contacts(residues, 4.0, "A", polar=True) == [("OG", "N", 1.0)]

    def test_polar_filter_accepts_phosphorus_and_sulfur(self):
        residues = [
            make_residue("MET", [make_atom("SD", 0.0, element="S")],
                         chainid="A", seqid="1"),
            make_residue("G", [make_atom("P", 2.0, element="P")],
                         chainid="B", seqid="1"),
        ]
        assert contacts(residues, 4.0, "A", polar=True) == [("SD", "P", 2.0)]

    def test_unknown_chain_gives_no_contacts(self):
        residues = [make_residue("SER", [make_atom("OG", 0.0)], chainid="A", seqid="1")]
        assert find_contacts_kdtree(residues, 4.0, "Z", False) == []

    def test_single_chain_gives_no_contacts(self):
        residues = [make_residue("SER", [make_atom("OG", 0.0)], chainid="A", seqid="1")]
        assert find_contacts_kdtree(residues, 4.0, "A", False) == []

    def test_empty_input(self):
        assert find_contacts_kdtree([], 4.0, "A", False) == []

    def test_polar_filter_can_empty_a_side(self):
        residues = [
            make_residue("ALA", [make_atom("CB", 0.0)], chainid="A", seqid="1"),
            make_residue("GLY", [make_atom("N", 1.0)], chainid="B", seqid="1"),
        ]
        assert find_contacts_kdtree(residues, 4.0, "A", True) == []


class TestFindNearestCa:
    def test_finds_the_nearest_ca(self):
        pdb1 = [make_residue("SER", [make_atom("CA", 0.0)], seqid="1")]
        pdb2 = [make_residue("GLY", [make_atom("CA", 9.0)], seqid="10"),
                make_residue("ALA", [make_atom("CA", 2.0)], seqid="11")]
        results = find_nearest_ca(pdb1, pdb2)
        assert len(results) == 1
        resi1, resi2, dist = results[0]
        assert resi2.seqid == "11"
        assert dist == pytest.approx(2.0)

    def test_distance_recorded_on_the_residue(self):
        pdb1 = [make_residue("SER", [make_atom("CA", 0.0)], seqid="1")]
        pdb2 = [make_residue("ALA", [make_atom("CA", 3.0)], seqid="2")]
        find_nearest_ca(pdb1, pdb2)
        assert pdb1[0].CA.xyz_change == pytest.approx(3.0)

    def test_ca_does_not_match_c1_prime(self):
        pdb1 = [make_residue("SER", [make_atom("CA", 0.0)], seqid="1")]
        pdb2 = [make_residue("G", [make_atom("C1'", 1.0)], seqid="2")]
        assert find_nearest_ca(pdb1, pdb2) == []

    def test_c1_prime_matches_c1_prime(self):
        pdb1 = [make_residue("G", [make_atom("C1'", 0.0)], seqid="1")]
        pdb2 = [make_residue("SER", [make_atom("CA", 0.5)], seqid="2"),
                make_residue("U", [make_atom("C1'", 4.0)], seqid="3")]
        results = find_nearest_ca(pdb1, pdb2)
        assert [(r[1].seqid, round(r[2], 6)) for r in results] == [("3", 4.0)]

    def test_residue_without_ca_is_skipped(self):
        pdb1 = [make_residue("MG", [make_atom("MG", 0.0, element="MG")], seqid="1")]
        pdb2 = [make_residue("SER", [make_atom("CA", 1.0)], seqid="2")]
        assert find_nearest_ca(pdb1, pdb2) == []

    def test_mixed_structures_handle_both_atom_types(self):
        pdb1 = [make_residue("SER", [make_atom("CA", 0.0)], seqid="1"),
                make_residue("G", [make_atom("C1'", 0.0)], seqid="2")]
        pdb2 = [make_residue("ALA", [make_atom("CA", 1.0)], seqid="3"),
                make_residue("U", [make_atom("C1'", 2.0)], seqid="4")]
        results = find_nearest_ca(pdb1, pdb2)
        assert [(r[0].seqid, r[1].seqid) for r in results] == [("1", "3"), ("2", "4")]

    def test_input_order_is_preserved(self):
        pdb1 = [make_residue("SER", [make_atom("CA", float(i))], seqid=str(i))
                for i in range(5)]
        pdb2 = [make_residue("ALA", [make_atom("CA", 100.0)], seqid="x")]
        results = find_nearest_ca(pdb1, pdb2)
        assert [r[0].seqid for r in results] == ["0", "1", "2", "3", "4"]

    def test_no_targets_gives_no_results(self):
        pdb1 = [make_residue("SER", [make_atom("CA", 0.0)], seqid="1")]
        pdb2 = [make_residue("MG", [make_atom("MG", 1.0, element="MG")], seqid="2")]
        assert find_nearest_ca(pdb1, pdb2) == []

    def test_empty_input(self):
        assert find_nearest_ca([], []) == []
