"""
Tests for the PDB and mmCIF parsers and the extension-based dispatch.

Both parsers are fed the same small structure (see the tiny_pdb / tiny_cif fixtures)
"""
import shutil

import pytest

from pdb_python_tools.core import (_dequote, _element_from_pdb, _is_hydrogen,
                                   _safe_float, get_resi_from_cif,
                                   get_resi_from_pdb, load_residues)

from conftest import atom_names, by_key, pdb_atom_line, write_pdb


class TestHelpers:
    @pytest.mark.parametrize("token,expected", [
        ("  1.5 ", 1.5),
        ("", 0.0),
        ("   ", 0.0),
        ("-3.25", -3.25),
    ])
    def test_safe_float(self, token, expected):
        assert _safe_float(token) == expected

    @pytest.mark.parametrize("token,expected", [
        ('"C1\'"', "C1'"),
        ("CA", "CA"),
        ('"O4\'"', "O4'"),
        ('"', '"'),      
        ("", ""),
    ])
    def test_dequote(self, token, expected):
        assert _dequote(token) == expected

    @pytest.mark.parametrize("element,expected", [
        ("H", True), ("h", True), ("D", True), ("d", True),
        ("C", False), ("MG", False), ("", False),
    ])
    def test_is_hydrogen(self, element, expected):
        assert _is_hydrogen(element) is expected

    def test_element_read_from_columns_77_78(self):
        line = pdb_atom_line(1, "CA", "SER", "A", 1, 0.0, 0.0, 0.0, element="C")
        assert _element_from_pdb(line, "CA") == "C"

    def test_element_falls_back_to_atom_name(self):
        line = pdb_atom_line(1, "CB", "SER", "A", 1, 0.0, 0.0, 0.0,
                             blank_element=True)
        assert _element_from_pdb(line, "CB") == "C"

    def test_element_fallback_skips_leading_digits(self):
        assert _element_from_pdb(" " * 80, "1HB") == "H"

    def test_element_fallback_without_letters(self):
        assert _element_from_pdb(" " * 80, "123") == ""


class TestPdbParser:
    def test_residues_and_atoms(self, tiny_pdb):
        residues = get_resi_from_pdb(tiny_pdb, False, False)
        keyed = by_key(residues)
        # HETATM excluded, hydrogens excluded
        assert set(keyed) == {("A", "10"), ("A", "10A"), ("B", "1")}
        assert atom_names(keyed[("A", "10")]) == ["N", "CA", "OG", "CB"]

    def test_fields_are_parsed(self, tiny_pdb):
        resi = by_key(get_resi_from_pdb(tiny_pdb, False, False))[("A", "10")]
        atom = resi.atom_list[1]
        assert atom.altid == "CA"
        assert atom.restyp == "SER"
        assert atom.chainid == "A"
        assert atom.seqid == "10"
        assert atom.element == "C"
        assert (atom.x, atom.y, atom.z) == (1.0, 0.0, 0.0)
        assert atom.occ == 1.0
        assert atom.biso == 20.0
        assert atom.atomid == "2"

    def test_blank_element_column_falls_back(self, tiny_pdb):
        resi = by_key(get_resi_from_pdb(tiny_pdb, False, False))[("A", "10")]
        cb = [a for a in resi.atom_list if a.altid == "CB"][0]
        assert cb.element == "C"

    def test_hydrogens_excluded_by_default(self, tiny_pdb):
        resi = by_key(get_resi_from_pdb(tiny_pdb, False, False))[("A", "10")]
        assert "HA" not in atom_names(resi)

    def test_hydrogens_included_on_request(self, tiny_pdb):
        resi = by_key(get_resi_from_pdb(tiny_pdb, False, True))[("A", "10")]
        assert "HA" in atom_names(resi)

    def test_hetatm_excluded_by_default(self, tiny_pdb):
        assert ("C", "1") not in by_key(get_resi_from_pdb(tiny_pdb, False, False))

    def test_hetatm_included_on_request(self, tiny_pdb):
        keyed = by_key(get_resi_from_pdb(tiny_pdb, True, False))
        assert ("C", "1") in keyed
        assert keyed[("C", "1")].restyp == "MG"

    def test_insertion_code_starts_a_new_residue(self, tiny_pdb):
        keyed = by_key(get_resi_from_pdb(tiny_pdb, False, False))
        # 10 and 10A must not be merged
        assert keyed[("A", "10")].restyp == "SER"
        assert keyed[("A", "10A")].restyp == "GLY"

    def test_ca_recorded_on_residue(self, tiny_pdb):
        resi = by_key(get_resi_from_pdb(tiny_pdb, False, False))[("A", "10")]
        assert resi.CA.altid == "CA"
        assert (resi.CA.x, resi.CA.y, resi.CA.z) == (1.0, 0.0, 0.0)

    def test_ca_is_a_separate_object_from_the_atom_list_copy(self, tiny_pdb):
        resi = by_key(get_resi_from_pdb(tiny_pdb, False, False))[("A", "10")]
        listed = [a for a in resi.atom_list if a.altid == "CA"][0]
        assert resi.CA is not listed

    def test_c1_prime_fills_the_ca_slot(self, tiny_pdb):
        resi = by_key(get_resi_from_pdb(tiny_pdb, False, False))[("B", "1")]
        assert resi.CA.altid == "C1'"
        assert resi.CA.z == 9.0

    def test_residue_without_ca_gets_the_placeholder(self, tiny_pdb):
        resi = by_key(get_resi_from_pdb(tiny_pdb, True, False))[("C", "1")]
        assert resi.CA.altid not in ("CA", "C1'")

    def test_non_atom_records_ignored(self, tmp_path):
        path = write_pdb(tmp_path / "hdr.pdb", [
            "HEADER    A TEST STRUCTURE",
            "REMARK   2 RESOLUTION.    2.00 ANGSTROMS.",
            "ANISOU    1  N   SER A  10     1000   1000   1000",
            pdb_atom_line(1, "CA", "SER", "A", 10, 0.0, 0.0, 0.0),
            "TER",
        ])
        residues = get_resi_from_pdb(path, True, True)
        assert len(residues) == 1
        assert atom_names(residues[0]) == ["CA"]

    def test_negative_coordinates(self, tmp_path):
        path = write_pdb(tmp_path / "neg.pdb", [
            pdb_atom_line(1, "CA", "SER", "A", 1, -12.345, -0.5, -100.25),
        ])
        atom = get_resi_from_pdb(path, False, False)[0].atom_list[0]
        assert (atom.x, atom.y, atom.z) == (-12.345, -0.5, -100.25)

    def test_empty_file_gives_no_residues(self, tmp_path):
        path = tmp_path / "empty.pdb"
        path.write_text("")
        assert get_resi_from_pdb(str(path), True, True) == []


class TestCifParser:
    def test_residues_and_atoms(self, tiny_cif):
        keyed = by_key(get_resi_from_cif(tiny_cif, False, False))
        assert set(keyed) == {("A", "10"), ("A", "10A"), ("B", "1")}
        assert atom_names(keyed[("A", "10")]) == ["N", "CA", "OG", "CB"]

    def test_fields_are_parsed(self, tiny_cif):
        resi = by_key(get_resi_from_cif(tiny_cif, False, False))[("A", "10")]
        atom = resi.atom_list[1]
        assert atom.altid == "CA"
        assert atom.restyp == "SER"
        assert atom.element == "C"
        assert (atom.x, atom.y, atom.z) == (1.0, 0.0, 0.0)
        assert atom.occ == 1.0
        assert atom.biso == 20.0

    def test_primed_atom_names_are_dequoted(self, tiny_cif):
        resi = by_key(get_resi_from_cif(tiny_cif, False, False))[("B", "1")]
        assert atom_names(resi) == ["C1'", "O4'"]
        # A quoted name must still fill the CA slot
        assert resi.CA.altid == "C1'"

    def test_auth_fields_win_over_label_fields(self, tiny_cif):
        # The fixture puts label_asym_id/label_seq_id before their auth
        # counterparts; auth must still be the one used
        keyed = by_key(get_resi_from_cif(tiny_cif, True, False))
        assert ("C", "1") in keyed

    def test_insertion_code_appended_to_seqid(self, tiny_cif):
        keyed = by_key(get_resi_from_cif(tiny_cif, False, False))
        assert keyed[("A", "10A")].restyp == "GLY"

    def test_placeholder_ins_codes_ignored(self, tiny_cif):
        # '?' and '.' mean "no insertion code" and must not reach the seqid
        keyed = by_key(get_resi_from_cif(tiny_cif, False, False))
        assert ("A", "10") in keyed
        assert not any(k[1].endswith("?") or k[1].endswith(".") for k in keyed)

    def test_hydrogens_excluded_by_default(self, tiny_cif):
        resi = by_key(get_resi_from_cif(tiny_cif, False, False))[("A", "10")]
        assert "HA" not in atom_names(resi)

    def test_hydrogens_included_on_request(self, tiny_cif):
        resi = by_key(get_resi_from_cif(tiny_cif, False, True))[("A", "10")]
        assert "HA" in atom_names(resi)

    def test_hetatm_excluded_by_default(self, tiny_cif):
        assert ("C", "1") not in by_key(get_resi_from_cif(tiny_cif, False, False))

    def test_hetatm_included_on_request(self, tiny_cif):
        keyed = by_key(get_resi_from_cif(tiny_cif, True, False))
        assert keyed[("C", "1")].restyp == "MG"

    def test_pdb_and_cif_agree(self, tiny_pdb, tiny_cif):
        """The two fixtures describe the same structure, so both parsers must agree."""
        from_pdb = by_key(get_resi_from_pdb(tiny_pdb, True, True))
        from_cif = by_key(get_resi_from_cif(tiny_cif, True, True))
        assert set(from_pdb) == set(from_cif)
        for key, resi in from_pdb.items():
            other = from_cif[key]
            assert resi.restyp == other.restyp
            assert atom_names(resi) == atom_names(other)
            for a, b in zip(resi.atom_list, other.atom_list):
                assert (a.x, a.y, a.z) == (b.x, b.y, b.z)
                assert a.element == b.element


class TestLoadResidues:
    @pytest.mark.parametrize("suffix", [".pdb", ".ent", ".PDB"])
    def test_pdb_extensions(self, tiny_pdb, tmp_path, suffix):
        target = tmp_path / ("copy" + suffix)
        shutil.copy(tiny_pdb, target)
        assert len(load_residues(str(target), False, False)) == 3

    @pytest.mark.parametrize("suffix", [".cif", ".mmcif", ".CIF"])
    def test_cif_extensions(self, tiny_cif, tmp_path, suffix):
        target = tmp_path / ("copy" + suffix)
        shutil.copy(tiny_cif, target)
        assert len(load_residues(str(target), False, False)) == 3

    def test_unknown_extension_raises(self, tmp_path):
        path = tmp_path / "structure.xyz"
        path.write_text("")
        with pytest.raises(ValueError, match="Unrecognized structure extension"):
            load_residues(str(path), False, False)

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_residues(str(tmp_path / "nope.pdb"), False, False)
