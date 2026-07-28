"""
Tests for the PDB and mmCIF parsers and the extension-based dispatch.

Both parsers are fed the same small structure (see the tiny_pdb / tiny_cif fixtures)
"""
import gzip
import os
import shutil

import pytest

from pdb_python_tools.core import (_element_from_pdb, _is_hydrogen,
                                   _safe_float, _split_cif_tokens,
                                   get_resi_from_cif, get_resi_from_pdb,
                                   load_residues, load_residues_or_exit)

from conftest import (atom_names, by_key, cif_atom_row, pdb_atom_line,
                      write_cif, write_pdb)


class TestHelpers:
    @pytest.mark.parametrize("token,expected", [
        ("  1.5 ", 1.5),
        ("", 0.0),
        ("   ", 0.0),
        ("-3.25", -3.25),
    ])
    def test_safe_float(self, token, expected):
        assert _safe_float(token) == expected

    @pytest.mark.parametrize("line,expected", [
        # Plain values
        ("ATOM 1 C CA . SER A 1", ["ATOM", "1", "C", "CA", ".", "SER", "A", "1"]),
        # A primed atom name, double-quoted
        ('ATOM 1 C "C1\'" . G A 1', ["ATOM", "1", "C", "C1'", ".", "G", "A", "1"]),
        ('ATOM 1 O "O4\'" .', ["ATOM", "1", "O", "O4'", "."]),
        # The same name written bare
        ("ATOM 1 O O5' .", ["ATOM", "1", "O", "O5'", "."]),
        ("ATOM 1 C C4'", ["ATOM", "1", "C", "C4'"]),
        # Single quotes
        ("ATOM 'a value' X", ["ATOM", "a value", "X"]),
        ('ATOM "two words" X', ["ATOM", "two words", "X"]),
        # Irregular whitespace and trailing newline
        ("  ATOM   1  \t C \n", ["ATOM", "1", "C"]),
        ("", []),
        ("   \n", []),
    ])
    def test_split_cif_tokens(self, line, expected):
        assert _split_cif_tokens(line) == expected

    def test_split_cif_tokens_keeps_an_interior_quote(self):
        # A quote that is not at the start of a token is part of the value
        assert _split_cif_tokens("O5' C1'") == ["O5'", "C1'"]

    def test_split_cif_tokens_matches_whitespace_split_for_plain_rows(self):
        line = "ATOM 12 N N . ALA A 3 ? 1.000 2.000 3.000 1.00 20.00 1\n"
        assert _split_cif_tokens(line) == line.split()

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


MINIMAL_ROW = "ATOM 1 C CA . SER A 1 ? 1.000 2.000 3.000 1.00 20.00 10 B"
MINIMAL_TAGS = ["group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id",
                "label_comp_id", "label_asym_id", "label_seq_id",
                "pdbx_PDB_ins_code", "Cartn_x", "Cartn_y", "Cartn_z",
                "occupancy", "B_iso_or_equiv", "auth_seq_id", "auth_asym_id"]


def write_loop(path, tags, rows, prefix="data_test\n#\n", suffix="#\n"):
    """Write a bare mmCIF loop with the given _atom_site tags and data rows."""
    text = prefix + "loop_\n"
    text += "".join("_atom_site.%s\n" % tag for tag in tags)
    text += "".join(row.rstrip("\n") + "\n" for row in rows)
    text += suffix
    path.write_text(text)
    return str(path)


class TestCifLoopHeader:
    """
    The _atom_site loop is located by reading the loop_ header, so column order,
    extra columns and neighbouring loops must not matter.
    """

    def test_minimal_loop(self, tmp_path):
        path = write_loop(tmp_path / "m.cif", MINIMAL_TAGS, [MINIMAL_ROW])
        residues = get_resi_from_cif(path, False, False)
        assert by_key(residues).keys() == {("B", "10")}

    def test_column_order_does_not_matter(self, tmp_path):
        # Reverse every column
        tags = list(reversed(MINIMAL_TAGS))
        row = " ".join(reversed(MINIMAL_ROW.split()))
        path = write_loop(tmp_path / "rev.cif", tags, [row])
        resi = get_resi_from_cif(path, False, False)[0]
        assert (resi.chainid, resi.seqid, resi.restyp) == ("B", "10", "SER")
        assert (resi.atom_list[0].x, resi.atom_list[0].y, resi.atom_list[0].z) == (1.0, 2.0, 3.0)

    def test_unknown_extra_columns_are_ignored(self, tmp_path):
        tags = MINIMAL_TAGS + ["pdbx_formal_charge", "pdbx_PDB_model_num"]
        path = write_loop(tmp_path / "x.cif", tags, [MINIMAL_ROW + " ? 1"])
        assert len(get_resi_from_cif(path, False, False)) == 1

    def test_esd_columns_do_not_shadow_their_values(self, tmp_path):
        """Cartn_x_esd must never be mistaken for Cartn_x."""
        tags = MINIMAL_TAGS + ["Cartn_x_esd", "Cartn_y_esd", "Cartn_z_esd",
                               "occupancy_esd", "B_iso_or_equiv_esd"]
        path = write_loop(tmp_path / "esd.cif", tags,
                          [MINIMAL_ROW + " 9.999 9.999 9.999 9.99 9.99"])
        atom = get_resi_from_cif(path, False, False)[0].atom_list[0]
        assert (atom.x, atom.y, atom.z) == (1.0, 2.0, 3.0)
        assert atom.occ == 1.0
        assert atom.biso == 20.0

    def test_esd_columns_work_without_a_trailing_newline(self, tmp_path):
        tags = MINIMAL_TAGS + ["Cartn_x_esd"]
        target = tmp_path / "nonl.cif"
        text = "data_t\nloop_\n"
        text += "".join("_atom_site.%s\n" % t for t in tags)
        text += MINIMAL_ROW + " 9.999"          # no trailing newline
        target.write_text(text)
        atom = get_resi_from_cif(str(target), False, False)[0].atom_list[0]
        assert atom.x == 1.0

    def test_other_loops_are_skipped(self, tmp_path):
        target = tmp_path / "multi.cif"
        text = "data_test\n#\nloop_\n_entity.id\n_entity.type\n1 polymer\n2 water\n#\n"
        text += "loop_\n"
        text += "".join("_atom_site.%s\n" % t for t in MINIMAL_TAGS)
        text += MINIMAL_ROW + "\n#\n"
        # A related loop whose tags start with _atom_site_ but are not _atom_site.
        text += "loop_\n_atom_site_anisotrop.id\n_atom_site_anisotrop.U[1][1]\n1 0.1\n2 0.2\n#\n"
        target.write_text(text)
        residues = get_resi_from_cif(str(target), False, False)
        assert len(residues) == 1
        assert atom_names(residues[0]) == ["CA"]

    def test_a_second_atom_site_loop_is_also_read(self, tmp_path):
        target = tmp_path / "two.cif"
        text = "data_test\n#\nloop_\n"
        text += "".join("_atom_site.%s\n" % t for t in MINIMAL_TAGS)
        text += MINIMAL_ROW + "\n#\n"
        text += "loop_\n"
        text += "".join("_atom_site.%s\n" % t for t in MINIMAL_TAGS)
        text += MINIMAL_ROW.replace(" 10 B", " 11 B") + "\n#\n"
        target.write_text(text)
        assert len(get_resi_from_cif(str(target), False, False)) == 2

    def test_key_value_items_end_the_data_block(self, tmp_path):
        target = tmp_path / "kv.cif"
        text = "data_test\n#\nloop_\n"
        text += "".join("_atom_site.%s\n" % t for t in MINIMAL_TAGS)
        text += MINIMAL_ROW + "\n"
        text += "_cell.length_a 100.0\n"
        text += "some stray data line that must not be parsed as an atom\n"
        target.write_text(text)
        assert len(get_resi_from_cif(str(target), False, False)) == 1

    def test_missing_atom_site_loop_raises(self, tmp_path):
        target = tmp_path / "none.cif"
        target.write_text("data_test\n#\nloop_\n_entity.id\n1\n#\n")
        with pytest.raises(ValueError, match="No _atom_site loop found"):
            get_resi_from_cif(str(target), False, False)

    def test_empty_cif_raises_rather_than_returning_nothing(self, tmp_path):
        target = tmp_path / "empty.cif"
        target.write_text("")
        with pytest.raises(ValueError, match="No _atom_site loop found"):
            get_resi_from_cif(str(target), False, False)

    @pytest.mark.parametrize("drop", ["label_atom_id", "label_comp_id",
                                      "Cartn_x", "Cartn_y", "Cartn_z"])
    def test_missing_required_column_names_it(self, tmp_path, drop):
        """A malformed loop must say what is missing, not raise UnboundLocalError."""
        tags = [t for t in MINIMAL_TAGS if t != drop]
        row = " ".join(v for t, v in zip(MINIMAL_TAGS, MINIMAL_ROW.split()) if t != drop)
        path = write_loop(tmp_path / "bad.cif", tags, [row])
        with pytest.raises(ValueError, match="missing required column"):
            get_resi_from_cif(path, False, False)

    def test_missing_chain_column_is_reported(self, tmp_path):
        tags = [t for t in MINIMAL_TAGS if t not in ("label_asym_id", "auth_asym_id")]
        row = " ".join(v for t, v in zip(MINIMAL_TAGS, MINIMAL_ROW.split())
                       if t not in ("label_asym_id", "auth_asym_id"))
        path = write_loop(tmp_path / "nochain.cif", tags, [row])
        with pytest.raises(ValueError, match="chainid"):
            get_resi_from_cif(path, False, False)

    def test_optional_columns_may_be_absent(self, tmp_path):
        keep = ["label_atom_id", "label_comp_id", "label_asym_id", "label_seq_id",
                "Cartn_x", "Cartn_y", "Cartn_z"]
        row = "CA SER A 1 1.000 2.000 3.000"
        path = write_loop(tmp_path / "min.cif", keep, [row])
        resi = get_resi_from_cif(path, False, False)[0]
        atom = resi.atom_list[0]
        # With no group_PDB column a row is taken as an ATOM record unless the
        # line itself begins with HETATM
        assert atom.altid == "CA"
        assert atom.occ == 1.0      # defaulted
        assert atom.biso == 0.0     # defaulted
        assert atom.element == "C"  # derived from the atom name, no type_symbol

    def test_truncated_row_is_skipped(self, tmp_path):
        path = write_loop(tmp_path / "short.cif", MINIMAL_TAGS,
                          [MINIMAL_ROW, "ATOM 2 C CB"])
        assert sum(len(r.atom_list) for r in get_resi_from_cif(path, False, False)) == 1

    def test_null_coordinates_skip_the_atom(self, tmp_path):
        bad = MINIMAL_ROW.replace(" 1.000 2.000 3.000 ", " ? ? ? ")
        path = write_loop(tmp_path / "null.cif", MINIMAL_TAGS, [MINIMAL_ROW, bad])
        assert sum(len(r.atom_list) for r in get_resi_from_cif(path, False, False)) == 1

    def test_group_pdb_selects_hetatm(self, tmp_path):
        # The HETATM row sits on its own residue so the two cases are distinct
        het = (MINIMAL_ROW.replace("ATOM ", "HETATM ").replace(" SER ", " MG ")
               .replace(" CA ", " MG ").replace(" 10 B", " 99 B"))
        path = write_loop(tmp_path / "het.cif", MINIMAL_TAGS, [MINIMAL_ROW, het])
        assert by_key(get_resi_from_cif(path, False, False)).keys() == {("B", "10")}
        assert by_key(get_resi_from_cif(path, True, False)).keys() == {("B", "10"),
                                                                       ("B", "99")}

    def test_group_pdb_is_preferred_over_the_line_text(self, tmp_path):
        """
        A row whose group_PDB says HETATM is a HETATM 
        """
        tags = ["label_atom_id", "group_PDB", "label_comp_id", "label_asym_id",
                "label_seq_id", "Cartn_x", "Cartn_y", "Cartn_z"]
        path = write_loop(tmp_path / "grp.cif", tags,
                          ["MG HETATM MG C 1 0.000 0.000 0.000"])
        assert get_resi_from_cif(path, False, False) == []
        assert len(get_resi_from_cif(path, True, False)) == 1


class TestAltlocs:

    def pdb_with_altlocs(self, tmp_path):
        return write_pdb(tmp_path / "alt.pdb", [
            pdb_atom_line(1, "N", "SER", "A", 1, 0.0, 0.0, 0.0),
            pdb_atom_line(2, "CA", "SER", "A", 1, 1.0, 0.0, 0.0),
            pdb_atom_line(3, "CB", "SER", "A", 1, 2.0, 0.0, 0.0, altloc="A", occ=0.6),
            pdb_atom_line(4, "CB", "SER", "A", 1, 9.0, 0.0, 0.0, altloc="B", occ=0.4),
            pdb_atom_line(5, "OG", "SER", "A", 1, 3.0, 0.0, 0.0, altloc="A", occ=0.6),
            pdb_atom_line(6, "OG", "SER", "A", 1, 8.0, 0.0, 0.0, altloc="B", occ=0.4),
        ])

    def cif_with_altlocs(self, tmp_path):
        rows = [
            cif_atom_row(1, "N", "N", "SER", 1, 0.0, 0.0, 0.0),
            cif_atom_row(2, "C", "CA", "SER", 1, 1.0, 0.0, 0.0),
            cif_atom_row(3, "C", "CB", "SER", 1, 2.0, 0.0, 0.0, altloc="A", occ=0.6),
            cif_atom_row(4, "C", "CB", "SER", 1, 9.0, 0.0, 0.0, altloc="B", occ=0.4),
            cif_atom_row(5, "O", "OG", "SER", 1, 3.0, 0.0, 0.0, altloc="A", occ=0.6),
            cif_atom_row(6, "O", "OG", "SER", 1, 8.0, 0.0, 0.0, altloc="B", occ=0.4),
        ]
        return write_cif(tmp_path / "alt.cif", rows)

    def test_pdb_keeps_every_conformer(self, tmp_path):
        resi = get_resi_from_pdb(self.pdb_with_altlocs(tmp_path), False, False)[0]
        assert atom_names(resi) == ["N", "CA", "CB", "CB", "OG", "OG"]
        assert [a.x for a in resi.atom_list] == [0.0, 1.0, 2.0, 9.0, 3.0, 8.0]

    def test_cif_keeps_every_conformer(self, tmp_path):
        resi = get_resi_from_cif(self.cif_with_altlocs(tmp_path), False, False)[0]
        assert atom_names(resi) == ["N", "CA", "CB", "CB", "OG", "OG"]
        assert [a.x for a in resi.atom_list] == [0.0, 1.0, 2.0, 9.0, 3.0, 8.0]

    def test_both_parsers_agree(self, tmp_path):
        from_pdb = get_resi_from_pdb(self.pdb_with_altlocs(tmp_path), False, False)[0]
        from_cif = get_resi_from_cif(self.cif_with_altlocs(tmp_path), False, False)[0]
        assert atom_names(from_pdb) == atom_names(from_cif)
        assert [a.x for a in from_pdb.atom_list] == [a.x for a in from_cif.atom_list]

    def test_conformers_stay_on_one_residue(self, tmp_path):
        assert len(get_resi_from_pdb(self.pdb_with_altlocs(tmp_path), False, False)) == 1

    def test_repeated_atom_names_without_an_altloc_are_also_kept(self, tmp_path):
        path = write_pdb(tmp_path / "plain.pdb", [
            pdb_atom_line(1, "CB", "SER", "A", 1, 0.0, 0.0, 0.0),
            pdb_atom_line(2, "CB", "SER", "A", 1, 5.0, 0.0, 0.0),
        ])
        assert atom_names(get_resi_from_pdb(path, False, False)[0]) == ["CB", "CB"]

    def test_the_ca_slot_takes_the_primary_conformer(self, tmp_path):
        path = write_pdb(tmp_path / "altca.pdb", [
            pdb_atom_line(1, "CA", "SER", "A", 1, 1.0, 0.0, 0.0, altloc="A"),
            pdb_atom_line(2, "CA", "SER", "A", 1, 7.0, 0.0, 0.0, altloc="B"),
        ])
        resi = get_resi_from_pdb(path, False, False)[0]
        assert resi.CA.x == 1.0
        assert resi.CA.altloc == "A"
        assert [a.x for a in resi.atom_list] == [1.0, 7.0]

    def test_altloc_is_recorded_on_each_atom(self, tmp_path):
        resi = get_resi_from_pdb(self.pdb_with_altlocs(tmp_path), False, False)[0]
        assert [(a.altid, a.altloc) for a in resi.atom_list] == [
            ("N", ""), ("CA", ""), ("CB", "A"), ("CB", "B"),
            ("OG", "A"), ("OG", "B")]

    def test_cif_altloc_is_recorded_on_each_atom(self, tmp_path):
        resi = get_resi_from_cif(self.cif_with_altlocs(tmp_path), False, False)[0]
        assert [(a.altid, a.altloc) for a in resi.atom_list] == [
            ("N", ""), ("CA", ""), ("CB", "A"), ("CB", "B"),
            ("OG", "A"), ("OG", "B")]

    def test_cif_null_altloc_becomes_blank(self, tmp_path):
        # '.' and '?' mean "no alternate conformation" and must not become ids
        rows = [cif_atom_row(1, "N", "N", "SER", 1, 0.0, 0.0, 0.0, altloc="."),
                cif_atom_row(2, "C", "CA", "SER", 1, 1.0, 0.0, 0.0, altloc="?")]
        resi = get_resi_from_cif(write_cif(tmp_path / "null.cif", rows), False, False)[0]
        assert [a.altloc for a in resi.atom_list] == ["", ""]

    def test_atom_key_pairs_name_with_conformation(self, tmp_path):
        resi = get_resi_from_pdb(self.pdb_with_altlocs(tmp_path), False, False)[0]
        keys = [a.key for a in resi.atom_list]
        assert keys == [("N", ""), ("CA", ""), ("CB", "A"), ("CB", "B"),
                        ("OG", "A"), ("OG", "B")]
        assert len(set(keys)) == len(keys)

    def test_each_residue_keeps_its_own_conformers(self, tmp_path):
        path = write_pdb(tmp_path / "two.pdb", [
            pdb_atom_line(1, "CB", "SER", "A", 1, 0.0, 0.0, 0.0, altloc="A"),
            pdb_atom_line(2, "CB", "SER", "A", 1, 9.0, 0.0, 0.0, altloc="B"),
            pdb_atom_line(3, "CB", "SER", "A", 2, 0.0, 5.0, 0.0, altloc="A"),
            pdb_atom_line(4, "CB", "SER", "A", 2, 9.0, 5.0, 0.0, altloc="B"),
        ])
        residues = get_resi_from_pdb(path, False, False)
        assert [atom_names(r) for r in residues] == [["CB", "CB"], ["CB", "CB"]]

    def test_load_residues_keeps_every_conformer(self, tmp_path):
        path = self.pdb_with_altlocs(tmp_path)
        assert len(load_residues(path, False, False)[0].atom_list) == 6


class TestGzipInput:

    def gzipped(self, path, tmp_path):
        target = tmp_path / (os.path.basename(path) + ".gz")
        with open(path, "rb") as src, gzip.open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
        return str(target)

    def test_gzipped_pdb(self, tiny_pdb, tmp_path):
        plain = get_resi_from_pdb(tiny_pdb, True, True)
        packed = load_residues(self.gzipped(tiny_pdb, tmp_path), True, True)
        assert by_key(packed).keys() == by_key(plain).keys()
        assert atom_names(packed[0]) == atom_names(plain[0])

    def test_gzipped_cif(self, tiny_cif, tmp_path):
        plain = get_resi_from_cif(tiny_cif, True, True)
        packed = load_residues(self.gzipped(tiny_cif, tmp_path), True, True)
        assert by_key(packed).keys() == by_key(plain).keys()
        assert atom_names(packed[0]) == atom_names(plain[0])

    def test_gzipped_coordinates_are_identical(self, tiny_pdb, tmp_path):
        plain = get_resi_from_pdb(tiny_pdb, True, True)
        packed = load_residues(self.gzipped(tiny_pdb, tmp_path), True, True)
        for r1, r2 in zip(plain, packed):
            for a, b in zip(r1.atom_list, r2.atom_list):
                assert (a.x, a.y, a.z) == (b.x, b.y, b.z)

    @pytest.mark.parametrize("suffix", [".pdb.gz", ".ent.gz", ".cif.gz",
                                        ".mmcif.gz", ".CIF.GZ"])
    def test_extension_dispatch_ignores_the_gz_suffix(self, tiny_pdb, tiny_cif,
                                                      tmp_path, suffix):
        source = tiny_cif if "cif" in suffix.lower() else tiny_pdb
        target = tmp_path / ("copy" + suffix)
        with open(source, "rb") as src, gzip.open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
        assert len(load_residues(str(target), False, False)) == 3

    def test_corrupt_gzip_raises_oserror(self, tmp_path):
        target = tmp_path / "broken.cif.gz"
        target.write_text("ATOM   this is not gzipped at all\n")
        with pytest.raises(OSError):
            load_residues(str(target), False, False)

    def test_gz_without_a_known_extension_is_rejected(self, tmp_path):
        target = tmp_path / "structure.xyz.gz"
        target.write_bytes(b"")
        with pytest.raises(ValueError, match="Unrecognized structure extension"):
            load_residues(str(target), False, False)


class TestLoadResiduesOrExit:
    """The CLI wrapper turns user-triggerable errors into clean exits."""

    def test_returns_residues_on_success(self, tiny_pdb):
        assert len(load_residues_or_exit(tiny_pdb, False, False)) == 3

    def test_missing_file(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as excinfo:
            load_residues_or_exit(str(tmp_path / "nope.pdb"), False, False)
        assert "no such file" in str(excinfo.value)
        assert "Traceback" not in str(excinfo.value)

    def test_directory_instead_of_a_file(self, tmp_path):
        with pytest.raises(SystemExit) as excinfo:
            load_residues_or_exit(str(tmp_path), False, False)
        # The extension check rejects a bare directory name first
        assert "error:" in str(excinfo.value)

    def test_directory_with_a_structure_extension(self, tmp_path):
        target = tmp_path / "looks_like.pdb"
        target.mkdir()
        with pytest.raises(SystemExit) as excinfo:
            load_residues_or_exit(str(target), False, False)
        assert "error:" in str(excinfo.value)

    def test_unknown_extension(self, tmp_path):
        target = tmp_path / "structure.xyz"
        target.write_text("")
        with pytest.raises(SystemExit) as excinfo:
            load_residues_or_exit(str(target), False, False)
        assert "Unrecognized structure extension" in str(excinfo.value)

    def test_corrupt_gzip(self, tmp_path):
        target = tmp_path / "broken.cif.gz"
        target.write_text("not gzipped\n")
        with pytest.raises(SystemExit) as excinfo:
            load_residues_or_exit(str(target), False, False)
        assert "error:" in str(excinfo.value)

    def test_unreadable_file(self, tiny_pdb):
        os.chmod(tiny_pdb, 0o000)
        try:
            if os.access(tiny_pdb, os.R_OK):
                pytest.skip("running as a user that ignores file permissions")
            with pytest.raises(SystemExit) as excinfo:
                load_residues_or_exit(tiny_pdb, False, False)
            assert "cannot read" in str(excinfo.value)
        finally:
            os.chmod(tiny_pdb, 0o644)


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
