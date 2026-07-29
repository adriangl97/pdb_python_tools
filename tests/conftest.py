"""
Shared builders and fixtures for the pdb_python_tools test suite.

"""
import os

import pytest

from pdb_python_tools.core import Atom, Residue

# The ribosome structures for the README examples
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_FILES_DIR = os.path.join(REPO_ROOT, "test_files")


def make_atom(name, x=0.0, y=0.0, z=0.0, element=None, restyp="ALA",
              chainid="A", seqid="1", atomid="1", occ=1.0, biso=20.0, altloc=""):
    """
    Build an Atom. `element` defaults to the first character of the atom name.
    `altloc` is the alternate conformation id ("" for none). `xyz_change` is left
    at its None default: nothing has been compared yet.
    """
    if element is None:
        element = name[0]
    return Atom(atomid=atomid, element=element, altid=name, restyp=restyp,
                chainid=chainid, seqid=seqid, x=x, y=y, z=z, occ=occ, biso=biso,
                altloc=altloc)


def make_residue(restyp, atoms, chainid="A", seqid="1"):
    """
    Build a Residue around `atoms`, stamping the residue identity onto each atom
    and filling the CA/C1' slot the way the parsers do
    A residue with neither a CA nor a C1' keeps CA = None.
    """
    for atom in atoms:
        atom.restyp = restyp
        atom.chainid = chainid
        atom.seqid = seqid
    resi = Residue(chainid, seqid, restyp, atoms)
    for atom in atoms:
        if atom.altid in ("CA", "C1'") and resi.CA is None:
            resi.CA = make_atom(atom.altid, atom.x, atom.y, atom.z,
                                restyp=restyp, chainid=chainid, seqid=seqid,
                                altloc=atom.altloc)
    return resi


def pdb_atom_line(serial, name, resname, chain, resseq, x, y, z, record="ATOM",
                  altloc=" ", icode=" ", occ=1.0, biso=20.0, element=None,
                  blank_element=False):
    """
    Format one ATOM/HETATM record at the wwPDB fixed-column offsets.

    Columns: 1-6 record, 7-11 serial, 13-16 name, 17 altLoc, 18-20 resName,
    22 chainID, 23-26 resSeq, 27 iCode, 31-38 x, 39-46 y, 47-54 z,
    55-60 occupancy, 61-66 B-factor, 77-78 element.
    """
    if element is None:
        element = name[0]
    if blank_element:
        element = ""
    # Four-character atom names start in column 13; shorter ones in column 14
    name_field = name if len(name) == 4 else " " + name
    return ("%-6s%5d %-4s%1s%3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s"
            % (record, serial, name_field, altloc, resname, chain, resseq, icode,
               x, y, z, occ, biso, element))


def write_pdb(path, lines):
    """Write ATOM/HETATM record lines out as a minimal PDB file."""
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\nEND\n")
    return str(path)


# Minimal mmCIF _atom_site loop
CIF_HEADER = """data_test
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
"""


def cif_atom_row(serial, element, name, resname, seqid, x, y, z, group="ATOM",
                 altloc=".", icode="?", occ=1.0, biso=20.0, chain="A",
                 label_chain="A", label_seq=None):
    """
    Format one mmCIF _atom_site row matching CIF_HEADER's column order.
    """
    if label_seq is None:
        label_seq = seqid
    if "'" in name:
        name = '"%s"' % name
    return ("%-6s %-5s %-2s %-6s %s %-4s %s %-4s %s %8.3f %8.3f %8.3f %5.2f %6.2f %-4s %s"
            % (group, serial, element, name, altloc, resname, label_chain,
               label_seq, icode, x, y, z, occ, biso, seqid, chain))


def write_cif(path, rows):
    """Write _atom_site rows out as a minimal mmCIF file."""
    with open(path, "w") as handle:
        handle.write(CIF_HEADER)
        handle.write("\n".join(rows) + "\n#\n")
    return str(path)


@pytest.fixture
def tiny_pdb(tmp_path):
    """
    A small PDB covering the parser's edge cases: a protein residue with a CA, a
    hydrogen, a blank element column, an insertion code, a HETATM and a
    nucleotide with a C1'.
    """
    lines = [
        pdb_atom_line(1, "N", "SER", "A", 10, 0.0, 0.0, 0.0),
        pdb_atom_line(2, "CA", "SER", "A", 10, 1.0, 0.0, 0.0),
        pdb_atom_line(3, "OG", "SER", "A", 10, 2.0, 0.0, 0.0),
        # Element column left blank on purpose: falls back to the atom name
        pdb_atom_line(4, "CB", "SER", "A", 10, 3.0, 0.0, 0.0, blank_element=True),
        pdb_atom_line(5, "HA", "SER", "A", 10, 1.0, 1.0, 0.0, element="H"),
        # Same residue number as above but with an insertion code: a new residue
        pdb_atom_line(6, "N", "GLY", "A", 10, 0.0, 5.0, 0.0, icode="A"),
        pdb_atom_line(7, "CA", "GLY", "A", 10, 1.0, 5.0, 0.0, icode="A"),
        # Nucleotide: the C1' fills the CA slot
        pdb_atom_line(8, "C1'", "G", "B", 1, 0.0, 0.0, 9.0),
        pdb_atom_line(9, "O4'", "G", "B", 1, 1.0, 0.0, 9.0),
        pdb_atom_line(10, "MG", "MG", "C", 1, 0.0, 0.0, 20.0, record="HETATM",
                      element="MG"),
    ]
    return write_pdb(tmp_path / "tiny.pdb", lines)


@pytest.fixture
def tiny_cif(tmp_path):
    """The same content as `tiny_pdb`, in mmCIF form."""
    rows = [
        cif_atom_row(1, "N", "N", "SER", 10, 0.0, 0.0, 0.0),
        cif_atom_row(2, "C", "CA", "SER", 10, 1.0, 0.0, 0.0),
        cif_atom_row(3, "O", "OG", "SER", 10, 2.0, 0.0, 0.0),
        cif_atom_row(4, "C", "CB", "SER", 10, 3.0, 0.0, 0.0),
        cif_atom_row(5, "H", "HA", "SER", 10, 1.0, 1.0, 0.0),
        cif_atom_row(6, "N", "N", "GLY", 10, 0.0, 5.0, 0.0, icode="A"),
        cif_atom_row(7, "C", "CA", "GLY", 10, 1.0, 5.0, 0.0, icode="A"),
        cif_atom_row(8, "C", "C1'", "G", 1, 0.0, 0.0, 9.0, chain="B",
                     label_chain="B"),
        cif_atom_row(9, "O", "O4'", "G", 1, 1.0, 0.0, 9.0, chain="B",
                     label_chain="B"),
        cif_atom_row(10, "MG", "MG", "MG", 1, 0.0, 0.0, 20.0, group="HETATM",
                     chain="C", label_chain="C"),
    ]
    return write_cif(tmp_path / "tiny.cif", rows)


def by_key(residues):
    """Index a residue list by (chainid, seqid) for convenient assertions."""
    return {(r.chainid, r.seqid): r for r in residues}


def atom_names(resi):
    """The atom names of a residue, in file order."""
    return [atom.altid for atom in resi.atom_list]
