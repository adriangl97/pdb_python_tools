#!/usr/bin/env python3
import math
import csv
import gzip
import os
import re
import sys
from dataclasses import dataclass, field, replace
from typing import List, Optional
import numpy as np
from . import __version__


@dataclass
class Atom:
    """
    Define atom class.


    Attributes
    ----------
    atomid : id for the atom
    element : C, N, O ...
    altid : id within residue
    restyp : residue type
    chainid : chain id
    seqid : residue number
    x, y, z : location
    occ : occupancy
    biso : b factor
    xyz_change : movement compared to another pdb, or None when the atom has no
        counterpart there
    altloc : alternate conformation id ("" when the atom has none). Atoms of
        different conformations are kept apart.

    """
    atomid: str = ""
    element: str = ""
    altid: str = ""
    restyp: str = ""
    chainid: str = ""
    seqid: str = ""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    occ: float = 1.0
    biso: float = 0.0
    xyz_change: Optional[float] = None
    altloc: str = ""

    @property
    def key(self):
        """
        Identity of this atom within its residue: name plus conformation.

        """
        return (self.altid, self.altloc)


@dataclass
class Residue:
    """
    Define residue class.

    Attributes
    ----------
    chainid : chain id
    seqid : residue number
    restyp : residue type
    atom_list : list of Atom objects belonging to that Residue
    max_xyz : Atom with the largest xyz change within the residue, or None
    average_xyz : average xyz change within the residue, or None
    CA : Calpha/C1' Atom object, or None when the residue has neither
    """
    chainid: str = ""
    seqid: str = ""
    restyp: str = ""
    atom_list: List[Atom] = field(default_factory=list)
    max_xyz: Optional[Atom] = None
    average_xyz: Optional[float] = None
    CA: Optional[Atom] = None

# mmCIF null tokens: '.' means inapplicable, '?' means unknown
_CIF_NULLS = (".", "?")


def _safe_float(value):
    """
    Convert a coordinate/occupancy/B-factor token to float.

    A blank token, or an mmCIF null ('.' or '?'), becomes 0.0.
    """
    value = value.strip()
    if value == "" or value in _CIF_NULLS:
        return 0.0
    return float(value)


def _is_cif_null(token):
    """True for a blank token or an mmCIF null ('.' or '?')."""
    token = token.strip()
    return token == "" or token in _CIF_NULLS


def _split_cif_tokens(line):
    """
    Split one mmCIF data line into values, following quoting.
    """
    tokens = line.split()
    if not any(token[0] in "'\"" for token in tokens):
        return tokens
    tokens = []
    index = 0
    length = len(line)
    while index < length:
        char = line[index]
        if char.isspace():
            index += 1
            continue
        if char in ("'", '"'):
            quote = char
            index += 1
            start = index
            while index < length:
                if line[index] == quote and (index + 1 >= length
                                             or line[index + 1].isspace()):
                    break
                index += 1
            tokens.append(line[start:index])
            index += 1
        else:
            start = index
            while index < length and not line[index].isspace():
                index += 1
            tokens.append(line[start:index])
    return tokens


def _element_from_name(atom_name):
    """
    Guess an element symbol from an atom name: its first alphabetic character.
    """
    for char in atom_name:
        if char.isalpha():
            return char.upper()
    return ""


def _element_from_pdb(line, atom_name):
    """
    Return the element symbol for a PDB ATOM/HETATM line.

    Reads columns 77-78 first (the wwPDB element field); if that field is blank
    it falls back to the first alphabetic character of the atom name
    """
    element = line[76:78].strip()
    if element:
        return element
    return _element_from_name(atom_name)


def _is_hydrogen(element):
    """Hydrogen (or deuterium) detection based on the element symbol."""
    return element.upper() in ("H", "D")


def _open_text(path):
    """
    Open a structure file for reading, decompressing it when the name ends in
    .gz. Returns a text-mode handle.
    """
    if path.lower().endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def _res_key(atom):
    """Residue identity: chain id plus seq id (seq id  carries any insertion code)"""
    return (atom.chainid, atom.seqid)


def _euclid(a, b):
    """Euclidean distance between two Atom objects."""
    return math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))


def _add_atom(residues, atom, hydrogens):
    """
    Append an Atom to the growing list of Residues, starting a new Residue when
    the chain/seqid (including insertion code) changes. Hydrogens are skipped
    unless requested. Also records the CA/C1' atom on its residue.

    Every alternate conformation is kept, so a residue modelled in two
    conformations contributes both copies of each affected atom.
    """
    # Ignore hydrogens by default; include them only when requested
    if _is_hydrogen(atom.element) and not hydrogens:
        return
    same_residue = bool(residues) and _res_key(residues[-1].atom_list[0]) == _res_key(atom)
    if not same_residue:
        # CA stays None until a CA/C1' atom actually turns up
        residues.append(Residue(atom.chainid, atom.seqid, atom.restyp, [atom]))
    else:
        residues[-1].atom_list.append(atom)
    # Record the CA (protein) / C1' (nucleic) atom for the residue as a separate
    # Atom object (kept distinct from the copy in atom_list). With alternate
    # conformations the first one seen is used, so the slot holds the primary
    # conformer rather than whichever copy happens to come last.
    if atom.altid in ("CA", "C1'") and residues[-1].CA is None:
        residues[-1].CA = replace(atom)


def get_resi_from_pdb(file, hetatm, hydrogens):
    """
    Parses through a pdb file and generates a list of Residues with their list of atoms.

    Inputs
    ------
    file : path to the pdb file to parse.
    hetatm : boolean, include HETATM records when True.
    hydrogens : boolean, include hydrogen atoms when True.

    Returns
    -------
    List of residues as Residue class with list of Atom classes within each residue.
    Every alternate conformation present in the file is kept.
    """
    residues = []
    with _open_text(file) as fh:
        for line in fh:
            record = line[0:6].strip()
            # Only ATOM (always) and HETATM (when requested) carry atom info
            if record == "ATOM" or (hetatm and record == "HETATM"):
                # Fixed-column fields per the PDB format specification
                atom_name = line[12:16].strip()
                alt_id = line[16:17].strip()
                resname = line[17:20].strip()
                chainid = line[21:22].strip()
                # Residue key = residue sequence number + insertion code
                seqid = line[22:26].strip() + line[26:27].strip()
                element = _element_from_pdb(line, atom_name)
                atom = Atom(atomid=line[6:11].strip(), element=element, altid=atom_name,
                            restyp=resname, chainid=chainid, seqid=seqid,
                            x=_safe_float(line[30:38]), y=_safe_float(line[38:46]),
                            z=_safe_float(line[46:54]),
                            occ=_safe_float(line[54:60]), biso=_safe_float(line[60:66]),
                            altloc=alt_id)
                _add_atom(residues, atom, hydrogens)
    return residues


# The _atom_site tags this parser understands, mapped to the field each fills.
# Tags are matched exactly (after lower-casing), so neighbouring tags such as
# Cartn_x_esd, label_entity_id or pdbx_formal_charge are  ignored
_ATOM_SITE_TAGS = {
    "group_pdb": "group",
    "id": "atomid",
    "type_symbol": "element",
    "label_atom_id": "name",
    "label_alt_id": "altloc",
    "label_comp_id": "restyp",
    "auth_asym_id": "chainid",
    "label_asym_id": "chainid_alt",
    "auth_seq_id": "seqid",
    "label_seq_id": "seqid_alt",
    "pdbx_pdb_ins_code": "icode",
    "cartn_x": "x",
    "cartn_y": "y",
    "cartn_z": "z",
    "occupancy": "occ",
    "b_iso_or_equiv": "biso",
}

# auth_* ids are what users see, label_* is the fallback
_ATOM_SITE_FALLBACKS = {"chainid": "chainid_alt", "seqid": "seqid_alt"}

# Without these there is no usable atom to build
_ATOM_SITE_REQUIRED = ("name", "restyp", "chainid", "seqid", "x", "y", "z")


def _atom_site_columns(tags):
    """
    Map the tags of one loop_ header to _atom_site column indices.

    """
    columns = {}
    is_atom_site = False
    for index, tag in enumerate(tags):
        if not tag.startswith("_atom_site."):
            continue
        is_atom_site = True
        field = _ATOM_SITE_TAGS.get(tag[len("_atom_site."):])
        if field is not None:
            columns[field] = index
    if not is_atom_site:
        return None
    for field, fallback in _ATOM_SITE_FALLBACKS.items():
        if field not in columns and fallback in columns:
            columns[field] = columns[fallback]
    return columns


def get_resi_from_cif(file, hetatm, hydrogens):
    """
    Parses through a cif file and generates a list of Residues with their list of atoms.

    Inputs
    ------
    file : path to the cif file to parse.
    hetatm : bool, include HETATM records when True.
    hydrogens : bool, include hydrogen atoms when True.

    Returns
    -------
    List of residues as Residue class with list of Atom classes within each residue.
    Every alternate conformation present in the file is kept.

    Raises
    ------
    ValueError if the file has no _atom_site loop, or if that loop is missing a
    column the parser needs.
    """
    residues = []
    tags = None          # collecting a loop_ header
    columns = None       # column indices, while inside _atom_site data rows
    widest = 0           # highest column index used, to spot truncated rows
    found_atom_site = False
    # Column indices are hoisted into locals once per loop
    c_group = c_atomid = c_element = c_altloc = c_icode = c_occ = c_biso = None
    c_name = c_restyp = c_chain = c_seqid = c_x = c_y = c_z = 0
    with _open_text(file) as fh:
        for line in fh:
            stripped = line.strip()
            # Blank lines and comments carry no state
            if not stripped or stripped.startswith("#"):
                continue
            if stripped == "loop_":
                tags, columns = [], None
                continue
            if stripped.startswith("_"):
                if tags is None:
                    # A key-value item outside a loop ends any data block
                    columns = None
                else:
                    tags.append(stripped.split()[0].lower())
                continue
            if stripped.startswith("data_") or stripped.startswith("save_"):
                tags, columns = None, None
                continue
            if stripped.startswith(";"):
                # Multi-line text value; _atom_site never uses one
                continue
            # First data line after a header
            if tags is not None:
                columns = _atom_site_columns(tags)
                tags = None
                if columns is not None:
                    missing = [f for f in _ATOM_SITE_REQUIRED if f not in columns]
                    if missing:
                        raise ValueError(
                            "_atom_site loop is missing required column(s): %s"
                            % ", ".join(sorted(missing)))
                    widest = max(columns.values())
                    found_atom_site = True
                    c_group = columns.get("group")
                    c_atomid = columns.get("atomid")
                    c_element = columns.get("element")
                    c_altloc = columns.get("altloc")
                    c_icode = columns.get("icode")
                    c_occ = columns.get("occ")
                    c_biso = columns.get("biso")
                    c_name = columns["name"]
                    c_restyp = columns["restyp"]
                    c_chain = columns["chainid"]
                    c_seqid = columns["seqid"]
                    c_x, c_y, c_z = columns["x"], columns["y"], columns["z"]
            if columns is None:
                continue
            values = _split_cif_tokens(line)
            if len(values) <= widest:
                continue
            # ATOM always, HETATM only when asked for
            if c_group is not None:
                group = values[c_group].upper()
            else:
                group = "HETATM" if stripped.startswith("HETATM") else "ATOM"
            if group != "ATOM" and not (hetatm and group == "HETATM"):
                continue
            atom_name = values[c_name]
            # An atom with no name or no coordinates is not usable
            if (_is_cif_null(atom_name) or _is_cif_null(values[c_x])
                    or _is_cif_null(values[c_y]) or _is_cif_null(values[c_z])):
                continue
            # Residue key = auth_seq_id + insertion code (blank for '.'/'?')
            res_seq = values[c_seqid]
            if c_icode is not None and not _is_cif_null(values[c_icode]):
                res_seq += values[c_icode]
            # '.'/'?' mean "no alternate conformation" and normalise to ""
            alt_id = values[c_altloc] if c_altloc is not None else ""
            if _is_cif_null(alt_id):
                alt_id = ""
            element = values[c_element] if c_element is not None else ""
            if _is_cif_null(element):
                element = _element_from_name(atom_name)
            atom = Atom(atomid=values[c_atomid] if c_atomid is not None else "",
                        element=element, altid=atom_name, restyp=values[c_restyp],
                        chainid=values[c_chain], seqid=res_seq,
                        x=_safe_float(values[c_x]),
                        y=_safe_float(values[c_y]),
                        z=_safe_float(values[c_z]),
                        occ=_safe_float(values[c_occ]) if c_occ is not None else 1.0,
                        biso=_safe_float(values[c_biso]) if c_biso is not None else 0.0,
                        altloc=alt_id)
            _add_atom(residues, atom, hydrogens)
    if not found_atom_site:
        raise ValueError("No _atom_site loop found")
    return residues


def load_residues(path, hetatm, hydrogens):
    """
    Dispatch to the correct structure parser based on the file extension.

    A trailing .gz is stripped before the extension is examined and the file is decompressed on the fly
    """
    p = path.lower()
    if p.endswith(".gz"):
        p = p[:-len(".gz")]
    if p.endswith(".pdb") or p.endswith(".ent"):
        return get_resi_from_pdb(path, hetatm, hydrogens)
    if p.endswith(".cif") or p.endswith(".mmcif"):
        return get_resi_from_cif(path, hetatm, hydrogens)
    raise ValueError(f"Unrecognized structure extension: {path}")


def load_residues_or_exit(path, hetatm, hydrogens):
    """
    load_residues() for command-line use.

    Turns the errors a user can actually trigger - a missing file, a directory,
    an unreadable file, an unknown extension, a corrupt gzip - into a short
    message on stderr and exit status 1, instead of a traceback.
    """
    try:
        return load_residues(path, hetatm, hydrogens)
    except FileNotFoundError:
        sys.exit("error: no such file: %s" % path)
    except IsADirectoryError:
        sys.exit("error: not a file: %s" % path)
    except PermissionError:
        sys.exit("error: cannot read: %s" % path)
    except (OSError, ValueError) as exc:
        sys.exit("error: %s: %s" % (path, exc))

# Interchangeable atom-name pairs per residue type. Within each pair the two
# atoms are equivalent, so swapping them is not a real movement: the
# displacement for such an atom is the minimum over itself and its partner.
_SYMMETRIC = {
    "TYR": (("CD1", "CD2"), ("CE1", "CE2")),
    "PHE": (("CD1", "CD2"), ("CE1", "CE2")),
    "GLU": (("OE1", "OE2"),),
    "ASP": (("OD1", "OD2"),),
    "ARG": (("NH1", "NH2"),),
    "LEU": (("CD1", "CD2"),),
    "VAL": (("CG1", "CG2"),),
}


def _build_swap_map(symmetric):
    """Expand the symmetric pairs into per-residue atom-name -> partner-name maps."""
    swap = {}
    for restyp, pairs in symmetric.items():
        partners = {}
        for first, second in pairs:
            partners[first] = second
            partners[second] = first
        swap[restyp] = partners
    return swap


_SWAP = _build_swap_map(_SYMMETRIC)


def compare_pdb_resi_xyz(pdb1, pdb2):
    """
    Compares two lists of Residues (class)

    Inputs
    ------
    pdb1, pdb2 : List of Residues (class)

    Returns
    -------
    Modifies self.xyz_change from pdb1 Atoms within the list in the Residue (class) based on the
    x, y, z change between pdb1 and pdb2. Also records the CA/C1' displacement on
    each pdb1 residue's CA attribute when both structures have that atom.

    An atom with no counterpart in pdb2 is left at xyz_change = None

    """
    # Index pdb2 residues by (chainid, seqid)
    # seqid already carries any insertion code
    #  Keep the first residue seen for a given key to mirror the previous first-match behaviour.
    index = {}
    for resi2 in pdb2:
        index.setdefault((resi2.chainid, resi2.seqid), resi2)
    for resi1 in pdb1:
        resi2 = index.get((resi1.chainid, resi1.seqid))
        if resi2 is None:
            continue
        # CA/C1' displacement: computed explicitly and only when both residues
        # have a real CA/C1' atom. Otherwise resi1.CA.xyz_change stays None.
        if resi1.CA is not None and resi2.CA is not None:
            resi1.CA.xyz_change = _euclid(resi1.CA, resi2.CA)
        # Index resi2's atoms by (name, conformation). Keying on the pair keeps
        # alternate conformations independent: conformer A of an atom is only
        # ever compared with conformer A in the other structure, never with B.
        atoms2 = {}
        for atom2 in resi2.atom_list:
            atoms2.setdefault(atom2.key, atom2)
        # Which atom names are interchangeable for this residue type
        swap = _SWAP.get(resi1.restyp, {})
        for atom1 in resi1.atom_list:
            # Same atom, same conformation: its displacement
            match = atoms2.get(atom1.key)
            if match is not None:
                atom1.xyz_change = _euclid(atom1, match)
            # For an interchangeable atom, also measure against its symmetry
            # partner and keep whichever distance is shorter.
            partner_name = swap.get(atom1.altid)
            partner = atoms2.get((partner_name, atom1.altloc)) if partner_name else None
            if partner is not None:
                xyz = _euclid(atom1, partner)
                if atom1.xyz_change is None or xyz < atom1.xyz_change:
                    atom1.xyz_change = xyz


_POLAR_ELEMENTS = {"N", "O", "P", "S"}


def find_contacts_kdtree(residues, distance, chain, polar):
    """
    Find all inter-chain atom contacts within `distance` of a query chain.

    Uses a scipy cKDTree. The query set is every atom in `chain`; the target
    set is every atom in all other chains.
    Hydrogens/HETATM are already filtered upstream by the parser flags
    when `polar` is True both sets are restricted to N/O/P/S.

    Inputs
    ------
    residues : list of Residue (class)
    distance : float, contact cutoff in Angstrom
    chain : chain id (str) to analyze
    polar : boolean, restrict to polar (N/O/P/S) atoms only

    Returns
    -------
    List of [atom1, atom2, distance] with atom1 in `chain` and atom2 in another chain.
    """
    from scipy.spatial import cKDTree

    # Split atoms into the query chain and everything else
    query_atoms = []
    target_atoms = []
    for resi in residues:
        is_query = resi.chainid == chain
        for atom in resi.atom_list:
            if polar and atom.element not in _POLAR_ELEMENTS:
                continue
            (query_atoms if is_query else target_atoms).append(atom)

    if not query_atoms or not target_atoms:
        return []

    query_coords = np.array([(a.x, a.y, a.z) for a in query_atoms])
    target_coords = np.array([(a.x, a.y, a.z) for a in target_atoms])
    # Build the tree on the (usually larger) target set and query each atom
    tree = cKDTree(target_coords)
    neighbours = tree.query_ball_point(query_coords, r=distance)

    atom_pairs = []
    for qi, hits in enumerate(neighbours):
        atom1 = query_atoms[qi]
        for ti in hits:
            atom2 = target_atoms[ti]
            atom_pairs.append([atom1, atom2, _euclid(atom1, atom2)])
    return atom_pairs


def find_nearest_ca(pdb1, pdb2):
    """
    For every residue in pdb1 that has a CA/C1' atom, find the nearest residue
    in pdb2 whose CA/C1' atom is of the same type (CA->CA, C1'->C1') using a
    scipy cKDTree nearest-neighbour query.

    The two structures do not need to be equivalent or share residue numbering, but
    they should be pre-aligned.

    Inputs
    ------
    pdb1, pdb2 : list of Residue (class)

    Returns
    -------
    List of (resi1, resi2, distance) in pdb1 order. Also stores the
    nearest-neighbour distance on resi1.CA.xyz_change.
    """
    from scipy.spatial import cKDTree

    # Build one tree per CA/C1' type from the pdb2 residues that have that atom
    trees = {}
    targets_by_name = {}
    for name in ("CA", "C1'"):
        targets = [resi for resi in pdb2 if resi.CA is not None and resi.CA.altid == name]
        if targets:
            coords = np.array([(r.CA.x, r.CA.y, r.CA.z) for r in targets])
            trees[name] = cKDTree(coords)
            targets_by_name[name] = targets

    results = []
    for resi1 in pdb1:
        if resi1.CA is None or resi1.CA.altid not in trees:
            continue
        name = resi1.CA.altid
        dist, idx = trees[name].query((resi1.CA.x, resi1.CA.y, resi1.CA.z), k=1)
        resi2 = targets_by_name[name][idx]
        resi1.CA.xyz_change = float(dist)
        results.append((resi1, resi2, float(dist)))
    return results


# Standard RNA and DNA residue names, split into purines and pyrimidines. The
# glycosidic torsion chi is defined from a different base atom for each group.

_RNA_PURINES = {"A", "G"}
_RNA_PYRIMIDINES = {"C", "U"}
_DNA_PURINES = {"DA", "DG"}
_DNA_PYRIMIDINES = {"DC", "DT", "DU"}
_PURINES = _RNA_PURINES | _DNA_PURINES
_PYRIMIDINES = _RNA_PYRIMIDINES | _DNA_PYRIMIDINES
_NUCLEOTIDES = _PURINES | _PYRIMIDINES

# The four atoms chi is measured on, per base type
_PURINE_CHI = ("O4'", "C1'", "N9", "C4")
_PYRIMIDINE_CHI = ("O4'", "C1'", "N1", "C2")
# Pseudouridine and its derivatives are C-glycosides:
_C_GLYCOSIDE_CHI = ("O4'", "C1'", "C5", "C4")
# Tried in this order against the atoms a non-standard residue actually has
_CHI_TEMPLATES = (_PURINE_CHI, _PYRIMIDINE_CHI, _C_GLYCOSIDE_CHI)
_PYRIMIDINE_TEMPLATES = (_PYRIMIDINE_CHI, _C_GLYCOSIDE_CHI)
# Longest bond to the sugar accepted when deciding whether a residue that is not
# a standard nucleotide really is one, and which of its atoms carries the base.
# The bond is ~1.47 A, so anything past this is not the glycosidic one.
_MAX_GLYCOSIDIC_BOND = 1.8


def _dihedral(p0, p1, p2, p3):
    """
    Dihedral angle in degrees (range (-180, 180]) defined by four points.

    Each point is an (x, y, z) sequence. Uses the standard projection formulation
    """
    b0 = np.array(p0, dtype=float) - np.array(p1, dtype=float)
    b1 = np.array(p2, dtype=float) - np.array(p1, dtype=float)
    b2 = np.array(p3, dtype=float) - np.array(p2, dtype=float)
    # Normalise the central bond so it does not scale the projections
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return math.degrees(math.atan2(y, x))


def _bonded(p0, p1):
    """True when two points are within glycosidic bonding distance."""
    return math.dist(p0, p1) <= _MAX_GLYCOSIDIC_BOND


def nucleotide_chi_atoms(resi):
    """
    The four atom names chi is measured on for a residue, or None when the
    residue is not a nucleotide.

    Standard RNA/DNA residues are recognised by name. Anything else (a modified
    or otherwise non-standard nucleotide, usually read from HETATM records) is
    recognised from its own atoms instead: it must carry the sugar atoms
    O4'/C1', and the base type is the one whose glycosidic atom is bonded
    to the C1', N9 for a purine, N1 for a pyrimidine and C5 for a C-glycoside
    such as pseudouridine.

    Inputs
    ------
    resi : Residue (class)

    Returns
    -------
    (O4', C1', glycosidic atom, next base atom) atom names, or None
    """
    if resi.restyp in _PURINES:
        return _PURINE_CHI
    if resi.restyp in _PYRIMIDINES:
        return _PYRIMIDINE_CHI
    coords = {}
    for atom in resi.atom_list:
        coords.setdefault(atom.altid, (atom.x, atom.y, atom.z))
    if not {"O4'", "C1'"} <= set(coords):
        return None
    for names in _CHI_TEMPLATES:
        if (set(names) <= set(coords)
                and _bonded(coords["C1'"], coords[names[2]])):
            return names
    return None


def is_pyrimidine(resi):
    """
    True when chi is measured on a pyrimidine base for this residue,
    C-glycosides such as pseudouridine included.
    """
    return nucleotide_chi_atoms(resi) in _PYRIMIDINE_TEMPLATES


def classify_nucleotide_conformation(residues):
    """
    Compute the glycosidic torsion angle chi for every RNA or DNA nucleotide and
    classify it as syn or anti.

    chi is measured O4'-C1'-N1-C2 for pyrimidines (C, U, DC, DT, DU) and
    O4'-C1'-N9-C4 for purines (A, G, DA, DG). A nucleotide is 'syn' when chi
    lies in [-90, +90] degrees and 'anti' otherwise.

    Modified nucleotides that keep the standard atom names are measured too, and a
    non-nucleotide that happens to reuse the names is not mistaken for a base.

    Inputs
    ------
    residues : list of Residue (class)

    A nucleotide modelled in more than one conformation is measured once per
    conformation

    Returns
    -------
    List of (residue, chi, conformation, altloc) for every nucleotide that has
    all four chi atoms, where altloc is the alternate conformation id ("" when
    the residue has none).
    """
    results = []
    for resi in residues:
        # Chi atom names
        names = nucleotide_chi_atoms(resi)
        if names is None:
            continue
        # Split the chi atoms by conformation. Atoms with no alternate id are
        # shared, so they seed every conformation's set.
        shared = {}
        per_altloc = {}
        for atom in resi.atom_list:
            if atom.altid not in names:
                continue
            if atom.altloc:
                per_altloc.setdefault(atom.altloc, {}).setdefault(
                    atom.altid, (atom.x, atom.y, atom.z))
            else:
                shared.setdefault(atom.altid, (atom.x, atom.y, atom.z))
        if per_altloc:
            groups = [(alt, dict(shared, **coords))
                      for alt, coords in sorted(per_altloc.items())]
        else:
            groups = [("", shared)]
        for alt, coords in groups:
            # Skip a conformation that is missing any of the four atoms
            if len(coords) != 4:
                continue
            if (resi.restyp not in _NUCLEOTIDES
                    and not _bonded(coords[names[1]], coords[names[2]])):
                continue
            chi = _dihedral(coords[names[0]], coords[names[1]],
                            coords[names[2]], coords[names[3]])
            conformation = "syn" if -90 <= chi <= 90 else "anti"
            results.append((resi, chi, conformation, alt))
    return results


# The base groups the syn/anti counts are reported for, in reporting order.
# "nucleotides" is the total, so every measured residue counts towards it as
# well as towards its own group.
CONFORMATION_GROUPS = ("pyrimidines", "purines", "nucleotides")


def count_nucleotide_conformations(results, is_borderline=None):
    """
    Count, per base group, how many measured nucleotides came out syn.

    Counting follows the same rule as the table: a nucleotide modelled in more
    than one conformation is counted once per conformation, and C-glycosides
    such as pseudouridine count as pyrimidines.

    Inputs
    ------
    results : list of (residue, chi, conformation, altloc), as returned by
        classify_nucleotide_conformation
    is_borderline : callable taking chi and returning True when the angle sits
        close to the syn/anti boundary, or None to skip the borderline count

    Returns
    -------
    dict keyed by CONFORMATION_GROUPS, each holding a (syn, borderline, total)
    tuple of counts
    """
    counts = {name: [0, 0, 0] for name in CONFORMATION_GROUPS}
    for resi, chi, conformation, _ in results:
        group = "pyrimidines" if is_pyrimidine(resi) else "purines"
        for name in (group, "nucleotides"):
            counts[name][2] += 1
            if conformation == "syn":
                counts[name][0] += 1
            if is_borderline is not None and is_borderline(chi):
                counts[name][1] += 1
    return {name: tuple(values) for name, values in counts.items()}


def format_percentage(count, total, precision=2, full_precision=False):
    """
    "count/total (pct%)" as a single string, following the same rounding rules
    as the table cells. An empty total has no percentage, so it prints as NA.
    """
    if total == 0:
        return "%d/%d (NA)" % (count, total)
    pct = _format_cell(100.0 * count / total, precision, full_precision)
    return "%d/%d (%s%%)" % (count, total, pct)


def add_version_arg(parser):
    """Add the shared --version flag, reporting the installed package version."""
    parser.add_argument('--version', action='version',
                        version='pdb_python_tools ' + __version__)


def add_output_args(parser):
    """
    Add the shared output-formatting flags to an argparse parser so every
    tool exposes the same output interface
    """
    parser.add_argument('-f', '--format', choices=('tsv', 'csv'), default='tsv',
                        help='output format (default: tsv)')
    parser.add_argument('-o', '--output', default=None,
                        help='write to this file instead of stdout '
                             '(refuses to overwrite unless --force is given)')
    parser.add_argument('--force', action='store_true',
                        help='allow overwriting an existing --output file')
    parser.add_argument('--precision', type=int, default=2,
                        help='decimal places for distance values (default: 2; '
                             'use a negative value for raw floats)')
    parser.add_argument('--full-precision', action='store_true', dest='full_precision',
                        help='print raw float values without rounding')
    parser.add_argument('--coot', default=None, metavar='PATH',
                        help='also write a Coot (0.9 or 1) Python script '
                             'to PATH; opening it in Coot shows the same results as a '
                             'clickable list that recenters the view on each residue, '
                             'and for atom_tracker a second window with a bar graph '
                             'of the displacement per chain '
                             "(refuses to overwrite unless --force is given)")

def _format_cell(value, precision, full_precision):
    """
    Format a single table cell, rounding floats unless full precision is
    requested. None prints as NA.
    """
    if value is None:
        return "NA"
    if isinstance(value, float):
        if full_precision or precision is None or precision < 0:
            return str(value)
        return format(value, f".{precision}f")
    return str(value)

def write_table(header, rows, fmt="tsv", output=None, force=False,
                precision=2, full_precision=False, comments=()):
    """
    Write a table (header + rows) as TSV or CSV to stdout or a file.

    Float cells are rounded to `precision` decimals unless `full_precision` is
    set. Writing to a path that already exists is refused unless `force` is True.

    Inputs
    ------
    header : sequence of column names
    rows : iterable of row sequences
    fmt : "tsv" or "csv"
    output : path to write to, or None for stdout
    force : allow overwriting an existing output file
    precision : decimal places for float cells (negative -> raw)
    full_precision : if True, never round float cells
    comments : lines written above the header, each prefixed with "# "
    """
    fmt = fmt.lower()
    if fmt not in ("tsv", "csv"):
        raise ValueError(f"Unsupported output format: {fmt}")
    delimiter = "\t" if fmt == "tsv" else ","
    if output is not None and os.path.exists(output) and not force:
        raise FileExistsError(f"Refusing to overwrite existing file: {output} (use --force)")
    handle = open(output, "w", newline="") if output is not None else sys.stdout
    try:
        for comment in comments:
            handle.write("# %s\n" % comment)
        writer = csv.writer(handle, delimiter=delimiter, lineterminator="\n")
        writer.writerow(list(header))
        for row in rows:
            writer.writerow([_format_cell(v, precision, full_precision) for v in row])
    except BrokenPipeError:
        _exit_on_broken_pipe()
    finally:
        if output is not None:
            handle.close()


def _exit_on_broken_pipe():
    """
    Leave quietly after a broken pipe.

    """
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
    except (OSError, ValueError):
        pass
    raise SystemExit(1)


# Bar colours for the graph the generated script can draw, following the
# green -> yellow -> orange -> red ladder, the last bound is
# None and catches everything above.
COOT_GRAPH_BANDS = ((0.5, "#55dd55"), (1.0, "#eecc22"),
                    (2.0, "#ee9933"), (None, "#dd4444"))


_LEADING_INT = re.compile(r"-?\d+")


def _residue_number(seqid):
    """
    The residue number in a seqid, as an int for plotting.

    An insertion code ("10A") keeps the number it is attached to; a seqid with
    no number at all gives None, and its residue is left out of the graph.
    """
    match = _LEADING_INT.match(str(seqid).strip())
    return int(match.group()) if match else None


# Template for the generated Coot script. It is Python-2 compatible, so that
# Coot 0.9 (Python 2, PyGTK) and Coot 1 (Python 3, PyGObject) can both run it,
# and only relies on `set_rotation_centre`
_COOT_TEMPLATE = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Coot script generated by pdb_python_tools.
# Open it from Coot:  Calculate -> Run Script...   (or:  coot --script this_file.py)
# Requires Coot (0.9 or 1). Clicking a row recenters the view on the
# listed residue's CA/C1' (or, for find_contacts, the contact midpoint).
from __future__ import print_function

import math

TITLE = {title!r}

# Each entry: (label, value_text, x, y, z)
MARKERS = [
{markers}
]

# The bar graph, drawn in a second window when it is not empty. Each entry
# holds one value per series, in the order of GRAPH_SERIES:
#   (label, chain, residue number, values, value_texts, x, y, z)
# A value of None means that residue has none of that kind, and gets no bar.
GRAPH_TITLE = {graph_title!r}
GRAPH_SERIES = [
{series}
]
GRAPH_SELECTED = [{selected}]   # which series the bars stand for; "Value..." moves it
GRAPH = [
{graph}
]

# Bar colours, lowest band first: (upper bound, colour), the last bound None
GRAPH_BANDS = [
{bands}
]


def _recentre_function():
    """Coot's set_rotation_centre, or None when this is not running in Coot."""
    function = globals().get("set_rotation_centre")   # Coot 0.9 runs scripts
    if function is not None:                          # in its own namespace
        return function
    try:
        import coot                                   # Coot 1 keeps them here
    except ImportError:
        return None
    return getattr(coot, "set_rotation_centre", None)


def _gtk():
    """
    The GTK module Coot brought and which one it is, or (None, None).

    Coot 0.9 embeds PyGTK (GTK 2), Coot 1 PyGObject (GTK 4, GTK 3 in early
    builds); the three differ enough to be told apart here.
    """
    try:
        import gtk                                    # Coot 0.9
    except ImportError:
        pass
    else:
        # PyGObject ships a stand-in "gtk" module that imports and then raises
        # on every attribute, so PyGTK has to be recognised by one of its own
        if getattr(gtk, "pygtk_version", None) is not None:
            return gtk, "pygtk"
    try:
        import gi                                     # Coot 1
    except ImportError:
        return None, None
    # Coot has settled on a version already; asking for another one would fail
    try:
        version = gi.get_required_version("Gtk")
    except Exception:
        version = None
    if version is None:
        for candidate in ("4.0", "3.0"):
            try:
                gi.require_version("Gtk", candidate)
            except (ValueError, AttributeError):
                continue
            version = candidate
            break
    if version is None:
        return None, None
    try:
        from gi.repository import Gtk
    except ImportError:
        return None, None
    return Gtk, "gtk4" if version.startswith("4") else "gtk3"


def _box(gtk, kind, spacing, border=0):
    """A vertical box."""
    if kind == "pygtk":
        box = gtk.VBox(False, spacing)
    else:
        box = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=spacing)
    if border:
        if kind == "gtk4":
            for side in ("start", "end", "top", "bottom"):
                getattr(box, "set_margin_" + side)(border)
        else:
            box.set_border_width(border)
    return box


def _pack(kind, box, child, grow=False):
    """Add a child to a vertical box, growing with the window or not."""
    if kind == "gtk4":
        if grow:
            child.set_hexpand(True)
            child.set_vexpand(True)
        box.append(child)
    else:
        box.pack_start(child, grow, grow, 0)


def _label(gtk, kind, text):
    """A left-aligned label."""
    if kind == "pygtk":
        label = gtk.Label(text)
        label.set_alignment(0, 0.5)
    else:
        label = gtk.Label(label=text)
        label.set_xalign(0)
    return label


def _button(gtk, kind, text):
    """A button with its text against the left edge."""
    button = gtk.Button(text) if kind == "pygtk" else gtk.Button(label=text)
    child = button.get_child()
    if child is not None:
        if hasattr(child, "set_xalign"):
            child.set_xalign(0)
        elif hasattr(child, "set_alignment"):
            child.set_alignment(0, 0.5)
    return button


def _scrolled(gtk, kind, child):
    """A scrolled window around `child`."""
    scrolled = gtk.ScrolledWindow()
    if kind == "pygtk":
        scrolled.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
        scrolled.add_with_viewport(child)
    else:
        scrolled.set_policy(gtk.PolicyType.AUTOMATIC, gtk.PolicyType.AUTOMATIC)
        if kind == "gtk4":
            scrolled.set_child(child)
        else:
            scrolled.add(child)
    return scrolled


def _make_cb(recentre, x, y, z):
    """Return a click handler that recenters the Coot view on (x, y, z)."""
    def _cb(*args):
        recentre(x, y, z)
    return _cb


def show_coot_dialog():
    recentre = _recentre_function()
    gtk, kind = _gtk() if recentre is not None else (None, None)
    # Outside Coot, or without a GUI, just print the list.
    if gtk is None:
        for label, value_text, x, y, z in MARKERS:
            print("%-28s %12s  (%.3f, %.3f, %.3f)" % (label, value_text, x, y, z))
        return
    window = gtk.Window(gtk.WINDOW_TOPLEVEL) if kind == "pygtk" else gtk.Window()
    window.set_title(TITLE)
    window.set_default_size(380, 540)
    outer = _box(gtk, kind, 2, border=4)
    _pack(kind, outer, _label(gtk, kind, TITLE))
    inner = _box(gtk, kind, 0)
    for label, value_text, x, y, z in MARKERS:
        button = _button(gtk, kind, "%s    %s" % (label, value_text))
        button.connect("clicked", _make_cb(recentre, x, y, z))
        _pack(kind, inner, button)
    _pack(kind, outer, _scrolled(gtk, kind, inner), True)
    close_button = _button(gtk, kind, "Close")
    close_button.connect("clicked", lambda *args: window.destroy())
    _pack(kind, outer, close_button)
    if kind == "gtk4":
        window.set_child(outer)
        window.present()
    else:
        window.add(outer)
        window.show_all()


# ---------------------------------------------------------------------------
# The bar graph
# ---------------------------------------------------------------------------


GRAPH_MIN_WIDTH = 560                        # the legend, and the window itself
GRAPH_HEIGHT = 130
GRAPH_MARGINS = (48.0, 14.0, 10.0, 26.0)     # left, right, top, bottom
GRAPH_CHAIN_MIN_WIDTH = 240

# What the "Options" window can change. Each is a one-item list, so that the
# window can drop a new value into it without the drawing code having to be
# handed it. The defaults are the graph as it opens.
#
# Every chain is drawn at the same bar width, so one bar means the same as any
# other and two chains can be held against each other along x. A chain that
# would need a graph wider than GRAPH_MAX_WIDTH is the one exception: there the
# bar and the gap are scaled down together, keeping their proportion, so that
# asking for a wider bar still draws a wider bar up to the point where the
# residues fill the canvas and nothing wider will fit.
GRAPH_BAR_WIDTH = [7.0]
GRAPH_TICK_RESIDUES = [10]   # residues from one x-axis number to the next
GRAPH_Y_STEP = [1.0]         # angstrom from one gridline to the next
GRAPH_YMAX = [None]          # one top for every chain, or None for its own
GRAPH_MAX_WIDTH = [30000]    # the widest one chain's graph may be drawn

GRAPH_BAR_GAP = 2.0          # clear space between two neighbouring bars
GRAPH_BAR_MAX_WIDTH = 60.0
# GTK 2 keeps widget coordinates in 16 bits, so nothing may be drawn wider
GRAPH_WIDTH_LIMIT = 32000
GRAPH_TICK_MIN_GAP = 30.0    # x numbers closer than this are thinned out
GRAPH_Y_LABEL_MIN_GAP = 11.0 # and so are y numbers
GRAPH_Y_MAX_LINES = 500      # a floor under a very small gridline step


def _bar_pitch():
    """Points from one residue to the next: a bar and the gap after it."""
    return GRAPH_BAR_WIDTH[0] + GRAPH_BAR_GAP


def _selected():
    """Which of GRAPH_SERIES the bars are drawn from."""
    return GRAPH_SELECTED[0]


def _entry_value(entry):
    """
    The number the shown series holds for one residue.

    None when that residue has no such value gets no bar.
    """
    return entry[3][_selected()]


def _entry_text(entry):
    """The same number, ready to show."""
    return entry[4][_selected()]


def _band_colour(value):
    """The bar colour for one value, from the GRAPH_BANDS ladder."""
    for upper, colour in GRAPH_BANDS:
        if upper is None or value < upper:
            return colour
    return GRAPH_BANDS[-1][1]


def _rgb(colour):
    """'#rrggbb' as the three 0-1 components cairo takes."""
    colour = colour.lstrip("#")
    return (int(colour[0:2], 16) / 255.0,
            int(colour[2:4], 16) / 255.0,
            int(colour[4:6], 16) / 255.0)


def _ymax_of(entries):
    """
    The top of the scale for `entries`, rounded up to a whole gridline.

    Each chain gets its own, worked out from the series on show. A top set in the Options window
    overrides all of that, and every chain is drawn against it.
    """
    if GRAPH_YMAX[0] is not None:
        return GRAPH_YMAX[0]
    step = GRAPH_Y_STEP[0]
    values = [_entry_value(entry) for entry in entries]
    values = [value for value in values if value is not None]
    highest = max(values) if values else 0.0
    return max(step, math.ceil(highest / step) * step)


def _graph_chains():
    """The graph entries grouped by chain, in the order the chains appear."""
    order = []
    by_chain = dict()
    for entry in GRAPH:
        chain = entry[1]
        if chain not in by_chain:
            by_chain[chain] = []
            order.append(chain)
        by_chain[chain].append(entry)
    return [(chain, by_chain[chain]) for chain in order]


class _ChainGraph(object):
    """One chain's bar chart: it draws itself and answers clicks on its bars."""

    def __init__(self, chain, entries):
        self.chain = chain
        self.entries = entries
        numbers = [entry[2] for entry in entries]
        self.first = min(numbers)
        self.last = max(numbers)
        # (left edge, right edge, entry) per bar, refreshed on every draw so
        # that a resized window still maps a click to the right residue
        self.bars = []

    def span(self):
        """Residue numbers covered."""
        return max(1, self.last - self.first)

    def width(self):
        """
        The width this chain asks for, at the shared points-per-residue.

        A short chain gets a short graph rather than a stretched one to keep the bars of every chain the same width.
        """
        left, right, _top, _bottom = GRAPH_MARGINS
        needed = (left + right + self.span() * _bar_pitch()
                  + GRAPH_BAR_WIDTH[0])
        return int(max(GRAPH_CHAIN_MIN_WIDTH, min(GRAPH_MAX_WIDTH[0], needed)))

    def size(self):
        """The width and height this chain's graph asks for."""
        return (self.width(), GRAPH_HEIGHT)

    def _scale(self, plot_w):
        """
        Points per residue, and the bar width, for this draw.

        Both are what the Options window asks for, unless this chain needs
        more room than the canvas has.
        """
        pitch = _bar_pitch()
        bar_w = GRAPH_BAR_WIDTH[0]
        # What is left once the last bar has its own width to sit in
        room = max(0.0, plot_w - bar_w)
        needed = self.span() * pitch
        if needed > room:
            factor = room / needed
            pitch = pitch * factor
            bar_w = max(1.0, bar_w * factor)
        return pitch, bar_w

    def squeezed(self, plot_w):
        """Whether this chain had to be scaled down to fit the canvas."""
        _pitch, bar_w = self._scale(plot_w)
        return bar_w < GRAPH_BAR_WIDTH[0] - 0.01

    def draw(self, cr, width, height):
        left, right, top, bottom = GRAPH_MARGINS
        plot_w = max(1.0, width - left - right)
        plot_h = max(1.0, height - top - bottom)
        pitch, bar_w = self._scale(plot_w)
        # The axes stop where the residues do, so a short chain in a wide
        # window is a short graph and not a long empty one
        used_w = min(plot_w, self.span() * pitch + bar_w)
        ymax = self.ymax()
        cr.set_line_width(1.0)
        cr.select_font_face("Sans")
        cr.set_font_size(9.0)
        cr.set_source_rgb(1.0, 1.0, 1.0)
        cr.rectangle(0.0, 0.0, width, height)
        cr.fill()
        self._draw_y_axis(cr, left, top, used_w, plot_h, ymax)
        self._draw_bars(cr, left, top, plot_h, pitch, bar_w, ymax)
        self._draw_x_axis(cr, left, top, used_w, plot_h, pitch)
        if bar_w < GRAPH_BAR_WIDTH[0] - 0.01:
            # so that a bar width that cannot be honoured does not look as
            # though the setting was ignored
            text = "bars %.1f pt: too many residues for the graph width" % bar_w
            cr.set_source_rgb(0.55, 0.55, 0.55)
            cr.move_to(left + used_w - cr.text_extents(text)[4] - 4.0, top + 9.0)
            cr.show_text(text)

    def ymax(self):
        """The top of this chain's own scale."""
        return _ymax_of(self.entries)

    def _label_every(self, plot_h, ymax):
        """How many gridlines to a number, so that the numbers stay apart."""
        gap = plot_h / (ymax / GRAPH_Y_STEP[0])
        for candidate in (1, 2, 5, 10, 20, 50):
            if gap * candidate >= GRAPH_Y_LABEL_MIN_GAP:
                return candidate
        return 100

    def _draw_y_axis(self, cr, left, top, used_w, plot_h, ymax):
        """A gridline every step, up to this chain's maximum."""
        step = GRAPH_Y_STEP[0]
        # A step far smaller than the scale would draw thousands of lines
        lines = int(min(GRAPH_Y_MAX_LINES, round(ymax / step, 6)))
        every = self._label_every(plot_h, ymax)
        for index in range(lines + 1):
            value = index * step
            y = top + plot_h - (value / ymax) * plot_h
            cr.set_source_rgb(0.87, 0.87, 0.87)
            cr.move_to(left, y)
            cr.line_to(left + used_w, y)
            cr.stroke()
            if index % every:
                continue
            text = "%g" % round(value, 6)
            cr.set_source_rgb(0.35, 0.35, 0.35)
            cr.move_to(left - 6.0 - cr.text_extents(text)[4], y + 3.0)
            cr.show_text(text)

    def _draw_bars(self, cr, left, top, plot_h, pitch, bar_w, ymax):
        """One bar per residue, at its own residue number along the chain."""
        self.bars = []
        for entry in self.entries:
            value = _entry_value(entry)
            if value is None:
                continue                # nothing to draw for this residue here
            bar_h = plot_h * min(1.0, value / ymax)
            x = left + (entry[2] - self.first) * pitch
            cr.set_source_rgb(*_rgb(_band_colour(value)))
            cr.rectangle(x, top + plot_h - bar_h, bar_w, bar_h)
            cr.fill()
            self.bars.append((x, x + bar_w, entry))

    def _tick_step(self, pitch):
        """
        How many residues one x-axis label is from the next.

        The step from the Options window, unless that many would not be far
        enough apart to read, which happens on a chain squeezed into
        GRAPH_MAX_WIDTH: there it goes up in whole steps until they fit.
        """
        step = float(GRAPH_TICK_RESIDUES[0])
        if pitch <= 0.0:
            return step
        while step * pitch < GRAPH_TICK_MIN_GAP:
            step += GRAPH_TICK_RESIDUES[0]
        return step

    def _draw_x_axis(self, cr, left, top, used_w, plot_h, pitch):
        """The baseline, numbered every so many residues."""
        cr.set_source_rgb(0.45, 0.45, 0.45)
        cr.move_to(left, top + plot_h)
        cr.line_to(left + used_w, top + plot_h)
        cr.stroke()
        cr.set_source_rgb(0.35, 0.35, 0.35)
        step = self._tick_step(pitch)
        number = math.ceil(self.first / step) * step
        numbers = []
        while number <= self.last:
            numbers.append(number)
            number += step
        # A chain that does not reach a round ten still says where it sits
        if not numbers:
            numbers = [self.first]
        for number in numbers:
            x = left + (number - self.first) * pitch
            text = "%d" % int(number)
            cr.move_to(x - cr.text_extents(text)[4] / 2.0, top + plot_h + 14.0)
            cr.show_text(text)

    def clicked(self, x):
        """The entry whose bar is under `x`, or None when no bar is."""
        for start, end, entry in self.bars:
            if start - 2.0 <= x <= end + 2.0:
                return entry
        return None


def _legend_text(index):
    """The range one colour band stands for, as the legend spells it out."""
    lower = GRAPH_BANDS[index - 1][0] if index > 0 else None
    upper = GRAPH_BANDS[index][0]
    if lower is None:
        return "< %g" % upper
    if upper is None:
        return "> %g" % lower
    return "%g - %g" % (lower, upper)


def _draw_legend(cr, width, height):
    """A row of swatches naming the band each bar colour stands for."""
    cr.set_source_rgb(1.0, 1.0, 1.0)
    cr.rectangle(0.0, 0.0, width, height)
    cr.fill()
    cr.select_font_face("Sans")
    cr.set_font_size(9.0)
    x = 6.0
    for index in range(len(GRAPH_BANDS)):
        cr.set_source_rgb(*_rgb(GRAPH_BANDS[index][1]))
        cr.rectangle(x, height / 2.0 - 5.0, 12.0, 10.0)
        cr.fill()
        text = _legend_text(index)
        x += 16.0
        cr.set_source_rgb(0.25, 0.25, 0.25)
        cr.move_to(x, height / 2.0 + 4.0)
        cr.show_text(text)
        x += cr.text_extents(text)[4] + 14.0


def _button_press_mask(gtk, kind):
    """The GDK mask for a button press, however this GTK spells it."""
    if kind == "pygtk":
        return gtk.gdk.BUTTON_PRESS_MASK
    try:
        from gi.repository import Gdk
    except ImportError:
        return 256                  # GDK_BUTTON_PRESS_MASK, 1 << 8 in every GDK
    return Gdk.EventMask.BUTTON_PRESS_MASK


def _drawing_area(gtk, kind, on_draw, on_click):
    """
    A DrawingArea calling on_draw(cr, width, height) and on_click(x).

    The three GTKs disagree on both halves: PyGTK draws on an expose event and
    only hears a click once the button mask is switched on, GTK 3 has a draw
    signal, and GTK 4 takes a draw function and a click gesture.
    """
    area = gtk.DrawingArea()
    if kind == "gtk4":
        area.set_draw_func(lambda widget, cr, width, height, *args:
                           on_draw(cr, width, height))
        gesture = gtk.GestureClick()
        gesture.connect("pressed", lambda control, presses, x, y: on_click(x))
        area.add_controller(gesture)
        return area
    if kind == "pygtk":
        def exposed(widget, event):
            allocation = widget.get_allocation()
            on_draw(widget.window.cairo_create(),
                    allocation.width, allocation.height)
            return False
        area.connect("expose-event", exposed)
    else:
        area.connect("draw", lambda widget, cr: on_draw(
            cr, widget.get_allocated_width(), widget.get_allocated_height()))
    area.add_events(_button_press_mask(gtk, kind))
    area.connect("button-press-event", lambda widget, event: on_click(event.x))
    return area


def _make_pick(graph, recentre, status):
    """Return a click handler that recenters on the residue whose bar was hit."""
    def _pick(x):
        entry = graph.clicked(x)
        if entry is None:
            return
        recentre(entry[5], entry[6], entry[7])
        status.set_text("%s    %s" % (entry[0], _entry_text(entry)))
    return _pick


def _hbox(gtk, kind, spacing):
    """A horizontal box."""
    if kind == "pygtk":
        return gtk.HBox(False, spacing)
    return gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=spacing)


def _entry(gtk, text, width_chars=6):
    """A short text entry, filled in."""
    entry = gtk.Entry()
    entry.set_width_chars(width_chars)
    entry.set_text(text)
    return entry


def _separator(gtk, kind):
    """A horizontal rule."""
    if kind == "pygtk":
        return gtk.HSeparator()
    return gtk.Separator(orientation=gtk.Orientation.HORIZONTAL)


def _refresh(areas):
    """
    Put every graph back on the screen with the current settings.

    Each area comes with the size it asks for, or None when it has a fixed
    one, so that a change of bar width can widen the chains that need it.
    """
    for area, size in areas:
        if size is not None:
            area.set_size_request(*size())
        area.queue_draw()


def _swatch(gtk, kind, colour):
    """A small square of one band's colour, which stands in for its name."""
    def draw(cr, width, height):
        cr.set_source_rgb(*_rgb(colour))
        cr.rectangle(0.0, 0.0, width, height)
        cr.fill()
    area = _drawing_area(gtk, kind, draw, lambda x: None)
    area.set_size_request(16, 16)
    return area


class _Rejected(Exception):
    """One entry of the Options window that cannot be used, and why."""


def _positive(text, name, whole=False):
    """One entry read as a number greater than zero."""
    text = text.strip()
    try:
        value = float(text)
    except ValueError:
        raise _Rejected("%s: '%s' is not a number." % (name, text))
    if value <= 0:
        raise _Rejected("%s has to be greater than zero." % name)
    if whole and value != int(value):
        raise _Rejected("%s has to be a whole number." % name)
    return value


def _apply_options(cutoffs, x_step, y_step, ymax, bar_width, max_width):
    """
    Put everything the Options window says into effect.

    Every entry is checked before any of them is applied, so one bad number
    leaves the graph exactly as it was. `ymax` may be blank, which puts each
    chain back on a scale of its own. Returns "" when the settings were taken,
    and what is wrong with them when they were not.
    """
    try:
        values = [_positive(text, "Cutoff") for text in cutoffs]
        for earlier, later in zip(values, values[1:]):
            if later <= earlier:
                raise _Rejected("Each cutoff has to be larger than the one above it.")
        residues = _positive(x_step, "Residues per label", whole=True)
        step = _positive(y_step, "Gridline every")
        width = _positive(bar_width, "Bar width")
        if width > GRAPH_BAR_MAX_WIDTH:
            raise _Rejected("Bar width has to be %g or less." % GRAPH_BAR_MAX_WIDTH)
        canvas = _positive(max_width, "Graph width")
        if canvas < GRAPH_CHAIN_MIN_WIDTH:
            raise _Rejected("Graph width has to be %d or more."
                            % GRAPH_CHAIN_MIN_WIDTH)
        if canvas > GRAPH_WIDTH_LIMIT:
            raise _Rejected("Graph width has to be %d or less: a wider one is "
                            "past what GTK can draw." % GRAPH_WIDTH_LIMIT)
        top = None
        if ymax.strip():
            top = _positive(ymax, "Highest y value")
    except _Rejected as problem:
        return str(problem)
    for index, value in enumerate(values):
        GRAPH_BANDS[index] = (value, GRAPH_BANDS[index][1])
    GRAPH_TICK_RESIDUES[0] = int(residues)
    GRAPH_Y_STEP[0] = step
    GRAPH_BAR_WIDTH[0] = width
    GRAPH_MAX_WIDTH[0] = int(canvas)
    GRAPH_YMAX[0] = top
    return ""


# The side windows are built inside a callback, so they are kept here rather
# than left to the garbage collector
_OPEN_WINDOWS = []


def _side_window(gtk, kind, title, parent):
    """An empty window belonging to the graph window."""
    window = gtk.Window(gtk.WINDOW_TOPLEVEL) if kind == "pygtk" else gtk.Window()
    window.set_title(title)
    window.set_transient_for(parent)
    return window


def _show_side_window(kind, window, outer):
    """Fill one in and put it on the screen, keeping a reference to it."""
    def forget(*args):
        if window in _OPEN_WINDOWS:
            _OPEN_WINDOWS.remove(window)

    _OPEN_WINDOWS.append(window)
    window.connect("destroy", forget)
    if kind == "gtk4":
        window.set_child(outer)
        window.present()
    else:
        window.add(outer)
        window.show_all()
    return window


def _make_choose(index, window, areas, header):
    """A button handler that puts series `index` on show and closes `window`."""
    def _choose(*args):
        GRAPH_SELECTED[0] = index
        header.set_text(GRAPH_SERIES[index])
        _refresh(areas)
        window.destroy()
    return _choose


def show_value_dialog(gtk, kind, parent, areas, header):
    """
    The window that picks which number the bars stand for.

    One button per series in GRAPH_SERIES; picking one redraws the graphs,
    against a scale worked out afresh for it, and closes the window. Residues
    with no value in that series simply lose their bar.
    """
    window = _side_window(gtk, kind, "Value shown", parent)
    outer = _box(gtk, kind, 4, border=8)
    _pack(kind, outer, _label(gtk, kind, "Draw the bars from:"))
    for index, name in enumerate(GRAPH_SERIES):
        mark = "* " if index == _selected() else "   "
        button = _button(gtk, kind, mark + name)
        button.connect("clicked", _make_choose(index, window, areas, header))
        _pack(kind, outer, button)
    close_button = _button(gtk, kind, "Close")
    close_button.connect("clicked", lambda *args: window.destroy())
    _pack(kind, outer, close_button)
    return _show_side_window(kind, window, outer)


def _settings_row(gtk, kind, text, value, note=""):
    """A labelled entry, and the entry itself so it can be read back."""
    row = _hbox(gtk, kind, 6)
    label = _label(gtk, kind, text)
    label.set_size_request(150, -1)
    _pack(kind, row, label)
    entry = _entry(gtk, value)
    _pack(kind, row, entry)
    if note:
        _pack(kind, row, _label(gtk, kind, note))
    return row, entry


def show_options_dialog(gtk, kind, parent, areas):
    """
    The window that sets out how the graphs are drawn.

    The colour bands, how often each axis is numbered, the top of the y axis
    and how wide a bar is. Applying redraws `areas` straight away, without
    running the tool again; the numbers last as long as the graph window does.
    """
    window = _side_window(gtk, kind, "Options", parent)
    outer = _box(gtk, kind, 4, border=8)

    _pack(kind, outer, _label(gtk, kind,
                              "A bar takes the colour of the band its value falls in."))
    cutoffs = []
    for upper, colour in GRAPH_BANDS:
        row = _hbox(gtk, kind, 6)
        _pack(kind, row, _swatch(gtk, kind, colour))
        if upper is None:
            _pack(kind, row, _label(gtk, kind, "everything above"))
        else:
            _pack(kind, row, _label(gtk, kind, "up to"))
            entry = _entry(gtk, "%g" % upper)
            cutoffs.append(entry)
            _pack(kind, row, entry)
        _pack(kind, outer, row)

    _pack(kind, outer, _separator(gtk, kind))
    rows = [("Number x axis every", "%g" % GRAPH_TICK_RESIDUES[0], "residues"),
            ("Gridline every", "%g" % GRAPH_Y_STEP[0], "Å"),
            ("Highest y value",
             "" if GRAPH_YMAX[0] is None else "%g" % GRAPH_YMAX[0],
             "Å  (blank: each chain to its own)"),
            ("Bar width", "%g" % GRAPH_BAR_WIDTH[0], "points per residue"),
            ("Widest graph", "%d" % GRAPH_MAX_WIDTH[0],
             "points  (a longer chain is squeezed into it)")]
    entries = []
    for text, value, note in rows:
        row, entry = _settings_row(gtk, kind, text, value, note)
        entries.append(entry)
        _pack(kind, outer, row)
    message = _label(gtk, kind, "")

    def on_apply(*args):
        problem = _apply_options([entry.get_text() for entry in cutoffs],
                                 *[entry.get_text() for entry in entries])
        if problem:
            message.set_text(problem)
            return
        message.set_text("Applied.")
        _refresh(areas)

    for entry in cutoffs + entries:
        entry.connect("activate", on_apply)
    buttons = _hbox(gtk, kind, 6)
    apply_button = _button(gtk, kind, "Apply")
    apply_button.connect("clicked", on_apply)
    close_button = _button(gtk, kind, "Close")
    close_button.connect("clicked", lambda *args: window.destroy())
    _pack(kind, buttons, apply_button)
    _pack(kind, buttons, close_button)
    _pack(kind, outer, message)
    _pack(kind, outer, buttons)
    return _show_side_window(kind, window, outer)


def show_coot_graph():
    """
    Open the graph window: one bar chart per chain, next to the list dialog.

    "Value..." picks which number the bars stand for and "Options" how they
    are drawn; both redraw without running the tool again. Nothing is opened
    when the run has no graph data, or when this is not running inside a Coot
    with a GUI.
    """
    recentre = _recentre_function()
    if not GRAPH or recentre is None:
        return
    gtk, kind = _gtk()
    if gtk is None:
        return
    window = gtk.Window(gtk.WINDOW_TOPLEVEL) if kind == "pygtk" else gtk.Window()
    window.set_title(GRAPH_TITLE)
    window.set_default_size(660, 460)
    outer = _box(gtk, kind, 2, border=4)
    _pack(kind, outer, _label(gtk, kind, GRAPH_TITLE))
    header = _label(gtk, kind, GRAPH_SERIES[_selected()])
    _pack(kind, outer, header)
    status = _label(gtk, kind, "Click a bar to recentre on that residue.")

    inner = _box(gtk, kind, 6)
    areas = []
    for chain, entries in _graph_chains():
        graph = _ChainGraph(chain, entries)
        area = _drawing_area(gtk, kind, graph.draw,
                             _make_pick(graph, recentre, status))
        area.set_size_request(*graph.size())
        # with the size to ask for again when the bar width changes
        areas.append((area, graph.size))
        _pack(kind, inner, _label(gtk, kind, "Chain %s" % chain))
        _pack(kind, inner, area)
    _pack(kind, outer, _scrolled(gtk, kind, inner), True)

    legend = _drawing_area(gtk, kind, _draw_legend, lambda x: None)
    legend.set_size_request(GRAPH_MIN_WIDTH, 22)
    areas.append((legend, None))
    _pack(kind, outer, legend)
    _pack(kind, outer, status)
    buttons = _hbox(gtk, kind, 6)
    value_button = _button(gtk, kind, "Value...")
    value_button.connect("clicked", lambda *args:
                         show_value_dialog(gtk, kind, window, areas, header))
    options_button = _button(gtk, kind, "Options")
    options_button.connect("clicked", lambda *args:
                           show_options_dialog(gtk, kind, window, areas))
    close_button = _button(gtk, kind, "Close")
    close_button.connect("clicked", lambda *args: window.destroy())
    _pack(kind, buttons, value_button)
    _pack(kind, buttons, options_button)
    _pack(kind, buttons, close_button)
    _pack(kind, outer, buttons)
    if kind == "gtk4":
        window.set_child(outer)
        window.present()
    else:
        window.add(outer)
        window.show_all()


show_coot_dialog()
show_coot_graph()
'''


def write_coot_script(markers, title, output, force=False, precision=2,
                      full_precision=False, graph=None, graph_title=None,
                      graph_series=(), graph_selected=0, graph_bands=None):
    """
    Write a Coot (0.9 or 1) Python script.

    Running the script inside Coot opens a dialog listing `markers` in the given
    order; clicking a row recenters the view via `set_rotation_centre`. When
    `graph` is given, a second window opens with one bar chart per
    chain: one bar per residue at its residue number, coloured by the band its value falls into,
    and clickable in the same way as a row of the list. `graph_bands` sets the
    colours it starts with; the graph window can move the boundaries between
    them afterwards.

    Inputs
    ------
    markers : iterable of (label, value, unit, x, y, z)
        label : text shown for the row (residue/contact identity)
        value : the relevant number (distance or angle), rounded
        unit  : unit string appended after the value (e.g. "Å", "°"), or "" for none
        x, y, z : coordinate the click recenters on (CA/C1' or contact midpoint)
    title : dialog window title
    output : path to write the script to (required)
    force : allow overwriting an existing file
    precision, full_precision : control rounding of the displayed value
    graph : iterable of (label, chain, seqid, values, unit, x, y, z), or None
        the residues to draw as bars. `values` holds one number per entry of
        `graph_series`, or None where that residue has none of that kind. A
        seqid with no residue number in it is left out of the graph, and no
        graph window opens when nothing is left
    graph_title : title of the graph window (defaults to `title`)
    graph_series : what the bars can be drawn from, e.g.
        ("Max displacement (Å)", "Average displacement (Å)"); the graph
        window's "Value..." button switches between them
    graph_selected : which of `graph_series` the window opens on (default: the
        first)
    graph_bands : the colour ladder as ((upper bound, colour), ...), lowest
        band first and the last bound None (defaults to COOT_GRAPH_BANDS)
    """
    if output is None:
        raise ValueError("write_coot_script requires an output path")
    if graph and not graph_series:
        raise ValueError("write_coot_script requires graph_series with a graph")
    if graph_series and not 0 <= graph_selected < len(graph_series):
        raise ValueError("graph_selected must name one of graph_series")
    if os.path.exists(output) and not force:
        raise FileExistsError(f"Refusing to overwrite existing file: {output} (use --force)")
    lines = []
    for label, value, unit, x, y, z in markers:
        value_text = _format_cell(value, precision, full_precision)
        if unit:
            value_text = value_text + " " + unit
        # %r keeps full-precision coordinates and safely escapes the strings
        lines.append("    (%r, %r, %r, %r, %r)," % (label, value_text, float(x), float(y), float(z)))
    graph_lines = []
    for label, chain, seqid, values, unit, x, y, z in (graph or []):
        number = _residue_number(seqid)
        if number is None:
            continue
        numbers = tuple(None if value is None else float(value) for value in values)
        texts = []
        for value in values:
            text = _format_cell(value, precision, full_precision)
            texts.append(text + " " + unit if unit and value is not None else text)
        graph_lines.append("    (%r, %r, %r, %r, %r, %r, %r, %r)," % (
            label, chain, number, numbers, tuple(texts),
            float(x), float(y), float(z)))
    band_lines = ["    (%r, %r)," % (upper, colour)
                  for upper, colour in (graph_bands or COOT_GRAPH_BANDS)]
    series_lines = ["    %r," % name for name in graph_series]
    content = _COOT_TEMPLATE.format(title=title, markers="\n".join(lines),
                                    graph_title=graph_title or title,
                                    series="\n".join(series_lines),
                                    selected=int(graph_selected),
                                    graph="\n".join(graph_lines),
                                    bands="\n".join(band_lines))
    with open(output, "w") as handle:
        handle.write(content)