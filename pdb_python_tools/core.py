#!/usr/bin/env python3
import math
import csv
import gzip
import os
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
                        help='also write a Coot (0.9) Python script '
                             'to PATH; opening it in Coot shows the same results as a '
                             'clickable list that recenters the view on each residue '
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


# Template for the generated Coot script. It is Python-2 compatible (Coot 0.9) 
# and only relies on `set_rotation_centre`
_COOT_TEMPLATE = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Coot script generated by pdb_python_tools.
# Open it from Coot:  Calculate -> Run Script...   (or:  coot --script this_file.py)
# Requires Coot (0.9). Clicking a row recenters the view on the
# listed residue's CA/C1' (or, for find_contacts, the contact midpoint).
from __future__ import print_function

TITLE = {title!r}

# Each entry: (label, value_text, x, y, z)
MARKERS = [
{markers}
]

try:
    import gtk
except ImportError:
    gtk = None


def _make_cb(x, y, z):
    """Return a click handler that recenters the Coot view on (x, y, z)."""
    def _cb(*args):
        set_rotation_centre(x, y, z)
    return _cb


def show_coot_dialog():
    # Without a GUI (e.g. run outside Coot) just print the list.
    if gtk is None:
        for label, value_text, x, y, z in MARKERS:
            print("%-28s %12s  (%.3f, %.3f, %.3f)" % (label, value_text, x, y, z))
        return
    window = gtk.Window(gtk.WINDOW_TOPLEVEL)
    window.set_title(TITLE)
    window.set_default_size(380, 540)
    outer = gtk.VBox(False, 2)
    outer.set_border_width(4)
    heading = gtk.Label(TITLE)
    heading.set_alignment(0, 0.5)
    outer.pack_start(heading, False, False, 2)
    scrolled = gtk.ScrolledWindow()
    scrolled.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
    inner = gtk.VBox(False, 0)
    for label, value_text, x, y, z in MARKERS:
        button = gtk.Button("%s    %s" % (label, value_text))
        child = button.get_child()
        if child is not None:
            child.set_alignment(0, 0.5)
        button.connect("clicked", _make_cb(x, y, z))
        inner.pack_start(button, False, False, 0)
    scrolled.add_with_viewport(inner)
    outer.pack_start(scrolled, True, True, 0)
    close_button = gtk.Button("Close")
    close_button.connect("clicked", lambda w: window.destroy())
    outer.pack_start(close_button, False, False, 2)
    window.add(outer)
    window.show_all()


show_coot_dialog()
'''


def write_coot_script(markers, title, output, force=False, precision=2, full_precision=False):
    """
    Write a Coot (0.9) Python script.

    Running the script inside Coot opens a dialog listing `markers` in the given
    order; clicking a row recenters the view via `set_rotation_centre`.

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
    """
    if output is None:
        raise ValueError("write_coot_script requires an output path")
    if os.path.exists(output) and not force:
        raise FileExistsError(f"Refusing to overwrite existing file: {output} (use --force)")
    lines = []
    for label, value, unit, x, y, z in markers:
        value_text = _format_cell(value, precision, full_precision)
        if unit:
            value_text = value_text + " " + unit
        # %r keeps full-precision coordinates and safely escapes the strings
        lines.append("    (%r, %r, %r, %r, %r)," % (label, value_text, float(x), float(y), float(z)))
    content = _COOT_TEMPLATE.format(title=title, markers="\n".join(lines))
    with open(output, "w") as handle:
        handle.write(content)