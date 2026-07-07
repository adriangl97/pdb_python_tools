#!/usr/bin/env python3
import math
import numpy as np
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
    xyz_change : movement compared to another pdb
    
    """   
    def __init__(self, atomid, element, altid, restyp, chainid, seqid, x, y, z, occ, biso, xyz_change):
        self.atomid = atomid
        self.element = element
        self.altid = altid
        self.restyp = restyp
        self.chainid = chainid
        self.seqid = seqid
        self.x = x
        self.y = y
        self.z = z
        self.occ = occ
        self.biso = biso
        self.xyz_change = xyz_change

class Residue:
    """
    Define residue class.

    Attributes
    ----------
    chainid : chain id
    seqid : residue number
    restyp : residue type
    atom_list : list of Atom objects belonging to that Residue
    max_xyz : maximum xyz change within the residue
    average_xyz : average xyz change within the residue
    CA : Calpha/C1' Atom object
    """
    def __init__(self, chainid, seqid, restyp, atom_list, max_xyz, average_xyz, CA):
        self.chainid = chainid
        self.seqid = seqid
        self.restyp = restyp
        self.atom_list = atom_list
        self.max_xyz = max_xyz
        self.average_xyz = average_xyz
        self.CA = CA

def _safe_float(value):
    """Convert a coordinate/occupancy/B-factor token to float, defaulting to 0.0 when blank."""
    value = value.strip()
    if value == "":
        return 0.0
    return float(value)


def _dequote(token):
    """
    Strip surrounding double quotes from an mmCIF token.

    mmCIF writes atom names that contain a prime (e.g. C1') as double-quoted
    tokens ("C1'").
    """
    if len(token) >= 2 and token[0] == '"' and token[-1] == '"':
        return token[1:-1]
    return token


def _element_from_pdb(line, atom_name):
    """
    Return the element symbol for a PDB ATOM/HETATM line.

    Reads columns 77-78 first (the wwPDB element field); if that field is blank
    it falls back to the first alphabetic character of the atom name
    """
    element = line[76:78].strip()
    if element:
        return element
    for char in atom_name:
        if char.isalpha():
            return char.upper()
    return ""


def _is_hydrogen(element):
    """Hydrogen (or deuterium) detection based on the element symbol."""
    return element.upper() in ("H", "D")


def _res_key(atom):
    """Residue identity: chain id plus seq id (seq id  carries any insertion code)"""
    return (atom.chainid, atom.seqid)


def _euclid(a, b):
    """Euclidean distance between two Atom objects."""
    return math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))


def _has_ca(resi):
    """True when the residue has a real CA/C1' atom"""
    return resi.CA.altid == "CA" or resi.CA.altid == "C1'"


def _add_atom(residues, atom, hydrogens):
    """
    Append an Atom to the growing list of Residues, starting a new Residue when
    the chain/seqid (including insertion code) changes. Hydrogens are skipped
    unless requested. Also records the CA/C1' atom on its residue.
    """
    # Ignore hydrogens by default; include them only when requested
    if _is_hydrogen(atom.element) and not hydrogens:
        return
    if not residues or _res_key(residues[-1].atom_list[0]) != _res_key(atom):
        residues.append(Residue(atom.chainid, atom.seqid, atom.restyp, [atom], 0, 0,
                                Atom(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))
    else:
        residues[-1].atom_list.append(atom)
    # Record the CA (protein) / C1' (nucleic) atom for the residue as a separate Atom object (kept distinct from the copy in atom_list)
    if atom.altid == "CA" or atom.altid == "C1'":
        residues[-1].CA = Atom(atom.atomid, atom.element, atom.altid, atom.restyp,
                               atom.chainid, atom.seqid, atom.x, atom.y, atom.z,
                               atom.occ, atom.biso, 0)


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
    """
    residues = []
    with open(file, 'r') as fh:
        for line in fh:
            record = line[0:6].strip()
            # Only ATOM (always) and HETATM (when requested) carry atom info
            if record == "ATOM" or (hetatm and record == "HETATM"):
                # Fixed-column fields per the PDB format specification
                atom_name = line[12:16].strip()
                resname = line[17:20].strip()
                chainid = line[21:22].strip()
                # Residue key = residue sequence number + insertion code
                seqid = line[22:26].strip() + line[26:27].strip()
                element = _element_from_pdb(line, atom_name)
                atom = Atom(line[6:11].strip(), element, atom_name, resname, chainid, seqid,
                            _safe_float(line[30:38]), _safe_float(line[38:46]), _safe_float(line[46:54]),
                            _safe_float(line[54:60]), _safe_float(line[60:66]), 0)
                _add_atom(residues, atom, hydrogens)
    return residues


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
    """
    residues = []
    chainid = 999
    seqid = 999
    icode = 999
    # Track column order within the _atom_site loop
    count = -1
    with open(file, 'r') as fh:
        for line in fh:
            count += 1
            # Reset the column counter at the start of a loop_
            if "loop_" in line:
                count = -1
            # Learn the column order for each attribute from the loop header
            if "esd" not in line[-4:].lower():
                if "_atom_site.id" in line.lower():
                    atomid = count
                if "_atom_site.type_symbol" in line.lower():
                    element = count
                if "_atom_site.label_atom_id" in line.lower():
                    altid = count
                if "_atom_site.label_comp_id" in line.lower():
                    restyp = count
                if "_atom_site.auth_asym_id" in line.lower():
                    chainid = count
                if chainid == 999 and "_atom_site.label_asym_id" in line.lower():
                    chainid = count
                if "_atom_site.auth_seq_id" in line.lower():
                    seqid = count
                if seqid == 999 and "_atom_site.label_seq_id" in line.lower():
                    seqid = count
                if "_atom_site.pdbx_pdb_ins_code" in line.lower():
                    icode = count
                if "_atom_site.cartn_x" in line.lower():
                    x = count
                if "_atom_site.cartn_y" in line.lower():
                    y = count
                if "_atom_site.cartn_z" in line.lower():
                    z = count
                if "_atom_site.occupancy" in line.lower():
                    occ = count
                if "_atom_site.b_iso" in line.lower():
                    biso = count
            # Parse ATOM (always) and HETATM (when requested) rows
            record = line[:10]
            if "ATOM" in record or (hetatm and "HETATM" in record):
                fields = line.split()
                # Dequote tokens that mmCIF may wrap in double quotes
                atom_name = _dequote(fields[altid])
                resname = _dequote(fields[restyp])
                chain = _dequote(fields[chainid])
                # Residue key = auth_seq_id + insertion code (blank for '.'/'?')
                res_seq = _dequote(fields[seqid])
                if icode != 999:
                    ins = _dequote(fields[icode])
                    if ins not in (".", "?"):
                        res_seq += ins
                atom = Atom(fields[atomid], fields[element], atom_name, resname, chain, res_seq,
                            float(fields[x]), float(fields[y]), float(fields[z]),
                            float(fields[occ]), float(fields[biso]), 0)
                _add_atom(residues, atom, hydrogens)
    return residues


def load_residues(path, hetatm, hydrogens):
    """
    Dispatch to the correct structure parser based on the file extension.

    """
    p = path.lower()
    if p.endswith(".pdb") or p.endswith(".ent"):
        return get_resi_from_pdb(path, hetatm, hydrogens)
    if p.endswith(".cif") or p.endswith(".mmcif"):
        return get_resi_from_cif(path, hetatm, hydrogens)
    raise ValueError(f"Unrecognized structure extension: {path}")

#Define function to compare atom distances and write into the atom attribute
def compare_pdb_xyz(pdb1, pdb2):
    """
    Compares two lists of Atoms (class)

    Inputs
    ------
    pdb1, pdb2 : List of Atoms (class)

    Returns
    -------
    Modifies self.xyz_change from pdb1 atoms (class) based on the
    x, y, z change between pdb1 and pdb2.

    """
    # Iterate through the first atom list
    for atom1 in pdb1:
        # Iterate through the second atom list
        for atom2 in pdb2:
            # Make sure it is the same atom being compared (same chain, seq number and atom name)
            if atom1.chainid == atom2.chainid:
                if atom1.seqid == atom2.seqid:
                    if atom1.altid == atom2.altid and isinstance(atom1.xyz_change, int):
                        # Get coordinates from each atom
                        x1, y1, z1 = atom1.x, atom1.y, atom1.z
                        x2, y2, z2 = atom2.x, atom2.y, atom2.z
                        # Calculate vector distance
                        xyz = (x1-x2)**2+(y1-y2)**2+(z1-z2)**2
                        xyz = math.sqrt(xyz)
                        # Write distance to attribute on the first list
                        atom1.xyz_change = xyz
                    if atom1.restyp == "TYR" or atom1.restyp == "PHE":
                            if "CE" in atom1.altid or "CD" in atom1.altid:
                                if "CE" in atom2.altid or "CD" in atom2.altid:
                                    x1, y1, z1 = atom1.x, atom1.y, atom1.z
                                    x2, y2, z2 = atom2.x, atom2.y, atom2.z
                                    xyz = (x1-x2)**2+(y1-y2)**2+(z1-z2)**2
                                    xyz = math.sqrt(xyz)
                                    if xyz < atom1.xyz_change or isinstance(atom1.xyz_change, int):
                                        atom1.xyz_change = xyz

def find_max_res(pdb):
    """
    Finds the atom with most xyz_change and returns a list of these atoms sorted by the xyz_change.

    Inputs
    ------
    pdb : List of Atoms (class)

    Returns
    -------
    resi_list_max : list of Atoms from pdb with the largest xyz_change per residue.

    """
    # Prepare dummy variables
    atom_p = Atom(0,0,0,0,0,0,0,0,0,0,0,0)
    resi = []
    resi_list_atom = []
    resi_list_max = []
    # Iterate through the atoms in the pdb list
    for atom in pdb:
        # Check that the atom belongs to the same residue as the one before
        if atom.seqid == atom_p.seqid:
            if atom.chainid == atom_p.chainid:
                #Build the residue list with that atom
                resi += [atom]
        # Once it finishes going though all atoms of that residue (it considers that they are correctly ordered)
        else:
            # Double check that it is not empty
            if resi != []:
                # Build a list with the residues
                resi_list_atom += [resi]
                # Start a new residue
                resi = []
        # Reset so it compares with the previous
        atom_p = atom
    # add last residue:
    resi_list_atom += [resi]
    # Order the residue list by distance per residue and keep the max
    for residue in resi_list_atom:
        residue.sort(key=lambda x: x.xyz_change, reverse=True)
        if len(residue) > 0:
            resi_list_max += [residue[0]]
    return resi_list_max

# Symmetric/interchangeable atom-name substrings per residue type. For these
# residues the named atoms are (near-)equivalent, so the residue's displacement
# for such an atom is taken as the minimum distance over the swappable partners.
_SYMMETRIC = {
    "TYR": ("CE", "CD"),
    "PHE": ("CE", "CD"),
    "GLU": ("OE1", "OE2"),
    "ASP": ("OD1", "OD2"),
    "ARG": ("NH1", "NH2"),
    "LEU": ("D1", "D2"),
    "VAL": ("CG1", "CG2"),
}


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
        # have a real CA/C1' atom. Otherwise resi1.CA stays the dummy placeholder.
        if _has_ca(resi1) and _has_ca(resi2):
            resi1.CA.xyz_change = _euclid(resi1.CA, resi2.CA)
        # Which (if any) atom names are interchangeable for this residue type
        symmetric = _SYMMETRIC.get(resi1.restyp)
        for atom1 in resi1.atom_list:
            for atom2 in resi2.atom_list:
                # Same atom by name: record its displacement once
                if atom1.altid == atom2.altid and isinstance(atom1.xyz_change, int):
                    atom1.xyz_change = _euclid(atom1, atom2)
                # Interchangeable atoms: keep the minimum distance over the pair
                if symmetric and (symmetric[0] in atom1.altid or symmetric[1] in atom1.altid):
                    xyz = _euclid(atom1, atom2)
                    if isinstance(atom1.xyz_change, int) or xyz < atom1.xyz_change:
                        atom1.xyz_change = xyz


def find_contacts(pdb, distance, chain, polar):
    """
    Finds all atoms from different chains within a specific distance and returns a list of pairs.
    """
    atom_pairs = []
    # Iterate through atoms
    for atom1 in pdb:
        # Only consider atoms for a given chain
        if atom1.chainid == chain:
            if polar == True:
                if atom1.element == "O" or atom1.element == "N" or atom1.element == "P" or atom1.element == "S":
                    for atom2 in pdb:
                        if atom1.chainid != atom2.chainid:
                            if atom2.element == "O" or atom2.element == "N" or atom2.element == "P" or atom2.element == "S":
                                #Get coordinates from each atom
                                x1, y1, z1 = atom1.x, atom1.y, atom1.z
                                x2, y2, z2 = atom2.x, atom2.y, atom2.z
                                #Calculate vector distance
                                xyz = (x1-x2)**2+(y1-y2)**2+(z1-z2)**2
                                xyz = math.sqrt(xyz)
                                if xyz <= distance:
                                    atom_pairs += [[atom1,atom2,xyz]]
        # Compare with all other atoms. Slowest part - possibility for improvement
            else:
                for atom2 in pdb:
                    # Consider only atoms from different chain for the comparison
                    if atom1.chainid != atom2.chainid:
                        # Get coordinates from each atom
                        x1, y1, z1 = atom1.x, atom1.y, atom1.z
                        x2, y2, z2 = atom2.x, atom2.y, atom2.z
                        # Calculate vector distance
                        xyz = (x1-x2)**2+(y1-y2)**2+(z1-z2)**2
                        xyz = math.sqrt(xyz)
                        if xyz <= distance:
                            # Ignore duplicate (the opposite pair, reversed)
                            if [atom2,atom1,xyz] not in atom_pairs:
                                atom_pairs += [[atom1,atom2,xyz]]
    return(atom_pairs)

def find_contacts_resi(pdb, distance, chain, polar):
    """
    Finds all atoms from different chains within a specific distance and returns a list of pairs.
    """
    atom_pairs = []
    # Iterate through atoms
    for resi1 in pdb:
        # Only consider atoms for a given chain
        if resi1.chainid == chain:
            if polar == True:
                for resi2 in pdb:
                    # Consider only atoms from different chain for the comparison
                    if resi1.chainid != resi2.chainid:
                        #iterate over residue
                        for atom1 in resi1.atom_list:
                            if atom1.element == "O" or atom1.element == "N" or atom1.element == "P" or atom1.element == "S":
                                for atom2 in resi2.atom_list:
                                    if atom2.element == "O" or atom2.element == "N" or atom2.element == "P" or atom2.element == "S":
                                        # Get coordinates from each atom
                                        x1, y1, z1 = atom1.x, atom1.y, atom1.z
                                        x2, y2, z2 = atom2.x, atom2.y, atom2.z
                                        # Calculate vector distance
                                        xyz = (x1-x2)**2+(y1-y2)**2+(z1-z2)**2
                                        xyz = math.sqrt(xyz)
                                        if xyz <= distance:
                                            # Ignore duplicate (the opposite pair, reversed)
                                            if [atom2,atom1,xyz] not in atom_pairs:
                                                atom_pairs += [[atom1,atom2,xyz]]
        # Compare with all other atoms. Slowest part - possibility for improvement
            else:
                for resi2 in pdb:
                    # Consider only atoms from different chain for the comparison
                    if resi1.chainid != resi2.chainid:
                        #iterate over residue
                        for atom1 in resi1.atom_list:
                            for atom2 in resi2.atom_list:
                                # Get coordinates from each atom
                                x1, y1, z1 = atom1.x, atom1.y, atom1.z
                                x2, y2, z2 = atom2.x, atom2.y, atom2.z
                                # Calculate vector distance
                                xyz = (x1-x2)**2+(y1-y2)**2+(z1-z2)**2
                                xyz = math.sqrt(xyz)
                                if xyz <= distance:
                                    # Ignore duplicate (the opposite pair, reversed)
                                    if [atom2,atom1,xyz] not in atom_pairs:
                                        atom_pairs += [[atom1,atom2,xyz]]
    return(atom_pairs)