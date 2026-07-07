#!/usr/bin/env python3
from pdb_python_tools import Atom
from pdb_python_tools import Residue
from pdb_python_tools import load_residues
from pdb_python_tools import compare_pdb_resi_xyz
import argparse
# Check for flags
parser = argparse.ArgumentParser(
                    prog='track_xyz.py',
                    description='Track xyz changes between two equivalent and aligned pdb/cif files',
                    epilog='Usage: pdb1/cif1 pdb2/cif2 -arguments')
parser.add_argument('pdb1', help='first coordinate file (pdb/cif)')
parser.add_argument('pdb2', help='second coordinate file (pdb/cif)')
parser.add_argument('-HET','--HETATM', action='store_true', dest='hetatm', help='include hetatms')
parser.add_argument('-hy','--hydrogens', action='store_true', dest='hydrogens', help='include hydrogens')
args = parser.parse_args()
pdb1 = args.pdb1
pdb2 = args.pdb2
hetatm = args.hetatm
hydrogens = args.hydrogens
#  Check format and parse with appropriate function
pdb1 = load_residues(pdb1, hetatm, hydrogens)
pdb2 = load_residues(pdb2, hetatm, hydrogens)

# Compare both pdbs 
compare_pdb_resi_xyz(pdb1,pdb2)
for resi in pdb1:
    resi.max_xyz = max(resi.atom_list, key=lambda x: x.xyz_change)
    resi.average_xyz = sum(atom.xyz_change for atom in resi.atom_list) / len(resi.atom_list)

pdb1.sort(key=lambda x: x.max_xyz.xyz_change, reverse=True)
# Print table
print("Chain\tResidue\tResidue name\tMax_Distance\tMax_atom\tAverage_distance\tCA/C1'_distance")
for resi in pdb1:
    if resi.max_xyz.xyz_change > 0.01:
        print("%s\t%s\t%s\t%s\t%s\t%s\t%s" % (resi.chainid, resi.seqid, resi.restyp, resi.max_xyz.xyz_change, resi.max_xyz.altid, resi.average_xyz, resi.CA.xyz_change))