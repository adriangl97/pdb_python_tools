#!/usr/bin/env python3
from pdb_python_tools import Atom
from pdb_python_tools import Residue
from pdb_python_tools import load_residues
from pdb_python_tools import compare_pdb_resi_xyz
from pdb_python_tools import add_output_args
from pdb_python_tools import write_table
import argparse
import sys
# Check for flags
parser = argparse.ArgumentParser(
                    prog='track_xyz.py',
                    description='Track xyz changes between two equivalent and aligned pdb/cif files',
                    epilog='Usage: pdb1/cif1 pdb2/cif2 -arguments')
parser.add_argument('pdb1', help='first coordinate file (pdb/cif)')
parser.add_argument('pdb2', help='second coordinate file (pdb/cif)')
parser.add_argument('-HET','--HETATM', action='store_true', dest='hetatm', help='include hetatms')
parser.add_argument('-hy','--hydrogens', action='store_true', dest='hydrogens', help='include hydrogens')
parser.add_argument('--min-change', type=float, default=0.01, dest='min_change',
                    help='only report residues whose maximum displacement exceeds this value (default: 0.01)')
add_output_args(parser)
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
# Build the table, keeping only residues that moved more than --min-change
header = ["Chain", "Residue", "Residue name", "Max_Distance", "Max_atom",
          "Average_distance", "CA/C1'_distance"]
rows = []
for resi in pdb1:
    if resi.max_xyz.xyz_change > args.min_change:
        # Blank/NA when the residue has no real CA/C1' atom in both structures
        ca = resi.CA.xyz_change if resi.CA.altid in ("CA", "C1'") else "NA"
        rows.append([resi.chainid, resi.seqid, resi.restyp, resi.max_xyz.xyz_change,
                     resi.max_xyz.altid, resi.average_xyz, ca])

try:
    write_table(header, rows, fmt=args.format, output=args.output, force=args.force,
                precision=args.precision, full_precision=args.full_precision)
except FileExistsError as e:
    sys.exit(str(e))
