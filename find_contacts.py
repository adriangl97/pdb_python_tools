#!/usr/bin/env python3
from pdb_python_tools import Atom
from pdb_python_tools import Residue
from pdb_python_tools import load_residues
from pdb_python_tools import find_contacts_kdtree
from pdb_python_tools import add_output_args
from pdb_python_tools import write_table
import argparse
import sys
import numpy as np

# Check for flags
parser = argparse.ArgumentParser(
                    prog='find_contacts.py',
                    description='Find possible contacts between chains for a given chain and within a given distance',
                    epilog='Usage: pdb1/cif1 -arguments')
parser.add_argument('pdb', help='coordinate file (pdb/cif)')
parser.add_argument('-c','--chain', help='chain id to analyze', required=True)
parser.add_argument('-d','--distance', help='distance to check',type=float, required=True)
parser.add_argument('-HET','--HETATM', action='store_true', dest='hetatm', help='include hetatms')
parser.add_argument('-hy','--hydrogens', action='store_true', dest='hydrogens', help='include hydrogens')
parser.add_argument('-p','--polar_only', action='store_true', dest='polar', help='check only polar')
parser.add_argument('-a','--all', action='store_true', dest='all', help='all output: display all atoms involved and distances')
add_output_args(parser)
args = parser.parse_args()
pdb = args.pdb
chain = args.chain
distance = args.distance
hetatm = args.hetatm
hydrogens = args.hydrogens
polar = args.polar
show_all = args.all

# Check format and parse with appropriate function
pdb = load_residues(pdb, hetatm, hydrogens)

# Find the inter-chain contacts within that distance (scipy cKDTree)
atom_pairs = find_contacts_kdtree(pdb, distance, chain, polar)

# Check if all output is requested
if not show_all:
    # Collapse to one contact per residue pair, keeping the shortest distance
    # Key on the full residue identity of both partners (chain + seqid)
    best = {}
    for atom1, atom2, dist in atom_pairs:
        key = (atom1.chainid, atom1.seqid, atom2.chainid, atom2.seqid)
        if key not in best or dist < best[key][2]:
            best[key] = [atom1, atom2, dist]
    header = ["Residue1", "Residue1 number", "Chain2", "Residue2", "Residue2 number", "Distance"]
    rows = [[atom1.restyp, atom1.seqid, atom2.chainid, atom2.restyp, atom2.seqid, dist]
            for atom1, atom2, dist in best.values()]
else:
    header = ["Chain1", "Residue1", "Residue1 number", "Atom1",
              "Chain2", "Residue2", "Residue2 number", "Atom2", "Distance"]
    rows = [[atom1.chainid, atom1.restyp, atom1.seqid, atom1.altid,
             atom2.chainid, atom2.restyp, atom2.seqid, atom2.altid, dist]
            for atom1, atom2, dist in atom_pairs]

try:
    write_table(header, rows, fmt=args.format, output=args.output, force=args.force,
                precision=args.precision, full_precision=args.full_precision)
except FileExistsError as e:
    sys.exit(str(e))