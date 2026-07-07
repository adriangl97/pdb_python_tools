#!/usr/bin/env python3
from pdb_python_tools import Atom
from pdb_python_tools import Residue
from pdb_python_tools import load_residues
from pdb_python_tools import find_contacts_kdtree
import argparse
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
    print("Residue1\tResidue1 number\tChain2\tResidue2\tResidue2 number\tDistance")
    for atom1, atom2, dist in best.values():
        print("%s\t%s\t%s\t%s\t%s\t%s" % (atom1.restyp, atom1.seqid, atom2.chainid, atom2.restyp, atom2.seqid, dist))
else:
    # Print table
    print("Chain1\tResidue1\tResidue1 number\tAtom1\tChain2\tResidue2\tResidue2 number\tAtom2\tDistance")
    for atom1, atom2, dist in atom_pairs:
        print("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (atom1.chainid, atom1.restyp, atom1.seqid, atom1.altid, atom2.chainid, atom2.restyp, atom2.seqid, atom2.altid, dist))