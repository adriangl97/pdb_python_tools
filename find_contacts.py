#!/usr/bin/env python3
from pdb_python_tools import Atom
from pdb_python_tools import Residue
from pdb_python_tools import load_residues
from pdb_python_tools import find_contacts_resi
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
all = args.all

# Check format and parse with appropriate function
pdb = load_residues(pdb, hetatm, hydrogens)

# Find the contacts within that distance
atom_pairs = find_contacts_resi(pdb, distance, chain, polar)

# Check if all output is requested
if not all:
        n=0
        for i in atom_pairs:
            if n == 0:
                atom_pairs_simple = [i]
                n+=1
            else:
                count = -1
                for j in atom_pairs_simple:
                    check = False
                    count += 1
                    if i[0].seqid == j[0].seqid:
                        if i[1].seqid == j[1].seqid:
                            check = True
                            if i[2] < j[2]:
                                atom_pairs_simple[count][2] = i[2]
                                break
                if not check:
                    atom_pairs_simple.append(i)
        print("Residue1\tResidue1 number\tChain2\tResidue2\tResidue2 number\tDistance")
        for i in atom_pairs_simple:
            print("%s\t%s\t%s\t%s\t%s\t%s" % (i[0].restyp,i[0].seqid,i[1].chainid,i[1].restyp,i[1].seqid,i[2]))
else:
    # Print table
    print("Chain1\tResidue1\tResidue1 number\tAtom1\tChain2\tResidue2\tResidue2 number\tAtom2\tDistance")
    for i in atom_pairs:
        print("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (i[0].chainid, i[0].restyp, i[0].seqid, i[0].altid, i[1].chainid, i[1].restyp, i[1].seqid, i[1].altid, i[2]))