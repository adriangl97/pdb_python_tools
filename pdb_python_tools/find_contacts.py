#!/usr/bin/env python3
"""
find_contacts.py - inter-chain atom contacts for a given chain.

Find all atoms of other chains within a cutoff distance of a chosen chain, using
a scipy cKDTree. By default one contact per residue pair is reported (the
shortest); use -a/--all to list every atom pair.
"""
from .core import Atom
from .core import Residue
from .core import load_residues
from .core import find_contacts_kdtree
from .core import add_output_args
from .core import write_table
from .core import write_coot_script
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog='find_contacts.py',
        description='Find possible contacts between chains for a given chain and within a given distance',
        epilog='Usage: pdb1/cif1 -arguments')
    parser.add_argument('pdb', help='coordinate file (pdb/cif)')
    parser.add_argument('-c', '--chain', help='chain id to analyze', required=True)
    parser.add_argument('-d', '--distance', help='distance to check', type=float, required=True)
    parser.add_argument('-HET', '--HETATM', action='store_true', dest='hetatm', help='include hetatms')
    parser.add_argument('-hy', '--hydrogens', action='store_true', dest='hydrogens', help='include hydrogens')
    parser.add_argument('-p', '--polar_only', action='store_true', dest='polar', help='check only polar')
    parser.add_argument('-a', '--all', action='store_true', dest='all',
                        help='all output: display all atoms involved and distances')
    add_output_args(parser)
    args = parser.parse_args()

    pdb = load_residues(args.pdb, args.hetatm, args.hydrogens)

    # Find the inter-chain contacts within that distance (scipy cKDTree)
    atom_pairs = find_contacts_kdtree(pdb, args.distance, args.chain, args.polar)

    def coot_marker(atom1, atom2, dist, with_atoms):
        # Center on the midpoint of the two contacting atoms
        mid = ((atom1.x + atom2.x) / 2.0, (atom1.y + atom2.y) / 2.0, (atom1.z + atom2.z) / 2.0)
        if with_atoms:
            label = "%s %s %s/%s - %s %s %s/%s" % (
                atom1.chainid, atom1.restyp, atom1.seqid, atom1.altid,
                atom2.chainid, atom2.restyp, atom2.seqid, atom2.altid)
        else:
            label = "%s %s - %s %s %s" % (
                atom1.restyp, atom1.seqid, atom2.chainid, atom2.restyp, atom2.seqid)
        return (label, dist, "Å", mid[0], mid[1], mid[2])

    if not args.all:
        # Collapse to one contact per residue pair, keeping the shortest distance.
        # Key on the full residue identity of both partners (chain + seqid).
        best = {}
        for atom1, atom2, dist in atom_pairs:
            key = (atom1.chainid, atom1.seqid, atom2.chainid, atom2.seqid)
            if key not in best or dist < best[key][2]:
                best[key] = [atom1, atom2, dist]
        header = ["Residue1", "Residue1 number", "Chain2", "Residue2", "Residue2 number", "Distance"]
        rows = [[atom1.restyp, atom1.seqid, atom2.chainid, atom2.restyp, atom2.seqid, dist]
                for atom1, atom2, dist in best.values()]
        markers = [coot_marker(atom1, atom2, dist, False)
                   for atom1, atom2, dist in best.values()] if args.coot else []
    else:
        header = ["Chain1", "Residue1", "Residue1 number", "Atom1",
                  "Chain2", "Residue2", "Residue2 number", "Atom2", "Distance"]
        rows = [[atom1.chainid, atom1.restyp, atom1.seqid, atom1.altid,
                 atom2.chainid, atom2.restyp, atom2.seqid, atom2.altid, dist]
                for atom1, atom2, dist in atom_pairs]
        markers = [coot_marker(atom1, atom2, dist, True)
                   for atom1, atom2, dist in atom_pairs] if args.coot else []

    try:
        write_table(header, rows, fmt=args.format, output=args.output, force=args.force,
                    precision=args.precision, full_precision=args.full_precision)
        if args.coot:
            write_coot_script(markers, "find_contacts: contacts within cutoff", args.coot,
                              force=args.force, precision=args.precision,
                              full_precision=args.full_precision)
    except FileExistsError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
