#!/usr/bin/env python3
"""
CA_difference.py - nearest CA/C1' displacement between two structures.

For every residue in the first structure, find the nearest CA (protein) or C1'
(nucleic) atom in the second structure and report that distance. Unlike
atom_tracker.py the two structures do not need to be equivalent or share residue
numbering, but they should be pre-aligned first (e.g. in ChimeraX). The table is
sorted by CA/C1' distance, largest first.
"""
from .core import Atom
from .core import Residue
from .core import load_residues
from .core import find_nearest_ca
from .core import add_output_args
from .core import write_table
from .core import write_coot_script
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog='CA_difference.py',
        description="For every residue of the first structure, report the nearest "
                    "CA/C1' distance in the second structure"
                    " (structures do not need to be equivalent)",
        epilog='Usage: pdb1/cif1 pdb2/cif2 -arguments')
    parser.add_argument('pdb1', help='first coordinate file (pdb/cif)')
    parser.add_argument('pdb2', help='second coordinate file (pdb/cif)')
    parser.add_argument('-HET', '--HETATM', action='store_true', dest='hetatm', help='include hetatms')
    parser.add_argument('-hy', '--hydrogens', action='store_true', dest='hydrogens', help='include hydrogens')
    add_output_args(parser)
    args = parser.parse_args()

    # Parse both structures
    pdb1 = load_residues(args.pdb1, args.hetatm, args.hydrogens)
    pdb2 = load_residues(args.pdb2, args.hetatm, args.hydrogens)

    # Nearest CA/C1' in pdb2 for each residue of pdb1 (scipy cKDTree)
    results = find_nearest_ca(pdb1, pdb2)
    # Sort by distance, largest first
    results.sort(key=lambda t: t[2], reverse=True)

    header = ["Chain1", "Residue1", "Residue name1", "Chain2", "Residue2",
              "Residue name2", "CA/C1'_distance"]
    rows = [[r1.chainid, r1.seqid, r1.restyp, r2.chainid, r2.seqid, r2.restyp, dist]
            for r1, r2, dist in results]
    # Coot markers: center on the first structure's CA/C1'
    markers = [("%s %s %s -> %s %s %s" % (r1.chainid, r1.seqid, r1.restyp,
                                          r2.chainid, r2.seqid, r2.restyp),
                dist, "Å", r1.CA.x, r1.CA.y, r1.CA.z)
               for r1, r2, dist in results]
    try:
        write_table(header, rows, fmt=args.format, output=args.output, force=args.force,
                    precision=args.precision, full_precision=args.full_precision)
        if args.coot:
            write_coot_script(markers, "CA_difference: nearest CA/C1' distance", args.coot,
                              force=args.force, precision=args.precision,
                              full_precision=args.full_precision)
    except FileExistsError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
