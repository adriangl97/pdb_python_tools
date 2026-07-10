#!/usr/bin/env python3
"""
nucleotide_conformation.py - syn/anti glycosidic conformation of RNA nucleotides.

Compute the glycosidic torsion angle chi for every standard RNA nucleotide and
classify it as syn or anti (syn when chi is in [-90, +90] degrees). chi is
measured O4'-C1'-N1-C2 for pyrimidines (C, U) and O4'-C1'-N9-C4 for purines
(A, G).

By default only RNA pyrimidines (C, U) in the syn conformation are reported,
since a syn pyrimidine is unusual and often points to a modeling error. 
Use -s/--syn to list every syn nucleotide, purines (A, G) included, and no anti ones
use -a/--all to list every RNA nucleotide with its chi angle and conformation.
"""
from pdb_python_tools import Atom
from pdb_python_tools import Residue
from pdb_python_tools import load_residues
from pdb_python_tools import classify_rna_conformation
from pdb_python_tools import add_output_args
from pdb_python_tools import write_table
import argparse
import sys

# RNA pyrimidines - the residues flagged when found in the syn conformation
_PYRIMIDINES = {"C", "U"}


def main():
    parser = argparse.ArgumentParser(
        prog='nucleotide_conformation.py',
        description='Classify RNA nucleotides as syn or anti from the glycosidic '
                    'torsion chi and flag unlikely syn pyrimidines (C, U)',
        epilog='Usage: pdb/cif -arguments')
    parser.add_argument('pdb', help='coordinate file (pdb/cif)')
    # The three views are alternatives: default (syn pyrimidines), -s, -a
    view = parser.add_mutually_exclusive_group()
    view.add_argument('-a', '--all', action='store_true', dest='all',
                      help='report every RNA nucleotide with its chi angle and '
                           'conformation (default: only syn pyrimidines C/U)')
    view.add_argument('-s', '--syn', action='store_true', dest='syn',
                      help='report every syn nucleotide, purines (A, G) included, '
                           'and no anti ones')
    parser.add_argument('-m', '--margin', type=float, default=0.0,
                        help='degrees around the +/-90 syn/anti boundary to treat '
                             'as borderline; adds a Borderline column and, in the '
                             'default view, also lists borderline-anti pyrimidines '
                             '(default: 0, off)')
    add_output_args(parser)
    args = parser.parse_args()

    pdb = load_residues(args.pdb, False, False)

    # chi + syn/anti call for every standard RNA nucleotide
    results = classify_rna_conformation(pdb)

    use_margin = args.margin > 0

    def is_borderline(chi):
        # Distance to the nearest syn/anti boundary at +/-90 degrees
        return use_margin and min(abs(chi - 90), abs(chi + 90)) <= args.margin

    def build_row(resi, chi, conf):
        row = [resi.chainid, resi.seqid, resi.restyp, chi, conf]
        if use_margin:
            row.append("yes" if is_borderline(chi) else "no")
        return row

    if args.all:
        rows = [build_row(resi, chi, conf) for resi, chi, conf in results]
    elif args.syn:
        # Every syn nucleotide regardless of base, plus borderline-anti ones
        rows = [build_row(resi, chi, conf)
                for resi, chi, conf in results
                if conf == "syn" or is_borderline(chi)]
    else:
        # Unlikely cases: syn pyrimidines, plus borderline-anti pyrimidines that
        # sit close enough to the boundary to plausibly be syn
        rows = [build_row(resi, chi, conf)
                for resi, chi, conf in results
                if resi.restyp in _PYRIMIDINES and (conf == "syn" or is_borderline(chi))]

    header = ["Chain", "Residue", "Residue name", "Chi", "Conformation"]
    if use_margin:
        header.append("Borderline")

    try:
        write_table(header, rows, fmt=args.format, output=args.output, force=args.force,
                    precision=args.precision, full_precision=args.full_precision)
    except FileExistsError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
