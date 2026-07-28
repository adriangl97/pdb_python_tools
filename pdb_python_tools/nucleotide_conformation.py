#!/usr/bin/env python3
"""
nucleotide_conformation.py: syn/anti glycosidic conformation of nucleotides.

Compute the glycosidic torsion angle chi for every standard RNA or DNA
nucleotide and classify it as syn or anti (syn when chi is in [-90, +90]
degrees). chi is measured O4'-C1'-N1-C2 for pyrimidines (C, U, DC, DT, DU) and
O4'-C1'-N9-C4 for purines (A, G, DA, DG).

By default only pyrimidines in the syn conformation are reported, since a syn
pyrimidine is unusual and often points to a modeling error.
Use -s/--syn to list every syn nucleotide, purines included, and no anti ones
use -a/--all to list every nucleotide with its chi angle and conformation.
"""
from .core import Atom
from .core import Residue
from .core import load_residues_or_exit
from .core import classify_nucleotide_conformation
from .core import add_output_args
from .core import add_version_arg
from .core import write_table
from .core import write_coot_script
from .core import _PYRIMIDINES
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog='nucleotide_conformation.py',
        description='Classify RNA and DNA nucleotides as syn or anti from the '
                    'glycosidic torsion chi and flag unlikely syn pyrimidines '
                    '(C, U, DC, DT, DU)',
        epilog='Usage: pdb/cif -arguments')
    parser.add_argument('pdb', help='coordinate file (pdb/cif)')
    # The three views are alternatives: default (syn pyrimidines), -s, -a
    view = parser.add_mutually_exclusive_group()
    view.add_argument('-a', '--all', action='store_true', dest='all',
                      help='report every RNA/DNA nucleotide with its chi angle and '
                           'conformation (default: only syn pyrimidines)')
    view.add_argument('-s', '--syn', action='store_true', dest='syn',
                      help='report every syn nucleotide, purines (A, G, DA, DG) '
                           'included, and no anti ones')
    parser.add_argument('-m', '--margin', type=float, default=0.0,
                        help='degrees around the +/-90 syn/anti boundary to treat '
                             'as borderline; adds a Borderline column and, in the '
                             'default view, also lists borderline-anti pyrimidines '
                             '(default: 0, off)')
    add_version_arg(parser)
    add_output_args(parser)
    args = parser.parse_args()

    pdb = load_residues_or_exit(args.pdb, False, False)

    # chi + syn/anti call for every standard RNA/DNA nucleotide
    results = classify_nucleotide_conformation(pdb)

    use_margin = args.margin > 0

    def is_borderline(chi):
        # Distance to the nearest syn/anti boundary at +/-90 degrees
        return use_margin and min(abs(chi - 90), abs(chi + 90)) <= args.margin

    if args.all:
        selected = list(results)
    elif args.syn:
        # Every syn nucleotide regardless of base, plus borderline-anti ones
        selected = [r for r in results
                    if r[2] == "syn" or is_borderline(r[1])]
    else:
        # Unlikely cases: syn pyrimidines, plus borderline-anti pyrimidines that
        # sit close enough to the boundary to plausibly be syn
        selected = [r for r in results
                    if r[0].restyp in _PYRIMIDINES and (r[2] == "syn" or is_borderline(r[1]))]

    # A nucleotide modelled in alternate conformations gives one row per
    # conformation, so the id is shown to tell those rows apart. The column is
    # omitted entirely when nothing in the table has an alternate conformation.
    show_altloc = any(alt for _, _, _, alt in selected)

    def build_row(resi, chi, conf, alt):
        row = [resi.chainid, resi.seqid, resi.restyp, chi, conf]
        if show_altloc:
            row.append(alt or ".")
        if use_margin:
            row.append("yes" if is_borderline(chi) else "no")
        return row

    rows = [build_row(*entry) for entry in selected]
    # Coot markers: center on the C1' (recorded as the residue's CA atom)
    markers = [("%s %s %s%s (%s)" % (resi.chainid, resi.seqid, resi.restyp,
                                     " alt " + alt if alt else "", conf),
                chi, "°", resi.CA.x, resi.CA.y, resi.CA.z)
               for resi, chi, conf, alt in selected] if args.coot else []

    header = ["Chain", "Residue", "Residue name", "Chi", "Conformation"]
    if show_altloc:
        header.append("Altloc")
    if use_margin:
        header.append("Borderline")

    try:
        write_table(header, rows, fmt=args.format, output=args.output, force=args.force,
                    precision=args.precision, full_precision=args.full_precision)
        if args.coot:
            write_coot_script(markers, "nucleotide_conformation: glycosidic chi", args.coot,
                              force=args.force, precision=args.precision,
                              full_precision=args.full_precision)
    except FileExistsError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
