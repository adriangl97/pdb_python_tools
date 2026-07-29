#!/usr/bin/env python3
"""
nucleotide_conformation.py: syn/anti glycosidic conformation of nucleotides.

Compute the glycosidic torsion angle chi for every RNA or DNA nucleotide and
classify it as syn or anti (syn when chi is in [-90, +90] degrees). chi is
measured O4'-C1'-N1-C2 for pyrimidines (C, U, DC, DT, DU) and O4'-C1'-N9-C4 for
purines (A, G, DA, DG).

Modified nucleotides written as HETATM are included as long as they keep the
standard atom names. Which atoms chi is measured on follows from the atom that
is bonded to the C1': N9 for a purine, N1 for a pyrimidine, and C5 for a C-glycoside
such as pseudouridine, whose chi therefore runs O4'-C1'-C5-C4.

By default only pyrimidines in the syn conformation are reported, since a syn
pyrimidine is unusual and often points to a modeling error.
Use -s/--syn to list every syn nucleotide, purines included, and no anti ones
use -a/--all to list every nucleotide with its chi angle and conformation.

How many nucleotides came out syn is reported, as comment lines ('#')
above the table. The counts follow the view: the default view counts syn
pyrimidines out of every pyrimidine, while -s and -a also count purines and all
nucleotides.
"""
from .core import load_residues_or_exit
from .core import classify_nucleotide_conformation
from .core import add_output_args
from .core import add_version_arg
from .core import write_table
from .core import write_coot_script
from .core import is_pyrimidine
from .core import count_nucleotide_conformations
from .core import format_percentage
from .core import CONFORMATION_GROUPS
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog='nucleotide_conformation.py',
        description='Classify RNA and DNA nucleotides, including modified ones, '
                    'as syn or anti from the glycosidic torsion chi and flag '
                    'unlikely syn pyrimidines (C, U, DC, DT, DU)',
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

    # HETATM records are read too, so modified nucleotides are covered
    pdb = load_residues_or_exit(args.pdb, True, False)

    # chi + syn/anti call for every RNA/DNA nucleotide
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
        # Syn pyrimidines, plus borderline-anti pyrimidines that
        # sit close enough to the boundary
        selected = [r for r in results
                    if is_pyrimidine(r[0]) and (r[2] == "syn" or is_borderline(r[1]))]

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

    # The default view only deals with pyrimidines, so only those are counted;
    # -s and -a cover every base, so purines and the overall figure are
    # reported too
    groups = CONFORMATION_GROUPS if (args.all or args.syn) else ("pyrimidines",)
    stats = stats_comments(results, groups, is_borderline if use_margin else None,
                           args.precision, args.full_precision)

    try:
        write_table(header, rows, fmt=args.format, output=args.output, force=args.force,
                    precision=args.precision, full_precision=args.full_precision,
                    comments=stats)
        if args.coot:
            write_coot_script(markers, "nucleotide_conformation: glycosidic chi", args.coot,
                              force=args.force, precision=args.precision,
                              full_precision=args.full_precision)
    except FileExistsError as e:
        sys.exit(str(e))


def stats_comments(results, groups, is_borderline, precision, full_precision):
    """
    The syn counts for `groups`, as the comment lines written above the table.

    Every measured nucleotide is counted, including those the chosen view does
    not list, so the counts describe the structure and not the table. A
    borderline count is added per group when -m/--margin is in use
    """
    counts = count_nucleotide_conformations(results, is_borderline)

    def line(label, count, total):
        return "%s: %s" % (label, format_percentage(count, total, precision,
                                                    full_precision))

    lines = []
    for group in groups:
        syn, borderline, total = counts[group]
        lines.append(line("Syn " + group, syn, total))
        if is_borderline is not None:
            lines.append(line("Borderline " + group, borderline, total))
    return lines


if __name__ == "__main__":
    main()
