#!/usr/bin/env python3
"""
atom_tracker.py - per-residue/atom coordinate changes between two aligned structures.

Given two equivalent, pre-aligned pdb/cif files, report for each residue the
largest per-atom displacement, the average atom displacement and the CA/C1'
displacement. The table is sorted by the largest displacement first. Inputs must
be aligned beforehand (e.g. in ChimeraX) if they do not come from the same refinement.
"""
from pdb_python_tools import Atom
from pdb_python_tools import Residue
from pdb_python_tools import load_residues
from pdb_python_tools import compare_pdb_resi_xyz
from pdb_python_tools import add_output_args
from pdb_python_tools import write_table
from pdb_python_tools import write_coot_script
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog='atom_tracker.py',
        description='Track xyz changes between two equivalent and aligned pdb/cif files',
        epilog='Usage: pdb1/cif1 pdb2/cif2 -arguments')
    parser.add_argument('pdb1', help='first coordinate file (pdb/cif)')
    parser.add_argument('pdb2', help='second coordinate file (pdb/cif)')
    parser.add_argument('-HET', '--HETATM', action='store_true', dest='hetatm', help='include hetatms')
    parser.add_argument('-hy', '--hydrogens', action='store_true', dest='hydrogens', help='include hydrogens')
    parser.add_argument('--min-change', type=float, default=0.01, dest='min_change',
                        help='only report residues whose maximum displacement exceeds this value (default: 0.01)')
    add_output_args(parser)
    args = parser.parse_args()

    # Parse both structures with the appropriate parser
    pdb1 = load_residues(args.pdb1, args.hetatm, args.hydrogens)
    pdb2 = load_residues(args.pdb2, args.hetatm, args.hydrogens)

    # Compare both structures
    compare_pdb_resi_xyz(pdb1, pdb2)
    for resi in pdb1:
        resi.max_xyz = max(resi.atom_list, key=lambda x: x.xyz_change)
        resi.average_xyz = sum(atom.xyz_change for atom in resi.atom_list) / len(resi.atom_list)

    pdb1.sort(key=lambda x: x.max_xyz.xyz_change, reverse=True)
    # Build the table, keeping only residues that moved more than --min-change
    header = ["Chain", "Residue", "Residue name", "Max_Distance", "Max_atom",
              "Average_distance", "CA/C1'_distance"]
    rows = []
    markers = []
    for resi in pdb1:
        if resi.max_xyz.xyz_change > args.min_change:
            # Blank/NA when the residue has no real CA/C1' atom in both structures
            has_ca = resi.CA.altid in ("CA", "C1'")
            ca = resi.CA.xyz_change if has_ca else "NA"
            rows.append([resi.chainid, resi.seqid, resi.restyp, resi.max_xyz.xyz_change,
                         resi.max_xyz.altid, resi.average_xyz, ca])
            if args.coot:
                # Center on the CA/C1' when present, else the largest-moving atom
                center = resi.CA if has_ca else resi.max_xyz
                label = "%s %s %s" % (resi.chainid, resi.seqid, resi.restyp)
                markers.append((label, resi.max_xyz.xyz_change, "Å",
                                center.x, center.y, center.z))

    try:
        write_table(header, rows, fmt=args.format, output=args.output, force=args.force,
                    precision=args.precision, full_precision=args.full_precision)
        if args.coot:
            write_coot_script(markers, "atom_tracker: max displacement", args.coot,
                              force=args.force, precision=args.precision,
                              full_precision=args.full_precision)
    except FileExistsError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
