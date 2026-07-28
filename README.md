# pdb_python_tools

Small command-line tools for analyzing and comparing PDB/mmCIF structures, useful during modeling or structural analysis. Every tool reads `.pdb`/`.ent`, `.cif`/`.mmcif` and Gzipped files with a hand-written parser (no external structure library), writes a tab- or comma-separated table to stdout (or a file with `-o`), and has a `-h/--help` describing its flags.

## Requirements

- Python >= 3.8
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)

## Installation

Install the package (this also pulls in numpy and scipy):

```bash
pip install .
```

This puts four commands on your `PATH`:

- `pdb_python_tools.atom_tracker`
- `pdb_python_tools.find_contacts`
- `pdb_python_tools.CA_difference`
- `pdb_python_tools.nucleotide_conformation`

Each command name matches its module path, so without installing you can run the
same tool from a clone with `python -m <command>`, for example
`python -m pdb_python_tools.atom_tracker ...`.

Conda users can instead create an environment with the dependencies and then
`pip install .` into it:

```bash
conda env create -f environment.yml
conda activate pdb_python_tools
pip install .
```

## Tools

| Script | Purpose | Example |
|---|---|---|
| `pdb_python_tools.atom_tracker` | Per-residue/atom coordinate change between two **equivalent** structures | `pdb_python_tools.atom_tracker a.cif b.cif` |
| `pdb_python_tools.find_contacts` | Inter-chain atom contacts for a chosen chain within a cutoff | `pdb_python_tools.find_contacts a.cif -c 4 -d 4.5` |
| `pdb_python_tools.CA_difference` | Nearest CA/C1' distance in a second structure for every residue (structures **do not need** to be equivalent) | `pdb_python_tools.CA_difference a.cif b.cif` |
| `pdb_python_tools.nucleotide_conformation` | Glycosidic syn/anti conformation of RNA and DNA nucleotides, flags unlikely syn pyrimidines (C, U, DC, DT, DU) | `pdb_python_tools.nucleotide_conformation a.cif` |

### Shared flags

All analysis tools share the same interface:

- `-f/--format {tsv,csv}` — output format (default `tsv`).
- `-o/--output PATH` — write to a file instead of stdout; refuses to overwrite an existing file unless `--force` is given.
- `--precision N` — decimal places for distances or angles (default `2`); `--full-precision` or negative valued prints raw floats.
- `--coot PATH` — additionally write a [Coot](https://www2.mrc-lmb.cam.ac.uk/personal/pemsley/coot/) script (for 0.9, unsure if it works for Coot 1) to `PATH`. Open it in Coot (`Calculate → Run Script…`) to get a dialog listing the results in the same order as the table, each row showing the relevant number; clicking a row recenters the view on that residue's CA/C1' (or, for `pdb_python_tools.find_contacts`, on the contact midpoint). Refuses to overwrite `PATH` unless `--force` is given.
- `--version` — print the installed version and exit.

> **Alternate conformations:** every alternate conformation in a file is read and kept, and the conformations are tracked separately, but the global (e.g. Max distance and CA/C1' distance) are shared for now.

> **Alignment note:** `pdb_python_tools.atom_tracker` and `pdb_python_tools.CA_difference` compare coordinates directly, so the two inputs must be pre-aligned first (e.g. in ChimeraX). If you just ran a refinement and are comparing the input and output, no alignment is needed.

## Usage examples

The `test_files/` folder contains two aligned cryo-EM ribosome structures frozen at different time-points (`6ot3.cif` and `6ouo_aligned.cif`), which the examples below use.

### pdb_python_tools.atom_tracker

Track per-residue coordinate changes between two aligned structures:

```bash
pdb_python_tools.atom_tracker test_files/6ot3.cif test_files/6ouo_aligned.cif
```

```
Chain   Residue Residue name    Max_Distance    Max_atom        Average_distance        CA/C1'_distance
t       52      SER     6.67    OG      4.26    4.00
t       53      ARG     5.59    CB      4.21    4.09
t       48      LYS     4.18    CB      2.38    2.57
```

Showing just the top three above.

Add `-HET` to include HETATMs, `-hy` to include hydrogens, and `--min-change` to change the reporting threshold (default `0.01`). Residues without a CA/C1' atom in both structures show `NA` in the last column.

Only atoms present in both structures are measured. An atom with no counterpart in the second file is left out of both `Max_Distance` and `Average_distance`. A residue with no atom in common at all is omitted from the table.

### pdb_python_tools.find_contacts

Find contacts of the mRNA (chain `4`) with other chains within 4.5 Å:

```bash
pdb_python_tools.find_contacts test_files/6ouo_aligned.cif -c 4 -d 4.5
```

```
Residue1        Residue1 number Chain2  Residue2        Residue2 number Distance
U       13      p       VAL     129     4.15
U       13      2       G       693     2.99
U       13      l       GLY     82      3.13
```

Showing just the top three above

By default one (shortest) contact per residue pair is shown. Use `-a/--all` to list every atom pair, `-p/--polar_only` to restrict to N/O/P/S atoms, and `-HET`/`-hy` to include HETATMs/hydrogens.

### pdb_python_tools.CA_difference

For every residue of the first structure, report the nearest CA/C1' distance in the second (the two structures need not be equivalent or share numbering):

```bash
pdb_python_tools.CA_difference test_files/6ot3.cif test_files/6ouo_aligned.cif
```

```
Chain1  Residue1        Residue name1   Chain2  Residue2        Residue name2   CA/C1'_distance
6       79      PHE     A       252     GLN     9.23
6       78      PHE     A       252     GLN     9.06
t       52      SER     t       55      GLY     3.46
```

Showing just the top three above

### pdb_python_tools.nucleotide_conformation

Classify every standard RNA (A, C, G, U) and DNA (DA, DC, DG, DT, DU) nucleotide as *syn* or *anti* from the glycosidic torsion angle χ (measured O4'-C1'-N1-C2 for pyrimidines, O4'-C1'-N9-C4 for purines) and report the unlikely cases, pyrimidines, modeled in the *syn* conformation:

```bash
pdb_python_tools.nucleotide_conformation test_files/6ouo_aligned.cif
```

```
Chain   Residue Residue name    Chi     Conformation
1       102     U       48.91   syn
1       138     U       43.87   syn
1       139     U       44.55   syn
```

Showing just the top three above.

A nucleotide is called *syn* when χ is in `[-90°, +90°]` and *anti* otherwise. Three views are available (`-s` and `-a` are mutually exclusive):

- *default* — only syn pyrimidines (C, U, DC, DT, DU), the unlikely cases.
- `-s/--syn` — every syn nucleotide, purines (A, G, DA, DG) included, and no anti ones.
- `-a/--all` — every nucleotide with its χ angle and conformation.

`--precision` controls the decimal places on χ.

A nucleotide modelled in more than one conformation is measured once per conformation. An `Altloc` column is added to tell those rows apart

Because the ±90° cutoff is sharp, use `-m/--margin DEG` to catch χ values sitting close to the boundary: it adds a `Borderline` column (`yes`/`no`) and, in the default and `-s` views, also lists borderline-*anti* nucleotides that are within `DEG` of the boundary and could plausibly be syn. For example, `-m 5` flags a residue at χ = -88° as borderline and surfaces one at χ = -92° that the plain call would hide.

## Test files

`test_files/` also contains reference outputs (`.tsv`) produced by these commands:

```bash
pdb_python_tools.atom_tracker -HET test_files/6ot3.cif test_files/6ouo_aligned.cif -o test_atom_tracker.tsv
pdb_python_tools.find_contacts test_files/6ouo_aligned.cif -c 4 -d 4.5 -o test_find_contacts.tsv
pdb_python_tools.nucleotide_conformation test_files/6ouo_aligned.cif -o test_nucleotide_conformation.tsv
```
