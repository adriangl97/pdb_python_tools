# pdb_python_tools

Small command-line tools for analyzing and comparing PDB/mmCIF structures, useful during modeling or structural analysis. Every tool reads `.pdb`/`.ent` and `.cif`/`.mmcif` files with a hand-written parser (no external structure library), writes a tab- or comma-separated table to stdout (or a file with `-o`), and has a `-h/--help` describing its flags.

## Requirements

- Python >= 3.8
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)

Install the dependencies with either:

```bash
pip install -r requirements.txt
# or, for conda users:
conda env create -f environment.yml
```

## Tools

| Script | Purpose | Example |
|---|---|---|
| `atom_tracker.py` | Per-residue/atom coordinate change between two **equivalent** structures | `atom_tracker.py a.cif b.cif` |
| `find_contacts.py` | Inter-chain atom contacts for a chosen chain within a cutoff | `find_contacts.py a.cif -c 4 -d 4.5` |
| `CA_difference.py` | Nearest CA/C1' distance in a second structure for every residue (structures **do not need** to be equivalent) | `CA_difference.py a.cif b.cif` |

### Shared output flags

All analysis tools share the same output interface:

- `-f/--format {tsv,csv}` — output format (default `tsv`).
- `-o/--output PATH` — write to a file instead of stdout; refuses to overwrite an existing file unless `--force` is given.
- `--precision N` — decimal places for distances (default `2`); `--full-precision` or negative valued prints raw floats.

> **Alignment note:** `atom_tracker.py` and `CA_difference.py` compare coordinates directly, so the two inputs must be pre-aligned first (e.g. in ChimeraX). If you just ran a refinement and are comparing the input and output, no alignment is needed.

## Usage examples

The `test_files/` folder contains two aligned cryo-EM ribosome structures frozen at different time-points (`6ot3.cif` and `6ouo_aligned.cif`), which the examples below use.

### atom_tracker.py

Track per-residue coordinate changes between two aligned structures:

```bash
atom_tracker.py test_files/6ot3.cif test_files/6ouo_aligned.cif
```

```
Chain   Residue Residue name    Max_Distance    Max_atom        Average_distance        CA/C1'_distance
t       52      SER     6.67    OG      4.26    4.00
t       53      ARG     5.59    CB      4.21    4.09
t       48      LYS     4.18    CB      2.38    2.57
```

Showing just the top three above.

Add `-HET` to include HETATMs, `-hy` to include hydrogens, and `--min-change` to change the reporting threshold (default `0.01`). Residues without a CA/C1' atom show `NA` in the last column.

### find_contacts.py

Find contacts of the mRNA (chain `4`) with other chains within 4.5 Å:

```bash
find_contacts.py test_files/6ouo_aligned.cif -c 4 -d 4.5
```

```
Residue1        Residue1 number Chain2  Residue2        Residue2 number Distance
U       13      p       VAL     129     4.15
U       13      2       G       693     2.99
U       13      l       GLY     82      3.13
```

Showing just the top three above

By default one (shortest) contact per residue pair is shown. Use `-a/--all` to list every atom pair, `-p/--polar_only` to restrict to N/O/P/S atoms, and `-HET`/`-hy` to include HETATMs/hydrogens.

### CA_difference.py

For every residue of the first structure, report the nearest CA/C1' distance in the second (the two structures need not be equivalent or share numbering):

```bash
CA_difference.py test_files/6ot3.cif test_files/6ouo_aligned.cif
```

```
Chain1  Residue1        Residue name1   Chain2  Residue2        Residue name2   CA/C1'_distance
6       79      PHE     A       252     GLN     9.23
6       78      PHE     A       252     GLN     9.06
t       52      SER     t       55      GLY     3.46
```

Showing just the top three above

## Test files

`test_files/` also contains reference outputs (`.tsv`) produced by these commands:

```bash
atom_tracker.py -HET test_files/6ot3.cif test_files/6ouo_aligned.cif -o test_atom_tracker.tsv
find_contacts.py test_files/6ouo_aligned.cif -c 4 -d 4.5 -o test_find_contacts.tsv
```
