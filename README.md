# pdb_python_tools
Small tools for analyzing pdb/cif files.  
The objective of this project is to set up python script that are useful during modeling or analyzing structures.  
  
# For usage of any of these tools, run them with -h  

# Requirements
Python3 and the modules argparse, math, numpy, mpi4py (optional)  

## atom_tracker.py usage
This script will find the maximum xyz coordinate change between two pdb/cif files. The output will be a tabulated table sorted by the residues with the largest coordinate change as well as the CA/C1' change.
For now, the files should be aligned first with some other program (e.g. ChimeraX).  
To include hetatm (ignored by default) add as an argument after the inputs:  
--HETATM or -HET  
To include hydrogens (ignored by default) add as an argument after the inputs:  
--hydrogens or -hy  
### Non-mpi
python atom_tracker.py *filename1 filename2* > output.txt
### Mpi (only if very slow with 1 core)
mpiexec -n *number of mpi processes* python atom_tracker_mpi.py *filename1 filename2* > output.txt  
If you encounter an error about host, try:  
mpiexec -n *number of mpi processes* -host localhost python atom_tracker_mpi.py *filename1 filename2* > output.txt  

## find_contacts.py usage
This script will find all atoms between chains within a certain distance (for a given chain). The output will be a tabulated table following the original pdb atom order.  
To include hetatm (ignored by default) add as an argument after the inputs:  
--HETATM or -HET  
To include hydrogens (ignored by default) add as an argument after the inputs:  
--hydrogens or -hy  
To only include possible polar contacts add as an argument after the inputs:  
--polar_only or -p  
To only get one contact per residue to another residue:  
--simple or -s  
### Non-mpi
python find_contacts.py *filename --distance distance --chain chainid(auth)* > output.txt
### Mpi (due to overhead, it can be slower in small structures)  
mpiexec -n *number of mpi processes* -host localhost python find_contacts.py *filename --distance distance --chain chainid(auth)* > output.txt

## CA_difference.py usage  
This script will find the maximum xyz CA/C1' coordinate change between two pdb/cif files (they do not need to be equivalent). The output will be a tabulated table sorted by the residues with the largest coordinate change.
For now, the files should be aligned first with some other program (e.g. ChimeraX)  
To include hetatm (ignored by default) add as an argument after the inputs:  
--HETATM or -HET  
To include hydrogens (ignored by default) add as an argument after the inputs:  
--hydrogens or -hy  

# Test files  
In test_files folder, you can try the test files. There are also outputs of what you would get with the command indicated in the .tsv files.  
This test is comparing two cryoEM ribosome structures frozen at different time-points. You can track the changes between both structures with the atom_tracker program (atom_tracker.py -HET 6ot3.cif 6ouo_aligned.cif -o test_atom_tracker.tsv).  
The other test finds all the contacts of the mRNA (chain 4) with other chains with find_contacts (../find_contacts.py --chain 4 6ouo_aligned.cif --distance 4.5 -o test_find_contacts.tsv).  
