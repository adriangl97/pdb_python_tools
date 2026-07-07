#!/usr/bin/env python3
from pdb_python_tools import Atom
from pdb_python_tools import load_residues
from pdb_python_tools import Residue
import argparse
import math
import numpy as np
from mpi4py import MPI
# Set up MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
# Check for flags
parser = argparse.ArgumentParser(
                    prog='track_xyz_mpi.py',
                    description='Track xyz changes between two equivalent and aligned pdb/cif files',
                    epilog='Usage: pdb1/cif1 pdb2/cif2 -arguments')
parser.add_argument('pdb1', help='first coordinate file (pdb/cif)')
parser.add_argument('pdb2', help='second coordinate file (pdb/cif)')
parser.add_argument('-HET','--HETATM', action='store_true', dest='hetatm', help='include hetatms')
args = parser.parse_args()
pdb1 = args.pdb1
pdb2 = args.pdb2
hetatm = args.hetatm
hydrogens = False

# Check format and parse with appropriate function
pdb2 = load_residues(pdb2, hetatm, hydrogens)
# Parse the other pdb and split in equal chunks for the mpi processes
if rank == 0:
    pdb1 = load_residues(pdb1, hetatm, hydrogens)
    df_pdb1 = np.array_split(pdb1,size)
    
else:
    df_pdb1 = None

def compare_resi_CA(df_pdb1,pdb2):
    """
    Compares two lists of Residues CA and returns the smallest difference

    Inputs
    ------
    pdb1, pdb2 : List of Residues (class)

    Returns
    -------
    Modifies self.xyz_change from pdb1 Atoms within the list in the Residue (class) based on the
    x, y, z change between pdb1 and pdb2.

    """
    # Set up MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # Scatter the lists
    sc_pdb1 = comm.scatter(df_pdb1, root=0)
    # Set up variable
    resi_pairs = []
    count = -1
    # Iterate through the part of the lists assigned to each rank
    for i in zip(sc_pdb1):
        for resi1 in i:
            for resi2 in pdb2:
                    # Make sure it is the same atom being compared
                    if resi1.CA.altid == resi2.CA.altid and not isinstance(resi1.CA.xyz_change, float):
                        #Get coordinates from each atom
                        x1, y1, z1 = resi1.CA.x, resi1.CA.y, resi1.CA.z
                        x2, y2, z2 = resi2.CA.x, resi2.CA.y, resi2.CA.z
                        #Calculate vector distance
                        xyz = (x1-x2)**2+(y1-y2)**2+(z1-z2)**2
                        xyz = math.sqrt(xyz)
                        #Write distance to attribute
                        resi1.CA.xyz_change = xyz
                        resi_pairs += [[resi1,resi2]]
                        count += 1
                    else:
                        # Check if the distance is the same or less
                        # Get coordinates from each atom
                        x1, y1, z1 = resi1.CA.x, resi1.CA.y, resi1.CA.z
                        x2, y2, z2 = resi2.CA.x, resi2.CA.y, resi2.CA.z
                        #Calculate vector distance
                        xyz = (x1-x2)**2+(y1-y2)**2+(z1-z2)**2
                        xyz = math.sqrt(xyz)
                        if xyz < resi1.CA.xyz_change:
                            resi1.CA.xyz_change = xyz
                            resi_pairs[count] = [resi1,resi2]
            
    # Gather results on rank 0
    resi_pairs = comm.gather(resi_pairs, root=0)
    if rank == 0:
            while [] in resi_pairs:
                    resi_pairs.remove([])
            # Clean list
            flat_resi_pairs = [item for sublist in resi_pairs for item in sublist]
            return(flat_resi_pairs)

# Compare both pdbs through mpi    
resi_pairs = compare_resi_CA(df_pdb1,pdb2)

# Root process does the rest
if rank == 0:
     # Print table
    print("Chain1\tResidue1\tResidue name1\tChain2\tResidue2\tResidue name2\tCA/C1'_distance")
    for resi_pair in resi_pairs:
            print("%s\t%s\t%s\t%s\t%s\t%s\t%s" % (resi_pair[0].chainid, resi_pair[0].seqid, resi_pair[0].restyp, resi_pair[1].chainid, resi_pair[1].seqid, resi_pair[1].restyp, resi_pair[0].CA.xyz_change))