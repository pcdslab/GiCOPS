#!@PYTHON_EXECUTABLE@
#   Copyright (c) 2020 Muhammad Haseeb, Fahad Saeed
#    School of Computing and Information Sciences
#      Florida International University   (FIU)
#         Email: {mhaseeb, fsaeed} @fiu.edu
# 
#  License
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.


# Import Packages
import os
import sys
import glob
import pandas as pd


# Sanity Checking

# The main function
if __name__ == '__main__':

    if len(sys.argv) <= 1:
        print ("ERROR: Please provide the path to partial TSV files")
        exit(0)

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = './'

    if len(sys.argv) > 2:
        output = sys.argv[2]
    else:
        output = ''

    # Open the TSV files
    data_dir = os.path.expanduser(data_dir)

    # Check if directory exists
    if not os.path.isdir(data_dir):
        print ("Directory does not exist\n")
        sys.exit (-1)

    # Get all files with TSV
    tsv_files = glob.glob(data_dir + '/*.tsv')
    #print (tsv_files)


    # Extract TSV data

    # Matrix where the df will be collected
    matrix = []

    # Read all TSVs into data matrix
    if (len(tsv_files) > 1):
        for kk in tsv_files:
            print ('Loading File: ', kk)
            dat = pd.read_csv(kk, sep='\t', index_col=None, header=0)
            matrix.append(dat)
            os.remove(kk)


    # Construct data frame
    print ("Constructing DataFrame...")

    # Concatenate data into a single data frame
    frame = pd.concat(matrix, axis=0, ignore_index=True)

    # Print the new data frame shape
    if (frame.shape[1] == 0):
        print ('ERROR: Empty data frame')
        sys.exit(-2)
    # else:
    #    print(frame.shape)

    # Write to Excel file

    print ('Writing to Excel...')

    # Check if output file provided
    if output == '':
        output = data_dir + '/Concat.xlsx'

    # Write to Excel format
    frame.to_excel(output)

    # Print the output address
    print ('DONE: ', output)