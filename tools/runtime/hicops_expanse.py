#!@PYTHON_EXECUTABLE@
#  This file is a part of HiCOPS software
#
#  Copyright (C) 2021 Parallel Computing and Data Science (PCDS) Laboratory
#  School of Computing and Information Sciences
#  Florida International University (FIU)
#  Authors: Muhammad Haseeb, Fahad Saeed
#  Email: {mhaseeb, fsaeed}@fiu.edu
#
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
#

# Required Imports

import os
import sys
import math
import glob
import time
import os.path
import filecmp
import datetime
import argparse
import subprocess
from subprocess import call
from shutil import copyfile
from functools import reduce
from simple_slurm import Slurm

# time right now to be used
timerightnow = (datetime.datetime.now()).strftime("%Y_%m_%d_%H_%M_%Z")

#
# ------------------------------ Helper Functions ------------------------------
#
# Returns environment variables
def getEnvVar(var):
    return os.environ[var]

#
# ------------------------------------------------------------------------------
#
# Computes factors of a number in descending order
def factors(n):
    return list(set(reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0))))

#
# ------------------------------------------------------------------------------
#
# Checks if any jobs are running
def checkRunningJobs(username):
    squeue = subprocess.run('squeue -u ' + username + ' | wc -l', stdout=subprocess.PIPE, shell=True)
    proc = int(squeue.stdout.decode('utf-8'))
    if (proc == 1):
        return False
    else:
        return True

# Check completion status of a job
def checkJobStatus(job_id):
    squeue = subprocess.run('sacct --format JobID,State  -j ' + str(job_id) + ' | grep ' + str(job_id), stdout=subprocess.PIPE, shell=True)
    log = squeue.stdout.decode('utf-8')

    if log == '':
        return False

    else:
        log = log.split('\n')

        # last line is just empty
        for lines in log[:-1]:
            lines.lstrip().rstrip()
            id, stat = lines.split()
            if stat == 'COMPLETED':
                continue
            elif stat == 'FAILED':
                print ('FATAL: JobID# ' + str(job_id) + ' failed. Aborting...')
                exit(-10)
            else:
                return False

    return True
#
# ------------------------------------------------------------------------------
#
# Generates a normal unicore job script
def genSimpleScript(workspace, jobname, account, output, partition, nodes, ntasks_per_node, time, mailto, username, events, command, args=''):
    script = open(workspace + '/autogen/' + jobname, 'w+')
    script.write('#!/bin/bash\n')
    script.write('\n')
    script.write('#SBATCH --account=' + account +'\n')
    script.write('#SBATCH --job-name=' + jobname +'\n')
    script.write('#SBATCH --output=' + output + '\n')
    script.write('#SBATCH --partition=' + partition + '\n')
    script.write('#SBATCH --nodes=' + nodes + '\n')
    script.write('#SBATCH --ntasks-per-node=' + ntasks_per_node + '\n')
    script.write('#SBATCH --export=ALL\n')
    script.write('#SBATCH -t ' + time + '\n')

    if (mailto):
        script.write('#SBATCH --mail-type=' + events + '\n')
        script.write('#SBATCH --mail-type=' + username + '\n')

    script.write('\n')
    script.write(command + ' ' + args)
    script.write('\n')

    return

#
# ------------------------------------------------------------------------------
#
# Generates a multithreaded OpenMP job script
def genOpenMPScript(workspace, jobname, account, outname, partition, nodes, ntasks_per_node, time, cpus_per_task, mailto, username, events, command, args):
    script = open(workspace + '/autogen/' + jobname, 'w+')
    script.write('#!/bin/bash\n')
    script.write('\n')
    script.write('#SBATCH --account=' + account + '\n')
    script.write('#SBATCH --job-name="' + jobname +'"\n')
    script.write('#SBATCH --output=' + workspace + '/autogen/' + outname + '.out\n')
    script.write('#SBATCH --partition=' + partition + '\n')
    script.write('#SBATCH --nodes=' + nodes + '\n')
    script.write('#SBATCH --ntasks-per-node='+ ntasks_per_node +'\n')
    script.write('#SBATCH --cpus-per-task=' + cpus_per_task + '\n')
    script.write('#SBATCH --export=ALL\n')
    script.write('#SBATCH -t ' + time + '\n')
    if (mailto):
        script.write('#SBATCH --mail-type=' + events + '\n')
        script.write('#SBATCH --mail-type=' + username + '\n')

    script.write('\n')
    script.write ('export OMP_NUM_THREADS      ' + cpus_per_task + '\n')
    script.write('\n')
    script.write(command + ' ' + args)
    script.write('\n')

    return

#
# ------------------------------------------------------------------------------
#
# Generates a Hybrid MPI/OpenMP job script
def genMPI_OpenMPScript(workspace, jobname, account, output, partition, nodes, ntasks_per_socket, time, cpus_per_task, ntasks, bindto, mapby, mailto, username, events, command, args):
    script = open(workspace + '/autogen/' + jobname, 'w+')
    script.write('#!/bin/bash\n')
    script.write('\n')
    script.write('#SBATCH --account=' + account + '\n')
    script.write('#SBATCH --job-name=' + jobname +'\n')
    script.write('#SBATCH --output=' + output + '\n')
    script.write('#SBATCH --partition=' + partition + '\n')
    script.write('#SBATCH --nodes=' + nodes + '\n')
    script.write('#SBATCH --ntasks-per-socket=' + ntasks_per_socket + '\n')
    script.write('#SBATCH --cpus-per-task=' + cpus_per_task + '\n')
    script.write('#SBATCH --export=ALL\n')
    script.write('#SBATCH -t ' + time + '\n')
    if (mailto):
        script.write('#SBATCH --mail-type=' + events + '\n')
        script.write('#SBATCH --mail-type=' + username + '\n')
    script.write('\n')
    script.write ('export OMP_NUM_THREADS      ' + cpus_per_task + '\n')
    script.write('\n')
    script.write('srun --mpi=pmi2 ' + ' -np ' + ntasks + ' --bind-to ' + bindto + ' --map-by ' + mapby + ' ' + command + ' ' + args)
    script.write('\n')

    return

#
# ------------------------------------------------------------------------------
#
# Generates a sampleparams.txt file
def genSampleParams(inpath):
    # get user name to make workspace path
    username = os.environ['USER']
    sample = open(inpath + '/sampleparams.txt', 'w+')

    sample.write('\n')
    sample.write('# HiCOPS on SGCI Expanse Gateway\n')
    sample.write('# Copyrights(c) 2022 PCDS Laboratory\n')
    sample.write('# Muhammad Haseeb, and Fahad Saeed\n')
    sample.write('# School of Computing and Information Sciences\n')
    sample.write('# Florida International University (FIU), Miami, FL\n')
    sample.write('# Email: {mhaseeb, fsaeed}@fiu.edu\n\n')

    sample.write('# Auto generated sampleparams.txt\n')
    sample.write('# Sample parameters generated for SDSC Expanse machine\n')
    sample.write('# More information: https://portal.xsede.org/sdsc-expanse\n\n')

    sample.write('# Generated on: ' + timerightnow + '\n\n')

    sample.write('# -------------- SLURM parameters --------------\n\n')

    sample.write('# Job time: hh:mm:ss (max: 2:00:00)\n')
    sample.write('jobtime=00:45:00\n\n')

    sample.write('# -------------- MPI parameters --------------\n\n')

    sample.write('# Number of nodes (max: 2)\n')
    sample.write('nodes=2\n\n')

    sample.write('# Cores per MPI process (choose from: 32, 64)\n')
    sample.write('cpus_per_task=32\n\n')

    sample.write('# Optimize MPI settings? on/off\n')
    sample.write('optimize=on\n\n')

    sample.write('# -------------- Search parameters --------------\n\n')

    #sample.write('# ABSOLUTE path to workspace directory\n')
    #sample.write('workspace=/expanse/lustre/scratch/'+ username + '/temp_project/workspaces/hicops_workspace\n\n')

    #sample.write('# ABSOLUTE path to processed protein database parts\n')
    #sample.write('dbparts=/path/to/processed/database/parts\n\n')

    #sample.write('# ABSOLUTE path to MS/MS dataset\n')
    #sample.write('ms2data=/path/to/ms2/dataset\n\n')

    sample.write('# Mods to include per peptide sequence\n')
    sample.write('nmods=3\n\n')

    sample.write('# Mods Information: AA(max 4) mod_mass mods_per_pep\n')
    sample.write('mod1=M 15.99 2\n')
    sample.write('mod2=X 0 0\n')
    sample.write('mod3=X 0 0\n')
    sample.write('mod4=X 0 0\n')
    sample.write('mod5=X 0 0\n')
    sample.write('mod6=X 0 0\n')
    sample.write('mod7=X 0 0\n')
    sample.write('mod8=X 0 0\n')
    sample.write('mod9=X 0 0\n')
    sample.write('mod10=X 0 0\n')
    sample.write('mod11=X 0 0\n')
    sample.write('mod12=X 0 0\n')
    sample.write('mod13=X 0 0\n')
    sample.write('mod14=X 0 0\n')
    sample.write('mod15=X 0 0\n\n')

    sample.write('# Min peptide length\n')
    sample.write('min_length=6\n\n')

    sample.write('# Max peptide length\n')
    sample.write('max_length=40\n\n')

    sample.write('# Min precursor mass (Da)\n')
    sample.write('min_prec_mass=500\n\n')

    sample.write('# Max precursor mass (Da)\n')
    sample.write('max_prec_mass=5000\n\n')

    sample.write('# Max fragment charge\n')
    sample.write('maxz=3\n\n')

    sample.write('# Min shared peaks\n')
    sample.write('shp=4\n\n')

    sample.write('# Required min shared peaks for candidacy\n')
    sample.write('min_hits=4\n\n')

    sample.write('# Base normalized Intensity for MS/MS data \n')
    sample.write('base_int=100000\n\n')

    sample.write('# Cutoff ratio w.r.t. base intensity \n')
    sample.write('cutoff_ratio=0.01\n\n')

    sample.write('# Resolution (Da)\n')
    sample.write('res=0.01\n\n')

    sample.write('# Precursor Mass Tolerance (+-Da): -1 means infinity \n')
    sample.write('dM=10\n\n')

    sample.write('# Fragment Mass Tolerance (+-Da)\n')
    sample.write('dF=0.02\n\n')

    # sample.write('# Top Matches to report\n')
    # sample.write('top_matches=10\n\n')

    sample.write('# Max expect value to report\n')
    sample.write('expect_max=20.0\n')

    print('Generating a sample params file...')
    print ('SUCCESS\n')
    print('Generated file: ' + inpath +'/sampleparams.txt\n')

#
# ------------------------------------------------------------------------------
#
def readParamFile(file):
    # initialize an empty dictionary
    settings = {}

    # initialize to an empty array
    settings['mods'] = []
    settings['madded'] = 0
    
    # Parse the params file
    with open(file) as f:
        for line in f:

            # Ignore the empty or comment lines
            if (line[0] == '\r' or line[0] == '#' or line[0] == '\n'):
                continue

            # Split line into param and value
            param, val = line.split('=', 1)

            # strip whitespaces from left and right if any
            param = param.lstrip().rstrip()
            val = val.rstrip().lstrip()

            if (param == 'dbparts'):
                settings['dbparts'] = os.path.abspath(val)
            elif (param == 'nmods'):
                settings['nmods'] = int(val)
            elif (param[:-1] == 'mod' or param[:-1] == 'mod1'):
                if (val != 'X 0 0'):
                    settings['mods'].append(val)
                    settings['madded'] += 1
            elif (param == 'min_length'):
                settings['min_length'] = int(val)
            elif (param == 'max_length'):
                settings['max_length'] = int(val)
            elif (param == 'min_prec_mass'):
                settings['min_prec_mass'] = int(val)
            elif (param == 'max_prec_mass'):
                settings['max_prec_mass'] = int(val)
    
    return settings

#
# ------------------------------------------------------------------------------
#
def compareSettings(file1, file2):
    a = readParamFile(file1)
    b = readParamFile(file2)

    return a == b

#
# ------------------------------ Main Function ------------------------------
#

# The main function
if __name__ == '__main__':

    # print header
    print ('\n-----------------------------')
    print   ('|  HiCOPS for XSEDE Gateway |')
    print   ('|    PCDS Lab, SCIS, FIU    |')
    print   ('-----------------------------\n')

    # Check the Python version
    pyversion = float(str(sys.version_info[0]) + '.' + str(sys.version_info[1]))
    if (pyversion < 3.6):
        print ('\nERROR: This software requires Python v3.6+')
        print (' Your Python version is: ' + str(pyversion) + '\n')
        exit(-1)

    # initialize parser
    parser = argparse.ArgumentParser(description='Automated execution of HiCOPS on SGCI Expanse Gateway')

    # input parameters
    parser.add_argument('-i', '--in', dest='params', type=str, required=('-g' not in sys.argv) and ('--gen' not in sys.argv),
                        help='Path to params file')

    # generate a sample file
    parser.add_argument('-g', '--gen', dest='gen', action='store_true',
                        help='Generate a sampleparams.txt file and exit.')

    # path to database parts
    parser.add_argument('-db', '--database', dest='db', required=('-g' not in sys.argv) and ('--gen' not in sys.argv), 
                        help='Path to database parts')

    parser.add_argument('-dat', '--dataset', dest='dataset', required=('-g' not in sys.argv) and ('--gen' not in sys.argv),
                        help = 'Path to MS/MS dataset')

    # parse arguments
    args = parser.parse_args()

    # variable to check if to generate a sampleparams file
    generate = args.gen

    # if generate a sampleparam
    if (generate == True):
        # gen sampleparams.txt
        genSampleParams(os.getcwd())
        sys.exit(0)

    # input path to the params file
    if (args.params is not None):
        paramfile = args.params.lstrip()
        paramfile = os.path.abspath(paramfile)

        # check if directory exists
        if not os.path.isfile(paramfile):
            print ('ERROR: Directory does not exist\n')
            sys.exit (-1)

    # path to database parts
    if (args.db is not None):
        db = args.db.lstrip()
        db = os.path.abspath(db)

        if (os.path.exists(db) == False):
            print ("FATAL: Invalid path to proteome database parts directory")
            sys.exit(-2)
    
    # path to dataset
    if (args.dataset is not None):
        dataset = args.dataset.lstrip()
        dataset = os.path.abspath(dataset)

        if (os.path.exists(dataset) == False):
            print ("FATAL: Invalid path to MS/MS dataset")
            sys.exit(-2)

#
# ------------------------------ Initialization ------------------------------
#
    # path to hicops binary
    hicopspath = os.path.dirname(getEnvVar('hicops_PATH')+'/bin')

    # Initialize the variables
    parameters = {}

    # node parameters
    parameters['nodes'] = 2
    parameters['cores'] = 128
    parameters['sockets'] = 2
    parameters['numa'] = 8
    parameters['numamem'] = math.inf
    parameters['cpus_per_socket'] = int(parameters['cores']/parameters['sockets'])
    parameters['cpus_per_numa'] = int (parameters['cores']/parameters['numa'])

    # max allowed nodes to SGSI community 
    parameters['MAXNODES'] = 2

    # mpi settings
    parameters['ntasks_per_socket'] = 2
    parameters['ntasks_per_node'] = parameters['sockets'] * parameters['ntasks_per_socket']
    parameters['cpus_per_task'] = int (parameters['cpus_per_socket']/parameters['ntasks_per_socket'])
    parameters['ntasks'] = int(parameters['ntasks_per_node'] * parameters['nodes'])
    parameters['prep_threads'] = int (parameters['cpus_per_task']/4)
    parameters['mapby'] = 'socket'
    parameters['bindto'] = 'socket'

    # optimize MPI settings?
    parameters['optimize'] = True

    # search settings
    parameters['dbparts'] = dataset
    parameters['ms2data'] = db
    parameters['nmods'] = 0
    parameters['madded'] = 0
    parameters['mods'] = []
    parameters['min_length'] = 6
    parameters['max_length'] = 40
    parameters['maxz'] = 3
    parameters['dF'] = 0.03
    parameters['dM'] = 12
    parameters['res'] = 0.01
    parameters['scale'] = int(1 / parameters['res'])
    parameters['min_prec_mass'] = 500
    parameters['max_prec_mass'] = 5000
    parameters['top_matches'] = 10
    parameters['shp_cnt'] = 4
    parameters['min_hits'] = 4
    parameters['base_int'] = 100000
    parameters['cutoff_ratio'] = 0.01
    parameters['policy'] = 'cyclic'
    parameters['spadmem'] = 2048
    parameters['indexsize'] = 0
    parameters['nions'] = 0
    parameters['size_mb'] = 0
    parameters['mb_per_numa'] = 0
    parameters['mb_per_task'] = 0
    parameters['nparts'] = 0
    parameters['expect_max'] = 20.0
    parameters['jobtime'] ='00:45:00'

    # slurm settings
    parameters['account'] = 'wmu101'
    parameters['username'] = os.environ['USER']
    parameters['workspace'] = '/expanse/lustre/scratch/' + parameters['username'] + '/temp_project/workspaces/hicops_workspace' + timerightnow
    parameters['mail'] = False
    parameters['evts'] = 'FAIL'
    
    # test if new params
    newparams = False

#
# ------------------------------ Parse params file -------------------------------------------
#

    print ("\n\n-------------- Parameter Parsing -------------- \n")

    # Print the parameters provided in the file
    print ('Reading parameters from:', paramfile, '\n')

    # Parse the params file
    with open(paramfile) as params:
        for line in params:

            # Ignore the empty or comment lines
            if (line[0] == '\r' or line[0] == '#' or line[0] == '\n'):
                continue

            # Split line into param and value
            param, val = line.split('=', 1)

            # strip whitespaces from left and right if any
            param = param.lstrip().rstrip()
            val = val.rstrip().lstrip()

            # Set the job time
            if (param == 'jobtime'):
                hh,mm,ss = map(int, val.split(':',2))
                if (hh == 0 and mm == 0 and ss == 0):
                    val = '00:45:00'

                parameters['jobtime'] = val
                print ('Job time =', parameters['jobtime'])

            # Set max nodes in the system [1,4] on Expanse
            elif (param == 'nodes'):
                parameters['nodes'] = int(val)
                if (parameters['nodes'] <= 0):
                    parameters['nodes'] = 1
                elif (parameters['nodes'] > parameters['MAXNODES']):
                    parameters['nodes'] = parameters['MAXNODES']
                print ('Nodes =', parameters['nodes'])

            # Set cores per MPI process
            elif (param == 'cpus_per_task'):
                parameters['cpus_per_task'] = int(val)
                if (parameters['cpus_per_task'] not in [32,64]):
                    parameters['ntasks_per_socket'] = 2
                    parameters['cpus_per_task'] = int(parameters['cpus_per_socket']/parameters['ntasks_per_socket'])

                else:
                    parameters['ntasks_per_socket'] = int(parameters['cpus_per_socket']/parameters['cpus_per_task'])

                parameters['ntasks_per_node'] = parameters['sockets'] * parameters['ntasks_per_socket']
                parameters['ntasks'] = int(parameters['ntasks_per_node'] * parameters['nodes'])

            # Set max mods
            elif (param == 'nmods'):
                i_val = int (val)
                if (i_val < 0):
                    i_val = 0
                if (i_val > 8):
                    i_val = 8
                parameters['nmods'] = i_val
                print ('Max mods/pep  =', parameters['nmods'])

            # There is a \n at the end of each string
            elif (param[:-1] == 'mod' or param[:-1] == 'mod1'):
                if (val != 'X 0 0'):
                    parameters['mods'].append(val)
                    print ('Adding mod   =', str(val))
                    parameters['madded'] += 1

            # Set the min digestion length
            elif (param == 'min_length'):
                i_val = int(val)
                if (i_val < 5):
                    i_val = 5 
                if (i_val > 60):
                    i_val = 60
                parameters['min_length'] = i_val
                print ('Min pep len  =', parameters['min_length'])

            # Set the max digestion length
            elif (param == 'max_length'):
                i_val = int(val)
                if (i_val < 5):
                    i_val = 5 
                if (i_val > 60):
                    i_val = 60
                parameters['max_length'] = i_val
                print ('Max pep len  =', parameters['max_length'])

            # Minimum precursor mass
            elif (param == 'min_prec_mass'):
                i_val = int(val)
                if (i_val < 0):
                    i_val = 0 
                if (i_val > 10000):
                    i_val = 10000
                parameters['min_prec_mass'] = i_val
                print ('Min precursor mass =', parameters['min_prec_mass'])

            # Maximum precursor mass
            elif (param == 'max_prec_mass'):
                i_val = int(val)
                if (i_val < 0):
                    i_val = 0 
                if (i_val > 10000):
                    i_val = 10000
                parameters['max_prec_mass'] = i_val
                print ('Max precursor mass =', parameters['max_prec_mass'])

            # Set the max fragment ion charge
            elif (param == 'maxz'):
                i_val = int(val)
                if (i_val < 1):
                    i_val = 1 
                if (i_val > 5):
                    i_val = 5
                parameters['maxz'] = i_val
                print ('Max frag chg =', parameters['maxz'])

            # Minimum Shared Peaks
            elif (param == 'shp'):
                i_val = int(val)
                if (i_val < 1):
                    i_val = 1 
                if (i_val > 20):
                    i_val = 20
                parameters['shp_cnt'] = i_val
                print ('Min shared peaks =', parameters['shp_cnt'])
                
            # Minimum required PSM hits
            elif (param == 'min_hits'):
                i_val = int(val)
                if (i_val < 4):
                    i_val = 4
                parameters['min_hits'] = i_val
                print ('Required min PSM hits =', parameters['min_hits'])

            # Base Normalized Intensity
            elif (param == 'base_int'):
                i_val = int(val)
                if (i_val < 10000):
                    i_val = 10000
                parameters['base_int'] = i_val
                print ('Base Normalized Intensity =', parameters['base_int'])
                
            # Intensity Cutoff Ratio
            elif (param == 'cutoff_ratio'):
                f_val = float(val)
                if (f_val >= 0.80):
                    f_val = 0.80
                if (f_val <= 0):
                    f_val = 0.01
                parameters['cutoff_ratio'] = f_val
                print ('Intensity Cutoff Ratio =', parameters['cutoff_ratio'])

            # m/z axis resolution
            elif (param == 'res'):
                f_val = float(val)
                if (f_val <= 0):
                    f_val = 0.01 
                if (f_val > 5.0):
                    f_val = 5.0
                parameters['res'] = f_val
                print ('Resolution   =', parameters['res'])

            # Peptide precursor mass tolerance
            elif (param == 'dM'):
                f_val = float(val)
                if (f_val < 0.001):
                    f_val = 0.001 
                parameters['dM'] = f_val
                print ('Peptide mass tolerance (dM) =', parameters['dM'])

            # Fragment mass tolerance
            elif (param == 'dF'):
                f_val = float(val)
                if (f_val < 0.0):
                    f_val = 0.0
                if (f_val > 0.1):
                    f_val = 0.1
                parameters['dF'] = f_val
                print ('Fragment mass tolerance (dF) =', parameters['dF'])

            elif (param == 'expect_max'):
                f_val = float(val)
                if (f_val < 0):
                    f_val = 0
                if (f_val > 100):
                    f_val = 100
                parameters['expect_max'] = f_val
                print ('Max expect value to report =', parameters['expect_max'])

            '''
            # Scorecard memory
            elif (param == 'spadmem'):
                i_val = int(val)
                if (i_val < 2048):
                    i_val = 2048
                parameters['spadmem'] = i_val
                print ('Scratch Memory (MB) =', parameters['spadmem'])

            # Maximum precursor mass
            elif (param == 'top_matches'):
                i_val = int(val)
                if (i_val < 1):
                    i_val = 1
                parameters['top_matches'] = i_val
                print ('Top matches =', parameters['top_matches'])
            '''

#    print ('Mods Added', mods)

    if (len(parameters['mods']) == 0):
        parameters['mods'].append("X 0 0")
        parameters['nmods'] = 0

    # Close the params file
    params.close()

    print('\n')

    #
    # ------------------------------ Workspace Creation -------------------------------------------
    #

    # Create a workspace directory
    print ('\nInit workspace at: ', parameters['workspace'])

    os.makedirs(parameters['workspace'], exist_ok=True)

    # Create the output directory for results
    os.makedirs(parameters['workspace'] + '/output', exist_ok=True)

    # Create directory where autogen stuff will be placed
    os.makedirs(parameters['workspace'] + '/autogen', exist_ok=True)

    # Check if the params have been changed from the last run
    if (os.path.isfile(parameters['workspace'] + '/autogen/settings.txt') == False or compareSettings(parameters['workspace'] + '/autogen/settings.txt', paramfile) == False):
        newparams = True
        copyfile(paramfile, parameters['workspace'] + '/autogen/settings.txt')

    # Sanity check
    if (parameters['min_length'] > parameters['max_length']):
        temp = parameters['min_length']
        parameters['min_length'] = parameters['max_length']
        parameters['max_length'] = temp
        print('WARNING: Provided: min_length > max_length. swapping...\n')

    # Check if all database parts are available
    for k in range(parameters['min_length'], parameters['max_length'] + 1):
        if (os.path.isfile(parameters['dbparts'] + '/' + str(k) + '.peps') == False):
            print ('FATAL: Database part(s) are missing\n')
            exit(-3)

#
# ------------------------------ Optimization -------------------------------------------
#

    # Optimizer
    if (parameters['optimize'] == True):
        print ("\n\n-------------- Optimizing MPI settings --------------\n")

        # invoke lsinfo and numastat to gather CPU information
        if (os.path.isfile(hicopspath + '/.nodeinfo') == False):
            mch_info = Slurm(account = parameters['account'], job_name = 'lstopo', partition = 'compute', export = 'ALL', time = '00:00:10', nodes = 1, ntasks_per_node = 1, cpus_per_task = 1, output = hicopspath +'/.nodeinfo')

            job_id = mch_info.sbatch('lscpu | tr -d " \\r" && numactl --hardware | tr -d " \\r"')

            # Wait for the lscpu process to complete 
            while (os.path.isfile(hicopspath + '/.nodeinfo') == False or checkJobStatus(job_id) == False):
                time.sleep(0.3)

        print ('Reading System Node Info\n')

        # Parse the machine info file
        with open(hicopspath + '/.nodeinfo') as minfo:
            for line in minfo:

                # Ignore the empty or comment lines
                if (line[0] == '\r' or line[0] == '#' or line[0] == '\n'):
                    continue

                # Split line into param and value
                param, val = line.split(':', 1)
                
                # strip left and right
                param.lstrip().rstrip()
                val.lstrip().rstrip()

                # Sockets per node
                if (param == 'Socket(s)'):
                    parameters['sockets'] = int(val)
                    print ('sockets_per_node  =', parameters['sockets'])

                # CPUS per node
                elif (param == 'CPU(s)'):
                    parameters['cores'] = int(val)
                    print ('cpus_per_node  =', parameters['cores'])

                # CPUS per socket
                elif (param == 'Core(s)persocket'):
                    parameters['cpus_per_socket'] = int(val)
                    print ('cpus_per_socket =', parameters['cpus_per_socket'])

                # NUMA nodes
                elif (param == 'NUMAnode(s)'):
                    parameters['numa'] = int(val)
                    print ('NUMA nodes =', parameters['numa'])

                # RAM per NUMA node
                elif (param[:4] == 'node' and param[-4:] == 'free'):
                    mem = int(val[:-3])
                    if (mem < parameters['numamem']):
                        parameters['numamem'] = mem
                
                # todo: NUMA distances to optimize job span
                elif (param == 'nodedistances'):
                    break
    
        minfo.close()

        # print RAM per NUMA node
        print ('RAM per NUMA node =', parameters['numamem'])

        # set cpus_per_numa
        parameters['cpus_per_numa'] = int(parameters['cores']/parameters['numa'])

        # Check if params file was modified
        if (newparams == True or os.path.isfile(parameters['workspace'] + '/autogen/counter.out') == False):
            # Prepare the pparams.txt file for seq generator
            pparams = parameters['workspace'] + '/autogen/pparams.txt'

            modfile = open(pparams, "w+")

            # Write params for the CFIR index
            modfile.write(parameters['dbparts'] + '\n')
            modfile.write(parameters['ms2data'] + '\n')
            modfile.write(str(parameters['cores']) + '\n')
            modfile.write(str(parameters['min_length']) + '\n')
            modfile.write(str(parameters['max_length']) + '\n')
            modfile.write(str(parameters['maxz']) + '\n')
            modfile.write(str(parameters['dF']) + '\n')
            modfile.write(str(parameters['dM']) + '\n')
            modfile.write(str(parameters['res']) + '\n')
            modfile.write(str(parameters['scale']) + '\n')
            modfile.write(str(parameters['min_prec_mass']) + '\n')
            modfile.write(str(parameters['max_prec_mass']) + '\n')
            modfile.write(str(parameters['top_matches']) + '\n')
            modfile.write(str(parameters['shp_cnt']) + '\n')
            modfile.write(str(parameters['spadmem']) + '\n')
            modfile.write(str(parameters['policy']) + '\n')

            modfile.write(str(len(parameters['mods'])) + '\n')
            modfile.write(str(parameters['nmods']) + '\n')
            for info in parameters['mods']:
                aa,ms,num = info.split(' ', 2)
                modfile.write(aa + '\n')
                modfile.write(str(ms) + '\n')
                modfile.write(str(num) + '\n')

            modfile.close()

            print ('\n')

            # Remove the previous counter file
            if (os.path.isfile(parameters['workspace'] + '/autogen/counter.out')):
                os.remove(parameters['workspace'] + '/autogen/counter.out')

            counter = Slurm(account=parameters['account'], job_name = 'counter', time='00:30:00', partition='compute', export = 'ALL', nodes = 1, ntasks_per_node = 1, cpus_per_task=parameters['cores'], output=parameters['workspace'] + '/autogen/counter.out')

            # sbatch the counter job
            job_id = counter.sbatch('export OMP_NUM_THREADS      ' + str(parameters['cores']) + ' ; ' + hicopspath + '/bin/counter ' + pparams)

            # Wait for the counter process to complete
            while (os.path.isfile(parameters['workspace'] + '/autogen/counter.out') == False or checkJobStatus(job_id) == False):
                time.sleep(0.3)

            # Remove the temporary pparams.txt
            if (os.path.isfile(pparams)):
                os.remove(pparams)

        # Parse the index size file
        with open(parameters['workspace'] + '/autogen/counter.out') as minfo:
            for line in minfo:

                # Ignore the empty or comment lines
                if (line[0] == '\r' or line[0] == '#' or line[0] == '\n' or line[0] == '/'):
                    continue

                # Split line into param and value
                param, val = line.split(':', 1)

                # strip left and right
                param.lstrip().rstrip()
                val.lstrip().rstrip()

                if (param == 'spectra'):
                    parameters['indexsize'] = int(val)
                    print ('Estimated Index Size (x1e6 spectra) =', float(parameters['indexsize'])/(1000 * 1000))

                elif (param == 'ions'):
                    parameters['nions'] = int(val)

        minfo.close()

        if (parameters['nions'] == 0 or parameters['indexsize'] == 0):
            print ('\nFATAL: counter.exe failed or index size = 0. Please check: ' + hicopspath + '/cnterr.txt\n')

            if (os.path.isfile(parameters['workspace'] + '/autogen/counter.out') == True):
                copyfile(parameters['workspace'] + '/autogen/counter.out', hicopspath + '/cnterr.txt')
                os.remove(parameters['workspace'] + '/autogen/counter.out')

            # Exit abnormally
            exit(-3)

        print ('\n')

#
# ------------------------------ Optimize -------------------------------------------
#

        # Case 1: sockets >= NUMA nodes (one or multiple sockets per NUMA)
    
        # Case 2: multiple NUMA nodes per socket (sockets < NUMA) -- True for Expanse
        # TODO: need to read the node distances to find the optimal range
        # FIXME: here we will do it manually

        parameters['bindto'] = 'socket'
        parameters['mapby'] = 'socket'

        parameters['cpus_per_task'] = int(parameters['cpus_per_socket'] / parameters['ntasks_per_socket'])

        parameters['ntasks_per_node'] = int(parameters['sockets'] * parameters['ntasks_per_socket'])

        parameters['ntasks'] = int(parameters['ntasks_per_node'] * parameters['nodes'])
    
        # Estimate index in MBs
        parameters['size_mb'] = ((parameters['nions'] * 4 + (parameters['ntasks'] * parameters['max_prec_mass'] * parameters['scale'] * 4)) / (1024 * 1024))  + (parameters['spadmem'] * parameters['ntasks'])

        parameters['size_per_task'] = parameters['size_mb'] / (parameters['ntasks'])

        parameters['mem_per_task'] = parameters['numamem'] * parameters['numa'] / parameters['ntasks_per_node']

        # Optimize based on the index size (in spectra) per MPI
        # If partition size > 25 million, then increase number of partitions
        max_tasks_per_node = parameters['numa']

        if parameters['size_per_task'] > (parameters['mem_per_task'] * 0.8):

            parameters['ntasks_per_socket'] = 1
            parameters['cpus_per_task'] = int(parameters['cpus_per_socket'] / parameters['ntasks_per_socket'])
            parameters['ntasks_per_node'] = int(parameters['sockets'] * parameters['ntasks_per_socket'])
            parameters['ntasks'] = int(parameters['ntasks_per_node'] * parameters['nodes'])
            parameters['mem_per_task'] = (parameters['numamem'] * parameters['numa']) / parameters['ntasks_per_node']

            parameters['size_mb'] = ((parameters['nions'] * 4 + (parameters['ntasks'] * parameters['max_prec_mass'] * parameters['scale'] * 4)) / (1024 * 1024))  + (parameters['spadmem'] * parameters['ntasks'])
            parameters['size_per_task'] = parameters['size_mb'] / (parameters['ntasks'])

            # We hope this never happens :)
            if parameters['size_per_task'] > (parameters['mem_per_task'] * 0.7):
                print ('WARNING: index size > available memory ==> ' + str(parameters['size_per_task']) + 'MB > ' + str(parameters['mem_per_task'] * 0.7) + 'MB \n')
                print ('TIP: Increase the no. of nodes or reduce the index size to avoid SEGFAULTS and/or performance degradations')

        # if very small index then the preprocessing threads may be increased to 50%
        elif (parameters['size_per_task']  < 10E6 or parameters['dM'] < 50):
            parameters['prep_threads'] = int(parameters['cpus_per_task']/3)

        print('\nOptimized settings for this machine:\n')
        print('Setting cpus_per_task = ', parameters['cpus_per_task'])
        print('Setting pre-process cpus_per_task = ', parameters['prep_threads'])
        print('Setting ntasks_per_node = ', parameters['ntasks_per_node'])
        print('Setting ntasks_per_socket = ', parameters['ntasks_per_socket'])
        print('Setting total tasks = ', parameters['ntasks'])
        print('Setting NUMA nodes per task = ', int(parameters['ntasks']/parameters['numa']))
        print('Setting task span = ', parameters['ntasks_per_socket'], ' tasks/socket')
        print('Setting MPI Binding to  =', parameters['bindto'])
        print('Setting MPI mappings to =', parameters['mapby'])
        print('Estimated index per task =',  round(parameters['size_per_task'], 2), 'MB')
        print('Availale RAM per task = ', round(parameters['mem_per_task'], 2), 'MB')

#
# ------------------------------ Write uparams.txt -------------------------------------------
#

    print ("\n\n-------------- Launching HiCOPS --------------\n")

    # Prepare the uparams.txt file for PCDSFrame
    uparams = parameters['workspace'] + '/autogen/uparams.txt'

    modfile = open(uparams, "w+")

    # Write params for the CFIR index
    modfile.write(parameters['dbparts'] + '\n')
    modfile.write(parameters['ms2data'] + '\n')
    modfile.write(parameters['workspace'] + '/output\n')
    modfile.write(str(parameters['cpus_per_task']) + '\n')
    modfile.write(str(parameters['prep_threads']) + '\n')
    modfile.write(str(parameters['min_length']) + '\n')
    modfile.write(str(parameters['max_length']) + '\n')
    modfile.write(str(parameters['maxz']) + '\n')
    modfile.write(str(parameters['dF']) + '\n')
    modfile.write(str(parameters['dM']) + '\n')
    modfile.write(str(parameters['res']) + '\n')
    modfile.write(str(parameters['scale']) + '\n')
    modfile.write(str(parameters['min_prec_mass']) + '\n')
    modfile.write(str(parameters['max_prec_mass']) + '\n')
    modfile.write(str(parameters['top_matches']) + '\n')
    modfile.write(str(parameters['expect_max']) + '\n')
    modfile.write(str(parameters['shp_cnt']) + '\n')
    modfile.write(str(parameters['min_hits']) + '\n')
    modfile.write(str(parameters['base_int']) + '\n')
    modfile.write(str(parameters['cutoff_ratio']) + '\n')
    modfile.write(str(parameters['spadmem']) + '\n')
    modfile.write(parameters['policy'] + '\n')

    modfile.write(str(len(parameters['mods'])) + '\n')
    modfile.write(str(parameters['nmods']) + '\n')
    for info in parameters['mods']:
        aa,ms,num = info.split(' ', 2)
        modfile.write(aa + '\n')
        modfile.write(str(ms) + '\n')
        modfile.write(str(num) + '\n')

    modfile.close()

#
# ------------------------------ Launch HiCOPS -------------------------------------------
#

    # Generate hicops job script
    genMPI_OpenMPScript(workspace = parameters['workspace'], jobname = 'hicops', account = parameters['account'], output = parameters['workspace'] + '/output/hicops.%j.%N.out', partition = 'compute', nodes = str(parameters['nodes']), ntasks_per_socket = str(parameters['ntasks_per_socket']), time = parameters['jobtime'], cpus_per_task = str(parameters['cpus_per_task']), ntasks = str(parameters['ntasks']), bindto = parameters['bindto'], mapby = parameters['mapby'], mailto = parameters['mail'], username = parameters['username'], events = parameters['evts'],  command = hicopspath + '/bin/hicops', args = uparams)

    # Generate psm2tsv job script
    genSimpleScript(workspace = parameters['workspace'], jobname = 'postprocess', account = parameters['account'], output = parameters['workspace'] + '/output/postprocess.%j.out', partition = 'shared', nodes = '1', ntasks_per_node = '1', time = '00:25:00', mailto = parameters['mail'], username = parameters['username'], events = parameters['evts'], command = hicopspath + '/bin/tools/psm2tsv ', args = '-i ' + parameters['workspace'] + '/output')

    # SLURM 

    # Run HiCOPS
    hicops = Slurm(account = parameters['account'], job_name = 'hicops', output = parameters['workspace'] + '/output/hicops.%j.%N.out', ntasks = parameters['ntasks'], time = parameters['jobtime'], partition='compute', nodes = parameters['nodes'], ntasks_per_socket = parameters['ntasks_per_socket'], cpus_per_task = parameters['cpus_per_task'], export = 'ALL')

    hicops_id = hicops.sbatch('export OMP_NUM_THREADS      ' + str(parameters['cpus_per_task']) + ' ; srun --mpi=pmi2 -n ' + str(parameters['ntasks']) + ' ' + hicopspath + '/bin/hicops ' + uparams)

    # Run post-processor with dependency=hicops_id
    postprocess = Slurm(account = parameters['account'], job_name = 'postprocess', output = parameters['workspace'] + '/output/postprocess.%j.out', time = '00:25:00' , partition = 'shared', nodes = 1, ntasks_per_node = 1, cpus_per_task = 1, export = 'ALL', dependency=dict(afterany=hicops_id))

    post_id = postprocess.sbatch(hicopspath + '/bin/tools/psm2tsv ' + ' -i ' + parameters['workspace'] + '/output')

    # Final prints
    print ('\nHiCOPS is now running. job_ids:', hicops_id, post_id, '\n')
    print ('You can check the job progress by: \n')
    print ('squeue -j ' + str(hicops_id) + ',' + str(post_id), ' \n')
    print ('\nOR\n')
    print ('sacct -j ' + str(hicops_id) + ',' + str(post_id), ' \n')
    print ('The output will be written at: '+ parameters['workspace'] + '/output')

    print ('\nSUCCESS\n')

    # print ('After job completion, run:\n')
    # print ('sbatch ' + parameters['workspace'] + '/autogen/postprocess')
    # print ('\nOR\n')
    # print ('srun -A=wmu101 --partition=compute --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 -t 00:25:00 --export=ALL ' + hicopspath + '/tools/psm2excel -i ' + parameters['workspace'] + '/output\n')

    print ('-----------------------------------------')
    print ('|  Read more: https://hicops.github.io  |')
    print ('-----------------------------------------\n\n')
