#  
#  This file is a part of HPC PCDSFrame software
#
#  Copyright (C) 2019 Parallel Computing and Data Science (PCDS) Laboratory
#                         School of Computing and Information Sciences
#                           Florida International University (FIU)
#                            Authors: Muhammad Haseeb, Fahad Saeed
#                              Email: {mhaseeb, fsaeed}@fiu.edu
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
import os.path
import filecmp
import datetime
import subprocess
from subprocess import call
from shutil import copyfile


# Generates a normal unicore job script
def genNormalScript(wkspc, jobname, outname, partition, nds, ntask_per_node, minust, comd):
	script = open(wkspc + '/autogen/' + jobname, 'w+')
	script.write('#!/bin/bash\n')
	script.write('\n')
	script.write('#SBATCH --job-name="' + jobname +'"\n')
	script.write('#SBATCH --output="' + wkspc + '/autogen/' + outname + '.out"\n')
	script.write('#SBATCH --partition=' + partition + '\n')
	script.write('#SBATCH --nodes=' + nds + '\n')
	script.write('#SBATCH --ntasks-per-node=' + ntask_per_node + '\n')
	script.write('#SBATCH -t ' + minust + '\n')
	script.write('\n')
	script.write(comd + '\n')

	return

# Generates a multithreaded OpenMP job script
def genOpenMPScript(wkspc, jobname, outname, partition, nds, ntask_per_node, minust, ompthrds, command, args):
	script = open(wkspc + '/autogen/' + jobname, 'w+')
	script.write('#!/bin/bash\n')
	script.write('\n')
	script.write('#SBATCH --job-name="' + jobname +'"\n')
	script.write('#SBATCH --output="' + wkspc + '/autogen/' + outname + '.out"\n')
	script.write('#SBATCH --partition=' + partition + '\n')
	script.write('#SBATCH --nodes=' + nds + '\n')
	script.write('#SBATCH --ntasks-per-node=' + ntask_per_node + '\n')
	script.write('#SBATCH -t ' + minust + '\n')
	script.write('\n')
	script.write ('export OMP_NUM_THREADS      ' + ompthrds + '\n')
	script.write('\n')
	script.write(command + ' ' + args)

	return

# Generates a Hybrid MPI/OpenMP job script
def genMPI_OpenMPScript(wkspc, jobname, outname, partition, nds, ntask_per_node, minust, ompthrds, command, npernode, blevel, bpolicy, args):
	script = open(wkspc + '/autogen/' + jobname, 'w+')
	script.write('#!/bin/bash\n')
	script.write('\n')
	script.write('#SBATCH --job-name="' + jobname +'"\n')
	script.write('#SBATCH --output="' + wkspc + '/output/' + outname + '.%j.%N.out"\n')
	script.write('#SBATCH --partition=' + partition + '\n')
	script.write('#SBATCH --nodes=' + nds + '\n')
	script.write('#SBATCH --ntasks-per-node=' + ntask_per_node + '\n')
	script.write('#SBATCH -t ' + minust + '\n')
	script.write('\n')
	script.write ('export OMP_NUM_THREADS      ' + ompthrds + '\n')
	script.write('\n')
	script.write('ibrun --npernode ' + npernode + ' -bl ' + blevel + ' -bp ' + bpolicy + ' ' + command + ' ' + args)

	return

# The main function
if __name__ == '__main__':

	# Read the arguments
	if len(sys.argv) > 1:
		paramfile = sys.argv[1]
	else:
		print ("ERROR: Enter the path to params file")
		print ("USAGE: py preprocess.py ./params.txt")
		print ("Generate a sampleparams.txt: py preprocess.py -g")
		sys.exit(-1)

	# Generate the sampleparams.txt file
	if (paramfile == '-g'):
		sample = open("./sampleparams.txt","w+")

		sample.write('# \n')
		sample.write('# HPC MS/MS Proteomics Pipeline\n')
		sample.write('# Copyrights(C) 2019 PCDS Laboratory\n')
		sample.write('# Muhammad Haseeb, and Fahad Saeed')
		sample.write('# School of Computing and Information Sciences\n')
		sample.write('# Florida International University (FIU), Miami, FL\n')
		sample.write('# Email: {mhaseeb, fsaeed}@fiu.edu\n')
		sample.write('# \n')
		sample.write('# Auto generated sampleparams.txt\n')
		sample.write('# Sample parameters generated for XSEDE Comet cluster\n')
		sample.write('# For more information: https://portal.xsede.org/sdsc-comet\n')
		sample.write('# \n')
		sample.write('# Generated on: ' + (datetime.datetime.now()).strftime("%Y-%m-%d %H:%M %Z") + '\n')
		sample.write('# \n')
		sample.write('# IMPORTANT: DO NOT put any spaces between variable=value\n')
		sample.write('# \n\n')

		sample.write('# Path (absolute or relative) to Workspace directory \n')
		sample.write('workspace=/path/to/workspace\n\n')

		sample.write('# Nodes available\n')
		sample.write('nodes=2\n\n')

		sample.write('# Cores per machine\n')
		sample.write('cores=24\n\n')

		sample.write('# OpenMP cores per MPI process\n')
		sample.write('cores_per_mpi=12\n\n')

		sample.write('# MPI binding policy: scatter, compact \n')
		sample.write('bp=scatter\n\n')

		sample.write('# MPI binding level: core, socket, numanode\n')
		sample.write('bl=socket\n\n')

		sample.write('# Recommended: Auto tune MPI/OpenMP settings based on \n')
		sample.write('# Index size, Sockets and NUMA nodes in the system? 1/0? \n')
		sample.write('autotune=1\n\n')

		sample.write('# ABSOLUTE path to proteome database\n')
		sample.write('database=/path/to/database.fasta\n\n')

		sample.write('# ABSOLUTE path to MS/MS dataset\n')
		sample.write('ms2data=/path/to/msms/dataset\n\n')

		sample.write('# Mods to include per peptide sequence\n')
		sample.write('nmods=3\n\n')

		sample.write('# Mods Information: AAs mass mods_per_pep\n')
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
		
		sample.write('# Missed cleavages\n')
		sample.write('missed_cleavages=1\n\n')
		
		sample.write('# Min peptide length\n')
		sample.write('min_length=6\n\n')
		
		sample.write('# Max peptide length\n')
		sample.write('max_length=40\n\n')

		sample.write('# Min precursor mass (Da)\n')
		sample.write('min_prec_mass=500\n\n')
		
		sample.write('# Max precursor mass (Da)\n')
		sample.write('max_prec_mass=5000\n\n')

		sample.write('# Digestion Enzyme\n')
		sample.write('enzyme=Trypsin\n\n')

		sample.write('# Index Distribution Policy: chunk, cyclic, zigzag\n')
		sample.write('policy=cyclic\n\n')

		sample.write('# Max fragment charge\n')
		sample.write('maxz=3\n\n')

		sample.write('# Min shared peak\n')
		sample.write('shp=4\n\n')
		
		sample.write('# Scratch pad memory for scorecard in MBs (min: 2048MB)\n')
		sample.write('spadmem=2048\n\n')

		sample.write('# Resolution (Da)\n')
		sample.write('res=0.01\n\n')
		
		sample.write('# Precursor Mass Tolerance (+-Da): -1 means infinity \n')
		sample.write('dM=500\n\n')

		sample.write('# Fragment Mass Tolerance (+-Da)\n')
		sample.write('dF=0.05\n\n')

		sample.write('# Top Matches to report\n')
		sample.write('top_matches=10\n')

		
		print('Generated: ./sampleparams.txt')
		print ("\nSUCCESS")
		
		sys.exit(0)

# ##################################################################################

	# Initialize the parameters
	cores = 24
	sockets = 2
	numa = 2
	nodes = 2
	numamem = math.inf
	mpi_per_node = sockets
	cores_per_socket = int(cores/sockets)
	cores_per_numa = int (cores/numa)
	threads = cores_per_socket
	bp = 'scatter'
	bl = 'socket'
	autotune = 1
	database = ''
	ms2data = ''
	nmods = 0
	madded = 0
	mods = []
	mcleavages = 2
	min_length = 6
	max_length = 40
	maxz       = 3
	enzyme = 'Trypsin'
	dF = 0.02
	dM = 500
	res = 0.01
	scale = int(1/res)
	min_prec_mass = 500
	max_prec_mass = 5000
	top_matches = 10
	shp_cnt = 4
	workspace = './workspace'
	policy = 'cyclic'
	spadmem = 2048
	indexsize = 0
	nions     = 0
	size_mb   = 0
	mb_per_numa = 0
	mb_per_mpi  = 0
	pcdsframepath = os.getcwd()

# ##################################################################################

	print ('\n************************************\n')
	print   ('*  HPC MS/MS Proteomics Pipeline   *\n')
	print   ('*  Copyrights PCDS Lab, SCIS, FIU  *\n')
	print   ('************************************\n\n')

	# Parse the params file
	with open(paramfile) as params:
		for line in params:

			# Ignore the empty or comment lines
			if (line[0] == '\r' or line[0] == '#' or line[0] == '\n'):
				continue

			# Split line into param and value
			param, val = line.split('=', 1)

			# Set database file 
			if (param == 'database'):
				if (val[-1] == '\n'):
					val = val[:-1]
				if (val[-1] == '\r'):
					val = val[:-1]

				database = val
				print ('Proteome DB   =', database)
				if (os.path.isfile(database) == False):
					print ("ERROR: Enter valid path to database.fasta")
					sys.exit(-2)

			elif (param == 'ms2data'):
				if (val[-1] == '\n'):
					val = val[:-1]
				if (val[-1] == '\r'):
					val = val[:-1]

				ms2data = val
				print ('MS/MS dataset =', ms2data)
				if (os.path.exists(ms2data) == False):
					print ("ERROR: Enter valid path to MS2 dataset")
					sys.exit(-3)

			# Set max nodes in the system [1,72] on COMET
			elif (param == 'nodes'):
				nodes = int(val)
				if (nodes <= 0):
					nodes = 1
				if (nodes > 72):
					nodes = 72
				print ('Using nodes =', nodes)

			# Cores per node
			elif (param == 'cores'):
				cores = int(val)
				if (cores <= 0 or cores > 24):
					cores = 24
				print ('Using cores/node  =', cores)

			# Autotune number of cores and MPI processes to run?
			elif (param == 'autotune'):
				autotune = int(val)
				if (autotune <= 0):
					autotune = 0
				if (autotune > 0):
					autotune = 1
				print ('Autotune =', autotune)

			# Set the MPI binding level
			elif (param == 'bl'):
				if (val[-1] == '\n'):
					val = val[:-1]
				if (val[-1] == '\r'):
					val = val[:-1]

				if (bl == 'socket' or bl == 'numanode' or bl == 'core'):
					bl = val
				print ('Using MPI bl =', bl)

				# Set the MPI binding policy
			elif (param == 'bp'):
				if (val[-1] == '\n'):
					val = val[:-1]
				if (val[-1] == '\r'):
					val = val[:-1]

				if (bp == 'scatter' or bp == 'compact'):
					bp = val
				print ('Using MPI bp =', bp)

			# Set OMP cores per MPI
			elif (param == 'cores_per_mpi'):
				threads = int(val)
				if (threads <= 0 or threads > cores):
					threads = int(cores)

			# Set the enzyme for digestion
			elif (param == 'enzyme'):
				if (val[-1] == '\n'):
					val = val[:-1]
				if (val[-1] == '\r'):
					val = val[:-1]
				enzyme = val
				print ('Using enzyme  =', enzyme)

			# Set the distribution policy
			elif (param == 'policy'):
				if (val[-1] == '\n'):
					val = val[:-1]
				if (val[-1] == '\r'):
					val = val[:-1]
				policy = val
				print ('Using policy =', policy)

			# Set max mods
			elif (param == 'nmods'):
				nmods = int (val)
				if (nmods < 0):
					nmods = 0 
				if (nmods > 8):
					nmods = 8 
				print ('Max mods/pep  =', nmods)

			# There is a \n at the end of each string
			elif (param[:-1] == 'mod'):
				if (val[-1] == '\n'):
					val = val[:-1]
				if (val[-1] == '\r'):
					val = val[:-1]

				if (val != 'X 0 0'):
					mods.append(val)
					print ('Adding mod   =', str(val))
					madded += 1

			# Set the min digestion length
			elif (param == 'min_length'):
				min_length = int(val)

				if (min_length < 4):
					min_length = 4 
				if (min_length > 60):
					min_length = 60 
				print ('Min pep len  =', min_length)

			# Set the max digestion length
			elif (param == 'max_length'):
				max_length = int(val)
				if (max_length < 4):
					max_length = 4 
				if (max_length > 60):
					max_length = 60
				print ('Max pep len  =', max_length)

			# Set the max missed cleavages
			elif (param == 'missed_cleavages'):
				mcleavages = int(val)
				if (mcleavages < 0):
					mcleavages = 0 
				if (mcleavages > 5):
					mcleavages = 5
				print ('Missed Cleavages =', mcleavages)

			# Set the max fragment ion charge
			elif (param == 'maxz'):
				maxz = int(val)
				if (maxz < 1):
					maxz = 1 
				if (maxz > 5):
					maxz = 5
				print ('Max frag chg =', maxz)

			# Fragment mass tolerance
			elif (param == 'dF'):
				if (dF < 0.001):
					dF = 0.001 
				if (dF > 5.0):
					dF = 5.0
				dF = float(val)
				print ('dF           =', dF)

			# Peptide precursor mass tolerance
			elif (param == 'dM'):
				dM = float(val)
				if (dM < 0.001):
					dM = 0.001 
				print ('dM           =', dM)

			# m/z axis resolution
			elif (param == 'res'):
				res = float(val)
				if (res <= 0):
					res = 0.01 
				if (res > 5.0):
					res = 5.0
				print ('resolution   =', res)

			# Minimum precursor mass
			elif (param == 'min_prec_mass'):
				min_prec_mass = int(val)
				if (min_prec_mass < 0):
					min_prec_mass = 0 
				if (min_prec_mass > 10000):
					min_prec_mass = 10000
				print ('min_prec_mass =', min_prec_mass)

			# Maximum precursor mass
			elif (param == 'max_prec_mass'):
				max_prec_mass = int(val)
				if (max_prec_mass < 0):
					max_prec_mass = 0 
				if (max_prec_mass > 10000):
					max_prec_mass = 10000
				print ('max_prec_mass =', max_prec_mass)				

			# Minimum Shared Peaks
			elif (param == 'shp_cnt'):
				shp_cnt = int(val)
				if (shp_cnt < 1):
					shp_cnt = 1 
				if (shp_cnt > 20):
					shp_cnt = 20
				print ('Min Shared Peaks =', shp_cnt)

			# Scorecard memory
			elif (param == 'spadmem'):
				spadmem = int(val)
				if (spadmem < 2048):
					spadmem = 2048
				print ('Scorecard Memory =', spadmem)

			# Workspace Path
			elif (param == 'workspace'):
				if (val[-1] == '\n'):
					val = val[:-1]
				if (val[-1] == '\r'):
					val = val[:-1]

				if (val[-1] == '/'):
					val = val[:-1]

				workspace = os.path.abspath(str(val))
				print ('workspace   =', workspace)

			# Maximum precursor mass
			elif (param == 'top_matches'):
				top_matches = int(val)
				print ('Top matches =', top_matches)

#	print ('Mods Added', mods)

	if (len(mods) == 0):
		mods.append("X 0 0")
		nmods = 0

	# Close the params file
	params.close()

# ##################################################################################

	# Create a workspace directory
	print ('\nInitializing Workspace at: ', workspace)

	if (os.path.exists(workspace) == False):	
		os.mkdir(workspace)

	# Create the output directory for results
	if (os.path.exists(workspace + '/output') == False):
		os.mkdir(workspace + '/output')

	# Create directory where autogen stuff will be placed
	if (os.path.exists(workspace + '/autogen') == False):
		os.mkdir(workspace + '/autogen')

	if (os.path.isfile(workspace + '/autogen/settings.txt') == False or filecmp.cmp(workspace + '/autogen/settings.txt', paramfile) == False):
		copyfile(paramfile, workspace + '/autogen/settings.txt')

# ##################################################################################

	# AUTOTUNER
	if (autotune == 1):
		print ("\n\n**** Autotuning parameters ****\n")

		# Call the lsinfo to gather CPU information
		if (os.path.isfile(workspace + '/autogen/lscpu.out') == False):
			genNormalScript(workspace, 'lscpu', 'lscpu', 'compute', '1','1', '00:00:05', 'lscpu | tr -d " \\r"')

			autotune = call("sbatch " + workspace + "/autogen/lscpu", shell=True)
			print ('\nWaiting for job scheduler\n')

		# Call the numactl --hardware to gather NUAM information
		if (os.path.isfile(workspace + '/autogen/numainfo.out') == False):
			genNormalScript(workspace, 'numainfo', 'numainfo', 'compute', '1','1', '00:00:05', 'numactl --hardware | tr -d " \\r"')
			
			autotune2 = call("sbatch " + workspace + "/autogen/numainfo", shell=True)
			print ('\nWaiting for job scheduler\n')

		# Wait for the lscpu process to complete 
		while (os.path.isfile(workspace + '/autogen/lscpu.out') == False):
			pass

		print ('\nExtracted System Settings\n')

		# Parse the machine info file
		with open(workspace + '/autogen/lscpu.out') as minfo:
			for line in minfo:

				# Ignore the empty or comment lines
				if (line[0] == '\r' or line[0] == '#' or line[0] == '\n'):
					continue

				# Split line into param and value
				param, val = line.split(':', 1)		

				# Set the sockets per node
				if (param == 'Socket(s)'):
					sockets = int(val)
					print ('Available sockets/machine  =', sockets)

				elif (param == 'CPU(s)'):
					cores = int(val)
					print ('Available cores/machine  =', cores)

				elif (param == 'Core(s)persocket'):
					cores_per_socket = int(val)
					print ('Available cores/socket  =', threads)

				elif (param == 'NUMAnode(s)'):
					numa = int(val)
					print ('Available NUMA nodes/machine =', numa)

		cores_per_numa = int(cores/numa)

		minfo.close()

		if (filecmp.cmp(workspace + '/autogen/settings.txt', paramfile) == False or os.path.isfile(workspace + '/autogen/counter.out')==False):

			# Prepare the pparams.txt file for seq generator
			pparam = workspace + '/autogen/pparams.txt'
			modfile = open(pparam, "w+")

			# Write params for the CFIR index
			modfile.write('/home/mhaseeb/database' + '/parts\n')
			modfile.write(ms2data + '\n')
			modfile.write(str(cores) + '\n')
			modfile.write(str(min_length) + '\n')
			modfile.write(str(max_length) + '\n')
			modfile.write(str(maxz) + '\n')
			modfile.write(str(dF) + '\n')
			modfile.write(str(dM) + '\n')
			modfile.write(str(res) + '\n')
			modfile.write(str(scale) + '\n')
			modfile.write(str(min_prec_mass) + '\n')
			modfile.write(str(max_prec_mass) + '\n')
			modfile.write(str(top_matches) + '\n')
			modfile.write(str(shp_cnt) + '\n')
			modfile.write(str(spadmem) + '\n')
			modfile.write(str(policy) + '\n')

			modfile.write(str(len(mods)) + '\n')
			modfile.write(str(nmods) + '\n')
			for info in mods:
				aa,ms,num = info.split(' ', 2)
				modfile.write(aa + '\n')
				modfile.write(str(ms) + '\n')
				modfile.write(str(num) + '\n')

			modfile.close()

			print ('\n')

			# Remove the previous counter file
			if (os.path.isfile(workspace + '/autogen/counter.out')):
				os.remove(workspace + '/autogen/counter.out')

			# Call the counter process
			cleancntr = call("make -C counter allclean", shell=True)
			makecntr = call("make -C counter", shell=True)
			genOpenMPScript(workspace, 'counter', 'counter', 'compute', '1', str(cores), '00:30:00', str(cores), pcdsframepath + '/counter/counter.exe', pparam)
			autotune3 = call('sbatch ' + workspace + '/autogen/counter', shell=True)

			print ('\nWaiting for job scheduler\n')


		# Wait for the numainfo process to complete 
		while (os.path.isfile(workspace + '/autogen/numainfo.out') == False):
			pass

		# Parse the machine info file
		with open(workspace + '/autogen/numainfo.out') as minfo:
			for line in minfo:

				# Ignore the empty or comment lines
				if (line[0] == '\r' or line[0] == '#' or line[0] == '\n'):
					continue

				# The distance table without the : splitter formatting this point onward
				if (line == 'nodedistances:'):
					break

				# Split line into param and value
				param, val = line.split(':', 1)

				if (param == 'nodedistances'):
					break

				# Get the available NUMA memory
				if (param[:4] == 'node' and param[-4:] == 'free'):
					mem = int(val[:-3]) - 512
					if (mem < numamem):
						numamem = mem

		print ('Available max NUMA memory (- 512 MB) =', numamem)

		minfo.close()

		# Wait for the counter process to complete
		while (os.path.isfile(workspace + '/autogen/counter.out') == False):
			pass

		print ('\nEstimating Index Size\n')

		# Parse the index size file
		with open(workspace + '/autogen/counter.out') as minfo:
			for line in minfo:

				# Ignore the empty or comment lines
				if (line[0] == '\r' or line[0] == '#' or line[0] == '\n' or line[0] == '/'):
					continue

				# Split line into param and value
				param, val = line.split(':', 1)	

				if (param == 'spectra'):
					indexsize = int(val)
					print ('Estimated Index Size (Spectra) =', indexsize)

				elif (param == 'ions'):
					nions = int(val)
					size_mb = (nions * 4 + (mpi_per_node * nodes * max_prec_mass * scale * 4)) / (1024 * 1024)
					print ('Estimated Index Size (Ions) =', nions)
					print ('Estimated Index Size (MBs)  =', round(size_mb, 3))

		minfo.close()
		
		if (size_mb == 0 or nions == 0 or indexsize == 0):
			print ('\nFATAL: counter.exe failed. Please check the ./cnterr.txt\n')

			if (os.path.isfile(workspace + '/autogen/counter.out') == True):
				copyfile(workspace + '/autogen/counter.out', pcdsframedir + '/cnterr.txt')
				os.remove(workspace + '/autogen/counter.out')

			# Exit abnormally
			exit(-3)

		print ('\n')

# ##################################################################################

		# Apply the optimizations 

		# Case 1: Sockets >= NUMA nodes (one or multiple sockets/NUMA)
		if (sockets >= numa):
			# Set the BL to socket, BP to scatter, mpi_per_node to sockets, and threads_per_mpi to cores_per_socket
			threads = cores_per_socket
			mpi_per_node = sockets
			bl = 'socket'
			bp = 'scatter'

		# Case 2: Socket mapped to multiple NUMA nodes
		elif (sockets < numa):
			threads = int(cores_per_socket/numa)
			mpi_per_node = int(sockets * numa)
			bl = 'numanode'
			bp = 'scatter'

		print('Tuning OpenMP and MPI settings...\n')
		print('Setting threads/MPI =', threads)
		print('Setting MPI/machine =', mpi_per_node)
		print('Setting MPI Policy  =', bp)
		print('Setting MPI Binding =', bl)
		print('Setting Index / MPI =', int(indexsize/(mpi_per_node * nodes)))

		print('\nSUCCESS\n')

	# Run the digestor now
	digesteddb = workspace + '/digested_db.fasta'
	digestcommand = "Digestor.exe -in " + database + " -out " + digesteddb + " -out_type fasta -threads " + str(threads) + " -missed_cleavages " + str(mcleavages) + " -enzyme " + enzyme +  " -min_length " + str(min_length) + " -max_length " + str(max_length) + " -FASTA:ID number -FASTA:description remove"
	
	print ('\nRunning: '+ digestcommand + '\n')

	# Run the Digester.exe
#	digestor = call(digestcommand, shell=True)

	print ("\nSUCCESS\n")


	# Print the next steps
	print ('\nRunning: '+ 'Separate by Peptide Length')
	print ('\nRunning: '+ 'Custom Lexicographical Sort\n')

	# Print the clustering command
	clustercommand = './bash/sep_by_len.sh ' + digesteddb + ' ' + str(min_length) + ' ' + str(max_length)

	print ('Running: ' + clustercommand)

	# Run the cluster command and pass arguments
#	cluster = subprocess.run(['./bash/sep_by_len.sh ', digesteddb, str(min_length), str(max_length)], stdout=subprocess.PIPE, shell=True)

	print ("\nSUCCESS\n")

	# Prepare the uparams.txt file for seq generator
	modfile = open(workspace + '/autogen/uparams.txt', "w+")

	# Write params for the CFIR index
	modfile.write(workspace + '/parts\n')
	modfile.write(ms2data + '\n')
	modfile.write(str(threads) + '\n')
	modfile.write(str(min_length) + '\n')
	modfile.write(str(max_length) + '\n')
	modfile.write(str(maxz) + '\n')
	modfile.write(str(dF) + '\n')
	modfile.write(str(dM) + '\n')
	modfile.write(str(res) + '\n')
	modfile.write(str(scale) + '\n')
	modfile.write(str(min_prec_mass) + '\n')
	modfile.write(str(max_prec_mass) + '\n')
	modfile.write(str(top_matches) + '\n')
	modfile.write(str(shp_cnt) + '\n')
	modfile.write(str(spadmem) + '\n')
	modfile.write(str(policy) + '\n')

	modfile.write(str(len(mods)) + '\n')
	modfile.write(str(nmods) + '\n')
	for info in mods:
		aa,ms,num = info.split(' ', 2)
		modfile.write(aa + '\n')
		modfile.write(str(ms) + '\n')
		modfile.write(str(num) + '\n')

	modfile.close()


	# Construct CFIR index and search spectra
	uparams = workspace + '/autogen/uparams.txt\n'
	
	genMPI_OpenMPScript(workspace, 'cfir', 'cfir', 'compute', str(nodes), str(cores), "01:00:00", str(threads), pcdsframedir + '/cfirindex/cfir.exe', str(mpi_per_node), bl, bp, uparams)
	# Clean and make a fresh copy of CFIR index
#	cleancfir = call("make -C cfirindex clean", shell=True)
#	makecfir = call("make -C cfirindex", shell=True)

#	Run the PCDSFrame (CFIR)
#	cfir = subprocess.run(['./cfirindex/cfir.exe ', uparams], stdout=subprocess.PIPE, shell=True)

#	print (cfir.stdout.decode('utf-8'))

	print ('\nSUCCESS\n')
	print ('Thanks for using PCDSFrame software')
	print ('Please report bugs (if any) at {mhaseeb, fsaeed}@fiu.edu\n')
