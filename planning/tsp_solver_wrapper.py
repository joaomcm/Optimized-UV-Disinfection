#		Interface the TSP LKH Solver
# 
# 		This script is a simple python interface to a compiled 
#		version of the LKH TSP Solver. It requires that the
#		solver is compiled at the given directories.
#
#	This script is part of the "utils" section of the StructuralInspectionPlanner
#	Toolbox. A set of elementary components are released together with this 
#	path-planning toolbox in order to make further developments easier. 
# 	
#	Authors: 
#	Kostas Alexis (kalexis@unr.edu)
#   From: https://github.com/unr-arl/LKH_TSP
# 	With slight modification by Ramya Ramalingam and Joao Marques
import os
import math
import numpy as np
import time

#root directory for Optimized-UV-Disinfection
lkh_dir = os.path.join(os.path.abspath(os.path.join(os.path.split(__file__)[0],'..')))
tsplib_dir = lkh_dir
lkh_cmd = '/LKH-3.0.6/LKH'


def writeTSPLIBfile_FE(fname_tsp,CostMatrix,user_comment):

	dims_tsp = len(CostMatrix)
	name_line = 'NAME : ' + fname_tsp + '\n'
	type_line = 'TYPE: TSP' + '\n'
	comment_line = 'COMMENT : ' + user_comment + '\n'
	tsp_line = 'TYPE : ' + 'TSP' + '\n'
	dimension_line = 'DIMENSION : ' + str(dims_tsp) + '\n'
	edge_weight_type_line = 'EDGE_WEIGHT_TYPE : ' + 'EXPLICIT' + '\n' # explicit only
	edge_weight_format_line = 'EDGE_WEIGHT_FORMAT: ' + 'FULL_MATRIX' + '\n'
	display_data_type_line ='DISPLAY_DATA_TYPE: ' + 'NO_DISPLAY' + '\n' # 'NO_DISPLAY'
	edge_weight_section_line = 'EDGE_WEIGHT_SECTION' + '\n'
	eof_line = 'EOF\n'
	Cost_Matrix_STRline = []
	for i in range(0,dims_tsp):
		cost_matrix_strline = ' '
		for j in range(0,dims_tsp-1):
			cost_matrix_strline = cost_matrix_strline + str(int(CostMatrix[i][j])) + '\t'

		j = dims_tsp-1
		cost_matrix_strline = cost_matrix_strline + str(int(CostMatrix[i][j]))
		cost_matrix_strline = cost_matrix_strline + '\n'
		Cost_Matrix_STRline.append(cost_matrix_strline)
	# print('\\n\n\n\n\n\n\n\n\n\n tsplib_dir = {} \n\n\n\n\n'.format(tsplib_dir))
	fileID = open((tsplib_dir + fname_tsp + '.tsp'), "w+")
	# print(name_line)
	fileID.write(name_line)
	fileID.write(comment_line)
	fileID.write(tsp_line)
	fileID.write(dimension_line)
	fileID.write(edge_weight_type_line)
	fileID.write(edge_weight_format_line)
	fileID.write(edge_weight_section_line)
	for i in range(0,len(Cost_Matrix_STRline)):
		fileID.write(Cost_Matrix_STRline[i])

	fileID.write(eof_line)
	fileID.close()

	fileID2 = open((tsplib_dir + fname_tsp + '.par'), "w+")

	problem_file_line = 'PROBLEM_FILE = ' + tsplib_dir + fname_tsp + '.tsp' + '\n' 
	optimum_line = 'OPTIMUM 378032' + '\n'
	move_type_line = 'MOVE_TYPE = 5' + '\n'
	patching_c_line = 'PATCHING_C = 3' + '\n'
	patching_a_line = 'PATCHING_A = 2' + '\n'
	TRACE_LEVEL = 'TRACE_LEVEL = 1000 ' + '\n'
	runs_line = 'RUNS = 1' + '\n'
	tour_file_line = 'TOUR_FILE = ' +tsplib_dir +  fname_tsp + '.txt' + '\n'

	fileID2.write(problem_file_line)
	# fileID2.write(optimum_line)
	fileID2.write(move_type_line)
	fileID2.write(patching_c_line)
	fileID2.write(patching_a_line)
	fileID2.write(runs_line)
	# fileID2.write(TRACE_LEVEL)
	fileID2.write(tour_file_line)
	fileID2.close()
	return fileID, fileID2

def run_LKHsolver_cmd(fname_basis):
	# print('\n\nExecuting LKH !\n\n')
	run_lkh_cmd = lkh_dir  + lkh_cmd + ' ' + tsplib_dir + fname_basis + '.par'
	os.system(run_lkh_cmd)
	# print('\n\n EXECUTED LKH \n\n\n')

def copy_toTSPLIBdir_cmd(fname_basis):
	copy_toTSPLIBdir_cmd = 'copy' + ' ' + fname_basis + '.txt' + ' ' +  tsplib_dir
	os.system(copy_toTSPLIBdir_cmd)

def rm_solution_file_cmd(fname_basis):
	rm_sol_cmd = 'del' + ' ' + '/' + fname_basis + '.txt'
	os.system(rm_sol_cmd) 

def readTourFile(filename):
    '''
        Reads the file which had the tour written to it and extracts the tour
    '''
    
    addLine = False
    getFirstNode = False
    firstNode = None
    tour = list()
    pathLength = 0
    fp = open(filename, 'r')
    lines = fp.read().splitlines()
    startTourIndex = lines.index('TOUR_SECTION')
    endTourIndex = lines.index('-1')
    stringTour = lines[startTourIndex + 1: endTourIndex]
    integerTour = [(int(val) - 1) for val in stringTour] # In the LKH solver, the nodes are indexed starting from 1, so we reduce 1 to start indexing from 0
    # tour.append(tour[0])

    return integerTour

def runTSP(CostMatrix, fname_tsp, user_comment = ' '):
	print('\n\n\n\n\n {}    \n\n\n'.format(fname_tsp))
	[fileID1,fileID2] = writeTSPLIBfile_FE(fname_tsp,CostMatrix,user_comment)
	run_LKHsolver_cmd(fname_tsp)

	final_path = lkh_dir + '/' + fname_tsp
	#copy_toTSPLIBdir_cmd(fname_tsp)
	#rm_solution_file_cmd(fname_tsp)
	# print('\n\n\n\n final_path = {} , lkh_dir = {}\n\n '.format(final_path,lkh_dir)	)
	return readTourFile(final_path+'.txt')
