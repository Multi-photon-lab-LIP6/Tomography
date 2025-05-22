import numpy as np
import scipy
import time
import math
import itertools
from scipy.stats import norm

from copy import deepcopy

from tomography import *

from NestedForLoop import get_iterator
from pathlib import Path
from scipy.linalg import sqrtm

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import os
import glob

import pandas as pd

from scipy.optimize import least_squares

import fnmatch
from efficiencies import *
from optimization import Optimizer, function_fidelity_U4, FidelityResults, function_fidelity_Rz
from constants import *

from densitymatrix import DensityMatrix, apply_unitary_to_dm
import u_to_wp_conversion
from pathlib import Path
import fnmatch



def ghz_tomo(file, working_dir, working_dir_data, correct_2emission, aq_time, get_u, get_wp,plot, density,save,qubit_number=4):

    ######################################################################################################
    #----- COUNTING THE FILES AND SAVING THEM IN AN ARRAY TO MAKES THE REST OF THE ANALYSIS EASIER -------
    ######################################################################################################

    n_files=0
    os.chdir(working_dir_data)

    filenames = [i for i in glob.glob(file)]
    filenames.sort(key=os.path.getmtime)

    index_to_file = {}

    for index, filename in enumerate(filenames):
        os.chdir(f"{working_dir_data}\\{filename}")
        filenames_aux=[i for i in glob.glob("counts*")]
        for index_second, filenames_aux_second in enumerate(filenames_aux):
            index_to_file[n_files] = f"{filename}\\{filenames_aux_second}"
            n_files+=1
    os.chdir(working_dir)

    # print("Analyse Files: ", filenames_aux)
    # print(filenames)
    # print("\n")

    #####################################################################
    #---------------------- DEFINING PARAMS ----------------------------#
    #####################################################################
    os.chdir(working_dir)

    ## Defining the columns of the data file we want to use as data to reconstruct the density matrix (eg.: HH HV VH and VV basis)
    BASIS_TO_CHANNEL={
        "HA": 1,
        "VA": 2,
        "HB": 3,
        "VB": 4,
        "HC": 5,
        "VC": 6,
        "HD": 7,
        "VD": 8,
        }

    ### 4 qubits GHZ ###
    eigenstates = [['HA','HB','HC','HD'],['HA','HB','HC','VD'],['HA','HB','VC','HD'],['HA','HB','VC','VD'],
                ['HA','VB','HC','HD'],['HA','VB','HC','VD'],['HA','VB','VC','HD'],['HA','VB','VC','VD'],
                ['VA','HB','HC','HD'],['VA','HB','HC','VD'],['VA','HB','VC','HD'],['VA','HB','VC','VD'],
                ['VA','VB','HC','HD'],['VA','VB','HC','VD'],['VA','VB','VC','HD'],['VA','VB','VC','VD']]

    fold = four_fold=[[BASIS_TO_CHANNEL[eigenstates[i][j]] for j in range(qubit_number)] for i in range(len(eigenstates))]

    datafile_channels = fold.copy()

    #####################################
    ### Ordering the output channels ####
    #####################################
    for clicks in fold:
        not_in_clicks = list(set(range(1, 2*qubit_number+1)) - set(clicks))
        not_in_clicks.sort()
        for rep in range(1, len(not_in_clicks)+1):
            for combo in itertools.combinations(not_in_clicks, rep):
                new_clicks = clicks + list(combo)
                new_clicks.sort()
                datafile_channels.append(new_clicks)
                
    datafile_channels = np.array(list(set(map(tuple, datafile_channels))), dtype=object)
    first_order = list(map(len, datafile_channels)) 
    order = np.lexsort((datafile_channels, first_order))
    datafile_channels = list(datafile_channels[order])
    datafile_channels = [list(t) for t in datafile_channels]

    ########################################
    ### Setting the fours and five fold ####
    ########################################
    coincidences_columns = []

    for i, iter in enumerate(eigenstates):
        proj = [BASIS_TO_CHANNEL[iter[m]] for m in range(qubit_number)]
        coincidences_columns.append(datafile_channels.index(proj))

    column_start = np.min(coincidences_columns)
    column_stop = np.max(coincidences_columns) + 1
    # print("Coincidences column_start:", column_start,"; column_stop: ", column_stop)

    column_start_5_emissions = 2**qubit_number
    column_stop_5_emissions = 2**qubit_number*qubit_number + column_start_5_emissions

    column_start_6_emissions = column_stop_5_emissions
    column_stop_6_emissions =  column_stop_5_emissions + 64*3

    # print("Coincidences column_start:", column_start_5_emissions,"; column_stop: ", column_stop_5_emissions)
    # print("Coincidences column_start:", column_start_6_emissions,"; column_stop: ", column_stop_6_emissions)


    statetomo = []
    state = []
    state_file = []

    xp_counts_corrected_with_eff=[]

    #####################################################################
    #---------------------- STATE TOMOGRAPHY ----------------------------
    #####################################################################
    for index in range(len(index_to_file)):
        os.chdir(f"{working_dir_data}\\{index_to_file[index]}\\")
        datafiles=[i for i in glob.glob("*")]
        
        ### Calculating the efficiencies of each detector
        efficiencies=get_channels_eff(datafiles, qubit_number, column_start, column_stop, os.getcwd())
        efficiencies_5_emissions=get_channels_eff(datafiles, qubit_number, column_start_5_emissions, column_stop_5_emissions, os.getcwd())
        efficiencies_6_emissions=get_channels_eff(datafiles, qubit_number, column_start_6_emissions, column_stop_6_emissions, os.getcwd())
    
        # print("Channels efficiencies: ", efficiencies)
        print("\n")

        ### Opening the data files and writing the data in counts_aux array
        counts_aux=set_raw_counts(datafiles, qubit_number, column_start, column_stop, os.getcwd())
        xp_counts=np.array(np.transpose(counts_aux))
        total_per_basis=np.sum(xp_counts, axis=1)
    
        counts_aux_5_emissions=set_raw_counts_double_emissions(datafiles, qubit_number, column_start_5_emissions, column_stop_5_emissions, os.getcwd())
        xp_counts_5_emissions=np.array(np.transpose(counts_aux_5_emissions))

        counts_aux_6_emissions=set_raw_counts_double_emissions(datafiles, qubit_number, column_start_6_emissions, column_stop_6_emissions, os.getcwd())
        xp_counts_6_emissions=np.array(np.transpose(counts_aux_6_emissions))

        statetomo.append(LRETomography(int(qubit_number), xp_counts, xp_counts_5_emissions,xp_counts_6_emissions))
        if correct_2emission is True:
            statetomo[-1].run(correct_eff=efficiencies,correct_double_emission_eff=efficiencies_5_emissions,correct_double_emission=None, four_emission_eff=None,GHZ = True)
        else:
            statetomo[-1].run(correct_eff=efficiencies)
        xp_counts_corrected_with_eff.append(statetomo[-1].xp_counts)
            
        state.append(statetomo[-1])
        state_file.append(index_to_file[index])

        ######################################
        #-- DEFINING THE TARGET GHZ STATE ---
        ######################################
        bell=(np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])+np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]))/np.sqrt(2)
        bellmatrix=np.array(np.outer(bell, np.conjugate(bell)))

        states=state
        # for index in range(len(states)):
        #     print("Fidelity before correction:")
        #     print(np.real(np.round(states[index].state.fidelity(bell),5)))
        #     print("\n")

        ##########################################################
        #----- OPTIMIZATION OF MAX FIDELITY UP TO UNITARIES ------
        ##########################################################
        fid=np.zeros((n_files))
        optimized_matrix=np.zeros((n_files,2**qubit_number,2**qubit_number), dtype='complex')

        guess=np.zeros(3*(qubit_number))
        bounds=[(-np.pi,np.pi)]*3*(qubit_number)
        results = []

        opt=Optimizer(guess, function_fidelity_U4, results=FidelityResults)
        for index in range(len(states)):
            result=opt.optimize(qubit_number,states[index].state, bell, bounds=bounds)
            results.append(result)

        if get_u is True:
            u_to_wp_conversion.get_u(results)
        if get_wp is True:
            u_to_wp_conversion.get_wp(results)

        ##########################################################
        #------------ ERRORS Input with BELL STATE---------------#
        ##########################################################
        error_runs=1

        U=[]
        bell_aux=[]
        target_ini=[]

        states=state
        states_file=state_file
        players=["Arya", "Bran", "Cersei", "Dany"]

        for index in range(len(states)):
            target=bellmatrix
            U.append(results[index].u)
            target_ini.append(np.transpose(np.conjugate(U[-1]))@bellmatrix@U[-1]) 

            # states[index].calculate_fidelity_error(players, error_runs, opt, target, optimization=True, bounds=bounds)
            
            # print('file, fidelity, fidelity_mean, fidelity_std: ',
            #       states_file[index], np.round(states[index].state.fidelity(target_ini[-1]),5), -np.round(states[index].fidelity_mu,5),
            #       np.round(states[index].fidelity_std,6), '\n')

        ##########################################################
        #--------------- PLOTTING DENSITY MATRIX-----------------#
        ##########################################################
        print(f"Filename - no unitary optimization: {file}")
        print(f"Double emission correction: {correct_2emission}")
        print(f"Average Rate: ({np.average(total_per_basis)/aq_time} +/- {np.std(total_per_basis)/aq_time}) Hz")
        print(f"Fidelity with no unitary correction = {np.round(states[-1].state.fidelity(bellmatrix),5)}%")
        # state[-1].state.plot_dm(cbar_real=True, cbar_im=False, save_pdf=None, save_svg=None)

        print(f"Filename - unitary optimization: {file}")
        print(f"Double emission correction: {correct_2emission}")
        print(f"Average Rate: ({np.average(total_per_basis)/aq_time} +/- {np.std(total_per_basis)/aq_time}) Hz")
        print(f"Fidelity with unitary optimization = {np.round(states[index].state.fidelity(target_ini[-1]),5)}%")
        
        if plot == True :
            results[-1].optimized_state.plot_dm(cbar_real=True, cbar_im=False, save_pdf=None, save_svg=None)

        if density == True :
            print('Density Matrix :', state[-1].state.state)
        
        if save == True :
            os.chdir(working_dir_data +'\\'+ filename)
            print(working_dir_data +'\\'+ filename)
            np.save("density", state[-1].state.state, allow_pickle=True)


#    def __init__(self, densitymatrix, working_dir):
#        self.densitymatrix=densitymatrix