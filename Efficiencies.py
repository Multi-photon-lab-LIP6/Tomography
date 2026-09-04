### Created on: 18-07-2022
### Author: Laura Martins

import numpy as np
import scipy
import time

import os
import glob

from pathlib import Path
import fnmatch
"""
This function finds files in the form 'general'+'containing'
This can be used if we want to find files that have 'general' in common but that can be distinguished by 'containing'
"""
def finding_file(general, containing, filenames):
    for file in filenames:
        if fnmatch.fnmatch(file, general+containing+'*'):
            return file
        elif fnmatch.fnmatch(file, general+containing+'*'):
            return file
    print('No file containing: ', general+containing, '...')
    pass

"""
select_lines_in_file() writes the data of a file in an array:
- each column (w iteration) corresponds to a new datafile (zz, za, az, aa)
- each line (iter iteration) corresponds to a different column of the datafile (to a different channel)
"""
def select_lines_in_file(start_line, finish_line, datafiles, shape, bases, directory):
    array=np.zeros((finish_line-start_line,shape[1]), dtype='float')
    os.chdir(directory)

    for w in range(shape[1]):
        file=finding_file('ABCD=', bases[w], datafiles)
        
        with open(file) as file:
            for line in file:
                fields = line.split()
                for iter, field in enumerate(fields[start_line:finish_line]):
                    array[iter][w]=float(field)
                    
    return(array)

"""
get_channels_eff() returns the normalized efficiency of each channel
Not all channels have the same efficiency for various reasons
We need to be able to characterize each efficiency to be able to correct the number of counts_array
"""
def get_channels_eff(datafiles,qubit_number, column_start, column_stop, directory):
    os.chdir(directory)
    letters = ['z', 'a']
    eff = []

    for i in letters:
        if qubit_number == 1:
            base = i 
            eff.append(base)
        for j in letters:
            if qubit_number == 2:
                base = i + j
                eff.append(base)
            for k in letters:
                if qubit_number == 3:
                    base = i + j + k 
                    eff.append(base)
                for l in letters:
                    if qubit_number == 4:
                        base = i + j + k + l
                        eff.append(base)
    

    efficiencies_aux=np.zeros(((column_stop)-column_start, len(eff)), dtype=float)
    
        ### !Don't forget that we transpose here. That's why we sum over the 0 axis and not 1 in the next line!
    efficiencies_aux=select_lines_in_file(column_start, column_stop, datafiles, np.shape(efficiencies_aux), eff, os.getcwd()).transpose()

    """
    Each column (channel) will be turned into the sum over the lines (that correspond to different measurement basis)
    When normalized we end up with a an array with the relative channel efficiencies in the order we saved the data
    The re-ordiring of the basis to match each channel to a measurement basis happens in
    correct_counts_with_channels_eff in ProjectorCounts.py
    """         
    efficiencies=np.sum(efficiencies_aux, 0)/np.max(np.sum(efficiencies_aux, 0))
    return(efficiencies)

### set_raw_counts() returns an array with the counts recorded in the columns 4:8 of the datafile (where the coincidence counts are written)
def set_raw_counts(datafiles, qubit_number, column_start, column_stop, directory):
    os.chdir(directory)
    counts_aux=np.zeros((2**qubit_number,3**qubit_number), dtype=float)
    
    letters = ['x', 'y', 'z']
    bases = []

    for i in letters:
        if qubit_number == 1:
            base = i 
            bases.append(base)
        for j in letters:
            if qubit_number == 2:
                base = i + j
                bases.append(base)
            for k in letters:
                if qubit_number == 3:
                    base = i + j + k 
                    bases.append(base)
                for l in letters:
                    if qubit_number == 4:
                        base = i + j + k + l
                        bases.append(base)

    counts=select_lines_in_file(column_start, column_stop, datafiles, np.shape(counts_aux), bases, os.getcwd())

    """
    Just ordering the array such that it matches with the covention counts_aux[x] (this changes depending on how we save data):
    - x=0: HH
    - x=1: HV
    - x=2: VH
    - x=3: VV
    If we change this order we need to do the same in correct_counts_with_channels_eff in projectorcounts.py
    """
    if qubit_number == 1:
        counts_aux=counts[[0,1]]
        return(counts_aux)
    if qubit_number == 2:
        counts_aux=counts[[0,1,2,3]]
        return(counts_aux)
    if qubit_number == 3:
        counts_aux=counts[[0,1,2,3,4,5,6,7]]
        return(counts_aux)
    if qubit_number == 4:
        counts_aux=counts[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]
        return(counts_aux)
    
def set_raw_counts_double_emissions(datafiles, qubit_number, column_start, column_stop, directory):
    os.chdir(directory)
    counts_aux=np.zeros((column_stop-column_start,3**qubit_number), dtype=float)
    
    letters = ['x', 'y', 'z']
    bases = []

    for i in letters:
        if qubit_number == 1:
            base = i 
            bases.append(base)
        for j in letters:
            if qubit_number == 2:
                base = i + j
                bases.append(base)
            for k in letters:
                if qubit_number == 3:
                    base = i + j + k 
                    bases.append(base)
                for l in letters:
                    if qubit_number == 4:
                        base = i + j + k + l
                        bases.append(base)

    counts=select_lines_in_file(column_start, column_stop, datafiles, np.shape(counts_aux), bases, os.getcwd())

    """
    Just ordering the array such that it matches with the covention counts_aux[x] (this changes depending on how we save data):
    - x=0: HH
    - x=1: HV
    - x=2: VH
    - x=3: VV
    If we change this order we need to do the same in correct_counts_with_channels_eff in projectorcounts.py
    """
    indice = []
    for i in range(column_stop-column_start):
        indice.append(i)
    counts_aux=counts[indice]
    return(counts_aux)

    
    