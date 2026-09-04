# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 16:20:21 2018

@author: Simon
"""

#%%

import numpy as np

def get_iterator(K,n):
    """
    Function to generate the indices used during a for loop of n indices
    starting from 0 to K-1  (n nested for loops of K iterations). 
    Returns a list of K^n tuples of n elements. 
    the r-th tuple contain the number "r" written in base K.
    """
    iterator = []
    for r in range(K**n):
        r_in_base_K = np.base_repr(r, K)
        list_r_in_base_K = [0]*n
        for c in range(len(r_in_base_K)):
            list_r_in_base_K[n - len(r_in_base_K) + c] = int(r_in_base_K[c])
        iterator.append(np.array(list_r_in_base_K))
    return iterator
