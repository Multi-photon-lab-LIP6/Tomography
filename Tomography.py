# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 14:41:58 2018

@author: Simon Neves, Robert Booth
"""

#from MeasurementData import *
from pauliexpectations import *
from projectorcounts import *
from QuantumStates import *
from QuantumStates.TriDiagonalMatrix import TriDiagonalMatrix
from densitymatrix import DensityMatrix, wp_rotation
from player import Player
import errors
#%%

import heapq
import os
import glob
from pathlib import Path
from itertools import combinations
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from scipy.optimize import least_squares
import scipy.linalg as lin
from tqdm import tqdm
import functools as ft

class Tomography:
    """
    'Tomography' is the class including all function needed to perform a
    complete tomography of a multi-qubits state. It provides the interface for
    running a tomography on experimental data. This class use a direct
    inversion to determine the density matrix - for more specific optimisation
    procedures, see the sub-class MaximumLikelihoodTomography.
    """

    def __init__(self,qbit_number,xp_counts,working_dir):
        """
        Initialisation of the tomography.
        - 'qbit_number' : number of qubits
        - 'xp_counts_array': array of experimental counts that can be passed
        as an argument to initialize an object of class XPCounts
        - 'working_dir' : directory to save and load data
        """
        self.qbit_number = qbit_number
        self.xp_counts = xp_counts

        self.working_dir = Path(working_dir)
        os.chdir(self.working_dir)
        print('Direct Inversion Tomography Initialized')

    def direct_inversion(self):
        """
        Calculates the density matrix of the system, using direct
        inversion of the data (experimental or simulated).

        WARNING: The resulting density matrix might not be physical,
        meaning that it is not necessarily positive semi-definite.
        """
        pauli_operators_array = np.load(
                self.working_dir / 'SavedVariables'/ f'Pauli{self.qbit_number}.npy')
        xp_pauli_expectations = ExperimentalPauliExpectations(
                self.qbit_number,self.xp_counts).pauli_expectations_array
        return np.sum(np.resize(xp_pauli_expectations,(
                2**self.qbit_number,2**self.qbit_number,4**self.qbit_number
                )).transpose((2,0,1))*pauli_operators_array,axis = 0)

    def run_pseudo_tomo(self):
        """
        Runs the simplest tomography possible by performing a direct inversion
        of the experimental data.

        WARNING: The resulting density matrix might not be physical, meaning
        that it is not necessarily positive semi-definite.
        """
        #Defining the "pseudo" density matrix calculated with direct inversion
        #method, from experimental data.
        self.pseudo_state = self.direct_inversion()
        print("Tomography terminated.")

class LRETomography():
    """
    'LRETomography' is the class including all function needed to perform a
    tomography of a multi-qubits state, via Linear Regression Estimation.
    It provides the interface for running a tomography on experimental data.
    This class uses the Linear Regression Estimation method and fast Maximum
    Likelihood estimation.
    """

    def __init__(self, qbit_number,xp_counts,xp_counts_2_emissions=None,xp_counts_4_emissions=None,working_dir=Path(__file__).parent):
        """
        Initialisation of the tomography.
        - 'qbit_number' : number of qubits
        - 'xp_counts_array': array of experimental counts that can be passed
        as an argument to initialize an object of class XPCounts
        xp_counts_2-emission_array': array of experimental counts of the double emissions that can be passed
        as an argument to initialize an object of class XPCounts
        - 'working_dir' : directory to save and load data
        """
        self.qbit_number = qbit_number
        self.xp_counts = XPCounts(xp_counts,self.qbit_number,xp_counts_2_emissions,xp_counts_4_emissions)
        self.working_dir = Path(working_dir)
        self.quantum_state = QuantumState(
            np.eye(2**self.qbit_number) / 2**self.qbit_number)
        os.chdir(self.working_dir)

    def get_theta_LS(self):
        """Function to get the vector of coordinates of the density matrix
        in the basis of Pauli operators."""
        X = np.load(self.working_dir / 'SavedVariables' / f'X_matrix{self.qbit_number}.npy')
        invXtX_matrix = np.load(
                self.working_dir / 'SavedVariables' / f'invXtX_matrix{self.qbit_number}.npy')
        Y = self.xp_counts.get_xp_probas() - 1/2**self.qbit_number
        XtY = np.sum(X*np.resize(Y,(4**self.qbit_number - 1,
                                    3**self.qbit_number,2**self.qbit_number
                                    )).transpose((1,2,0)),axis = (0,1))
        return np.dot(invXtX_matrix,XtY)


    def LREstate(self):
        """Function that calculates an approximation of the density matrix,
        using the linear regression estimation method"""
        pauli_operators_array = np.load(
                self.working_dir / 'SavedVariables'/ f'Pauli{self.qbit_number}.npy')
        theta_LS = self.get_theta_LS()
        #print(np.round((np.eye(2**self.qbit_number)/2**self.qbit_number + np.sum(
          #      np.resize(theta_LS,(2**self.qbit_number,2**self.qbit_number,
           #             4**self.qbit_number-1)).transpose((
            #                    2,0,1))*pauli_operators_array[1:,:,:],axis = 0)),3))
        return np.eye(2**self.qbit_number)/2**self.qbit_number + np.sum(
                np.resize(theta_LS,(2**self.qbit_number,2**self.qbit_number,
                        4**self.qbit_number-1)).transpose((
                                2,0,1))*pauli_operators_array[1:,:,:],axis = 0)

    def run_pseudo_tomo(self):
        """
        Runs the pseudo tomography to get generate a pseudo_state out of the
        experimental data, using the LRE method.

        WARNING: The resulting density matrix might not be physical, meaning
        that it is not necessarily positive semi-definite.
        """
        #Defining the "pseudo" density matrix calculated with LRE
        #method, from experimental data.
        self.pseudo_state = self.LREstate()

    def run(self, correct_eff=None, correct_double_emission_eff=None, correct_double_emission=None, four_emission_eff = None, GHZ = False, print_nc=False):
        """
        Runs the pseudo tomography to get generate a state out of the
        experimental data, using the LRE method and fast maximum likelihood
        estimation."""

        if correct_double_emission is not None:
            self.xp_counts.double_emission_columns=correct_double_emission
            self.xp_counts.correct_counts_with_double_emission()
        
        if GHZ == True :
            self.xp_counts.correct_counts_with_double_emission_for_GHZ()

        if correct_eff is not None:
            self.xp_counts.correct_counts_with_channels_eff(correct_eff, correct_double_emission_eff,four_emission_eff)
            #print("I'm correcting efficiencies")

        ### Sanity check: This prints the normalized number of counts for each measurement basis
        if print_nc:
            BasesO=['DD','DL','DH','LD','LL','LH', 'HD','HL', 'HH']
            for w in range(3**self.qbit_number):
                aux=0
                aux=np.max(np.sum(self.xp_counts.counts_array, 1))
                print('Output basis: ', BasesO[w], '; Sum of counts normalized: ', np.sum(self.xp_counts.counts_array[w])/aux)

        self.run_pseudo_tomo()
        self.quantum_state.set_density_matrix(self.pseudo_state)
        self.state=DensityMatrix(self.quantum_state.get_density_matrix())

    def fock(self):
        self.state.fock_basis(self.qbit_number)


    def permutation_elements(self, elements):
        permuted=[]
        
        el_num=len(elements)
        for m in range(el_num):
            if self.qbit_number==1:
                t=m
                permuted.append(f"{elements[m]}")
            elif self.qbit_number==2:
                for n in range(el_num):
                    t=n%2+2*(m%2)
                    permuted.append(f"{elements[m]}{elements[n]}")
            elif self.qbit_number==3:
                for n in range(el_num):
                    for o in range(el_num):
                        t=o%2+2*(n%2)+4*(m%2)
                        permuted.append(f"{elements[m]}{elements[n]}{elements[o]}")    
            elif self.qbit_number==4:
                for n in range(el_num):
                    for o in range(el_num):
                        for p in range(el_num):
                            t=p%2+2*(o%2)+4*(n%2)+8*(m%2)
                            permuted.append(f"{elements[m]}{elements[n]}{elements[o]}{elements[p]}")
                            
        return permuted
    
    def players_init(self, players):
        players_list=[]
        self.player_1=Player(players[0])
        players_list.append(self.player_1)
        if self.qbit_number>1:
            self.player_2=Player(players[1])
            players_list.append(self.player_2)
            if self.qbit_number>2:
                self.player_3=Player(players[2])
                players_list.append(self.player_3)
                if self.qbit_number>3:
                    self.player_4=Player(players[3])
                    players_list.append(self.player_4)
        return players_list


    def simulate_new_counts_with_uncertainties(self, players):
        ### lines are the 3 measurement basis and columns are the 2 possible outcomes for each measurement basis 
        
        lines, columns = 3**self.qbit_number,2**self.qbit_number

        self.xp_counts_err=np.zeros((3**self.qbit_number,2**self.qbit_number), dtype=int)
        self.players_list=self.players_init(players)

        if self.qbit_number == 2:
           self.xp_counts_err_2_emissions=np.zeros((3**self.qbit_number,2**self.qbit_number), dtype=int)
           self.players_list_2_emissions=self.players_init(players)

        proj=self.permutation_elements(["h","v"])
        proj_basis=self.permutation_elements(["x","y","z"])

        """
            we sample a value within a normal distribution for each waveplate we are using,
            given a mean value that is definied in the dictionaries in errors.py
        """
        shift_hwp=[]
        shift_qwp=[]
        for j in range(self.qbit_number):
            shift_hwp.append(np.random.normal(scale=self.players_list[j].sigma_wp[0], size=None))
            shift_qwp.append(np.random.normal(scale=self.players_list[j].sigma_wp[1], size=None))

        for k in range(lines):
            N_total=np.sum(self.xp_counts.counts_array[k])
            if self.qbit_number == 2:
               N_total_2_emissions=np.sum(self.xp_counts.counts_array_2_emissions[k])
            for l in range(columns):

                angle_hwp=[]
                angle_qwp=[]
                r=[]
                projectors=[]
                ### j defines the player
                for j in range(self.qbit_number):

                    angle_hwp.append(errors.HWP_DICT[proj_basis[k][j]] + shift_hwp[j])
                    angle_qwp.append(errors.QWP_DICT[proj_basis[k][j]] + shift_qwp[j])

                    """
                    r is the rotation matrix that arya's (cersei's) qubit goes through
                    before being measured in {V,H}
                    - the angle of rotation is determined based on the uncertainty of our waveplates
                    (determined the in lines above)
                    
                    when we do the tensor product of both (MB_change) and apply it to our state's density
                    matrix (dm_sim_WP), we are performing a basis rotation before the projection in {V, H}
                    (proj_basis_array)
                    """ 
                    
                    r.append(wp_rotation(angle_qwp[j],np.pi/2)@wp_rotation(angle_hwp[j],np.pi))

                    projectors.append(errors.PROJECTORS[proj[l][j]])

                MB_change=ft.reduce(np.kron, r)

                dm_sim_WP=MB_change@self.state.state@np.transpose(np.conjugate(MB_change))
                ###proj[x][y] x and y refer to the output port of the PBS and to the player, respectively
                proj_basis_array=ft.reduce(np.kron, projectors)
                """
                next we need to calculate the probability (p) of the state collapsing into the state represented
                by the projector

                with that we can generate a new xp_counts matrix (xp_counts_err), with each entry being a sample
                of the poissonian distibution with mean value (lam=p*N_total)
                """
                p=proj_basis_array@dm_sim_WP@np.transpose(np.conjugate(proj_basis_array))

                if np.imag(p)<1e-13:
                    self.xp_counts_err[k][l]=np.random.poisson(lam=np.real(p)*N_total)
                    if self.qbit_number == 2:
                       self.xp_counts_err_2_emissions[k][l]=np.random.poisson(lam=np.real(p)*N_total_2_emissions)
                else:
                    print('You are getting complex probabilities')
        if self.qbit_number == 2:
           return self.xp_counts_err + self.xp_counts_err_2_emissions
        else:
            return self.xp_counts_err
        

    def calculate_dm_from_simulated_counts(self, players):

        simulated_counts=self.simulate_new_counts_with_uncertainties(players)

        statetomo_err=LRETomography(int(self.qbit_number), simulated_counts, str(Path(__file__).parent))
        statetomo_err.run()
        error_simulation_dm=statetomo_err

        return(error_simulation_dm)
    

    def calculate_fidelity_error(self, players, error_runs, opt, target, optimization=False, bounds=None, penalty=None):
        self.players=players
        self.error_runs=error_runs

        self.error_simulation_dm=[]

        fidelity_sim=np.zeros((error_runs), dtype=float)

        self.Us=[]

        print("Simulating new states considering the uncertainties")
        for i in range(error_runs):
            self.error_simulation_dm.append(self.calculate_dm_from_simulated_counts(self.players))

        if optimization is True:
            print("Optimizing the fidelity between input and target up to a unitary")

            for i in range(error_runs):
                result=opt.optimize(int(self.qbit_number),self.error_simulation_dm[i].state, target, bounds=bounds, penalty=penalty)

                self.Us.append(result.u)
                fidelity_sim[i]=result.minimum()

        else:
            for i in range(error_runs):
                fidelity_sim[i]=self.error_simulation_dm[i].fidelity(target)
            
            ### I should make sure the fidelity is acutally real and not complex
            # fidelity_sim[i]=np.real(fid(dm_sim[i], target))

        """
        Should make sure that np.std is not assuming a normal distribution for the fidelities
        Otherwise I should fit a truncated normal distribution
        self.mu, self.std, skew, kurt = truncnorm.fit(fidelity_sim, 0 , 1)
        Also hould divide for the sqrt(samples)?
        """
        self.fidelity_mu = np.mean(fidelity_sim)
        self.fidelity_std = np.std(fidelity_sim)



    """
    Same as funciton above but for the case where the target state is also experimental
    Should add an option to optimize the fidelity with unitaries
    This is only to be used after haveing ran the calculate_fidelity_error() before
    """
    def calculate_fidelity_error_between_2_experimental_matrices(self, error_runs, players, target, apply_unitary_to_input=False, simulate_input_matrices=False):
    
        fidelity_sim=np.zeros((error_runs**2), dtype=float)
        target_sim=[]

        if simulate_input_matrices is True:
            print("Simulating input state matrices. If you've used calculate_fidelity_error() before, set simulate_input_matrices=False.")
            self.players=players
            self.error_runs=error_runs
            for j in range(self.error_runs):
                self.error_simulation_dm=[]
                self.error_simulation_dm.append(self.calculate_dm_from_simulated_counts(self.players))

        print("Simulating output matrices")
        for i in range(self.error_runs):
            target_sim.append(target.calculate_dm_from_simulated_counts(self.players))

        counter=0
        if apply_unitary_to_input is True:
            for l in range(self.error_runs):
                for m in range(self.error_runs):
                    fidelity_sim[counter]=self.error_simulation_dm[l].state.apply_unitary(self.Us[l]).fidelity(target_sim[m].state.state)
                    counter+=1

        else:
            for l in range(self.error_runs):
                for m in range(self.error_runs):
                    fidelity_sim[counter]=self.error_simulation_dm[l].state.fidelity(target_sim[m].state.state)
                    counter+=1

        """
        Should make sure that np.std is not assuming a normal distribution for the fidelities
        Otherwise I should fit a truncated normal distribution
        self.mu, self.std, skew, kurt = truncnorm.fit(fidelity_sim, 0 , 1)
        Also hould divide for the sqrt(samples)?
        """
        self.fidelity_2_experimental_dms_mu = np.mean(fidelity_sim)
        self.fidelity_2_experimental_dms_std = np.std(fidelity_sim)

class GeneticTomography(LRETomography):
    """
    Class for performing a tomography using a genetic algorithm optimisation.
    """

    def __init__(self, qbit_number, xp_counts_array, working_dir):
        """
        Initialisation of the tomography.
        'qbit_number' : number of qubits
        - 'xp_counts_array': array of experimental counts that can be passed
        as an argument to initialize an object of class XPCounts
        - 'working_dir' : working directory
        """
        self.qbit_number = qbit_number
        self.xp_counts = XPCounts(xp_counts_array)
        self.working_dir = working_dir
        #Mutable structure to save and interact with the different quantum
        #states generated during the algorithm. We change the quantum state by
        #setting either its denstiy matrix, or its t vector.
        self.quantum_state = QuantumState(
            np.eye(2**self.qbit_number) / 2**self.qbit_number)
        os.chdir(self.working_dir)
        print('Tomography initialized')

    def log_likelihood(self):
        """
        Returns the log likelihood for the current state in the tomography. We
        aim at maximizing this function.
        """
        #Updating the counts calculated from self.quantum_state
        self.theo_counts.update_counts(self.quantum_state)
        return -np.sum(np.power(np.real((
                self.theo_counts.counts_array - self.xp_counts.counts_array
                ) / np.sqrt(2 * np.abs(self.theo_counts.counts_array))),2))

    def crossover(self, parents):
        """
        Function that generate a new density operator, out of 2 parents density
        operators. We use the representation of density operators with vectors
        t, so that any output vector t represents a physical density matrix.
        The new vector is taking some mutations.

        Arguments : 'parents' is a couple of 2 parents vectors,
        representing a physical density operator.
        """

        #The new vector will take the same number of coefficient from its
        #mother and its father. We select these coefficients randomly.
        vect_select = np.random.permutation([0] * (
            (4**self.qbit_number) // 2) + [1] * ((4**self.qbit_number) // 2))

        # Cross-over
        vect_t_offspring = vect_select * parents[0][1] + (
            1 - vect_select) * parents[1][1]

        #Chosing randomly the amplitued of mutations.
        mutation_amount = 1/(np.random.uniform(1/(self.low_mutation_amount),
                                            1/(self.high_mutation_amount)))
        # Apply mutations
        vect_t_offspring += 2 * mutation_amount * (
            np.random.rand(4**self.qbit_number) - 0.5)

        return vect_t_offspring

    def update_population(self):
        """
        This function updates the population of density matrices, according to
        the rules of our genetic algorithm.
        It selects the best density matrices, throw away the others, and
        generate new offspring matrices from couples of remaining matrices.
        """
        ###############
        ## SELECTION ##
        ###############
        #Updating current population. Status : contains n_keep future parents.
        self.population = heapq.nlargest(self.n_keep, self.population)

        ##########################
        ## CROSSOVER + MUTATION ##
        ##########################
        for parents in combinations(self.population, 2):
            offspring_vector = self.crossover(parents)
            #Changing the state by setting the vector
            self.quantum_state.set_vector(offspring_vector)
            #Updating current population. Status : contains Children + Parents
            self.population.append([self.log_likelihood(), offspring_vector])

    def run(self,
            n_generation,
            n_population,
            low_mutation_amount,
            high_mutation_amount,
            verbose=True,test=False):
        """
        Runs the maximum likelihood tomography using a genetic algorithm.
        'n_generation' : number of updates of the population
        'n_population' : number of density operators in the population.
        'low_mutation_amount' and 'high_mutation_amount' : amplitudes of
        the noise applied to simulate mutation when generating new matrices.
        At each mutation step, a mutation amount is chosen randomly between
        these two mutation amounts.

        'verbose' : boolean. True if the function plots the progression of the
        algorithm, False if it shows nothing.
        """
        ####################
        ## INITIALISATION ##
        ####################
        start_time = time.time()
        self.n_generation = n_generation
        self.n_population = n_population
        self.low_mutation_amount = low_mutation_amount
        self.high_mutation_amount = high_mutation_amount

        #number of density matrices kept at each selection steps
        self.n_keep = int(np.ceil(0.5 * (np.sqrt(1 + 8 * n_population) - 1)))

        # `population` is a list of pairs (likelihood, t_vector) to use in the
        # genetic optimisation. It represents the current population.
        self.population = []

        # List to keep track of the best likelihood per generation.
        self.best_log_likelihood_per_generation = []

        # Initial state : direct inversion matrix
        self.run_pseudo_tomo()
        init_matrix = (1-10**(-10))*nearestPD(self.pseudo_state
                      )+((10**(-10))/2**(self.qbit_number))*np.eye(2**self.qbit_number)
        self.quantum_state.set_density_matrix(init_matrix)
        init_vector = self.quantum_state.get_vector()
        if test:
            elapsed_time = [start_time-time.time()]
            vectors = [init_vector]
        # Initialises the simulated counts object.
        self.theo_counts = TheoreticalCounts(
            self.quantum_state, self.xp_counts.total_counts_array)

        # Initial population
        for pop_k in range(self.n_population):
            self.quantum_state.set_vector(
                init_vector + 2 * self.low_mutation_amount *
                (np.random.rand(4**self.qbit_number) - 0.5))

            self.population.append(
                [self.log_likelihood(),
                 self.quantum_state.get_vector()])
        if verbose: print('Population Initialized')

        # Plotting the evolution of the likelihood
        if verbose: self.initialise_plot_log_likelihood()

        #############
        ## RUNTIME ##
        #############
        generation_iterator = tqdm(range(
                self.n_generation)) if verbose else range(self.n_generation)
        for k in generation_iterator:
            #Selection + Cross-over + Mutation
            self.update_population()
            #Finding best state to follow the evolution of the likelihood
            self.best_state = max(self.population)
            self.best_log_likelihood_per_generation.append(self.best_state[0])
            if (k + 1) % 10 == 0 and verbose:
                self.update_plot_log_likelihood(k)
            if test:
                elapsed_time.append(start_time-time.time())
                vectors.append(self.best_state[1])
        ############
        ## OUTPUT ##
        ############
        #best_state_index = self.population.index(self.best_state)
        self.quantum_state.set_vector(self.best_state[1])
        self.quantum_state.log_likelihood = self.best_state[0]
        if verbose:
            print("End of tomography. Best log likelihood : " +
                  str(self.quantum_state.log_likelihood))
        if test:
            return (elapsed_time,vectors)

    ########################################################################
    ## Functions to plot the evolution of likelihood during the algorithm ##
    ########################################################################

    def initialise_plot_log_likelihood(self):
        """
        Creating the plot of the likelihood.
        """
        self.fig = plt.figure(figsize=(9, 4))
        self.ax = self.fig.add_subplot(111)
        self.graph, = self.ax.plot([], [])
        plt.title(
            "Evolution of the log-likelihood throughout the optimisation.",
            fontsize=16)
        plt.xlabel("Generation", fontsize=16)
        plt.ylabel("Ln(Likelihood)", fontsize=16)
        plt.xlim((0, self.n_generation))
        plt.ylim(max(self.population)[0], 0)
        self.ax.tick_params(labelsize=14)
        plt.tight_layout()
        plt.show()

    def update_plot_log_likelihood(self, generation):
        """
        Updating the plot of the likelihood throughout the optimisation.
        """
        self.ax.plot(
            range(1 + generation),
            self.best_log_likelihood_per_generation,
            color='r')
        self.fig.canvas.draw()

#    def __init__(self, densitymatrix, working_dir):
#        self.densitymatrix=densitymatrix