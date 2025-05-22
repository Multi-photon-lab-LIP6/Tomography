from NestedForLoop import get_iterator
# from densitymatrix import DensityMatrix

# Measurement bases indexation :
# Comp, Diag, Circ -> 0, 1, 2

#Eigenstates indexation
# |+> , |-> -> 0,1

# Pauli operators indexation :
# I, X, Y, Z -> 0, 1, 2, 3

#%%

import os

import numpy as np

class XPCounts:
    """
    Class used to interact with the experimental numbers of event detected
    when simultaneously measuring the photons on different eigenvectors of
    tensor products of Pauli operators.
    Initialisation Arguments:
    - 'qbit_number' : number of qbits
    - 'counts_array' : is an array of shape (3**qbit_number,2**qbit_number)
    containing the numbers of of coincidence events for each
    Each line corresponds to a Pauli measurement basis
    Each column correspond to a eigenvector of the Pauli operator, on which
    we performed the measurement.
    - 'counts_2_emissions_array' : is an array of shape (3**qbit_number,2**qbit_number)
    containing the numbers of of coincidence events of the double emissions for each
    Each line corresponds to a Pauli measurement basis
    Each column correspond to a eigenvector of the Pauli operator, on which
    we performed the measurement.
    """

    def __init__(self,counts_array,qbit_number,counts_array_2_emissions=None,counts_array_4_emissions=None):
        self.counts_array = counts_array
        if counts_array_2_emissions is not None:
            self.counts_array_2_emissions = counts_array_2_emissions
        if counts_array_4_emissions is not None:
            self.counts_array_4_emissions = counts_array_4_emissions
        self.qbit_number = qbit_number # int(np.log2(np.shape(self.counts_array)[1]))

        self.base_2 = 2**np.linspace(0,self.qbit_number-1,self.qbit_number)
        self.base_3 = 3**np.linspace(0,self.qbit_number-1,self.qbit_number)

    def get_count_index(self,pauli_index,eigenvector_index):
        """
        Function that returns the number of coincidence events detected when
        the photons are measured on a certain product of eigenstates of a
        certain product of pauli operators, out of the experimental data
        contained in counts_array.
        Arguments : - 'pauli_index' is the index corresponding to the pauli
        measurement basis.
                    - 'eigenvector_index' is the index corresponding to the
        eigenstate of measurement.
        """
        return self.counts_array[pauli_index,eigenvector_index]

    def get_count(self,pauli_combination,eigenvector_combination):
        """
        Function that returns the number of coincidence events detected when
        the photons are measured on a certain product of eigenstates of a
        certain product of pauli operators, out of the experimental data
        contained in counts_array.
        Arguments : - 'pauli_combination' is an array of index corresponding
        to a product of pauli operator.
                    - 'eigenvector_index' is an array of index corresponding to
        the eigenstate of measurement
        """
        pauli_index = int(np.dot(self.base_3,pauli_combination))
        eigenvector_index = int(np.dot(self.base_2,eigenvector_combination))
        return self.counts_array[pauli_index,eigenvector_index]

    def get_total_counts_index(self,pauli_index):
        """
        Function to get the total number of coincidences of N-photon
        detection events, in a certain Pauli measurement basis.
        Arguments:  - `pauli_index` is the index corresponding to the pauli
        basis on which we need the total counts.
        """
        return np.sum(self.counts_array[pauli_index,:])

    def get_total_counts(self,pauli_combination):
        """
        Function to get the total number of coincidences of N-photon
        detection events, in a certain Pauli measurement basis.
        Arguments:  - `pauli_combination` is an array of index, each of which
        correponds to a pauli operator in the tensor product of operators.
        """
        pauli_index = int(np.dot(self.base_3, pauli_combination))
        return self.get_total_counts_index(pauli_index)

    @property
    def total_counts_array(self):
        """creates the atribute 'total_counts_array' for the object XPCounts.
        It's a (3**qbit_number,)-shaped array. The i-th elements of the array
        contains the total number of counts that were measure in the i-th pauli
        measurement basis.
        """
        total_counts_array = np.zeros(3**self.qbit_number)
        for pauli_index in range(3**self.qbit_number):
            total_counts_array[pauli_index] = self.get_total_counts_index(
                    pauli_index)
        return total_counts_array


    def get_xp_probas(self):
        """Returns a (3**qbit_number,2**qbit_number)-shaped array, similar to
        the array counts_array. Instead of containing the numbers of counts
        measured in each eigenvectors of each pauli operators, it contains the
        probabilities of the system to be in these eigenvectors."""
        return self.counts_array/np.resize(
                self.total_counts_array,(
                        2**self.qbit_number,3**self.qbit_number)).transpose()

    def correct_counts_with_channels_eff(self, channel_eff, double_emission_eff=None,four_emission_eff=None):

        for w in range(3**self.qbit_number):
            ### The ordering of the channel_eff matches with the covention: 0: HH; 1: HV; 2: VH; 3: VV(this changes depending on how we save data)
            ### The ordering of the channel_eff_2_emissinos matches with the covention: {123}, {124}, {134}, {234} (this changes depending on how we save data)
            ### If we change this order we need to do the same in set_raw_counts in efficiencies.py
            ### Correcting with the channel_eff
            # self.counts_array[w] /= channel_eff.astype(float)
            # if double_emission_eff is not None:
            #     self.counts_array_2_emissions[w] /= double_emission_eff.astype(float)
            # if four_emission_eff is not None:
            #     self.counts_array_4_emissions[w] /= four_emission_eff.astype(float)
            self.counts_array[w] = self.counts_array[w]/channel_eff.astype(float)
            if double_emission_eff is not None:
                self.counts_array_2_emissions[w] = self.counts_array_2_emissions[w]/double_emission_eff.astype(float)
            if four_emission_eff is not None:
                self.counts_array_4_emissions[w] = self.counts_array_4_emissions[w]/four_emission_eff.astype(float)
            
            '''
            Applying the correction of the double emission with respect of the order of the pseudo code : {123}, {124}, {134}, {234}
            {HH} = {13} = {13} - ({123} - {1234}) - ({134} - {1234}) - {1234}
            {HV} = {14} = {14} - {124} - {134};
            {VH} = {23} = {23} - {123} - {234};
            {VV} = {24} = {24} - {124} - {234};
            '''

    def correct_counts_with_double_emission(self):
        ### Subtracting the double emission counts from the coincidences
        for i in range(2**self.qbit_number):
            offset = np.min(self.double_emission_columns)
            doubles = np.sum([self.counts_array_2_emissions[j-offset][i]  for j in self.double_emission_columns[i][:]])
            self.counts_array[:,i] = self.counts_array[:,i]-doubles + self.counts_array_4_emissions[:,0]

        ### Counts need to be integers
        self.counts_array = np.round(self.counts_array)

    def correct_counts_with_double_emission_for_GHZ(self):

        for i in range(2**self.qbit_number):
                for j in range(3**self.qbit_number):
                    #triples = np.sum(self.counts_array_4_emissions[j,i*3:(i+1)*3]) 
                    doubles = np.sum(self.counts_array_2_emissions[j,i*4:(i+1)*4])
                    self.counts_array[j,i] = self.counts_array[j,i] - doubles

        self.counts_array = np.round(self.counts_array)

        

class TheoreticalCounts(XPCounts):
    """
    Derived class for holding and interacting with the simulated detection
    counts that would be recorded in a certain theoretical state.

    Initialisation Arguments:
    - `initial_state` : QuantumState object for the initial state, used to
            calcute the counts.
    - `total_counts`: is an array containing the total numbers of
    coincidences of N-photon detection events, in the differents measurement
    bases.
    """

    def __init__(self, initial_state,total_counts):
        # In the comments, N is the number of qbits.
        self.qbit_number = int(
            np.log2(np.shape(initial_state.density_representation(
                    ).matrix)[0]))
        self.counts_array = np.zeros((3**self.qbit_number,2**self.qbit_number))
        #definition of iterators. lists of tuples of indices, corresponding
        #to operators in tensors products
        self.pauli_iterator = get_iterator(3,self.qbit_number)
        self.eigenstate_iterator = get_iterator(2,self.qbit_number)

        self.total_counts = total_counts
        self.update_counts(initial_state)

    def update_counts(self, new_state):
        """
        Updates the array of counts for the a new state passed
        as argument.

        Arguments:
        - `new_state` : QuantumState object for the new state to update the
                expectation values with
        """
        for pauli_index in range(3**self.qbit_number):
            total_count = self.total_counts[pauli_index]
            for eigenvector_index in range(2**self.qbit_number):
                self.counts_array[pauli_index,eigenvector_index] = int(new_state.pauli_state_proba(
                                  self.pauli_iterator[pauli_index],self.eigenstate_iterator[eigenvector_index])*total_count)
        self.counts_array[self.counts_array<0] = 0
