import numpy as np


class HiddenMarkovModel:
    def __init__(self, path_list, distances, backwards_ratio = .3, distance_coefficient = .1, path_distance_coefficient = 1.3):
        # for path in path_list:
        #     for i, element in enumerate(path):
        #         if 'ch_' in element or 'Channel' in element or 'version' in element:
        #             path[i] = ''

        # Parameters to be optimized
        # backwards_ratio = .25
        # distance_coefficient = .5
        # path_distance_coefficient = 1
        pl = []
        for path in path_list:
            temp = []
            for element in path:
                if element.upper() == element and element != '':
                    temp.append(element)
            pl.append(temp)
        path_list = pl

        self.states = set()
        for path in path_list:
            for state in path:
                if state != '':
                    self.states.add(state)
        self.observations = self.states

        self.states = np.array(sorted(self.states))
        self.number_of_states = len(self.states)
        self.observations = self.states

        self.encode_observation_dictionary = {value: index for index, value in enumerate(self.observations)}

        initial_freq = np.zeros(len(self.states))
        for path in path_list:
            for state in path:
                if state != '':
                    state_index = np.where(self.states==state)[0][0]
                    initial_freq[state_index] += 1
        initial_freq_sum = np.sum(initial_freq)
        self.initial_prob = np.array([x/initial_freq_sum for x in initial_freq])

        self.transition_prob = np.zeros(shape=(len(self.states), len(self.states)))
        transition_frequency = np.zeros(shape=(len(self.states), len(self.states)))
        for path_index in range(len(path_list)):
            path = [x for x in path_list[path_index] if x!='']
            path_length = len(path)
            for i in range(path_length):
                for j in range(i+1, path_length):
                    left_state_index = np.where(self.states==path[i])[0][0]
                    right_state_index = np.where(self.states==path[j])[0][0]
                    # We create add a frequency for each x -> y. if it's a direct connection += 1/1
                    # if with one middle node 1/2 and so on.
                    # then the frequency is weighted with the distance of the path to the query path
                    transition_frequency[left_state_index][right_state_index] += (1/((j-i)*path_distance_coefficient)) * (1-distances[path_index])*distance_coefficient

                for j in range(0, i):
                    left_state_index = np.where(self.states==path[j])[0][0]
                    right_state_index = np.where(self.states==path[i])[0][0]
                    # We create add a frequency for each y -> x. if it's a direct connection += 1/1
                    # if with one middle node 1/2 and so on.
                    # then the frequency is weighted with the distance of the path to the query path.
                    # We also lower the weight of this one as it happens before x and not directly after
                    transition_frequency[right_state_index][left_state_index] += (1/(i-j)*path_distance_coefficient)* backwards_ratio * (1-distances[path_index])*distance_coefficient

        epsilon = 1e-6
        transition_frequency += epsilon
        self.transition_prob = transition_frequency / transition_frequency.sum(axis=1, keepdims=True)
        self.emission_prob = np.zeros(shape=(len(self.states), len(self.observations)))
        for i in range(len(self.states)):
            for j in range(len(self.observations)):
                if i == j:
                    self.emission_prob[i][j] = 1

    def encode_observation(self, observations):
        encoded_list = []
        n_observations = len(self.observations)
        for observation in observations:
            if observation in self.encode_observation_dictionary:
                current_observation = np.zeros(n_observations)
                current_observation[self.encode_observation_dictionary[observation]] = 1
                encoded_list.append(current_observation)
                # encoded_list.append([self.encode_observation_dictionary[observation]])
        return encoded_list

    def observations_to_stateid(self, observationslist):
        results = []
        states= []
        for observations in observationslist:
            for observation in observations:
                if observation in self.encode_observation_dictionary:
                    observation_id = self.encode_observation_dictionary[observation]
                    results.append([observation_id])
                    states.append(observation)
        return results, states

    def bayesian_optimziation(self, function, parameter):
        print('Bayesian Optimization')
