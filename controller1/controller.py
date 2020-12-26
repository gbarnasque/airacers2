import interfaces as controller_template
import numpy
import csv
import math
import sys

from itertools import product
from typing import Tuple, List


class State(controller_template.State):
    def __init__(self, sensors: list):
        self.sensors = sensors
        self.old_sensors = sensors

    def compute_features(self) -> Tuple:
        """
        This function should take the raw sensor information of the car (see below) and compute useful features for selecting an action
        The car has the following sensors:

        self.sensors contains (in order):
            0 track_distance_left: 1-100
            1 track_distance_center: 1-100
            2 track_distance_right: 1-100
            3 on_track: 0 if off track, 1 if on normal track, 2 if on ice
            4 checkpoint_distance: 0-???
            5 car_velocity: 10-200
            6 enemy_distance: -1 or 0-???
            7 enemy_position_angle: -180 to 180
            8 enemy_detected: 0 or 1
            9 checkpoint: 0 or 1
           10 incoming_track: 1 if normal track, 2 if ice track or 0 if car is off track
           11 bomb_detected = 0 or 1
           12 bomb_distance = -1 or 0-???
           13 bomb_position_angle = -180 to 180
           
          (see the specification file/manual for more details)
        :return: A Tuple containing the features you defined
        """

        features = []
        features.append(self.sensors[0] - self.sensors[2]) #[-100,100]
        if self.sensors[4] > 400:
            self.sensors[4] = 400
        
        features.append(self.sensors[4] - self.old_sensors[4]) #[-400,400]
        features.append(self.sensors[5])

        return tuple(features)
        

    def discretize_features(self, features: Tuple) -> Tuple:
        """
        This function should map the (possibly continuous) features (calculated by compute features) and discretize them.
        :param features 
        :return: A tuple containing the discretized features
        """
        features = list(features)
        discretized_features = []

        feature_levels = self.discretization_levels()
        feature_range = []
        
        for f in range(len(features)):
            feature_offset = 0

            if f == 0:
                feature_range = [-100, 100]
                feature_offset = 100
            elif f == 1:
                feature_range = [0, 0]
                feature_offset = 0
            elif f == 2:
                feature_range = [10, 200]
                feature_offset = -10

            cut_point = ((feature_range[1] + feature_offset)+(feature_range[0] + feature_offset))/feature_levels[f]
            for i in range(feature_levels[f]):
                if features[f] <= ((i+1)*cut_point - feature_offset):
                    discretized_features.append(i)
                    break

        return tuple(discretized_features)

    @staticmethod
    def discretization_levels() -> Tuple:
        """
        This function should return a vector specifying how many discretization levels to use for each state feature.
        :return: A tuple containing the discretization levels of each feature
        """

        levels = []
        levels.append(5)
        levels.append(2)
        levels.append(4)

        return tuple(levels)

    @staticmethod
    def enumerate_all_possible_states() -> List:
        """
        Handy function that generates a list with all possible states of the system.
        :return: List with all possible states
        """
        levels = State.discretization_levels()
        levels_possibilities = [(j for j in range(i)) for i in levels]
        return [i for i in product(*levels_possibilities)]


class QTable(controller_template.QTable):
    def __init__(self):
        """
        This class is used to create/load/store your Q-table. To store values we strongly recommend the use of a Python
        dictionary.
        """
        self.q_table  = {}
        
        for state in State.enumerate_all_possible_states():
            state_id = State.get_state_id(state)
            #lst = numpy.random.randint(-100, 100, 5)
            lst = [0,0,0,0,0]
            self.q_table[state_id] = lst

    def get_q_value(self, key: State, action: int) -> float:
        """
        Used to securely access the values within this q-table
        :param key: a State object 
        :param action: an action
        :return: The Q-value associated with the given state/action pair
        """
        action = action - 1
        
        return self.q_table[State.get_state_id(key.get_current_state())][action]
        

    def set_q_value(self, key: State, action: int, new_q_value: float) -> None:
        """
        Used to securely set the values within this q-table
        :param key: a State object 
        :param action: an action
        :param new_q_value: the new Q-value to associate with the specified state/action pair
        :return: 
        """
        action = action - 1 
        self.q_table[State.get_state_id(key.get_current_state())][action] = new_q_value

    @staticmethod
    def load(path: str) -> "QTable":
        """
        This method should load a Q-table from the specified file and return a corresponding QTable object
        :param path: path to file
        :return: a QTable object
        """
        Q_Table = QTable()

        file = open(path, 'r', newline='')
        reader = csv.reader(file, delimiter=';')
        for row in reader:
            state_id = int(row[0])
            row.remove(row[0])
            # Converte lista de strings para lista de float
            Q_Table.q_table[state_id] = [float(i) for i in row]
    
        return Q_Table

    def save(self, path: str, *args) -> None:
        """
        This method must save this QTable to disk in the file file specified by 'path'
        :param path: 
        :param args: Any optional args you may find relevant; beware that they are optional and the function must work
                     properly without them.
        """
        file = open(path, 'w', newline='')
        writer = csv.writer(file, delimiter=';')
        for state in State.enumerate_all_possible_states():
            state_id = State.get_state_id(state)
            row = []
            row.append(state_id)
            for q_value in self.q_table[state_id]:
                row.append(q_value)
            writer.writerow(row)
        

class Controller(controller_template.Controller):
    def __init__(self, q_table_path: str):
        if q_table_path is None:
            self.q_table = QTable()
        else:
            self.q_table = QTable.load(q_table_path)
        self.magic_number = 75

    def update_q(self, new_state: State, old_state: State, action: int, reward: float, end_of_race: bool) -> None:
        """
        This method is called by the learn() method in simulator.Simulation() to update your Q-table after each action is taken
        :param new_state: The state the car just entered
        :param old_state: The state the car just left
        :param action: the action the car performed to get to new_state
        :param reward: the reward the car received for getting to new_state  
        :param end_of_race: boolean indicating if a race timeout was reached
        """

        alpha = 0.2
        gamma = 0.9

        q_values_new_state = [] 
        for a in range(1,6):
            q_values_new_state.append(self.q_table.get_q_value(new_state, a))

        max_q_new_state = max(q_values_new_state)
        
        new_q_value = ((1-alpha)*self.q_table.get_q_value(old_state, action)) + (alpha*(reward + gamma*max_q_new_state))

        self.q_table.set_q_value(old_state, action, new_q_value)

    def compute_reward(self, new_state: State, old_state: State, action: int, n_steps: int,
                       end_of_race: bool) -> float:
        """
        This method is called by the learn() method in simulator.Simulation() to calculate the reward to be given to the agent
        :param new_state: The state the car just entered
        :param old_state: The state the car just left
        :param action: the action the car performed to get in new_state
        :param n_steps: number of steps the car has taken so far in the current race
        :param end_of_race: boolean indicating if a race timeout was reached
        :return: The reward to be given to the agent
        """
        reward = -0.1
        
        #try to get the car in the middle
        if new_state.get_current_state()[0] >= 1 and new_state.get_current_state()[0] <= 3:
            reward += 7
        else:
            reward += -0.5

        # Check if the car has made any progress in the track 
        if new_state.get_current_state()[1] == 0:
            reward += 5
        else:
            reward += -0.5

        # Maintain a "constant" velocity, not so fast, not so flow 
        if new_state.get_current_state()[2] == 2:
            reward += 3
        else:
            reward += -0.5

        return reward

    def take_action(self, new_state: State, episode_number: int) -> int:
        """
        Decides which action the car must execute based on its Q-Table and on its exploration policy
        :param new_state: The current state of the car 
        :param episode_number: current episode/race during the training period
        :return: The action the car chooses to execute
        1 - Right
        2 - Left
        3 - Accelerate
        4 - Brake
        5 - Nothing
        """

        action = 5
        
        guloso = False
        egreedy = False

        # Estratégia gulosa
        if guloso:
            q_values = []
            for a in range(1,6):
                q_values.append(self.q_table.get_q_value(new_state, a))
            action = q_values.index(max(q_values)) + 1
            if egreedy:
                probability = self.calculate_epsilon_greedy(episode_number)
                if numpy.random.ranf() < probability:
                    action = numpy.random.randint(1,6)
        # Exploração de Bolztamnn
        else:
            probabilities_of_actions = self.calculate_Boltzmann_Exploration(new_state, episode_number)
            action = numpy.random.choice([1,2,3,4,5], p=probabilities_of_actions)

        return action

    def calculate_epsilon_greedy(self, episode_number: int) -> float:
        epsilon = self.magic_number/100
        if episode_number != 0:
            epsilon = (self.magic_number/episode_number)/100
        return epsilon

    def calculate_Boltzmann_Exploration(self, new_state: State, episode_number: int) -> List:
        
        probabilities = []
        q_values = []
        for a in range(1,6):
            q_values.append(self.q_table.get_q_value(new_state, a))

        T = self.magic_number
        if episode_number != 0:
            T = self.magic_number/episode_number

        #absolute the exponentials otherwise the probabilities would be inverse, lower numbers would have a higher probability than bigger ones  
        exp_q_values = numpy.absolute(numpy.exp(numpy.true_divide(q_values,abs(T))))
        sum_of_exp_q_values = sum(exp_q_values)

        for i in range(5):
            probabilities.append(exp_q_values[i]/sum_of_exp_q_values)

        return probabilities