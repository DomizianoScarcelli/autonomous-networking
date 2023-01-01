from src.routing_algorithms.BASE_routing import BASE_routing
from src.utilities import utilities as util
from src.utilities import config
from src.simulation.metrics import Metrics
import numpy as np

class QlearningStepwiseRouting(BASE_routing):

    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone, simulator)
        self.LEARNING_RATE = 0.8 #Learning rate
        self.DISCOUNT_FACTOR = 0.1 #Discount factor (it represents the importance of future rewards)
        self.BETA = 0.8 #Coefficient weight value (to change!)
        self.OMEGA = 0.8 #Used in the reward function (to change!)
        self.RMAX = 2 #Maximum reward value
        self.RMIN = -1 #Minimum reward value

        self.random = np.random.RandomState(self.simulator.seed) #it generates a random value to be used in epsilon greedy
        #self.link_qualities = {} #In order to calculate it, we should considered the packet transmission time and packet delivery ratio but we made a simplification considering only the distance between two drones.
        self.q_table = self.instantiate_table() #{drone_1: [drone_1, drone_2, ..., drone_n], .., drone_n: [drone_1, drone_2, ..., drone_n]}
        #self.link_stability = self.instantiate_table()
        self.old_state = None

    #Function which is called at each step and it is used to compute the reward for each drone. After a reward has been chosen, another function is called to update the qtable accordingly.
    def compute_reward(self, old_action):
        reward = None
        distance_cur_drone_depot = util.euclidean_distance(self.drone.coords, self.simulator.depot.coords)
        distance_old_drone_depot = util.euclidean_distance(old_action.coords, self.simulator.depot.coords)
        # If the drone is in the communication range of the depot, then we give maximum reward
        if distance_cur_drone_depot <= config.DEPOT_COMMUNICATION_RANGE:
            reward = self.RMAX
        elif distance_old_drone_depot > distance_cur_drone_depot: #if the drone chosen as the best action moves even further away from the depot, then we give minimum reward 
            reward = self.RMIN
        else:
            #Otherwise (if the drone, the chosen action, is moving towards the depot but it is not in its communication range)
            if self.drone.identifier in self.simulator.depot.nodes_table.nodes_list: #In this case the drone is linked with the depot and hop != None
                hop = self.simulator.depot.nodes_table.nodes_list[self.drone.identifier].hop_count
                reward = self.OMEGA*np.exp(1/hop) + (1-self.OMEGA)*self.drone.link_stabilities[old_action.identifier]
            else:
                reward = self.drone.link_stabilities[old_action.identifier]
            if reward > self.RMAX:
                self.RMAX = reward
            elif reward < self.RMIN:
                self.RMIX = reward
        if self.old_state != None: #If not in the first step (in the first step there is no old_state)
            self.update_qtable(self.old_state.identifier, self.drone.identifier, self.drone.identifier, reward)
    
    #This function returns the best relay to send packets
    def relay_selection(self):
        self.old_state = self.drone #Update old_state with current_state (useful in the reward function)
        if len(self.drone.neighbor_table.neighbors_list) == 0: #If there are no neighbors, keep the packets
            return self.drone
        else: #Otherwise, choose the action (the neighbor) with the highest QValue
            best_qvalue = None
            best_neighbor = None
            for neighbor in self.drone.neighbor_table.neighbors_list:
                neighbor = self.simulator.drones[neighbor]
                if  best_neighbor == None or self.q_table[self.drone.identifier][neighbor.identifier] > best_qvalue:
                    best_neighbor = neighbor
                    best_qvalue = self.q_table[self.drone.identifier][neighbor.identifier]
            return best_neighbor

    #Create the QTable
    def instantiate_table(self):
        table = {}
        for drone_index in range(self.simulator.n_drones):
            table[drone_index] = [0 for _ in range(self.simulator.n_drones+1)]
        return table

    #Update the qtable
    def update_qtable(self, state, next_state, action, reward):
        #Update the reward function with the same formula used on the paper
        self.q_table[state][action] = (1-self.LEARNING_RATE)*self.q_table[state][action] + self.LEARNING_RATE*(reward + (self.DISCOUNT_FACTOR * max(self.q_table[next_state])))
