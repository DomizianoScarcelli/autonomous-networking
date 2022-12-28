from src.routing_algorithms.BASE_routing import BASE_routing
from src.utilities import utilities as util
from src.utilities import config
from src.simulation.metrics import Metrics
import numpy as np
import math

class QlearningStepwiseRouting(BASE_routing):

    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone, simulator)
        self.LEARNING_RATE = 0.8 #alpha (the learning rate)
        self.DISCOUNT_FACTOR = 0.1 #gamma (it represents the importance of future rewards)
        self.EPSILON = 0.4 #epsilon (it is used in episolon greedy)
        self.BETA = 0.1 #Coefficient weight value (to change!)
        self.OMEGA = 0.1 #Used in the reward function
        self.RMAX = 2
        self.RMIN = 0

        self.random = np.random.RandomState(self.simulator.seed) #it generates a random value to be used in epsilon greedy
        self.link_qualities = {} #In order to calculate it, we considered the packet transmission time and packet delivery ratio. To calculate this, the Window Mean with Exponentially Weighted Moving Average (WMEWMA) method[9] was used. Note: for the moment I consider the distance between node.
        self.q_table = {} #{drone_1: [drone_1, drone_2, ..., drone_n], .., drone_n: [drone_1, drone_2, ..., drone_n]}
        self.link_stability = {}
        self.old_state = None

        #print(self.link_quality)

    def feedback(self, drone, id_event, delay, outcome):
        """
        Feedback returned when the packet arrives at the depot or
        Expire. This function have to be implemented in RL-based protocols ONLY
        @param drone: The drone that holds the packet
        @param id_event: The Event id
        @param delay: packet delay
        @param outcome: -1 or 1 (read below)
        @return:
        """

    #TO CHANGE!
    def compute_reward(self, drone):
        reward = None
        distance_cur_drone_depot = util.euclidean_distance(drone.coords, self.simulator.depot.coords)
        distance_old_drone_depot = util.euclidean_distance(drone.coords, self.simulator.depot.coords)
        # If the drone is in the communication range of the depot and it has packets to deliver, then we give maximum reward
        if distance_cur_drone_depot and drone.buffer_length() != 0 <= config.DEPOT_COMMUNICATION_RANGE:
            reward = self.RMAX
        elif distance_old_drone_depot > distance_cur_drone_depot: #If the drone selected as the best action is far away from the depot with respect to the last drone, then we give minimum reward 
            reward = self.RMIN
        else: 
            if drone.identifier in self.simulator.depot.nodes_table.nodes_list:
                hop = self.simulator.depot.nodes_table.nodes_list[drone.identifier].hop_count
                if hop != None and len(self.link_stability.keys()) != 0:
                    reward = self.OMEGA*np.exp(1/hop) + (1-self.OMEGA)*self.link_stability[drone.identifier]
                    if reward > self.RMAX:
                        self.RMAX = reward
                    if reward < self.RMIN:
                        self.RMIX = reward
            else:
                if len(self.link_stability.keys()) != 0:
                    reward = self.link_stability[drone.identifier]
        if self.old_state != None and reward != None:
            self.update_qtable(self.old_state.identifier, self.drone.identifier, self.drone.identifier, reward)
        
    def relay_selection(self, opt_neighbors: list, packet):
        """
        This function returns the best relay to send packets.
        @param packet:
        @param opt_neighbors: a list of tuple (hello_packet, source_drone)
        @return: The best drone to use as relay
        """
        
        self.check_state(self.drone.identifier)
        #self.update_link_quality(self.drone) #WORKS GOOD BUT NEED OPTIMIZATION!
        #print(self.link_qualities)
        drones_speed = self.compute_nodes_speed(self.drone, self.simulator.drones)
        link_quality_sum = {}
        for drone in self.simulator.drones:
            #link_quality_sum[drone.identifier] = self.compute_past_link_quality(drone)
            link_quality_sum[drone.identifier] = 1
            self.link_stability[drone.identifier] = (1-self.BETA)*np.exp(1/drones_speed[drone.identifier])+self.BETA*(link_quality_sum[drone.identifier]/len(self.simulator.drones))
        
        self.old_state = self.drone
        if len(opt_neighbors) == 0:
            return self.drone
        else:
            best_neighbor = (None, None) # (Drone, QValue)
            for neighbor in opt_neighbors:
                if  best_neighbor[0] == None or self.q_table[self.drone.identifier][neighbor[-1].identifier] > best_neighbor[1]:
                    best_neighbor = (neighbor[-1], self.q_table[self.drone.identifier][neighbor[-1].identifier])
            return best_neighbor[0]

    #Check if the state exists in the QTable, otherwise it adds it
    def check_state(self, state):
        if not state in self.q_table:
            self.q_table[state] = [0 for _ in range(self.simulator.n_drones+1)]


    #Update the qtable
    def update_qtable(self, state, next_state, action, reward):
        self.check_state(state) #Add state to qtable if it doesn't exist
        self.check_state(next_state) #Add next_state to qtable if it doesn't exist
        self.q_table[state][action] = (1-self.LEARNING_RATE)*self.q_table[state][action] + self.LEARNING_RATE*(reward + (self.DISCOUNT_FACTOR * max(self.q_table[next_state])))


    #Save the link quality of the drone at current step
    def update_link_quality(self, cur_drone):
        starting_point = 0 if (self.simulator.cur_step < config.RETRANSMISSION_DELAY) or (len(self.link_qualities.keys()) == 0) else max(self.link_qualities.keys()) #self.simulator.cur_step-config.RETRANSMISSION_DELAY
        for step in range(starting_point, self.simulator.cur_step):
            i = cur_drone
            link_qualities = []
            for j in self.simulator.drones:
                link_quality_ij = np.exp(-7*(util.euclidean_distance(i.coords, j.coords)/config.COMMUNICATION_RANGE_DRONE)) if i != j else 0
                link_qualities.append(link_quality_ij)
            self.link_qualities[step] = link_qualities


    def compute_past_link_quality(self, neighbor):
        link_quality = 0
        sum_lower_bound = self.simulator.cur_step-self.simulator.n_drones if self.simulator.cur_step >= self.simulator.n_drones else 0
        for k in range(sum_lower_bound, self.simulator.cur_step-1):
            link_quality += self.link_qualities[k][neighbor.identifier]
        return link_quality

    #TO CHANGE TOO! (how to compute? - 𝑣_i,j represents the speed at which nodes 𝑖 and 𝑗 are moving away)
    def compute_nodes_speed(self, cur_drone, neighbors):
        curr_speed = np.array([drone.speed for drone in neighbors]) #to change!
        return curr_speed
