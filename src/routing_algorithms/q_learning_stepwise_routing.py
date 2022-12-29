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
        self.link_qualities = {} #In order to calculate it, we should considered the packet transmission time and packet delivery ratio but we made a simplification considering only the distance between two drones.
        self.q_table = self.instantiate_qtable() #{drone_1: [drone_1, drone_2, ..., drone_n], .., drone_n: [drone_1, drone_2, ..., drone_n]}
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

    #Function which is called at each step and it is used to compute the reward for each drone. After a reward has been chosen, another function is called to update the qtable.
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
        
        self.update_link_quality() #WORKS GOOD BUT NEED OPTIMIZATION!
        #print(self.link_qualities)
        drones_speed = self.compute_nodes_speed(self.drone, self.simulator.drones)
        link_quality_sum = {}
        for drone in self.simulator.drones:
            link_quality_sum[drone.identifier] = self.sum_n_last_link_qualities(drone)
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
    def instantiate_qtable(self):
        qtable = {}
        for drone_index in range(self.simulator.n_drones):
            qtable[drone_index] = [0 for _ in range(self.simulator.n_drones+1)]
        return qtable

    #Update the qtable
    def update_qtable(self, state, next_state, action, reward):
        self.q_table[state][action] = (1-self.LEARNING_RATE)*self.q_table[state][action] + self.LEARNING_RATE*(reward + (self.DISCOUNT_FACTOR * max(self.q_table[next_state])))

    #Save the link quality of the drone at current step. Only the last n link qualities are saved (where n is the number of drones in the simulator)
    def update_link_quality(self):
        #Why do we need a starting point?
        #Due to the retransmission delay, it seems that the relay selection function is only called every "RETRANSMISSION_DELAY" times. So, for example, it may happen that if "RETRANSMISSION_DELAY"=10
        #we have that link quality is only computed every 10 steps (e.g. 10, 20, 30, etc..) but since we need it in all steps we necessarily have to make an approcimation and assign the same link quality
        #at every steps in each retransmission interval (e.g. 0/9, 10/19, etc..). Starting point is the point from which we have to assign the last computed link quality.
        #So, at the first steps (self.simulator.cur_step < config.RETRANSMISSION_DELAY or the link quality is still empty) we assign 0 as the starting point, otherwise we start from the last step
        #for which we have computed the link quality.
        starting_point = 0 if (self.simulator.cur_step < config.RETRANSMISSION_DELAY) or (len(self.link_qualities.keys()) == 0) else max(self.link_qualities.keys())
        i = self.drone
        link_qualities = []
        for j in self.simulator.drones: #For each drones
            #We compute the link quality between i (self.drone) and the other j drones that depends on the distance. We use an exponential decay function so the closer they are, the higher the quality.
            link_quality_ij = np.exp(-7*(util.euclidean_distance(i.coords, j.coords)/config.COMMUNICATION_RANGE_DRONE)) if i != j else 0
            link_qualities.append(link_quality_ij) #We store the qualities between i and j
        for step in range(starting_point, self.simulator.cur_step): #Finally we assign the computed link qualities to each step from starting point to the current step.
            self.link_qualities[step] = link_qualities
        while (len(self.link_qualities.keys()) > self.simulator.n_drones): #Since we need only the last n link qualities, we delete the others (saves a lot of memory!)
            min_step = np.min(list(self.link_qualities.keys()))
            del self.link_qualities[min_step]

    #Compute the sum of the n last link qualities between self and and another drone (usually a neighbor) (n is the number of drones in the simulator). The output is used to compute the link stability.
    def sum_n_last_link_qualities(self, drone):
        #Since we keep only the last n link qualities computed in the link quality dictionary, we simply sum the all link quality values for the given drone
        return sum(link_qualities[drone.identifier] for link_qualities in self.link_qualities.values())

    #TO CHANGE TOO! (how to compute? - 𝑣_i,j represents the speed at which nodes 𝑖 and 𝑗 are moving away)
    def compute_nodes_speed(self, cur_drone, neighbors):
        curr_speed = np.array([drone.speed for drone in neighbors]) #to change!
        return curr_speed
