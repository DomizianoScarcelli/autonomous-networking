from src.routing_algorithms.BASE_routing import BASE_routing
from src.utilities import utilities as util
from src.utilities import config
from src.simulation.metrics import Metrics
import numpy as np
import math

class QlearningStepwiseRouting(BASE_routing):

    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone, simulator)
        self.LEARNING_RATE = 0.8 #Learning rate
        self.DISCOUNT_FACTOR = 0.1 #Discount factor (it represents the importance of future rewards)
        self.BETA = 0.1 #Coefficient weight value (to change!)
        self.OMEGA = 0.1 #Used in the reward function (to change!)
        self.RMAX = 2 #Maximum reward value
        self.RMIN = 0 #Minimum reward value

        self.random = np.random.RandomState(self.simulator.seed) #it generates a random value to be used in epsilon greedy
        self.link_qualities = {} #In order to calculate it, we should considered the packet transmission time and packet delivery ratio but we made a simplification considering only the distance between two drones.
        self.q_table = self.instantiate_qtable() #{drone_1: [drone_1, drone_2, ..., drone_n], .., drone_n: [drone_1, drone_2, ..., drone_n]}
        self.link_stability = {}
        self.old_state = None

    #Function which is called at each step and it is used to compute the reward for each drone. After a reward has been chosen, another function is called to update the qtable.
    def compute_reward(self, drone):
        reward = None
        distance_cur_drone_depot = util.euclidean_distance(drone.coords, self.simulator.depot.coords)
        distance_old_drone_depot = util.euclidean_distance(drone.coords, self.simulator.depot.coords)
        # If the drone is in the communication range of the depot and it has packets to deliver, then we give maximum reward
        if distance_cur_drone_depot and drone.buffer_length() != 0 <= config.DEPOT_COMMUNICATION_RANGE:
            reward = self.RMAX
        elif distance_old_drone_depot > distance_cur_drone_depot: #if the drone chosen as the best action moves even further away from the depot, then we give minimum reward 
            reward = self.RMIN
        else: 
            if drone.identifier in self.simulator.depot.nodes_table.nodes_list: #In this case the drone is linked with the depot and hop != None
                hop = self.simulator.depot.nodes_table.nodes_list[drone.identifier].hop_count
                if drone.identifier in self.link_stability:
                    reward = self.OMEGA*np.exp(1/hop) + (1-self.OMEGA)*self.link_stability[drone.identifier] #<----TO CHECK!
                    if reward > self.RMAX:
                        self.RMAX = reward
                    if reward < self.RMIN:
                        self.RMIX = reward
            else:
                if drone.identifier in self.link_stability:
                    reward = self.link_stability[drone.identifier]
        if self.old_state != None and reward != None:
            self.update_qtable(self.old_state.identifier, self.drone.identifier, self.drone.identifier, reward)
    
    #This function returns the best relay to send packets
    def relay_selection(self, opt_neighbors: list, packet): 
        self.update_link_quality() #Compute the link qualities based on the distance between the current drone and the others and save them to self.link_qualities dictionary
        drones_speed = self.compute_nodes_speed(self.drone, self.simulator.drones) #Compute the speed at which two nodes move away (YET TO IMPLEMENT - probabilmente va inserita nel ciclo for più in basso)
        link_quality_sum = {} #Stores the sum of link qualities in the last n steps (n is the number of drones) between the current drone and the neighbors

        for neighbor in self.drone.neighbor_table.neighbors_list:
            neighbor = self.simulator.drones[neighbor]
            link_quality_sum[neighbor.identifier] = self.sum_n_last_link_qualities(neighbor) #Sum the last n link qualities between self and the neighbor (n is the number of drones)
            self.link_stability[neighbor.identifier] = (1-self.BETA)*np.exp(1/drones_speed[neighbor.identifier])+self.BETA*(link_quality_sum[neighbor.identifier]/self.simulator.n_drones) #Computes the link stability between self and the neighbor

        self.old_state = self.drone
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
    def instantiate_qtable(self):
        qtable = {}
        for drone_index in range(self.simulator.n_drones):
            qtable[drone_index] = [0 for _ in range(self.simulator.n_drones+1)]
        return qtable

    #Update the qtable
    def update_qtable(self, state, next_state, action, reward):
        #Update the reward function with the same formula used on the paper
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

    #TO CHANGE! (how to compute? - 𝑣_i,j represents the speed at which nodes 𝑖 and 𝑗 are moving away)
    def compute_nodes_speed(self, cur_drone, neighbors):
        curr_speed = np.array([drone.speed for drone in neighbors]) #to change!
        return curr_speed
