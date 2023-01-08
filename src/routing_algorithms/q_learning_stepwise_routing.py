from src.routing_algorithms.BASE_routing import BASE_routing
from src.utilities import utilities as util
from src.utilities import config
#from src.simulation.metrics import Metrics
import numpy as np

class QlearningStepwiseRouting(BASE_routing):

    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone, simulator)
        self.LEARNING_RATE = 0.8 #Learning rate
        self.DISCOUNT_FACTOR = 0.1 #Discount factor (it represents the importance of future rewards)
        self.GAMMA = 0.7 #It is a coefficient weight value used in the reward function
        self.RMAX = 2 #Maximum reward value (updated dinamically)
        self.RMIN = -0.5 #Minimum reward value (updated dinamically)

        self.q_table = self.instantiate_qtable() #{drone_1: [drone_1, drone_2, ..., drone_n], .., drone_n: [drone_1, drone_2, ..., drone_n]}
        self.old_state = None #Stores the old_state once an action has been taken (used in the reward function)

    #Function which is called at each step and it is used to compute the reward for each drone. After a reward has been chosen, another function is called to update the qtable accordingly.
    def compute_reward(self, old_state):
        """
            Given the current state, the action taken at the last step and the old state, computes a reward and update the reward function.
            Notice that, in this model, both state and action is a drone. The agent is the packet. If we are a drone d1 that has a packet to send,
            the current state is d1 and the set of possible actions are the drone's neighbors. The drone d1 takes an action (a drone to send to
            packet to, let's say drone d2) and once the packet is sent, d2 becomes the new state. Notice also that we assign the reward to the
            old state that has taken an action so we update the qvalue of the old state.
            @param old_state: The old state from which the packet came from used to update the reward function.
            @return: None
        """
        reward = None
        distance_cur_drone_depot = util.euclidean_distance(self.drone.coords, self.simulator.depot.coords) #Computes the distance between the depot and the current drone (the chosen action)
        distance_old_drone_depot = util.euclidean_distance(old_state.coords, self.simulator.depot.coords) #Computes the distance between the depot and the last drone (the old state)
        # If the drone is in the communication range of the depot, then we give maximum reward
        if distance_cur_drone_depot <= config.DEPOT_COMMUNICATION_RANGE: #Case 1: If the selected drone (the current state) is in the communication range of the depot then it immediatly deliver the packets
            reward = self.RMAX #In this case we give maximum reward
        elif distance_old_drone_depot > distance_cur_drone_depot: #Case 2: If the drone chosen as the best action moves even further away from the depot
            reward = self.RMIN #Then we give minimum reward
        else:
            #Case 3: Otherwise (if the drone, the chosen action, is moving towards the depot but it is not in its communication range)
            if self.drone.identifier in self.simulator.depot.nodes_table.nodes_list: #If the drone has a path with the depot, so it's in its nodes table (in this case hop != None)
                hop = self.simulator.depot.nodes_table.nodes_list[self.drone.identifier].hop_count #Retrieve its hop count from the nodes table
                reward = self.GAMMA*np.exp(1/hop) + (1-self.GAMMA)*self.drone.link_stabilities[old_state.identifier] #Computes the reward considering the hop count and link stability computed from the chosen drone and the old one
            else:
                reward = self.drone.link_stabilities[old_state.identifier] #If the hop is not defined (the drone doesn't have a path with the depot), computes the reward with link stability only
            if reward > self.RMAX: #We want RMAX to be the maximum reward possible but we can't know it beforehand so, if a larger reward is found, update its current value
                self.RMAX = reward
            elif reward < self.RMIN: #Same for RMIN
                self.RMIX = reward
        if self.old_state != None or (self.old_state != self.old_state != self.drone): #If not in the first step (in the first step there is no old_state)
            self.update_qtable(self.old_state.identifier, self.drone.identifier, self.drone.identifier, reward) #Update the qtable with the computed reward
    
    #This function returns the best relay to send packets
    def relay_selection(self):
        """
            Selects the best action to send the packet to.
            @return: A drone.
        """
        self.old_state = self.drone #Update old_state with current_state (useful in the reward function)
        if len(self.drone.neighbor_table.neighbors_list) == 0: #If there are no neighbors, keep the packets
            return self.drone
        else: #Otherwise, choose the action (the neighbor) with the highest QValue
            drone_pool = self.drone.neighbor_table.get_drones() #Select all possible action (neighbors and the drone itself)
            curr_distances = np.array([util.euclidean_distance(drone.coords, self.simulator.depot_coordinates) for drone in drone_pool]) #Computes the current distances
            next_distances = np.array([util.euclidean_distance(drone.next_target(), self.simulator.depot_coordinates) for drone in drone_pool]) #Compute the next targets
            drone_step = curr_distances - next_distances #Compute the difference between the current distances and the next targets (if the value is positive then the drone is heading to the depot)
            max_qtable_value = best_choice = None
            for index, drone_distance in np.ndenumerate(drone_step):
                if drone_distance >= 0: #If the drone is heading to the depot
                    q_value = self.q_table[self.drone.identifier][drone_pool[index[0]].identifier] #Take its value from the Q-table
                    if max_qtable_value is None or q_value > max_qtable_value: #If a larger Q-table value is found
                        max_qtable_value = q_value #Update the max Q-table value found
                        best_choice = drone_pool[index[0]] #Update the associated drone
            if best_choice is None: return self.drone #If all drone heads in the opposite direction from the depot, choose self
            else: return best_choice

    def instantiate_qtable(self):
        """
            Create the Q-Table
            @return: Q-Table of the shape {drone_1: [drone_1, drone_2, ..., drone_n], .., drone_n: [drone_1, drone_2, ..., drone_n]}
        """
        qtable = {}
        for drone_index in range(self.simulator.n_drones):
            qtable[drone_index] = [0 for _ in range(self.simulator.n_drones+1)]
        return qtable

    def update_qtable(self, state, next_state, action, reward):
        """
            Update the Q-Table with the same formula used on the paper.
            @param state: The drone that had a packet to send and took an action (so it chose a drone to send the packet to).
            @param action: The drone chosen as the best action.
            @param next_state: The drone chosen as the best action that is now the next state (the drone that now has the packet)
            @return: None
        """
        self.q_table[state][action] = (1-self.LEARNING_RATE)*self.q_table[state][action] + self.LEARNING_RATE*(reward + (self.DISCOUNT_FACTOR * max(self.q_table[next_state])))
