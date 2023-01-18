from src.routing_algorithms.BASE_routing import BASE_routing
from src.utilities import utilities as util
import numpy as np

class EnhancedGeoQLRouting(BASE_routing):

    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone, simulator)
        self.LEARNING_RATE = 0.8
        self.DISCOUNT_FACTOR = 0.1
        self.EPSILON = 0.4

        self.random = np.random.RandomState(self.simulator.seed)
        self.taken_actions = {}  # id event : (old_state, old_action, next_state)
        self.q_table = {}

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
        if id_event in self.taken_actions:
            state, action = self.taken_actions[id_event] #Old state and action
            reward = compute_reward(self, drone, delay, outcome)
            update_qtable(self, action, state, reward)
            del self.taken_actions[id_event]

    def choose_best_drone(self,state,opt_neighbors):
        neighbors_drones = [drone[-1] for drone in opt_neighbors]
        self_drone_distance_to_depot = util.euclidean_distance(self.drone.coords, self.simulator.depot_coordinates)
        #EPSILON-GREEDY
        if self.random.rand() < self.EPSILON:
            if self.EPSILON > 0.1: self.EPSILON -= 0.05 #Each time we explore, reduce the probability of exploration at the next rounds
            #EXPLORATION
            # if len(neighbors_drones) == 0 or self_drone_distance_to_depot <= self.simulator.drone_com_range: #If there are no neighbors or the drone is in the communication range of the depot, I select the drone itself as the next action
            #     return self.drone
            return self.random.choice(neighbors_drones + [self.drone]) #Otherwise I select a random neighbor
        else:
            # if len(neighbors_drones) == 0 or self_drone_distance_to_depot <= self.simulator.drone_com_range: #If there are no neighbors, I select the drone itself as the next action
            #     return self.drone
            #Otherwise, if there are neighbors...
            drone_pool = [self.drone] + neighbors_drones #Select all possible action (neighbors and the drone itself)
            curr_distances = np.array([util.euclidean_distance(drone.coords, self.simulator.depot_coordinates) for drone in drone_pool])
            curr_distances_normalized = curr_distances / np.max(curr_distances)
            q_values = np.array([self.q_table[state][drone.identifier] for drone in drone_pool])
            if np.max(q_values) == 0:
                scores = curr_distances_normalized
            else:
                q_values_normalized = (q_values + np.min(q_values)) / np.max(q_values)
                weight = 1/4
                scores = (1-weight)*curr_distances_normalized + weight*(1-q_values_normalized)
            return drone_pool[np.argmin(scores)]


    def relay_selection(self, opt_neighbors: list, packet):
        """
        This function returns the best relay to send packets.
        @param packet:
        @param opt_neighbors: a list of tuple (hello_packet, source_drone)
        @return: The best drone to use as relay
        """
        state = compute_cell_index(self, self.drone, False)
        next_state = compute_cell_index(self, self.drone, True)
        check_state(self, state)

        action = self.choose_best_drone(state, opt_neighbors)
    
        self.taken_actions[packet.event_ref.identifier] = (state, action)
        check_state(self, next_state)
        return action


def compute_reward(self, drone, delay, outcome):
    distance_from_depot = 0
    if outcome == -1: 
        distance_from_depot = util.euclidean_distance(drone.coords, self.simulator.depot_coordinates)
    metrics_class = self.simulator.metrics
    curr_packet_delivery_ratio = len(metrics_class.drones_packets_to_depot)/metrics_class.all_data_packets_in_simulation
    delay_factor = (1/500) * delay 
    delivery_ratio_factor = (curr_packet_delivery_ratio * 20) if outcome == -1 else (-5*(1-curr_packet_delivery_ratio))
    reward = outcome - delay_factor + delivery_ratio_factor - distance_from_depot/10
    return reward
    


#Update the qtable
def update_qtable(self, drone, state, reward):
    next_state = compute_cell_index(self, drone, True) #Next state for the drone
    check_state(self, next_state) #Add state to qtable if it doesn't exist
    self.q_table[state][drone.identifier] += self.LEARNING_RATE * (reward + (self.DISCOUNT_FACTOR * max(self.q_table[next_state]) - self.q_table[state][drone.identifier]))

#Compute the cell index of the drone (if next = True returns the cell of the next target)
def compute_cell_index(self, drone, next):
    if not next:
        cell_index = util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell, width_area=self.simulator.env_width, x_pos=drone.coords[0], y_pos=drone.coords[1])[0]
    else:
        cell_index = util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell, width_area=self.simulator.env_width, x_pos=drone.next_target()[0], y_pos=drone.next_target()[1])[0] 
    return cell_index

#Check if the state exists in the QTable, otherwise it adds it
def check_state(self, state):
    try:
        self.q_table[state]
    except:
        self.q_table[state] = [0 for _ in range(self.simulator.n_drones)]