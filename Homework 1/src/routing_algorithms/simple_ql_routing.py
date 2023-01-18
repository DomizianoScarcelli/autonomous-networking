from src.routing_algorithms.BASE_routing import BASE_routing
from src.utilities import utilities as util
import numpy as np

class SimpleQLRouting(BASE_routing):

    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone, simulator)
        self.LEARNING_RATE = 0.9 #alpha (the learning rate)
        self.DISCOUNT_FACTOR = 0.1 #gamma (it represents the importance of future rewards)
        self.EPSILON = 0.5 #epsilon (it is used in episolon greedy)

        self.random = np.random.RandomState(self.simulator.seed) #it generates a random value to be used in epsilon greedy
        self.taken_actions = {}  # id event : (old_state, old_action)
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
        # Packets that we delivered and still need a feedback
        #print(self.taken_actions)

        # outcome can be:
        #
        # -1 if the packet/event expired;
        # 1 if the packets has been delivered to the depot
        #print(drone, id_event, delay, outcome)

        # remove the entry, the action has received the feedback
        # Be aware, due to network errors we can give the same event to multiple drones and receive multiple
        # feedback for the same packet!!
        if id_event in self.taken_actions:
            state, action = self.taken_actions[id_event] #Old state and action
            reward = compute_reward(self, drone, delay, outcome)
            update_qtable(self, action, state, reward)
            del self.taken_actions[id_event]


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

        action = None
        neighbors_drones = [drone[-1] for drone in opt_neighbors]
        self_drone_distance_to_depot = util.euclidean_distance(self.drone.coords, self.simulator.depot_coordinates)
        #EPSILON-GREEDY
        if self.random.rand() < self.EPSILON:
            if self.EPSILON > 0.1: self.EPSILON -= 0.1 #Each time we explore, reduce the probability of exploration at the next rounds
            #EXPLORATION
            if len(neighbors_drones) == 0 or self_drone_distance_to_depot <= (self.simulator.drone_com_range)*2: #If there are no neighbors or the drone is in the communication range of the depot, I select the drone itself as the next action
                action = self.drone
            else:
                action = self.random.choice(neighbors_drones) #Otherwise I select a random neighbor
        else:
            if len(neighbors_drones) == 0 or self_drone_distance_to_depot <= (self.simulator.drone_com_range)*2: #If there are no neighbors, I select the drone itself as the next action
                action = self.drone
            else: #Otherwise, if there are neighbors...
                q_values = [self.q_table[state][drone.identifier] for drone in neighbors_drones] #We retrieve the values from the q_table for all neighbors
                action = neighbors_drones[np.argmax(q_values)] #We select the drone with the best q_value
       
        self.taken_actions[packet.event_ref.identifier] = (state, action)
        check_state(self, next_state)
        return action

def compute_reward(self, drone, delay, outcome):
    reward = 0
    pkts = drone.buffer_length()
    if outcome == -1:
        if delay >= 2000 and delay < 3500:
            reward =- 15
        else:
            reward =- 30
    else:
        if delay < 1000:
            reward += (pkts * 2) + 45
        else:
            reward += (pkts * 2) + 30
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
        self.q_table[state] = [0 for _ in range(self.simulator.n_drones+1)]