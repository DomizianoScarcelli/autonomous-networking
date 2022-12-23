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

        self.random = np.random.RandomState(self.simulator.seed) #it generates a random value to be used in epsilon greedy
        self.taken_actions = {}  # id event : (old_state, old_action)
        self.link_qualities = {} #In order to calculate it, we considered the packet transmission time and packet delivery ratio. To calculate this, the Window Mean with Exponentially Weighted Moving Average (WMEWMA) method[9] was used. Note: for the moment I consider the distance between node.
        self.q_table = {} #{drone_1: [drone_1, drone_2, ..., drone_n], .., drone_n: [drone_1, drone_2, ..., drone_n]}

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
            reward = self.compute_reward(drone, delay, outcome)
            self.update_qtable(action, state, reward)
            del self.taken_actions[id_event]


    def relay_selection(self, opt_neighbors: list, packet):
        """
        This function returns the best relay to send packets.
        @param packet:
        @param opt_neighbors: a list of tuple (hello_packet, source_drone)
        @return: The best drone to use as relay
        """
        
        state = self.compute_cell_index(self.drone, False)
        self.check_state(state)

        action = self.drone
        
        self.update_link_quality(self.drone)
        #print(self.link_qualities)
        drones_speed = self.compute_nodes_speed(self.drone, self.simulator.drones)
        link_quality_sum = {}
        for j in opt_neighbors:
            link_quality_sum[j[-1].identifier] = self.compute_past_link_quality(j[-1])
        # print(link_quality_sum)
        link_stability = np.array([(1-self.BETA)*math.exp(1/drones_speed[j])+self.BETA for j in range(len(self.simulator.drones))])
       
        self.taken_actions[packet.event_ref.identifier] = (state, action)
        next_state = self.compute_cell_index(self.drone, True)
        self.check_state(next_state)
        return action

    #TO CHANGE!
    def compute_reward(self, drone, delay, outcome):
        reward = 0

    
    #Compute the cell index of the drone (if next = True returns the cell of the next target) (WE DON'T NEED IT ANYMORE - BUT LEAVE IT FOR NOW)
    def compute_cell_index(self, drone, next):
        pos = drone.next_target()[1] if next else drone.coords[1]
        cell_index = util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell, width_area=self.simulator.env_width, x_pos=drone.coords[0], y_pos=pos)[0]
        return cell_index

    #Check if the state exists in the QTable, otherwise it adds it (WE DON'T NEED IT ANYMORE - BUT LEAVE IT FOR NOW)
    def check_state(self, state):
        if not state in self.q_table:
            self.q_table[state] = [0 for _ in range(self.simulator.n_drones+1)]

    #Update the qtable (TO CHANGE!)
    def update_qtable(self, drone, state, reward):
        next_state = self.compute_cell_index(drone, True) #Next state for the drone
        self.check_state(next_state) #Add state to qtable if it doesn't exist
        self.q_table[state][drone.identifier] += self.LEARNING_RATE * (reward + (self.DISCOUNT_FACTOR * max(self.q_table[next_state])))

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
        sum_lower_bound = self.simulator.cur_step-len(self.simulator.drones) if self.simulator.cur_step >= len(self.simulator.drones) else 0
        for k in range(sum_lower_bound, self.simulator.cur_step-1):
            link_quality += self.link_qualities[k][neighbor.identifier]
        return link_quality

    #TO CHANGE TOO! (how to compute? - 𝑣_i,j represents the speed at which nodes 𝑖 and 𝑗 are moving away)
    def compute_nodes_speed(self, cur_drone, neighbors):
        curr_speed = np.array([drone.speed for drone in neighbors]) #to change!
        return curr_speed
