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
        self.BETA = 0.8 #Coefficient weight value (to change!)
        self.OMEGA = 0.1 #Used in the reward function (to change!)
        self.RMAX = 2 #Maximum reward value
        self.RMIN = 0 #Minimum reward value

        self.random = np.random.RandomState(self.simulator.seed) #it generates a random value to be used in epsilon greedy
        self.link_qualities = {} #In order to calculate it, we should considered the packet transmission time and packet delivery ratio but we made a simplification considering only the distance between two drones.
        self.q_table = self.instantiate_table() #{drone_1: [drone_1, drone_2, ..., drone_n], .., drone_n: [drone_1, drone_2, ..., drone_n]}
        self.link_stability = self.instantiate_table()
        self.old_state = None
        self.distance_vector = {} #{cur_step: [drone_i_j: [distance_i_j]]}

    #Function which is called at each step and it is used to compute the reward for each drone. After a reward has been chosen, another function is called to update the qtable accordingly.
    def compute_reward(self, old_action):
        #print(self.link_stability)
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
                reward = self.OMEGA*np.exp(1/hop) + (1-self.OMEGA)*self.link_stability[old_action.identifier][self.drone.identifier]
                if reward > self.RMAX:
                    self.RMAX = reward
                if reward < self.RMIN:
                    self.RMIX = reward
            else:
                reward = self.link_stability[old_action.identifier][self.drone.identifier]
        if self.old_state != None and reward != 0:
            self.update_qtable(self.old_state.identifier, self.drone.identifier, self.drone.identifier, reward)
    
    #This function returns the best relay to send packets
    def relay_selection(self): 
        self.update_link_quality() #Compute the link qualities based on the distance between the current drone and the others and save them to self.link_qualities dictionary
        link_quality_sum = {} #Stores the sum of link qualities in the last n steps (n is the number of drones) between the current drone and the neighbors
        
        print(self.drone.identifier)
        
        for neighbor in self.drone.neighbor_table.neighbors_list:
            neighbor = self.simulator.drones[neighbor]
            
            relative_speed = self.compute_nodes_speed(self.drone, neighbor, self.simulator.cur_step) #Compute the speed at which two nodes move away
            
            link_quality_sum[neighbor.identifier] = self.sum_n_last_link_qualities(neighbor) #Sum the last n link qualities between self and the neighbor (n is the number of drones)
            link_stability_ij = (1-self.BETA)*np.exp(1/relative_speed)+self.BETA*(link_quality_sum[neighbor.identifier]/self.simulator.n_drones) #Computes the link stability between self and the neighbor
            self.link_stability[self.drone.identifier][neighbor.identifier] = link_stability_ij
            self.link_stability[neighbor.identifier][self.drone.identifier]= link_stability_ij

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
    def instantiate_table(self):
        table = {}
        for drone_index in range(self.simulator.n_drones):
            table[drone_index] = [0 for _ in range(self.simulator.n_drones+1)]
        return table

    #Update the qtable
    def update_qtable(self, state, next_state, action, reward):
        #Update the reward function with the same formula used on the paper
        self.q_table[state][action] += (1-self.LEARNING_RATE)*self.q_table[state][action] + self.LEARNING_RATE*(reward + (self.DISCOUNT_FACTOR * max(self.q_table[next_state])))

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
            self.link_qualities[step] = {}
            self.link_qualities[step][i.identifier] = link_qualities
        while (len(self.link_qualities.keys()) > self.simulator.n_drones): #Since we need only the last n link qualities, we delete the others (saves a lot of memory!)
            min_step = np.min(list(self.link_qualities.keys()))
            del self.link_qualities[min_step]

    #Compute the sum of the n last link qualities between self and and another drone (usually a neighbor) (n is the number of drones in the simulator). The output is used to compute the link stability.
    def sum_n_last_link_qualities(self, drone):
        #Since we keep only the last n link qualities computed in the link quality dictionary, we simply sum the all link quality values for the given drone
        return sum([self.link_qualities[step][self.drone.identifier][drone.identifier] for step in self.link_qualities.keys()])

    #compute the speed at which nodes 𝑖 and 𝑗 are moving away, equals the change in distance between them divided by the change in time
    def compute_nodes_speed(self, drone_i, drone_j, step):
        # Ad ogni step, durante il calcolo della link stability, per ogni nodo vicino (j) al drone corrente (i) vado a calcolarmi una sorta di velocitá relativa in questo modo:
        # 1.	Utilizzo le coordinate della posizione corrente e le coordinate fornite da next_target del nodo i e j per capire in che direzione stanno procedendo.
        # 2.	Successivamente in base a questa direzione e alla loro DRONE_SPEED trovo le coordinate del punto che ogni drone raggiungerá al prossimo step (una sorta di next_step compreso nel percorso che parte da cur_pos e next_target).
        # 3.	A questo punto avró un vettore per ogni drone con due coordinate ([cur_pos], [next_step]), li chiameremo rispettivamente dist_i e dist_j.
        # 4.	Li sommiamo entrambi ottenendo la distanza percorsa in quello step da entrambi i droni.
        # 5.	Infine calcoliamo la velocitá relativa in questo modo: velocitá = distanza appena calcolata / tempo (inteso come time_step_duration)
        # 
        # La velocitá ottenuta, secondo il nostro ragionamento, dovrebbe essere compresa tra 0 (se i nodi si stanno muovendo nella stessa direzione) e 2*DRONE_SPEED (se vanno in direzioni opposte).
        
        # Vorreste calcolare la velocità relativa dei due punti calcolando la posizione corrente di entrambe i punti e misurando la distanza che c'è tra i due punti.
        # Dopo un delta t vi ricalcolate la posizione corrente dei punti e misurate la distanza che c'è.
        # A questo punto potete prendere la differenza nella distanza dei punti a tempo t e t + delta e dividerla per il delta e dovreste ottenere la velocità relativa.

        if step not in self.distance_vector:
            self.distance_vector[step] = {}
            identifier = str(drone_i.identifier) + "_" + str(drone_j.identifier)
            if identifier not in self.distance_vector[step]:
                cur_distance = util.euclidean_distance(drone_i.coords, drone_j.coords)
                self.distance_vector[step][identifier] = cur_distance

                cur_speed = 1
            else:
                old_distance = self.distance_vector[step][identifier]
                cur_distance = util.euclidean_distance(drone_i.coords, drone_j.coords)
                self.distance_vector[step][identifier] = cur_distance

                cur_speed = abs(old_distance - cur_distance) / config.TS_DURATION
        else:
            cur_speed = 1

        return cur_speed