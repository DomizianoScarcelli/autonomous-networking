import numpy as np
import src.utilities.utilities as util
import src.entities.uav_entities as entity

from src.routing_algorithms.BASE_routing import BASE_routing

class GeoRouting(BASE_routing):
    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone, simulator)

    def relay_selection(self, opt_neighbors, packet):
        #TODO: Original function description: This function returns a relay for packets according to geographic routing using C2S criteria.
        #TODO: the following is the description for the exercise2, in this case the only criteria that has to be applied is C2S
        """
            STEP 1: implement NFP, MFP, CR and C2S considering the current position of d_0 (self.drone) and
                    the last known position of N(d_0) (drones in opt_neighbours)
            STEP 2: implement NFP, MFP, CR and C2S considering the next target position of d_0 and N(d_0)
            STEP 3: compare the performances of STEP1 and STEP2
            OPTIONAL: design and implement your own heuristic (you can consider all the info in the hello packets)
        """
        # opt_neighbour -> list[(hellopacket, drone)]
        # hellopacket:
        #   i -> sic_drone
        #   ti -> time_step_creation
        #   (x0(ti), y0(ti)) -> cur_position
        #   v(ti) -> speed
        #   (x1(ti), y1(ti)) -> next_target

        # Nearest with forwarding progress (NFP) -> d = argmin proj(d0, D, di)
        # Most forwarding progress (MFP) -> d = argmax proj(d0, D, di)
        # Compass rounting (CR) -> d = argmin |THETA(d0, D, di)|
        # Closest to sink (C2S) -> d = argmin |di, D[]|
        D = self.simulator.depot_coordinates
        d_0 = self.drone.coords
        #STEP 1
        # implement NFP, MFP, CR and C2S considering the current position of d_0 (self.drone) and the last known position of N(d_0) (drones in opt_neighbours)
        d_NFP, d_MFP, d_CR, d_C2S = self.calculate_values_step_1(d_0, D, opt_neighbors) 
        # STEP 2   
        # implement NFP, MFP, CR and C2S considering the next target position of d_0 and N(d_0)
        d_0_2 = self.drone.next_target()
        d_NFP_2, d_MFP_2, d_CR_2, d_C2S_2 = self.calculate_values_step_2(d_0_2, D, opt_neighbors)

        # #Custom heuristic
        # d_custom = self.custom_heuristic(d_0, D, opt_neighbors)
        return opt_neighbors[d_C2S][-1]


    def calculate_values_step_1(self, d_0, D, N_d_0):
        projection_vector = [util.projection_on_line_between_points(d_0, D, d_i[-1].coords) for d_i in N_d_0]
        d_NFP = np.argmin(projection_vector) 
        d_MFP = np.argmax(projection_vector) 
        d_CR = np.argmin([util.angle_between_points(d_0, D, d_i[-1].coords) for d_i in N_d_0]) 
        d_C2S = np.argmin([util.euclidean_distance(d_i[-1].coords, D) for d_i in N_d_0]) 
        return (d_NFP, d_MFP, d_CR, d_C2S)
    
    def calculate_values_step_2(self, d_0, D, N_d_0):
        # TODO: for the homework, the step 2 it's not needed
        # TODO: These two lines are used to avoid division by zero, that happens when either d_0[0] == D[0] or if exists a d_i such that d_i[0] == D[0]
        if d_0[0] == D[0]: return (0,0,0,0)
        if any([d_i[-1].next_target()[0] == D[0]] for d_i in N_d_0): return (0,0,0,0)
        projection_vector = [util.projection_on_line_between_points(d_0, D, d_i[-1].next_target()) for d_i in N_d_0]
        d_NFP = np.argmin(projection_vector) 
        d_MFP = np.argmax(projection_vector) 
        d_CR = np.argmin([util.angle_between_points(d_0, D, d_i[-1].next_target()) for d_i in N_d_0]) 
        d_C2S = np.argmin([util.euclidean_distance(d_i[-1].next_target(), D) for d_i in N_d_0]) 
        return (d_NFP, d_MFP, d_CR, d_C2S)

