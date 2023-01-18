import src.utilities.utilities as util
from src.routing_algorithms.BASE_routing import BASE_routing


class RandomRouting(BASE_routing):

    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone, simulator)

    def relay_selection(self):
        """
        This function returns a random relay for packets.

        @param opt_neighbors: a list of tuples (hello_packet, drone)
        @return: a random drone as relay
        """
        opt_neighbors = self.drone.neighbor_table.get_drones()
        if len(opt_neighbors) == 0:
            return self.drone
        return self.simulator.rnd_routing.choice(opt_neighbors)
