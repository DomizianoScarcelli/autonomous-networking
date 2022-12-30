from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.simulation.simulator import Simulator
    from src.entities.uav_entities import Depot, Drone, AckDiscoveryPacket
from src.utilities import utilities
from src.utilities import printer
import sys

REDICRECT_STDOUT = not sys.stdout.isatty()

class Tester():
    def __init__(self, simulator):
        self.simulator: Simulator = simulator

    # Debugging for lost acks
    ##########################################################################      
    def add_ack(self, ack_packet: AckDiscoveryPacket, parent_node: Drone | Depot):
        """
        Adds an ack to the dictionary of ack sent
        """
        if ack_packet.sender_id not in self.simulator.metrics.sent_acks:
            self.simulator.metrics.sent_acks[parent_node.identifier] = []
        self.simulator.metrics.sent_acks[parent_node.identifier] += [self]

    def check_if_lost_ack(self, drone: Drone):
        """
        Checks if there is an ack that is sent but never received
        """
        if drone.identifier in self.simulator.metrics.sent_acks:
            acks_received = self.simulator.metrics.sent_acks[drone.identifier]
            neighbor_table = drone.neighbor_table.get_drones()
            lost_drones = set(acks_received).difference(neighbor_table)
            return lost_drones
        return set()

    def check_if_lost_ack_depot(self):
        if self.simulator.depot.identifier in self.simulator.metrics.sent_acks:
            acks_received = self.simulator.metrics.sent_acks[self.simulator.depot.identifier]
            neighbor_table = set(self.simulator.depot.nodes_table.nodes_list.keys())
            lost_drones = {drone.identifier for drone in acks_received}.difference(neighbor_table)
            return lost_drones
        return set()
    
    def lost_drones(self):
        for drone in self.drones:
            lost_drones = self.check_if_lost_ack(drone)
            if lost_drones != set():
                print(f"Lost drones: {lost_drones}")
        depot_lost_drones = self.check_if_lost_ack_depot()
        if depot_lost_drones != set():
            print(f"Depot lost drones: {depot_lost_drones}")
    
    def reset_state(self):
        self.simulator.metrics.sent_acks = {}
    ##########################################################################
    
    # Debugging for hop_count update
    ##########################################################################
    def hop_count_tester(self):
        drone: Drone
        for drone in [self.simulator.drones[id] for id in self.simulator.depot.nodes_table.nodes_list.keys() if utilities.euclidean_distance(self.simulator.drones[id].coords, self.simulator.depot.coords) <= self.simulator.depot.communication_range]:
            if drone.hop_from_depot > 1:
                printer.print_debug_colored(200, 0, 0, f"{drone} is in the neighborhood of the depot with hop count {drone.hop_from_depot}")
    
    def test_wrong_hop_from_depot(self):
        if len(self.simulator.depot.nodes_table.nodes_list) != 0:
            hop_counts = [node_info.hop_count > 1 for node_info in self.simulator.depot.nodes_table.nodes_list.values()]
            if any(hop_counts):
                print(f"Depot neighbors: {self.simulator.depot.nodes_table.nodes_list}")
    
    def print_hop_update(self, drone: Drone, new_hop, message):
        """
        Everytime the hop is update to some non-None hop, the cause of the update is printed, as well as the old and new hop values.
        """
        # if new_hop is not None:
        printer.print_debug_colored(252, 194, 3, f"Hop for {drone} changed from {drone.hop_from_depot} to {new_hop} because of {message}")
    ##########################################################################
    
    def print_send_dp(self, sender: Drone | Depot, receiver: Drone):
        """
        Prints out the sending of the discovery packet
        """
        printer.print_debug_colored(text=f"{sender} has sent a discovery packet to {receiver} at {self.simulator.cur_step}, they're distant {utilities.euclidean_distance(sender.coords, receiver.coords)}")

    def print_receive_dp(self, sender: Drone | Depot, receiver: Drone):
        """
        Prints out the reception of the discovery packet
        """
        printer.print_debug_colored(text=f"{receiver} has received a discovery packet from {sender} at {self.simulator.cur_step}, they're distant {utilities.euclidean_distance(sender.coords, receiver.coords)}")

    def print_neighborhood_flow(self, main: Drone, entry: str):
        """
        Prints out the adding ot "entry" Drone in the neighborhoo of "main"
        """
        printer.print_debug_colored(text=f"{main} has added Drone {entry} in their neighborhood, now it's composed from: {main.neighbor_table.get_drones()} at {self.simulator.cur_step}, they're distant {utilities.euclidean_distance(main.coords, self.simulator.drones[entry].coords)}")


    # Debugging for discovery correctness
    def check_drone_neighbors(self):
        """
        Check if the neighbors computed with the discovery are actually the correct neighbors.
        """
        drones = self.simulator.drones
        drone: Drone
        for drone in drones:
            # Compute the correct neighbor list
            correct_neighbor_list = {neighbor for neighbor in drones if utilities.euclidean_distance(drone.coords, neighbor.coords) <= drone.communication_range and drone != neighbor}
            # Retrieve the neighbor list computed in the discovery phase
            discovery_neighbor_list = set(drone.neighbor_table.get_drones())
            
            # #TODO: DEBUG: this is needed to bypass the fact that the parent node is not in the neighbor list right now
            # if drone.parent_node is not None:
            #     discovery_neighbor_list.add(drone.parent_node)

            # #TODO: DEBUG: this is needed to bypass the fact that the node is in its neighbor list
            # if drone in discovery_neighbor_list:
            #     discovery_neighbor_list.remove(drone)

            # #TODO: DEBUG: this is needed to bypass the fact that the depot is in the drone neighbor list
            # if self.simulator.depot in discovery_neighbor_list:
            #     discovery_neighbor_list.remove(self.simulator.depot)

            assert discovery_neighbor_list == correct_neighbor_list, f"""

                Discovery phase neighbors are not correct for {drone}. Computed: {discovery_neighbor_list}, correct: {correct_neighbor_list}
                Drone raw neighbor list: {drone.neighbor_table}
                Distance from each drone (communication range: {drone.communication_range}):
                {[(neighbor, utilities.euclidean_distance(drone.coords, neighbor.coords)) for neighbor in discovery_neighbor_list.union(correct_neighbor_list)]}
                Parent node: {drone.parent_node}
                """
    
    def check_depot_discovery(self):
        """
        Check if the drones that are connected to the depots are computed correctly.
        Note that this test doesn't check if the information about the drones that can reach the depot is updated.
        """
        computed_depot_discovery = set(self.simulator.depot.nodes_table.nodes_list.keys())
        correct_depot_discovery = {drone.identifier for drone in self.simulator.depot.get_chain()}
        assert computed_depot_discovery == correct_depot_discovery, f"""
        
            Drone connected to the depot are not computet correctly.
            Computed: {computed_depot_discovery}, correct: {correct_depot_discovery}
            Drones in the range of the depot: {[drone for drone in self.simulator.drones if utilities.euclidean_distance(drone.coords, self.simulator.depot.coords) <= self.simulator.depot.communication_range]}
        """

    def check_depot_information(self):
        from src.entities.uav_entities import NodeInfo, NodesTable #Just in time import in order to avoid circular imports
        """
        Check if the information about the drones that can reach the depot is up-to-date inside the depot's node table.
        """
        correct_depot_discovery = {drone.identifier for drone in self.simulator.depot.get_chain()}
        correct_information = NodesTable()

        for drone_id in correct_depot_discovery:
            drone: Drone = self.simulator.drones[drone_id]
            node_info = NodeInfo(drone_id, drone.speed, drone.coords, drone.hop_from_depot)
            correct_information.add_node(node_info)
        
        computed_information = self.simulator.depot.nodes_table
        assert correct_information == computed_information, f"""
        Information inside the depot's node table is not correct, or not up-to-date.
        Computed: {computed_information}
        Correct: {correct_information}
        """
    
    def print_real_computed_neighbors(self):
        """
        Prints out the nodes that are geographically inside the communication range and the one computed by the discovery phase
        """
        drones = self.simulator.drones
        drone: Drone
        for drone in drones:
            # Compute the correct neighbor list
            correct_neighbor_list = {neighbor for neighbor in drones if utilities.euclidean_distance(drone.coords, neighbor.coords) <= drone.communication_range and drone != neighbor}
            # Retrieve the neighbor list computed in the discovery phase
            discovery_neighbor_list = set(drone.neighbor_table.get_drones())

            printer.print_debug_colored(211, 3, 252, f"Real neighbors: {correct_neighbor_list}, discovered: {discovery_neighbor_list}")
