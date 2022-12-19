from src.entities.uav_entities import Entity, Drone, Depot, Event, DataPacket, ACKPacket, HelloPacket, Packet, DiscoveryPacket, NeighborPacket

from src.utilities import utilities as util
from src.utilities import config

from scipy.stats import norm
import abc


class BASE_routing(metaclass=abc.ABCMeta):

    def __init__(self, entity: Entity, simulator):
        """ The drone that is doing routing and simulator object. """
        self.entity = entity
        self.current_n_transmission = 0
        self.network_disp = simulator.network_dispatcher # net_routing.MediumDispatcher
        self.simulator = simulator
        self.ack_list = []
        if self.simulator.communication_error_type == config.ChannelError.GAUSSIAN:
            self.buckets_probability = self.__init_guassian()
        self.no_transmission = False

    @abc.abstractmethod
    def relay_selection(self, geo_neighbors, packet):
        pass

    def routing_close(self):
        self.no_transmission = False

    def initialize_discovery(self, current_ts):
        print(self.entity.neighbor_table, current_ts)
        self.entity.neighbor_table = {}
        for drone in self.simulator.drones:
            drone.neighbor_list = set()
            drone.parent_node = None
            drone.acks = set()
        discovery_packet = DiscoveryPacket(self.simulator.depot, 0, current_ts, self.simulator, Event(self.entity.coords, current_ts, self.simulator))
        self.simulator.depot.communication_range = 100000 #TODO: debug purposes
        neighbors = {drone for drone in self.simulator.drones if util.euclidean_distance(self.simulator.depot_coordinates, drone.coords) <= self.simulator.depot.communication_range}
        if len(neighbors) == 0:
            print("No neighbors found for the depot")
        else:
            print("Depot length of neighbors is", len(neighbors))
        self.broadcast_message(discovery_packet, self.simulator.depot, self.simulator.drones, current_ts)

    def drone_reception(self, src_drone, packet: Packet, current_ts):
        if not self.__is_depot():
            # self.entity.accept_packets([packet]) #TODO: I don't know if this is needed
            # print(f"{packet} received by {self.entity.identifier} from {src_drone.identifier}, parent node is {self.entity.parent_node} at {current_ts}")
            pass

        """ Handle receptions of all types of packets by the drone or the depot. """
        if isinstance(packet, DiscoveryPacket):
            if self.__is_depot():
                return # The depot does not need to handle discovery packets
            if self.entity.parent_node is not None:
                return # The drone has already received its discovery packet
            self.entity.parent_node = packet.sender_id
            self.ack_packet(packet, current_ts)
            self.broadcast_discovery(packet, current_ts)

        elif isinstance(packet, DataPacket):
            self.no_transmission = True
            self.drone.accept_packets([packet])
            # build ack for the reception
            ack_packet = ACKPacket(self.drone, src_drone,{}, packet, current_ts, self.simulator)
            self.unicast_message(ack_packet, self.drone, src_drone, current_ts)

        elif isinstance(packet, ACKPacket):
            if self.__is_depot():
                return # The depot does not need to handle acks
            # Send the neighbor packet to the parent node
            # print(self.entity.parent_node)
            if self.entity.parent_node is None:
                return
            self.entity.acks.add(packet.sender_id)
            self.entity.remove_packets([packet.acked_packet])
            neighbor_packet = NeighborPacket(self.entity, current_ts, self.simulator, self.entity.acks)
            self.entity.neighbor_list = self.entity.neighbor_list.union(self.entity.acks)
            self.unicast_message(neighbor_packet, self.entity, self.entity.parent_node, current_ts)

            # if self.entity.buffer_length() == 0:
            #     self.current_n_transmission = 0
            #     self.entity.move_routing = False

        elif isinstance(packet, NeighborPacket):
            #Handle global neighbor table update for the depot
            if self.__is_depot() and not isinstance(packet.sender_id, Depot):
                for drone in packet.neighbor_list:
                    self.entity.add_neighbor(packet.sender_id, drone)
            elif not self.__is_depot():
                if self.entity.parent_node is None:
                    return
                # Handle neighor list update for the drone
                self.entity.neighbor_list = self.entity.neighbor_list.union(packet.neighbor_list)
                updated_neighbor_packet = NeighborPacket(self.entity, current_ts, self.simulator, packet.neighbor_list, packet.event_ref)
                self.unicast_message(updated_neighbor_packet, self.entity, self.entity.parent_node, current_ts)


    def ack_packet(self, packet, cur_step):
        """ build ack for the reception """
        parent_node = self.entity.parent_node
        ack_packet_info = {"moving_speed": 10, "location": self.entity.coords, "hop_count_from_depot": packet.hop_count}
        ack_packet = ACKPacket(parent_node, self.entity, ack_packet_info, packet, cur_step, self.simulator, packet.event_ref)
        self.unicast_message(ack_packet, self.entity, parent_node, cur_step)

    def broadcast_discovery(self, packet, cur_step):
        """ broadcast a discovery packet to all drones """

        # Generates the ack and send it to the sender of the discovery packet.
        # Accepts the packet and broadcast it to the other drones.
        updated_discovery_packet = DiscoveryPacket(sender_id=self.entity, 
                                                    hop_count=packet.hop_count+1, 
                                                    time_step_creation=cur_step, 
                                                    simulator=self.simulator, 
                                                    event_ref=packet.event_ref)
        neighbors = set(self.simulator.drones) - {self.entity} - {self.entity.parent_node}
        self.broadcast_message(updated_discovery_packet, self.entity, neighbors, cur_step) #The drone send the discovery packet to all the drones, if the drone has already received one, it will ignore it

    def routing(self, depot, drones, cur_step):
        # set up this routing pass

        self.send_packets(cur_step)

        # close this routing pass
        self.routing_close()

    def send_packets(self, cur_step):
        """ procedure 3 -> choice next hop and try to send it the data packet """

        # FLOW 0
        if self.no_transmission or self.entity.buffer_length() == 0:
            return

        # FLOW 1
        if util.euclidean_distance(self.simulator.depot.coords, self.entity.coords) <= self.simulator.depot_com_range:
            # add error in case
            self.transfer_to_depot(self.entity.depot, cur_step)

            self.entity.move_routing = False
            self.current_n_transmission = 0
            return



        if cur_step % self.simulator.drone_retransmission_delta == 0:
            opt_neighbors = [(None, drone) for drone in self.simulator.depot.get_neighbors(self.entity)] #TODO: the first element is None, this is a problem
            # for dpk_id in self.discovery_packets:
            #     dpk: DiscoveryPacket = self.discovery_packets[dpk_id]

            #     # check if packet is too old
            #     if dpk.time_step_creation < cur_step - config.OLD_HELLO_PACKET:
            #         continue

            #     opt_neighbors.append((dpk, dpk.sender_id))


            if len(opt_neighbors) == 0:
                return

            # send packets
            for pkd in self.entity.all_packets():


                self.simulator.metrics.mean_numbers_of_possible_relays.append(len(opt_neighbors))

                best_neighbor = self.relay_selection(opt_neighbors, pkd)  # compute score

                if best_neighbor is not None:

                    self.unicast_message(pkd, self.entity, best_neighbor, cur_step)

                self.current_n_transmission += 1

    def geo_neighborhood(self, drones, no_error=False):
        """
        @param drones:
        @param no_error:
        @return: A list all the Drones that are in self.drone neighbourhood (no matter the distance to depot),
            in all direction in its transmission range, paired with their distance from self.drone
        """

        closest_drones = []  # list of this drone's neighbours and their distance from self.drone: (drone, distance)

        for other_drone in drones:

            if self.entity.identifier != other_drone.identifier:  # not the same drone
                drones_distance = util.euclidean_distance(self.entity.coords,
                                                          other_drone.coords)  # distance between two drones

                if drones_distance <= min(self.entity.communication_range,
                                          other_drone.communication_range):  # one feels the other & vv

                    # CHANNEL UNPREDICTABILITY
                    if self.channel_success(drones_distance, no_error=no_error):
                        closest_drones.append((other_drone, drones_distance))

        return closest_drones

    def channel_success(self, drones_distance, no_error=False):
        """
        Precondition: two drones are close enough to communicate. Return true if the communication
        goes through, false otherwise.
        """

        assert (drones_distance <= self.entity.communication_range)

        if no_error:
            return True

        if self.simulator.communication_error_type == config.ChannelError.NO_ERROR:
            return True

        elif self.simulator.communication_error_type == config.ChannelError.UNIFORM:
            return self.simulator.rnd_routing.rand() <= self.simulator.drone_communication_success

        elif self.simulator.communication_error_type == config.ChannelError.GAUSSIAN:
            return self.simulator.rnd_routing.rand() <= self.gaussian_success_handler(drones_distance)

    def broadcast_message(self, packet, src_drone, dst_drones, curr_step):
        """ send a message to my neigh drones"""
        for d_drone in dst_drones:
            self.unicast_message(packet, src_drone, d_drone, curr_step)

    def unicast_message(self, packet, src_drone, dst_drone, curr_step):
        """ send a message to my neigh drones"""
        # Broadcast using Network dispatcher
        self.simulator.network_dispatcher.send_packet_to_medium(packet, src_drone, dst_drone,
                                                                curr_step + config.LIL_DELTA)

    def gaussian_success_handler(self, drones_distance):
        """ get the probability of the drone bucket """
        bucket_id = int(drones_distance / self.radius_corona) * self.radius_corona
        return self.buckets_probability[bucket_id] * config.GUASSIAN_SCALE

    def transfer_to_depot(self, depot, cur_step):
        """ self.drone is close enough to depot and offloads its buffer to it, restarting the monitoring
                mission from where it left it
        """
        depot.transfer_notified_packets(self.entity, cur_step)
        self.entity.empty_buffer()
        self.entity.move_routing = False

    # --- PRIVATE ---
    def __is_depot(self):
        return isinstance(self.entity, Depot)

    def __init_guassian(self, mu=0, sigma_wrt_range=1.15, bucket_width_wrt_range=.5):

        # bucket width is 0.5 times the communication radius by default
        self.radius_corona = int(self.entity.communication_range * bucket_width_wrt_range)

        # sigma is 1.15 times the communication radius by default
        sigma = self.entity.communication_range * sigma_wrt_range

        max_prob = norm.cdf(mu + self.radius_corona, loc=mu, scale=sigma) - norm.cdf(0, loc=mu, scale=sigma)

        # maps a bucket starter to its probability of gaussian success
        buckets_probability = {}
        for bk in range(0, self.entity.communication_range, self.radius_corona):
            prob_leq = norm.cdf(bk, loc=mu, scale=sigma)
            prob_leq_plus = norm.cdf(bk + self.radius_corona, loc=mu, scale=sigma)
            prob = (prob_leq_plus - prob_leq) / max_prob
            buckets_probability[bk] = prob

        return buckets_probability
