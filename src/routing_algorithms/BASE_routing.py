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
        self.discovery_packets = {}
        self.network_disp = simulator.network_dispatcher # net_routing.MediumDispatcher
        self.simulator = simulator
        self.neighbor_list = {id: set() for id in range(self.simulator.n_drones)}

        if self.simulator.communication_error_type == config.ChannelError.GAUSSIAN:
            self.buckets_probability = self.__init_guassian()
        self.no_transmission = False

    @abc.abstractmethod
    def relay_selection(self, geo_neighbors, packet):
        pass

    def routing_close(self):
        self.no_transmission = False

    def initialize_discovery(self, current_ts):
        discovery_packet = DiscoveryPacket(self.simulator.depot, 0, current_ts, self.simulator, None)
        #TODO: might be useless since this is always done by the depot
        potential_neighbors = set(self.simulator.drones).difference({self.entity}) 
        self.broadcast_message(discovery_packet, self.simulator.depot, potential_neighbors, current_ts)

    def drone_reception(self, src_drone, packet: Packet, current_ts):
        """ Handle receptions of all types of packets by the drone or the depot. """
        if isinstance(packet, DiscoveryPacket):
            src_id = packet.sender_id.identifier
            self.discovery_packets[src_id] = packet

            # Handles the reception of a discovery packet
            if packet in self.entity.all_packets(): #TODO: don't know if it's correct
                    return # The packet has already been accepted by the drone.
            self.entity.accept_packets([packet])
            if not self.__is_depot():
                

                # Generates the ack and send it to the sender of the discovery packet.
                ack_packet_info = {"moving_speed": 10, "location": self.entity.coords, "hop_count_from_depot": packet.hop_count}
                ack_packet = ACKPacket(packet.sender_id, self.entity, ack_packet_info, packet, current_ts, self.simulator)
                self.unicast_message(ack_packet, self.entity, packet.sender_id, current_ts)
                # Accepts the packet and broadcast it to the other drones.
                updated_discovery_packet = DiscoveryPacket(sender_id=self.entity, 
                                                            hop_count=packet.hop_count+1, 
                                                            time_step_creation=current_ts, 
                                                            simulator=self.simulator, 
                                                            event_ref=Event(self.entity.coords, current_ts, self.simulator))
                potential_neighbors = set(self.simulator.drones).difference({self.entity, packet.sender_id})
                self.broadcast_message(updated_discovery_packet, self.entity, potential_neighbors, current_ts)

                neighbor_list_packet = NeighborPacket(self.entity, current_ts, self.simulator, self.neighbor_list[self.entity.identifier])
                self.neighbor_list[self.entity.identifier] = set()
                self.unicast_message(neighbor_list_packet, self.entity, packet.sender_id, current_ts)
        elif isinstance(packet, DataPacket):
            self.no_transmission = True
            self.entity.accept_packets([packet])
            # build ack for the reception
            ack_packet_info = {}
            ack_packet = ACKPacket(self.entity, src_drone, ack_packet_info, packet, current_ts, self.simulator) #TODO: might be wrong
            self.unicast_message(ack_packet, self.entity, src_drone, current_ts)


        elif isinstance(packet, ACKPacket):
            if self.__is_depot():
                return # The depot does not need to handle acks
            # TODO: Now i'm trying without the ack waiting part, this has to be implemented somehow
            self.neighbor_list[self.entity.identifier].add(packet.sender_id)
            self.entity.accept_packets([packet])
            self.entity.remove_packets([packet.acked_packet]) #TODO: there is a problem since the acked packet is not in the buffer
            # print(self.is_packet_received_drone_reward, "ACK", self.drone.identifier)

            if self.entity.buffer_length() == 0:
                self.current_n_transmission = 0
                self.entity.move_routing = False

        elif isinstance(packet, NeighborPacket):
            print(packet.neighbor_list)
            #Handle global neighbor table update for the depot
            if self.__is_depot():
                
                pass
            else:
                # Handle neighor list update for the drone
                # new_neighbor_table = packet.neighbor_list
                # new_neighbor_list = NeighborPacket(self.entity, current_ts, self.simulator, self.entity.neighbor_list)
                #TODO: this may cause a loop
                # self.unicast_message(new_neighbor_list, self.entity, parent_node, current_ts)
                #TODO: this is never called
                # Handle whole list of neighbors update for the depot
                #TODO: update neighbor list in a correct way
                # self.entity.whole_neighbor_table[packet.sender_id.identifier].union(packet.neighbor_list) 
                # print(self.entity.whole_neighbor_table)
                new_neighbor_table = packet.neighbor_list
                new_neighbor_table.union(self.entity.neighbor_list)
                parent_node = None
                
                self.unicast(message=new_neighbor_table, sender=self.entity, receiver=parent_node, current_ts=current_ts)
                pass
                


    # def drone_identification(self, drones, cur_step):
    #     """ handle drone hello messages to identify neighbors """
    #     # if self.drone in drones: drones.remove(self.drone)  # do not send hello to yourself
    #     if cur_step % config.HELLO_DELAY != 0:  # still not time to communicate
    #         return

    #     my_hello = HelloPacket(self.drone, cur_step, self.simulator, self.drone.coords,
    #                            self.drone.speed, self.drone.next_target())

    #     self.broadcast_message(my_hello, self.drone, drones, cur_step)


    def routing(self, depot, drones, cur_step):
        # set up this routing pass

        #TODO: deactivated for now
        # self.drone_identification(drones, cur_step)

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

            opt_neighbors = [(None, drone) for drone in self.entity.neighbor_list] #TODO: the first element is None, this is a problem
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
