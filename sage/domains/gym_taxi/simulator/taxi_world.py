"""
.. module:: taxi_world
   :synopsis: Simulates the taxi world environment based on actions passed in.
"""

from enum import Enum
import argparse
import numpy as np
import networkx as nx

from typing import List, NamedTuple

from sage.domains.gym_taxi.utils.representations import env_to_image, env_to_json
from sage.domains.gym_taxi.utils.config import MAX_EPISODE_LENGTH
from sage.domains.gym_taxi.utils.utils import generate_random_walls, generate_city_maze


ACTIONS = Enum("Actions", "north south east west pickup dropoff")

DIRECTIONS = {
    ACTIONS.north: (0, -1),
    ACTIONS.south: (0, 1),
    ACTIONS.west: (-1, 0),
    ACTIONS.east: (1, 0),
}

DEFAULT_REWARDS = {"base": -1, "failed-action": -10, "drop-off": 20}


def _get_destination(location, direction):
    """
    Returns the intended destination for the movement, does not check if road exists.
    :param location: current location of taxi
    :param direction: direction to move as ACTIONS enum
    :return: intended location after movement
    """
    return tuple([sum(x) for x in zip(location, DIRECTIONS[direction])])



class Taxi(NamedTuple):
    node: int
    location: int
    passenger: int

class Passenger(NamedTuple):
    location: int
    destination: int

class TaxiWorldSimulator(object):
    def __init__(
        self,
        random,
        size,
        passenger_locations=[],
        passenger_destinations=[],
        wall_locations=None,
        delivery_limit=1,
        concurrent_passengers=1,
        timeout=MAX_EPISODE_LENGTH,
        passenger_creation_probability=1,
        random_walls=True,
        taxi_locations=None,
        rewards=None,
        planning = False
    ):
        """
        Houses the game state and transition dynamics for the taxi world.

        :param size: size of gridworld
        :returns: this is a description of what is returned
        :raises keyError: raises an exception
        """
        self.random = random
        self.seed_id = hash(self.random)
        self.time = 0
        self.size = size
        self.passenger_locations = passenger_locations
        self.passenger_destinations = passenger_destinations
        self.delivery_limit = delivery_limit
        self.concurrent_passengers = concurrent_passengers
        self.timeout = timeout
        self.passenger_creation_probability = passenger_creation_probability
        self.rewards = rewards if rewards is not None else DEFAULT_REWARDS
        self.done = False
        self.planning = planning

        self.graph = self.generate_road_network(random_walls)

        self.taxi = self.add_taxi()

        self.passengers = {}
        self.add_passenger()
        

    def _get_state_json(self):
        return env_to_json(self)

    def act(self, action):
        """
        Advances the game state by one step
        :param action: action provided by the agent
        :returns: observation of the next state
        :raises assertionError: raises an exception if action is invalid
        """
        # action, param = action
        #assert action in ACTIONS
        node_type = self.graph.nodes[action]['attr']
        # if action == ACTIONS.noop:
        #     reward = self.rewards["base"]
        if node_type == [0,0,1]:
            reward = self.attempt_pickup(action)
        elif node_type == [0,1,0]:
            reward = self.attempt_dropoff(action)
        elif node_type == [1,0,0]:
            reward = self.attempt_move(action)

        self.try_spawn_passenger()
        if self.delivery_limit == 0:
            self.done = True
        self.time += 1
        return self._get_state_json(), reward, self.done, {}

    def attempt_dropoff(self,action):
        pid = self.taxi.passenger
        if pid is None:
            return self.rewards["failed-action"]
        passenger = self.passengers.pop(pid)
        assert passenger.location == self.taxi.node
        if self.taxi.location == passenger.destination:
            self.taxi = Taxi(self.taxi.node, self.taxi.location, None)
            self.delivery_limit -= 1
            # print(f"seed-id: {self.seed_id}. Time: {self.time} Delivered passenger.")
            self.graph.remove_node(pid)
            self.resort_passengers()
            return self.rewards["drop-off"]
        else:
            self.passengers[pid]=passenger
            return self.rewards["failed-action"]

    def attempt_pickup(self,pid):
        if self.taxi.passenger is not None:
            return self.rewards["failed-action"]
        passenger = self.passengers[pid]
        if self.taxi.location == passenger.location:
            self.taxi = Taxi(self.taxi.node, self.taxi.location, pid)
            self.passengers[pid] = Passenger(self.taxi.node, passenger.destination)
            self.graph.remove_edge(pid,passenger.location)
            self.graph.add_edge(pid, 0, attr=[0,1,0,1])
            self.graph.add_edge(0, pid, attr=[0,1,0,-1])
            return self.rewards["base"]
        return self.rewards["failed-action"]

    def attempt_move(self,action):
        start = self.taxi.location
        if start ==action:
            pass #no-op
        elif self.graph.edges[(start,action)]['attr']==[1,0,0,1]:
            self.taxi = Taxi(self.taxi.node, action, self.taxi.passenger)
            self.graph.remove_edge(0,start)
            self.graph.add_edge(0, action, attr=[0,1,0,1])
            self.graph.add_edge(action, 0, attr=[0,1,0,-1])
        return self.rewards["base"]

    def try_spawn_passenger(self):
        """
        Spawns a passenger
        :param a: 
        :returns: 
        """
        if (
            len(self.passengers) < self.concurrent_passengers
            and self.random.uniform() < self.passenger_creation_probability
        ):
            self.add_passenger()
            # print(f"seed-id: {self.seed_id}. Time: {self.time} Spawned passenger.")

    def generate_road_network(self, random_walls):
        if random_walls:
            network = generate_city_maze(self.random)
        else:
            network = nx.grid_2d_graph(self.size, self.size,create_using=nx.DiGraph)
        mapping = {k:v+1 for v,k in enumerate(network.nodes)}
        nx.relabel_nodes(network,mapping,copy=False)
        for n in network.nodes:
            network.nodes[n]['attr']=[1,0,0]
        
        for e in network.edges:
            network.edges[e]['attr']=[1,0,0,1]
        return network

    def add_taxi(self):
        location = self.random.choice(range(1,self.size*self.size+1))
        passenger = None
        self.graph.add_node(0, attr=[0,1,0])
        self.graph.add_edge(0, location, attr=[0,1,0,1])
        self.graph.add_edge(location, 0, attr=[0,1,0,-1])

        return Taxi(0,location,passenger)

    def add_passenger(self):
        pid = self.graph.number_of_nodes()
        location, destination = self.random.choice(range(1,self.size*self.size+1),2,replace=False)
        self.graph.add_node(pid, attr=[0,0,1])
        self.graph.add_edge(pid, location, attr=[0,1,0,1])
        self.graph.add_edge(location, pid, attr=[0,1,0,-1])
        self.graph.add_edge(pid, destination, attr=[0,0,1,1])
        self.graph.add_edge(destination, pid, attr=[0,0,1,-1])

        self.passengers[pid] = Passenger(location,destination)
        
    def resort_passengers(self):
        mapping = {k:v for v,k in enumerate(sorted(self.graph.nodes))}
        nx.relabel_nodes(self.graph,mapping,copy=False)
        new_passengers={}
        for k,v in self.passengers.items():
            new_passengers[mapping[k]]=v
        self.passengers = new_passengers
        if self.taxi.passenger is not None:
            self.taxi = Taxi(self.taxi.node, self.taxi.location, mapping[self.taxi.passenger])

class TaxiWorldSimulatorImage(object):
    def __init__(
        self,
        random,
        size,
        passenger_locations=[],
        passenger_destinations=[],
        wall_locations=None,
        delivery_limit=1,
        concurrent_passengers=1,
        timeout=MAX_EPISODE_LENGTH,
        passenger_creation_probability=1,
        random_walls=True,
        taxi_locations=None,
        rewards=None,
        planning = False
    ):
        """
        Houses the game state and transition dynamics for the taxi world.

        :param size: size of gridworld
        :returns: this is a description of what is returned
        :raises keyError: raises an exception
        """
        self.random = random
        self.seed_id = hash(self.random)
        self.time = 0
        self.next_passenger = 500
        self.size = size
        self.passenger_locations = passenger_locations
        self.passenger_destinations = passenger_destinations
        self.delivery_limit = delivery_limit
        self.concurrent_passengers = concurrent_passengers
        self.timeout = timeout
        self.passenger_creation_probability = passenger_creation_probability
        self.rewards = rewards if rewards is not None else DEFAULT_REWARDS
        self.done = False
        self.planning = planning

        self.graph = self.generate_road_network(random_walls)
        self.mapping = {k+1:v for k,v in enumerate(self.graph.nodes)}

        self.taxi = self.add_taxi()

        self.passengers = {}
        self.add_passenger()
        

    def _get_state_image(self):
        return env_to_image(self)

    def act(self, action):
        """
        Advances the game state by one step
        :param action: action provided by the agent
        :returns: observation of the next state
        :raises assertionError: raises an exception if action is invalid
        """
        # action, param = action
        enum_action = ACTIONS(action+1)

        if enum_action == ACTIONS.pickup:
            reward = self.attempt_pickup()
        elif enum_action == ACTIONS.dropoff:
            reward = self.attempt_dropoff()
        else:
            reward = self.attempt_move(enum_action)

        self.try_spawn_passenger()
        if self.delivery_limit == 0:
            self.done = True
        self.time += 1
        return self._get_state_image(), reward, self.done, {}

    def attempt_dropoff(self):
        pid = self.taxi.passenger
        if pid is None:
            return self.rewards["failed-action"]
        passenger = self.passengers.pop(pid)
        assert passenger.location == self.taxi.node
        if self.taxi.location == passenger.destination:
            self.taxi = Taxi(self.taxi.node, self.taxi.location, None)
            self.delivery_limit -= 1
            # print(f"seed-id: {self.seed_id}. Time: {self.time} Delivered passenger.")
            #self.resort_passengers()
            return self.rewards["drop-off"]
        else:
            self.passengers[pid]=passenger
            return self.rewards["failed-action"]

    def attempt_pickup(self):
        if self.taxi.passenger is not None:
            return self.rewards["failed-action"]
        for pid, passenger in self.passengers.items():
            if self.taxi.location == passenger.location:
                self.taxi = Taxi(self.taxi.node, self.taxi.location, pid)
                self.passengers[pid] = Passenger(self.taxi.node, passenger.destination)

            return self.rewards["base"]
        return self.rewards["failed-action"]

    def attempt_move(self,action):
        start = self.taxi.location
        destination = (start[0] + DIRECTIONS[action][0], start[1] + DIRECTIONS[action][1])
        if self.graph.has_edge(start,destination):
            self.taxi = Taxi(self.taxi.node, destination, self.taxi.passenger)

        return self.rewards["base"]

    def try_spawn_passenger(self):
        """
        Spawns a passenger
        :param a: 
        :returns: 
        """
        if (
            len(self.passengers) < self.concurrent_passengers
            and self.random.uniform() < self.passenger_creation_probability
        ):
            self.add_passenger()
            # print(f"seed-id: {self.seed_id}. Time: {self.time} Spawned passenger.")

    def generate_road_network(self, random_walls):
        if random_walls:
            network = generate_city_maze(self.random)
        else:
            network = nx.grid_2d_graph(self.size, self.size,create_using=nx.DiGraph)
        return network

    def add_taxi(self):
        location = self.random.choice(range(1,self.size*self.size+1))
        node_location = self.mapping[location]
        passenger = None
        # self.graph.add_node(0)
        # self.graph.add_edge(0, node_location)
        # self.graph.add_edge(node_location, 0)

        return Taxi(0,node_location,passenger)

    def add_passenger(self):
        pid = self.next_passenger
        self.next_passenger+=1
        location, destination = self.random.choice(range(1,self.size*self.size+1),2,replace=False)
        
        node_location = self.mapping[location]
        node_destination = self.mapping[destination]
        # self.graph.add_node(pid, attr=[0,0,1])
        # self.graph.add_edge(pid, location, attr=[0,1,0,1])
        # self.graph.add_edge(location, pid, attr=[0,1,0,-1])
        # self.graph.add_edge(pid, destination, attr=[0,0,1,1])
        # self.graph.add_edge(destination, pid, attr=[0,0,1,-1])

        self.passengers[pid] = Passenger(node_location,node_destination)
        
    def resort_passengers(self):
        mapping = {k:v for v,k in enumerate(sorted(self.graph.nodes))}
        nx.relabel_nodes(self.graph,mapping,copy=False)
        new_passengers={}
        for k,v in self.passengers.items():
            new_passengers[mapping[k]]=v
        self.passengers = new_passengers
        if self.taxi.passenger is not None:
            self.taxi = Taxi(self.taxi.node, self.taxi.location, mapping[self.taxi.passenger])