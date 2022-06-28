from os import remove
import numpy as np
import torch as th
import networkx as nx

from typing import List, NamedTuple


class Taxi(NamedTuple):
    node: int
    location: int
    passenger: int

class Passenger(NamedTuple):
    node: int
    location: int
    destination: int

class State(NamedTuple):
    graph: int
    taxi: Taxi
    passengers: List[Passenger] 


class Planner:
    
    def plan(self,graph,goal):
        state = graph_to_networkx(graph)
        if goal==state.taxi.node:
            if state.taxi.passenger is not None:
                projection, actions =  deliver_current_passenger(graph,state)
            else:
                projection = graph
                actions = []
        elif goal in [p.node for p in state.passengers]:
            projection, actions = deliver_passenger(graph,state,goal)
        else:
            projection, actions = move(graph,state,goal)

        if actions == []:
            actions = [state.taxi.location]
        
        return increment_timer(projection,actions)

def graph_to_networkx(graph):
    nodes = graph.x.cpu().numpy()
    edges = graph.edge_index.cpu().numpy()
    edge_attr = graph.edge_attr.cpu().numpy()

    passengers=[]
    G = nx.Graph()
    for i,(l,t,p) in enumerate(nodes):
        if l:
            node_type = "location" 
        elif t:
            node_type = "taxi" 
            
            location = edges[1,np.logical_and(edges[0]==i,edge_attr[:,3]==1)]
            taxi=Taxi(i,location.item(),None)
        elif p:
            node_type = "passenger" 
            edge_index = edges[0]==i
            passenger_edges =  edges[:,edge_index]
            passenger_edge_attributes = edge_attr[edge_index,:]
            #quite hacky here. There should always be two edges, one for location and destination, so all we need to do is figure out which one is first.
            if passenger_edge_attributes[0,1] == 1: #location is first
                location = passenger_edges[1,0]
                destination = passenger_edges[1,1]
            else: #destination is first
                location = passenger_edges[1,1]
                destination = passenger_edges[1,0]
            passenger = Passenger(i,location,destination)
            passengers.append(passenger)
        else:
            raise ValueError("Invalid node is neither location, taxi or passenger.")
        G.add_node(i,type=node_type)

    edge_indices = edge_attr[:,0]==1
    G.add_edges_from(edges.T[edge_indices])
  
    return State(G,taxi,passengers)


def deliver_current_passenger(graph,state):
    passenger = [p for p in state.passengers if p.location==state.taxi.node][0]
    move = find_path_to(state,state.taxi.location,passenger.destination)
    move_taxi(graph,state.taxi.node,passenger.destination)
    remove_node_from_graph(graph,passenger.node)
    return graph,move + [state.taxi.node]

def deliver_passenger(graph,state,goal):
    passenger = [p for p in state.passengers if p.node==goal][0]
    if passenger.location == state.taxi.node:
        return deliver_current_passenger(graph,state)
    move1 = find_path_to(state,state.taxi.location,passenger.location)
    move2 = find_path_to(state,passenger.location,passenger.destination)
    move_taxi(graph,state.taxi.node,passenger.destination)
    remove_node_from_graph(graph,passenger.node)
    return  graph,move1 + [passenger.node] + move2 + [state.taxi.node]

def move(graph,state,goal):
    actions = find_path_to(state,state.taxi.location,goal)
    move_taxi(graph,state.taxi.node,goal)
    return graph, actions

def find_path_to(state,start,end):
    path =  nx.shortest_path(state.graph,start,end)
    return path[1:]


def remove_node_from_graph(graph,node):
  
    graph.x = graph.x[:-1] 
    #remove all incoming/outgoing edges 
    edge_index = th.logical_or(graph.edge_index[0]==node,graph.edge_index[1]==node)
    graph.edge_index = graph.edge_index[:,th.logical_not(edge_index)]
    graph.edge_attr = graph.edge_attr[th.logical_not(edge_index)]
    #resort nodes after removed node
    graph.edge_index = th.where(graph.edge_index>node,graph.edge_index-1,graph.edge_index)

def move_taxi(graph,taxi,node):
    graph.edge_index[1,graph.edge_index[0]==taxi] = node
    taxi_edge_backwards_index = th.logical_and(graph.edge_index[1]==taxi,graph.edge_attr[:,3]==-1)
    graph.edge_index[0,taxi_edge_backwards_index] = node
    
def increment_timer(projection,actions):
    projection.global_features[0,0] = projection.global_features[0,0] - (len(actions)/2000)
    return projection, actions