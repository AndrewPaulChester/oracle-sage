"""
.. module:: utils
   :synopsis: Contains miscellaneous utility functions.
"""
import numpy as np
from math import floor, ceil
import networkx as nx


GRID_SIZE = 20



def generate_maze(random, width=20, height=20, complexity=0.1, density=0.02):
    """
    Creates a randomly generated rectangular maze on a square grid.
    :param width: width of maze
    :param height: height of maze
    :param complexity: Controls the length of each maze segment (as a percentage)
    :param density: Controls the number of maze segments (as a percentage)
    :returns: Array containing maze layout (True = wall)
    """
    # Only odd shapes
    shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
    # Adjust complexity and density relative to maze size
    complexity = int(complexity * (5 * (shape[0] + shape[1])))  # number of components
    density = int(density * ((shape[0] // 2) * (shape[1] // 2)))  # size of components
    # Build actual maze
    Z = np.zeros(shape, dtype=bool)
    # Fill borders
    Z[0, :] = Z[-1, :] = 1
    Z[:, 0] = Z[:, -1] = 1
    # Make aisles
    for i in range(density):
        x, y = (
            random.randint(0, shape[1] // 2) * 2,
            random.randint(0, shape[0] // 2) * 2,
        )  # pick a random position
        Z[y, x] = 1
        for j in range(complexity):
            neighbours = []
            if x > 1:
                neighbours.append((y, x - 2))
            if x < shape[1] - 2:
                neighbours.append((y, x + 2))
            if y > 1:
                neighbours.append((y - 2, x))
            if y < shape[0] - 2:
                neighbours.append((y + 2, x))
            if len(neighbours):
                y_, x_ = neighbours[random.randint(0, len(neighbours) - 1)]
                if Z[y_, x_] == 0:
                    Z[y_, x_] = 1
                    Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                    x, y = x_, y_
    return Z


def generate_random_walls(grid_size, random, complexity=0.1, density=0.02):
    """
    Creates a list of random walls .
    :param grid_size: dimension of gridworld
    :param complexity: Controls the length of each wall section (as a percentage)
    :param density: Controls the number of wall section (as a percentage)
    :returns: List of walls represented as edges in lattice
    :raises AssertError: if complexity and density values are invalid, must be in (0,1]
    """
    assert (
        complexity > 0 and complexity <= 1
    ), f"complexity value must be in the range (0,1] (currently {complexity})"
    assert (
        density > 0 and density <= 1
    ), f"density value must be in the range (0,1] (currently {density})"
    maze_size = 2 * grid_size + 1
    m = generate_maze(random, maze_size, maze_size, complexity, density)[1:-1, 1:-1]
    walls = []
    for ((x, y), v) in np.ndenumerate(m):
        if v and x % 2 + y % 2 == 1:  # xor of odd(x) and odd(y)
            walls.append(
                (
                    (int(floor(x / 2)), int(floor(y / 2))),
                    (int(ceil(x / 2)), int(ceil(y / 2))),
                )
            )
    return walls

def is_boundary_edge(edge):
    (_, (x,y)) = edge
    if x in (3,16) or y in (3,16):
        return True
    return False

def generate_city_maze(random):
    city = None
    while city is None:
        city = try_generate_city_maze(random)
    return city

def try_generate_city_maze(random):
    """
    Creates a randomly generated city maze on a square grid.
    The city is divided into three sections, the city center (6x6), the suburbs (12x12) and the outskirts (20x20).
    Not very flexible at the moment, really only allows for generation of full 20x20 size cities.
    :param random: seeded numpy random state
    :returns: Networkx graph containing road connectivity.
    """

    #start with a fully connected grid graph
    network = nx.grid_2d_graph(GRID_SIZE, GRID_SIZE)
    
    #divide outskirts into nodes connected to the suburbs (initially the border nodes)
    # and those unconnected.
    outskirt_edge_to_keep = []
    outskirt_nodes_connected = set()
    outskirt_nodes_connected.update([(x,y) for x in [4,15] for y in range(4,16)])
    outskirt_nodes_connected.update([(x,y) for y in [4,15] for x in range(4,16)])
    
    outskirt_nodes_unconnected = set()
    outskirt_nodes_unconnected.update([(x,y) for x in [0,1,2,3,16,17,18,19] for y in range(20)])
    outskirt_nodes_unconnected.update([(x,y) for y in [0,1,2,3,16,17,18,19] for x in range(4,16)])

    outskirt_edges = list(network.edges(outskirt_nodes_unconnected))

    #for any remaining nodes that aren't connected to the suburbs, do a random walk until you hit a node that is connected, then keep all those edges.
    while outskirt_nodes_unconnected:  
        disconnected = True
        current = outskirt_nodes_unconnected.pop()
        path = []
        while disconnected:
            current_edges = list(network.edges(current))
            index_keep = random.randint(len(current_edges))
            edge = current_edges[index_keep]
            outskirt_edge_to_keep.append(edge)
            next_node = edge[1]
            if(next_node == current):
                print("edge starts with wrong node")
                next_node = edge[0]
            path.append(current)
            if next_node in outskirt_nodes_connected:
                disconnected = False
            else:
                current = next_node
                outskirt_nodes_unconnected.difference_update([current])
                
        outskirt_nodes_connected.update(path)
        
    #remove all outskirt edges except those we flagged to keep earlier.
    network.remove_edges_from(outskirt_edges)
    network.add_edges_from(outskirt_edge_to_keep)

    #for the suburbs, remove one edge from each node that still has all 4 edges connected.
    suburbs = set()
    suburbs.update([(x,y) for x in range(4,16) for y in range(4,16)])
    suburbs.difference_update([(x,y) for x in range(7,13) for y in range(7,13)])

    for node in suburbs:
        edges = network.edges(node)
        if len(edges) > 3:
            remove_index=random.randint(len(edges))
            to_remove = list(edges)[remove_index]
            if not is_boundary_edge(to_remove):
                network.remove_edge(*to_remove)


    #sometimes we accidentally disconnect sections of the graph. If so, try to add an edge back to recover it. 
    cc = list(nx.connected_components(network))

    if len(cc) > 1:
        small = min(cc, key=len)
        (x,y) = small.pop()
        to = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
        for node in to:
            if node not in small:
                network.add_edge((x,y),node)
                break                
    cc = list(nx.connected_components(network))
    # if we failed to combine all connected components (rare), just give up and start again.
    if len(cc) > 1:
        #print("did not recover")
        return None

    return nx.DiGraph(network)
