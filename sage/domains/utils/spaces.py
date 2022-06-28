import gym.spaces 
import numpy as np
from gym.utils import seeding
from sage.domains.utils.representations import json_to_graph

class List(gym.spaces.Space):
    """
    A json observation 
    """

    def __init__(self, dimensions, max_size=1000, dtype=np.int32):
        self.dimensions = dimensions

        self.max_size = max_size
       
        self.dtype = dtype

        import numpy as np  # takes about 300-400ms to import, so we load lazily

        self.shape = None if self.dimensions is None else tuple([max_size]*dimensions)
        self._np_random = None

    # def sample(self): #box
    #     pass

    # def contains(self, x):#box
    #     """ A method for validating x is a valid member of the Json observation.
    #     """
    #     pass

    def __eq__(self, other):  # box
        pass

class JsonGraph(gym.spaces.Box):
    """
    A json observation 
    """

    def __init__(self, converter=json_to_graph,planner=None,node_dimension=1,edge_dimension=2):
        import numpy as np  # takes about 300-400ms to import, so we load lazily

        self.converter = converter
        self.planner = planner
        self.shape = (1,)
        self.dtype = np.dtype("U100000")
        self.node_dimension = node_dimension
        self.edge_dimension = edge_dimension
        

        
        self._np_random = None


class BinaryAction(gym.spaces.MultiDiscrete):
    """
    A binary action predicate stub
    """
    def __init__(self):
        self.nvec = np.asarray([1,1], dtype=np.int64)

class NodeAction(gym.spaces.Space):
    """
    A json observation 
    """

    def __init__(self, dimensions, dtype=np.int32):
        self.dimensions = dimensions
       
        self.dtype = dtype

        import numpy as np  # takes about 300-400ms to import, so we load lazily

        self._np_random = None


class Autoregressive(gym.spaces.Space):
    """
    A json observation 
    """

    def __init__(self, spaces, dtype=np.int32):
        self.dimensions = len(spaces)

        self.spaces = spaces
       
        #self.dtype = dtype

        import numpy as np  # takes about 300-400ms to import, so we load lazily

        #self.shape = None if self.dimensions is None else tuple([max_size]*dimensions)
        self._np_random = None

