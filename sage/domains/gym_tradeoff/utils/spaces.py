from gym.spaces import Tuple
from functools import reduce
import operator
import numpy as np
from gym.utils import seeding


class Json(object):
    """
    A json observation 
    """

    def __init__(self, grid_size, image=None, converter=None):
        self.grid_size = grid_size

        self.image = image
        self.converter = converter
        self.controller_converter = lambda x: x
        self.shape = (1,)
        self.dtype = "U100000"

        import numpy as np  # takes about 300-400ms to import, so we load lazily

        self.shape = None if self.shape is None else tuple(self.shape)
        self.dtype = None if self.dtype is None else np.dtype(self.dtype)
        self.np_random = None
        self.seed()

    def seed(self, seed=None):
        """Seed the PRNG of this space. """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @property
    def size(self):
        if isinstance(self.image, Tuple):
            return sum([reduce(operator.mul, box.shape, 1) for box in self.image])
        return reduce(operator.mul, self.image.shape, 1)

    # def sample(self): #box
    #     pass

    # def contains(self, x):#box
    #     """ A method for validating x is a valid member of the Json observation.
    #     """
    #     pass

    def __eq__(self, other):  # box
        pass
