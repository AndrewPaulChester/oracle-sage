from sage.forks.stable_baselines3.stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import torch as th
from collections import Counter
    
class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        #value = np.random.random()
        if self.verbose > 1:
            self.logger.record('values', self.locals["values"])
            self.logger.record('action_probabilities', self.locals["log_probs"].exp())
            try:
                c = Counter([(x,y) for x,y in self.locals["actions"]])
                for k,v in c.items():
                    self.logger.record(f'actions/{k}',v)
            except TypeError:
                c = Counter(self.locals["actions"])
                for k,v in c.items():
                    self.logger.record(f'actions/{k}',v)
            for name, parameters in self.locals["self"].policy.named_parameters():
                self.logger.record(f'weights/{name}', parameters.T)
            
        return True
