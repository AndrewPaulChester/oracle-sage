import torch, numpy as np
import torch_geometric, torch_scatter

from torch.nn import *
from torch_geometric.nn import MessagePassing, GlobalAttention
from torch_geometric.data import Data, Batch

def get_start_indices(splits):
    splits = torch.roll(splits, 1)
    splits[0] = 0

    start_indices = torch.cumsum(splits, 0)
    return start_indices

def masked_segmented_softmax(energies, mask, start_indices, batch_ind):
    mask = mask + start_indices
    mask_bool = torch.ones_like(energies, dtype=torch.bool) # inverse mask matrix
    mask_bool[mask] = False

    energies[mask_bool] = -np.inf
    probs = torch_geometric.utils.softmax(energies, batch_ind) # to probs ; per graph

    return probs

def segmented_sample(probs, splits):
    probs_split = torch.split(probs, splits)
    samples = [torch.multinomial(x, 1) for x in probs_split]
    
    return torch.cat(samples)

def segmented_scatter_(dest, indices, start_indices, values):
    real_indices = start_indices + indices
    dest[real_indices] = values
    return dest

def segmented_gather(src, indices, start_indices):
    real_indices = start_indices + indices
    return src[real_indices]

EMB_SIZE = 32
# ----------------------------------------------------------------------------------------
class MultiMessagePassing(Module):
    def __init__(self, steps, node_in_size=EMB_SIZE, node_out_size=EMB_SIZE, edge_size=2, global_size=EMB_SIZE, agg_size=EMB_SIZE,activation_fn=LeakyReLU):
        super().__init__()

        # if node_in_size is None:
        #     node_in_size = [EMB_SIZE] * size
            

        self.gnns = ModuleList( [GraphNet(node_in_size, edge_size, global_size, agg_size, node_out_size,activation_fn) for i in range(steps)] )
        self.pools = ModuleList( [GlobalNode(node_out_size, global_size,activation_fn) for i in range(steps)] )

        self.steps = steps

    def forward(self, x, x_global, edge_attr, edge_index, batch_ind, num_graphs):
        for i in range(self.steps):
            x = self.gnns[i](x, edge_attr, edge_index, x_global, batch_ind)
            x_global = self.pools[i](x_global, x, batch_ind)

        return x, x_global

# ----------------------------------------------------------------------------------------
class GlobalNode(Module):       
    def __init__(self, node_size, global_size,activation_fn):
        super().__init__()

        att_mask = Linear(node_size, 1)
        att_feat = Sequential( Linear(node_size, node_size), activation_fn() )

        self.glob = GlobalAttention(att_mask, att_feat)
        self.tranform = Sequential( Linear(global_size * 2, global_size), activation_fn() )

    def forward(self, xg_old, x, batch):
        xg = self.glob(x, batch)

        xg = torch.cat([xg, xg_old], dim=1)
        xg = self.tranform(xg) + xg_old # skip connection

        return xg

# ----------------------------------------------------------------------------------------
class GraphNet(MessagePassing):
    def __init__(self, node_in_size, edge_size, global_size, agg_size, node_out_size,activation_fn):
        super().__init__(aggr='max')

        self.f_mess = Sequential( Linear(node_in_size + edge_size, agg_size), activation_fn() )
        self.f_agg  = Sequential( Linear(node_in_size + global_size + agg_size, node_out_size), activation_fn() )

    def forward(self, x, edge_attr, edge_index, xg, batch):
        xg = xg[batch]

        return self.propagate(edge_index, x=x, edge_attr=edge_attr, xg=xg)

    def message(self, x_j, edge_attr):
        z = torch.cat([x_j, edge_attr], dim=1)
        z = self.f_mess(z)

        return z 

    def update(self, aggr_out, x, xg):
        z = torch.cat([x, xg, aggr_out], dim=1)
        z = self.f_agg(z) + x # skip connection

        return z
 