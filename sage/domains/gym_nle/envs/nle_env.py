from collections import OrderedDict
from typing import Tuple, Union

import gym
import nle  # noqa: F401
import numpy as np

from typing import NamedTuple, List
from sage.domains.gym_nle.utils.representations import env_to_json
from sage.domains.gym_nle.simulator.planner import Planner
from sage.domains.utils.representations import EMB_SIZE, json_to_graph
from sage.domains.utils.spaces import JsonGraph, NodeAction, Autoregressive
from sage.domains.gym_nle.utils.glyphs import Glyphs
from sage.domains.gym_nle.utils.representations import decode_action

from nle import nethack
from sage.domains.gym_nle.envs.tasks import create_env

ORIGINAL_ACTION_COUNT = 8

DIRECTIONS = {
    (1,0):0, #N
    (0,-1):1, #E
    (-1,0):2, #S
    (0,1):3, #W
    (1,-1):4, #NE
    (-1,-1):5, #SE
    (-1,1):6, #SW
    (1,1):7, #NW
}
            # penalty_step=flags.,
            # penalty_time=flags.penalty_time,
            # penalty_mode=flags.fn_penalty_step,

class Config(NamedTuple):
    env: str
    obs_keys: str = "glyphs,chars,colors,specials,blstats,message,inv_glyphs,inv_letters,inv_oclasses,inv_strs"
    penalty_step: float = -0.001
    penalty_time: float = 0.0
    fn_penalty_step: str = "constant"
    model: str = "standard"
    save_tty: str = None
    state_counter: str = "none"
    seedspath: str = None
    planning: bool = False

ENV_TILESETS = {
    'small_room':'movement',
    'small_room_random':'movement',
    'big_room':'movement',
    'big_room_random':'movement',

    'lava':'lava',
    'lava_lev':'lava',
    'lava_lev_potion_inv':'lava',
    'lava_lev_potion_pick':'lava',

    'mini_wear':'lava',

    
    'hungry':'minimal',
    'hungry_rooms':'minimal',
    'hungry_fixed3':'minimal',
    'hungry_fixed5':'minimal',
    'hungry_danger5':'minimal',
    
}
class SAGENLEEnv(gym.Env):
    def __init__(self, env_config: dict) -> None:
        # We sort the observation keys so we can create the OrderedDict output
        # in a consistent order
        
        config = Config(**env_config)
        self.gym_env = create_env(config)
        self.glyphs = Glyphs(ENV_TILESETS[env_config['env']])
        node_dim = len(self.glyphs.get_glyph_encoding(2378))
        self.observation_space = JsonGraph(
            converter=json_to_graph,node_dimension=node_dim,edge_dimension=6, planner=Planner(self.glyphs)
        )
        self.planning = config.planning
        if self.planning:
            self.action_space = Autoregressive([NodeAction(EMB_SIZE),gym.spaces.Discrete(ORIGINAL_ACTION_COUNT)])
        else:
            self.action_space = gym.spaces.Discrete(ORIGINAL_ACTION_COUNT)
        self.action_mapping = {}
        for i,a in enumerate(self.gym_env._actions):
            self.action_mapping[a.value]=i

        self.env_name = env_config['env']
        if self.env_name == 'lava':
            self.log_progress = {"pickup":0,"lev":0,"freeze":0,"lev_or_freeze":0,"cross_lava":0,"cross_lava_lev":0,"cross_lava_freeze":0,"win_lev":0,"win_freeze":0}
            self.itemset = set([2131]).union(range(2049,2084),range(2178,2203),range(2289,2316))
        elif self.env_name == 'hungry_danger5':
            self.log_progress = {"len100":0,"len200":0}

    # @property
    # def action_space(self) -> gym.Space:
    #     return self.gym_env.action_space

    # @property
    # def observation_space(self) -> gym.Space:
    #     return self.gym_env.observation_space

    def _process_obs(self, obs: dict) -> dict:
        message = "".join([chr(c) for c in obs['message']])

        self.last_obs = env_to_json(obs,self.glyphs,self.planning)
 
        
        if self.env_name == 'lava': #performing logging of intermediate progress
            if not set(obs['inv_glyphs']).isdisjoint(self.itemset): #if item is in inventory
                self.log_progress['pickup']=1

            if (self.obs['glyphs'][:,self.lava_x]==2378).any(): #if there are floor squares in lava row
                self.log_progress['freeze']=1
                self.log_progress['lev_or_freeze']=1

            if self.obs['blstats'][nethack.NLE_BL_CONDITION] & nethack.BL_MASK_LEV: # if levitate flag is set
                self.log_progress['lev']=1
                self.log_progress['lev_or_freeze']=1

            if (np.where(self.obs['glyphs']==329)[1][0]) > self.lava_x: #if person is right of lava
                self.log_progress['cross_lava']=1
                if self.log_progress['lev']:
                    self.log_progress['cross_lava_lev']=1
                if self.log_progress['freeze']:
                    self.log_progress['cross_lava_freeze']=1
        elif self.env_name == 'hungry_danger5':
            self.log_progress['len100']=obs['blstats'][20]
            self.log_progress['len200']=obs['blstats'][20]


        return self.last_obs

    def _process_action(self, action: int) -> int:
        self.state = json_to_graph([[self.last_obs]])
        action,argument = decode_action(action)

        if action is None: #None action means pass argument directly to NLE (i.e. pickup, descend.)
            return [argument]
        state = self.state
        target_node = state.x[action]
        player_location = state.edge_index[1,np.logical_and(state.edge_index[0]==0,state.edge_attr[:,1]==1).bool()]
        if action == player_location.item():
            return [17] #wait, effectively a no-op
            
        item_id = self.glyphs.get_glyph_from_encoding(state.x[action])
        if (item_id >= 2373 and item_id <=2398) or item_id in(2411,2353):
            direction = state.edge_attr[np.logical_and(state.edge_index[0]==player_location,state.edge_index[1]==action).bool()]
            if len(direction):
                return [DIRECTIONS[tuple(direction[0,-2:].tolist())]]
            else:
                print("not in expected square, waiting")
                return [17] #wait, effectively a no-op
        else:
            processed_action = self.glyphs.get_default_action(item_id)
            item_index = np.where(self.obs['inv_glyphs']==item_id)[0][0]
            inv_char = self.obs['inv_letters'][item_index]
            item_action = self.action_mapping[inv_char]
            if processed_action == 51: #if its a ring to be put on
                return [processed_action,item_action,54] 
            if processed_action == 20 and argument is not None: #if it's a horn
                return [processed_action,item_action,7,argument]
            if processed_action == 75: #if it's a wand
                return [processed_action,item_action,argument]
            if processed_action == 72: #if it's equipment
                return [processed_action,item_action]
            if processed_action == 52: #if it's a potion
                return [processed_action,item_action]
            if processed_action == 73: #if it's a weapon
                return [processed_action,item_action]
            if processed_action == 29: #if it's a comestible
                return [processed_action,item_action]
            if processed_action == 66: #if it's a projectile
                return [processed_action,item_action,argument]
            

    def reset(self) -> dict:
        self.obs = self.gym_env.reset()
        if self.env_name == 'lava': #performing logging of intermediate progress
            self.lava_x = np.where(self.obs['glyphs']==2393)[1][0]
            self.log_progress = {"pickup":0,"lev":0,"freeze":0,"lev_or_freeze":0,"cross_lava":0,"cross_lava_lev":0,"cross_lava_freeze":0,"win_lev":0,"win_freeze":0}
        elif self.env_name == 'hungry_danger5':
            self.log_progress = {"len100":0,"len200":0}

        return self._process_obs(self.obs)

    def step(
        self, action: Union[int, np.int64]
    ) -> Tuple[dict, Union[np.number, int], Union[np.bool_, bool], dict]:
        action_sequence = self._process_action(action)
        for action in action_sequence:
            obs, reward, done, info = self.gym_env.step(action)
            if done:
                break

        self.obs = obs
        if done: #Task completed, but final obs is kinda broken?
            
            if self.env_name == 'lava': #performing logging of intermediate progress
                if info['end_status'] == 2:
                    self.log_progress['win_freeze']=self.log_progress['freeze']
                    self.log_progress['win_lev']=self.log_progress['lev']
                info.update(**self.log_progress)
            
            elif self.env_name == 'hungry_danger5':
                if info['end_status'] == 1: # if dead, set to either 200 or 100
                    self.log_progress['len100']=100
                    self.log_progress['len200']=200

                info.update(**self.log_progress)
            return "",reward,done,info #In theory this obs never gets used as vec env resets automatically on task completion?
        else:
            return self._process_obs(obs), reward, done, info

    def render(self,mode):
        return self.gym_env.render()

    def close(self):
        return self.gym_env.close()



def compare_obs(prev,new):
    for p,n in zip(prev.values(),new.values()):
        if not (p==n).all():
            print(p)
            print(n)