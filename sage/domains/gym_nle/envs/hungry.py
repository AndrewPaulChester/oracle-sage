# Copyright (c) Facebook, Inc. and its affiliates.
from minihack import MiniHackSkill, RewardManager, LevelGenerator
from minihack.reward_manager import Event
from gym.envs import registration
from nle.nethack import Command
from nle import nethack
import numpy as np

MOVE_ACTIONS = tuple(nethack.CompassDirection)
NAVIGATE_ACTIONS = tuple(
    list(MOVE_ACTIONS) + [Command.OPEN, Command.KICK, Command.SEARCH]
)

class StairEvent(Event):
    def __init__(self, *args):
        super().__init__(*args)

    def check(self, env, previous_observation, action, observation) -> float:
        coordinates = tuple(observation[env._blstats_index][:2])

        if (previous_observation[env._original_observation_keys.index("glyphs")][coordinates[1],coordinates[0]] == 2383 and
            observation[env._original_observation_keys.index("glyphs")][coordinates[1],coordinates[0]] in (329,337)):
            return self._set_achieved()
        return 0.0

class DeathEvent(Event):
    def __init__(self, *args):
        super().__init__(*args)

    def check(self, env, previous_observation, action, observation) -> float:
        coordinates = tuple(observation[env._blstats_index][:2])

        if coordinates==(0,0):
            return self._set_achieved()
        return 0.0

class MiniHackHungry(MiniHackSkill):
    """Environment for "hungry" task.

    The agent has to navigate itself through randomely generated corridors that
    connect several rooms and find the goal.
    """

    def __init__(self, *args, des_file, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 100)
        reward_manager = RewardManager()
        reward_manager.add_event(StairEvent(10,False,True,True))
        reward_manager.add_event(DeathEvent(-10,False,True,True))
        kwargs["reward_win"] = kwargs.pop("reward_win", 10)
        kwargs["reward_lose"] = kwargs.pop("reward_lose", -10)
        kwargs["penalty_mode"] = kwargs.pop("penalty_mode", "always")
        kwargs["penalty_step"] = kwargs.pop("penalty_step", -0.1)
        
        kwargs["reward_manager"] = reward_manager
        super().__init__(*args, des_file=des_file, **kwargs)




class MiniHackHungry5(MiniHackHungry):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, des_file="hungry5.des", **kwargs)

class MiniHackHungry1(MiniHackHungry):
    def __init__(self, *args, **kwargs):
        des_file = """
MAZE: "mylevel",' '
FLAGS:hardfloor,premapped
INIT_MAP: solidfill,' '
GEOMETRY:center,center
MAP
.................
.................
.................
.................
.................
.................
.................
.................
.................
.................
.................
.................
.................
.................
.................
.................
.................
ENDMAP
REGION:(0,0,20,20),lit,"ordinary"


$all_region = (0,0,17,17)


$top_left_region = selection:fillrect (0,0,9,9)
$top_right_region = selection:fillrect (10,0,16,9)
$bottom_left_region = selection:fillrect (0,10,9,16)
$bottom_right_region = selection:fillrect (10,10,16,16)

REPLACE_TERRAIN: $all_region, '.', 'F', 20%

[50%]: OBJECT:('%', "meatball"), rndcoord($top_right_region)
[50%]: OBJECT:('%', "meatball"), rndcoord($top_left_region)
[50%]: OBJECT:('%', "meatball"), rndcoord($bottom_left_region)
[50%]: OBJECT:('%', "meatball"), rndcoord($bottom_right_region)

BRANCH: (0,0,4,4),(7,7,7,7)
[20%]: STAIR:rndcoord($top_left_region),down
[80%]: STAIR:rndcoord($top_right_region),down
[80%]: STAIR:rndcoord($bottom_left_region),down
STAIR:rndcoord($bottom_right_region),down
"""
        super().__init__(*args, des_file=des_file, **kwargs)

class MiniHackHungryLarge(MiniHackHungry):
    def __init__(self, *args, **kwargs):
        des_file = """
MAZE: "mylevel",' '
FLAGS:hardfloor,premapped
INIT_MAP: solidfill,' '
GEOMETRY:center,center
MAP
........................................
........................................
........................................
........................................
........................................
........................................
........................................
........................................
........................................
........................................
........................................
........................................
........................................
........................................
........................................
........................................
........................................
........................................
........................................
........................................
ENDMAP
REGION:(0,0,40,20),lit,"ordinary"


$all_region = (1,1,39,39)


$top_left_region = selection:fillrect (0,0,19,9)
$top_right_region = selection:fillrect (19,0,39,9)
$bottom_left_region = selection:fillrect (0,10,19,19)
$bottom_right_region = selection:fillrect (20,10,39,39)

REPLACE_TERRAIN: $all_region, '.', 'F', 20%

[50%]: OBJECT:('%', "meatball"), rndcoord($top_right_region)
[50%]: OBJECT:('%', "meatball"), rndcoord($top_left_region)
[50%]: OBJECT:('%', "meatball"), rndcoord($bottom_left_region)
[50%]: OBJECT:('%', "meatball"), rndcoord($bottom_right_region)

BRANCH: (0,0,6,6),(7,7,7,7)
[20%]: STAIR:rndcoord($top_left_region),down
[80%]: STAIR:rndcoord($top_right_region),down
[80%]: STAIR:rndcoord($bottom_left_region),down
STAIR:rndcoord($bottom_right_region),down
"""
        super().__init__(*args, des_file=des_file, **kwargs)



class MiniHackHungryFixed3(MiniHackHungry):
    def __init__(self, *args, **kwargs):
        des_file = """
MAZE: "mylevel",' '
FLAGS:hardfloor,premapped
INIT_MAP: solidfill,' '
GEOMETRY:center,center
MAP
.....      ..........    .....
.....      .    .        .....
............    .        .....
.....      .    .        .....
.....      .    ..............
 .         .                  
 .         .... ..............
.....      .       .     .....
.....      .       .     .....
.....      .       .     .....
.....      .       .     .....
.....      .       .          
           .       .          
           .       .          
           .       .     .....
 ...........       .     .....
 .....     .       .     .....
 .....     .       .     .....
 .....     .       .     .....
 .....     .       ...........       
           .........          
ENDMAP
REGION:(0,0,35,21),lit,"ordinary"

BRANCH: (0,0,0,0),(1,1,1,1)
STAIR: (04,11,04,11),(0,0,0,0),down
STAIR: (29,04,29,04),(0,0,0,0),down
STAIR: (29,19,29,19),(0,0,0,0),down
"""
        super().__init__(*args, des_file=des_file, **kwargs)

class MiniHackHungryFixed5(MiniHackHungry):
    def __init__(self, *args, **kwargs):
        des_file = """
MAZE: "mylevel",' '
FLAGS:hardfloor,premapped
INIT_MAP: solidfill,' '
GEOMETRY:center,center
MAP
.....      ..........    .....
.....      .    .        .....
............    .        .....
.....      .    .        .....
.....      .    ..............
 .         .                  
 .         .... ..............
.....      .       .     .....
.....      .       .     .....
.....      .       .     .....
.....      .       .     .....
.....      .       .          
           .       .          
           .       .          
           .       .     .....
 ...........       .     .....
 .....     .       .     .....
 .....     .       .     .....
 .....     .       .     .....
 .....     .       ...........       
           .........          
ENDMAP
REGION:(0,0,35,21),lit,"ordinary"

BRANCH: (0,0,0,0),(1,1,1,1)
STAIR: (04,11,04,11),(0,0,0,0),down
STAIR: (05,19,05,19),(0,0,0,0),down
STAIR: (29,04,29,04),(0,0,0,0),down
STAIR: (29,10,29,10),(0,0,0,0),down
STAIR: (29,19,29,19),(0,0,0,0),down
"""
        super().__init__(*args, des_file=des_file, **kwargs)




class MiniHackHungryDanger5(MiniHackHungry):
    def __init__(self, *args, **kwargs):
        
        des = self.get_random_des()
        super().__init__(*args, des_file=des, **kwargs)

    def get_random_des(self):

        maze = """
.....FFFFFF............FF.....
.....FFFFFF.FFFFFFFFFF.FF.....
............FFFFFFFFF..FF.....
.....FFFFFF.FFFFFFFFF.FFF.....
.....FFFFFF.FFFFFFF...........
F.FFFFFFFFF.FFFFFFF.FFFFFFFFFF
F.......FFF....F..............
F.FFFFF.FFF.FF.FFFF.FFFFF.....
F.FFFFF.....FF.FFFF.FFFFF.....
.....FFFFFF.FF.FFFF.FFFFF.....
.....FFFFFF.FF.FFFF.FFFFF.....
.....FFFFFF.FF.FFFF.FFFFFFFFFF
.....FFFFFF.FF.FFFF.FFFFFFFFFF
.....FFFFFF.FF.FFFF.FFFFFFFFFF
FFFFFFFFFFF.FF......FFFFF.....
FFFF........FFFFFFF.FFFFF.....
FFFF.....FF.FFFFFFF.FFFFF.....
FFFF.....FF.FFFFFFF.FFFFF.....
FFFF.....FF.FFFFFFF.FFFFF.....
FFFF.....FF.FFFFFFF...........
FFFFFFFFFFF.........FFFFFFFFFF
"""

        lvl_gen = LevelGenerator(map=maze,flags=("hardfloor","premapped"))
        lvl_gen.set_start_pos((0, 0))
        lvl_gen.add_goal_pos((4, 13))
        lvl_gen.add_goal_pos((8, 19))
        lvl_gen.add_goal_pos((29, 4))
        lvl_gen.add_goal_pos((29, 10))
        lvl_gen.add_goal_pos((29, 19))
        
        lvl_gen.add_boulder((23,4))
        lvl_gen.add_boulder((23,6))
        lvl_gen.add_boulder((23,19))
        lvl_gen.add_boulder((1,8))
        lvl_gen.add_boulder((9,15))

        lvl_gen.add_trap('pit',(24,4))
        lvl_gen.add_trap('pit',(24,6))
        lvl_gen.add_trap('pit',(24,19))
        lvl_gen.add_trap('pit',(1,9))
        lvl_gen.add_trap('pit',(8,15))
        
        #always one mastodon in top right / bottom left
        if np.random.rand() < 0.5:
            lvl_gen.add_monster(name="mastodon", place=(29, 3))
        else:
            lvl_gen.add_monster(name="mastodon", place=(8, 18))
        # one in the other three
        rand = np.random.rand()
        if rand < 0.3333:
            lvl_gen.add_monster(name="mastodon", place=(29, 18))
        elif rand < 0.6666:
            lvl_gen.add_monster(name="mastodon", place=(29, 9))
        else:
            lvl_gen.add_monster(name="mastodon", place=(4, 12))





        rand = np.random.rand()
        if rand < 0.3333: 
            lvl_gen.add_terrain((13,0),'F') # A
            lvl_gen.add_terrain((11,7),'F') # F
            lvl_gen.add_terrain((2,6),'F') # X
        else: 
            
            lvl_gen.add_terrain((14,10),'F') # G
            coords = [(13,0),(19,5),(19,10),(15,20),(11,12),(11,7)] #A-F
            lvl_gen.add_terrain(coords[np.random.randint(6)],'F')
            
            if np.random.rand() < 0.8:
                lvl_gen.add_terrain((2,6),'F') 


        des = lvl_gen.get_des()
        return des

    def reset(self, *args, **kwargs):
        des_file = self.get_random_des()
        self.update(des_file)
        return super().reset(*args, **kwargs)

















































