"""
.. module:: taxi_env
   :synopsis: Provides gym environment wrappers for the underlying taxi simulator.
"""

import sys
import re
from contextlib import closing
from six import StringIO
import numpy as np
import gym
import json
from scipy.spatial import distance
import networkx as nx
from ast import literal_eval as make_tuple

from gym import error, spaces, utils
from gym.utils import seeding
from sage.domains.gym_taxi.simulator.taxi_world import TaxiWorldSimulator, ACTIONS, TaxiWorldSimulatorImage
from sage.domains.gym_taxi.simulator.planner import Planner
from sage.domains.utils.spaces import JsonGraph
from sage.domains.gym_taxi.utils.spaces import Json
from sage.domains.utils.representations import (
    json_to_graph,
)
from sage.domains.gym_taxi.utils.config import (
    MAP,
    MAX_EPISODE_LENGTH,
    LOCS,
    DISCRETE_ENVIRONMENT_STATES,
    ORIGINAL,
    EXPANDED,
    MULTI,
    FUEL,
    OPEN,
    PREDICTABLE,
    PREDICTABLE5,
    PREDICTABLE10,
    PREDICTABLE15,
    CITY
)
from matplotlib import pyplot as plt

# from simulator
ACTION_COUNT = len(ACTIONS)
ORIGINAL_ACTION_COUNT = 6  # matching standard environment

# gym environment specific
CHANNEL_COUNT = 4
OUTPUT_IMAGE_SIZE = 84


OBS_SPACES = {
    ("screen", "original"): (4, None),
    ("screen", "expanded"): (4, None),
    ("screen", "predictable"): (4, None),
    ("screen", "predictable5"): (4, None),
    ("screen", "predictable10"): (4, None),
    ("screen", "predictable15"): (4, None),
    ("screen", "multi"): (4, None),
    ("screen", "fuel"): (5, None),
    ("mixed", "original"): (2, 6),
    ("mixed", "expanded"): (2, 6),
    ("mixed", "predictable"): (2, 6),
    ("mixed", "predictable5"): (2, 6),
    ("mixed", "predictable10"): (2, 6),
    ("mixed", "predictable15"): (2, 6),
    ("mixed", "multi"): (2, 30),
    ("mixed", "fuel"): (2, 33),
    ("mlp", "original"): (None, 8),
    ("both", "original"): (4, 8),
    ("one-hot", "original"): (1, 75),
}

SCENARIOS = {
    "original": ORIGINAL,
    "expanded": EXPANDED,
    "multi": MULTI,
    "fuel": FUEL,
    "predictable": PREDICTABLE,
    "predictable5": PREDICTABLE5,
    "predictable10": PREDICTABLE10,
    "predictable15": PREDICTABLE15,
    "city": CITY,
}


CONVERTERS = {
    # "screen": json_to_screen,
    # "mixed": json_to_mixed,
    # "mlp": json_to_mlp,
    # "both": json_to_both,
    # "one-hot": json_to_one_hot,
}


DIRECTIONS = {
    (0, -1): "move-up",
    (0, 1): "move-down",
    (-1, 0): "move-left",
    (1, 0): "move-right",
}


def _decode(i):
    """
    Turns discrete representation into form needed to print the ascii map.
    from https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py
    """
    out = []
    out.append(i % 4)
    i = i // 4
    out.append(i % 5)
    i = i // 5
    out.append(i % 5)
    i = i // 5
    out.append(i)
    assert 0 <= i < 5
    return reversed(out)


def _construct_image(representation, scenario):
    channels, length = OBS_SPACES[(representation, scenario)]
    if channels is None:  # purely MLP input
        return spaces.Box(0, 4, (length,), dtype=np.float32)
    if length is None:  # purely image based input
        return spaces.Box(
            0, 1, (channels, OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE), dtype=np.uint8
        )
    else:  # mixed input
        screen = spaces.Box(
            0, 1, (channels, OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE), dtype=np.uint8
        )
        dense = spaces.Box(0, 4, (length,), dtype=np.float32)
        return spaces.Tuple((screen, dense))


class BaseTaxiEnv(gym.Env):
    """
    Base class for all gym taxi environments
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        self.steps = 0
        self.lastaction = None
        self.seed()
        self.sim = self._init_simulator()
        self.action_space = spaces.Discrete(ACTION_COUNT)
        self.score = 0

    def _init_simulator(self):  
        return TaxiWorldSimulator(self.np_random, 5)

    def _step(self, action):
        self.steps += 1
        self.lastaction = action
        obs, reward, done, info = self.sim.act(action)
        self.score += reward
        info["score"] = self.score
        if done:
            # print(f"completed, score of: {self.score}")
            self.score = 0

        if self.steps == self.sim.timeout:
            done = True
            info["bad_transition"] = True
            # print(f"timed out, score of: {self.score}")
            self.score = 0

        return obs, reward, done, info

    def reset(self):
        self.sim = self._init_simulator()
        self.steps = 0
        self.score = 0
        self.lastaction = None
        return self.sim._get_state_json()

    def close(self):
        return

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class FixedTaxiEnv(BaseTaxiEnv):
    """
    Base class for gym taxi environment for 5x5 gridworld
    """

    def __init__(self, rewards=None):
        self.rewards = rewards
        super().__init__()
        self.desc = np.asarray(MAP, dtype="c")
        self.action_space = spaces.Discrete(ORIGINAL_ACTION_COUNT)

    def _init_simulator(self):
        return TaxiWorldSimulator(self.np_random, **ORIGINAL, rewards=self.rewards)

    def render(self, mode="human"):
        outfile = StringIO() if mode == "ansi" else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode("utf-8") for c in line] for line in out]
        taxi_row, taxi_col, pass_idx, dest_idx = _decode(
            json_to_discrete(self.sim._get_state_json())
        )

        def ul(x):
            return "_" if x == " " else x

        if pass_idx < 4:
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                out[1 + taxi_row][2 * taxi_col + 1], "yellow", highlight=True
            )
            pi, pj = LOCS[pass_idx]
            out[1 + pi][2 * pj + 1] = utils.colorize(
                out[1 + pi][2 * pj + 1], "blue", bold=True
            )
        else:  # passenger in taxi
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                ul(out[1 + taxi_row][2 * taxi_col + 1]), "green", highlight=True
            )

        di, dj = LOCS[dest_idx]
        out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], "magenta")
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.lastaction is not None:
            outfile.write(f"({self.lastaction})\n")
        else:
            outfile.write("\n")

        # No need to return anything for human
        if mode != "human":
            with closing(outfile):
                return outfile.getvalue()


class DiscreteTaxiEnv(FixedTaxiEnv):
    """
    Gym taxi environment for 5x5 gridworld with discrete observations
    """

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Discrete(DISCRETE_ENVIRONMENT_STATES)

    def step(self, action):
        obs, reward, done, info = self._step(action)
        return json_to_discrete(obs), reward, done, info

    def reset(self):
        return json_to_discrete(super().reset())


class DiscretePredictableTaxiEnv(FixedTaxiEnv):
    """
    Gym taxi environment for 5x5 gridworld with discrete observations
    """

    def __init__(self, env_name):
        self.env_name = env_name
        super().__init__()
        self.observation_space = spaces.Discrete(DISCRETE_ENVIRONMENT_STATES + 25)

    def _init_simulator(self):
        if self.env_name == "v1":
            return TaxiWorldSimulator(
                self.np_random,
                **PREDICTABLE5,
                rewards={"base": -1, "failed-action": -10, "drop-off": 20},
            )
        elif self.env_name == "v2":
            return TaxiWorldSimulator(
                self.np_random,
                **PREDICTABLE5,
                rewards={"base": 0, "failed-action": 0, "drop-off": 1},
            )
        else:
            raise ValueError("invalid environment name")

    def step(self, action):
        obs, reward, done, info = self._step(action)
        return json_to_discrete_predictable(obs), reward, done, info

    def reset(self):
        return json_to_discrete_predictable(super().reset())


class BoxTaxiEnv(FixedTaxiEnv):
    """
    Gym taxi environment for 5x5 gridworld with image observations
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, rewards=None):
        self.rewards = rewards
        super().__init__(rewards)
        self.observation_space = spaces.Box(
            0, 255, (CHANNEL_COUNT, OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE), dtype=np.uint8
        )

    def _init_simulator(self):
        return TaxiWorldSimulatorImage(self.np_random, **CITY, rewards=self.rewards, planning=False)

    def step(self, action):

        return  self._step(action)

    def reset(self):
        self.sim = self._init_simulator()
        self.steps = 0
        self.score = 0
        self.lastaction = None
        return self.sim._get_state_image()


class JsonTaxiEnv(BaseTaxiEnv):
    def __init__(self, representation, scenario, rewards=None):
        self.rewards = rewards
        self.scenario = SCENARIOS[scenario]
        super().__init__()
        image = _construct_image(representation, scenario)
        self.observation_space = Json(
            self.scenario["size"], image=image, converter=CONVERTERS[representation]
        )
        self.first = True
        if scenario == "original":
            self.action_space = spaces.Discrete(ORIGINAL_ACTION_COUNT)

    def _init_simulator(self):
        return TaxiWorldSimulator(self.np_random, **self.scenario, rewards=self.rewards)

    def step(self, action):
        return self._step(action)

    def reset(self):
        return super().reset()

    def convert_to_human(self, js):
        img = json_to_image(js)

        env = json.loads(js)
        output = img[0:3].copy()
        output[0:3] = np.abs(1 - output[0]) * 255
        output[0] = output[0] + img[1] * 255
        output[1] = output[1] + img[2] * 255
        output[2] = output[2] + img[3] * 255
        if img.shape[0] > 4:
            output[0] = output[0] + img[4] * 25
            output[1] = output[1] + img[4] * 25
            output[2] = output[2] + img[4] * 25
        return (
            np.transpose(output, (1, 2, 0)),
            env["taxi"]["fuel"],
            env["taxi"]["money"],
        )

    def render(self, mode="human"):

        img, fuel, money = self.convert_to_human(self.sim._get_state_json())
        if self.first:
            plt.ion()
            self.first = False
            fig, ax = plt.subplots()
            self.ax = ax
            self.im = ax.imshow(img)
        self.ax.set_title(f"fuel: {fuel}, money: {money}")
        self.im.set_data(img)

        plt.pause(0.001)
        plt.draw()

    def _get_random_action(self, obs):
        obs = json.loads(obs)
        passengers = len(obs["passenger"])
        size = obs["n"]

        empty, delivered, location, passenger = np.random.randint(0, 2, 4)
        action = {
            "empty": bool(empty),
            "delivered": None,
            "location": None,
            "passenger": None,
        }
        if location:
            x, y = np.random.randint(0, size, 2)  
            action["location"] = (x, y)
        if delivered:
            action["delivered"] = [
                obs["passenger"][np.random.randint(passengers)]["pid"]
            ]
        if passenger:
            action["passenger"] = [
                obs["passenger"][np.random.randint(passengers)]["pid"]
            ]
        return action

    def convert_to_action(self, subgoal, obs):

        if isinstance(subgoal, int):
            return _action_decode(subgoal)
        elif len(subgoal) == 8:
            return self._convert_ordinal_action(subgoal, obs)
        else:
            return self._convert_factored_action(subgoal, obs)

    def _convert_factored_action(self, subgoal, obs):
        env = json.loads(obs)
        size = env["n"]

        empty, delivered, location, passenger, x1, y1, x2, y2, x3, y3 = np.clip(
            subgoal.cpu(), 0, 1
        ).tolist()
        action = {
            "empty": bool(empty),
            "delivered": None,
            "location": None,
            "passenger": None,
        }
        if location:
            action["location"] = self._scale_coordinates(size, x1, y1)
        if delivered:
            action["delivered"] = [self._pick_passenger(env, x2, y2)]
        if passenger:
            action["passenger"] = [self._pick_passenger(env, x3, y3)]
        return action

    def _convert_ordinal_action(self, subgoal, obs):
        env = json.loads(obs)
        size = env["n"]

        empty, delivered, location, passenger, x, y = np.clip(
            subgoal[0:6].cpu(), 0, 1
        ).tolist()
        dindex, pindex = subgoal[6:].cpu().tolist()
        action = {
            "empty": bool(empty),
            "delivered": None,
            "location": None,
            "passenger": None,
        }
        if location:
            action["location"] = self._scale_coordinates(size, x, y)
        if delivered:
            try:
                did = env["passenger"][int(dindex)]["pid"]
            except IndexError:
                did = None
            action["delivered"] = [did]
        if passenger:
            try:
                pid = env["passenger"][int(pindex)]["pid"]
            except IndexError:
                pid = None
            action["passenger"] = [pid]
        return action

    def _scale_coordinates(self, size, x, y):
        return (int(round(x * (size - 1), 0)), int(round(y * (size - 1), 0)))

    def _pick_passenger(self, env, x, y):
        target = self._scale_coordinates(env["n"], x, y)
        min_distance = 10
        pid = None
        for p in env["passenger"]:
            location = p["location"]
            d = distance.euclidean(target, location)
            if d < min_distance:
                min_distance = d
                pid = p["pid"]

        return pid

    def generate_pddl(self, ob, subgoal):
        pddl = json_to_pddl(ob)
        pddl = pddl.replace("$goal$", self._action_to_pddl(subgoal))
        pddl = pddl.replace(
            "$nodes$", " ".join(list(set(re.findall("sx\d+y\d+", pddl))))
        )
        return pddl

    def _action_to_pddl(self, action):
        goal = ""
        goal += "(empty t)\n" if action["empty"] else ""

        if action["delivered"] is not None:
            goal += "".join([f"(delivered p{p})\n" for p in action["delivered"]])
        if action["passenger"] is not None:
            goal += "".join(
                [f"(carrying-passenger t p{p})\n" for p in action["passenger"]]
            )
        if action["location"] is not None:
            x, y = action["location"]
            goal += f"(in t sx{x}y{y})\n"

        return goal

    def expand_actions(self, obs, actions):
        if not any([a.startswith("move ") for a in actions]):
            return [(a, None) for a in actions]  # to fit craft action signiature

        env = json.loads(obs)
        network = create_network(env)

        newlist = []
        for action in actions:
            if action.startswith("move "):
                start, end = action.split()[1:3]
                newlist.extend(find_path(network, make_tuple(start), make_tuple(end)))
            else:
                newlist.append(action)
        return [(n, None) for n in newlist]  # to fit craft action signiature

    def project_symbolic_state(self, obs, action):
        """ Calculates a projected state from the current observation and symbolic action.
            For taxi env this is pointless as action always succeeds instantly.
        """
        return None

    def check_projected_state_met(self, obs, projected):
        """ Checks if observation is compatibile with the projected partial state. 
        Given deterministic 1-step actions, this is always true"""
        return True



class GraphTaxiEnv(BaseTaxiEnv):
    def __init__(self, representation, scenario, mask=False, rewards=None):
        self.rewards = rewards
        self.scenario = SCENARIOS[scenario]
        self.mask=mask
        super().__init__()
        #image = _construct_image(representation, scenario)
        self.observation_space = JsonGraph(
            converter=json_to_graph,node_dimension=3,edge_dimension=4, planner=Planner()
        )
        self.first = True
        if scenario == "original":
            self.action_space = spaces.Discrete(ORIGINAL_ACTION_COUNT)

    def _init_simulator(self):
        return TaxiWorldSimulator(self.np_random, **self.scenario, rewards=self.rewards, planning= not self.mask)

    def step(self, action):
        obs, reward, done, info = self._step(action)
        info['s_true'] = obs
        info['d_true'] = done
        return obs, reward, done, info 

    def reset(self):
        return super().reset()

    def convert_to_human(self, js):
        img = json_to_image(js)

        env = json.loads(js)
        output = img[0:3].copy()
        output[0:3] = np.abs(1 - output[0]) * 255
        output[0] = output[0] + img[1] * 255
        output[1] = output[1] + img[2] * 255
        output[2] = output[2] + img[3] * 255
        if img.shape[0] > 4:
            output[0] = output[0] + img[4] * 25
            output[1] = output[1] + img[4] * 25
            output[2] = output[2] + img[4] * 25
        return (
            np.transpose(output, (1, 2, 0)),
            env["taxi"]["fuel"],
            env["taxi"]["money"],
        )

    def render(self, mode="human"):

        img, fuel, money = self.convert_to_human(self.sim._get_state_json())
        if self.first:
            plt.ion()
            self.first = False
            fig, ax = plt.subplots()
            self.ax = ax
            self.im = ax.imshow(img)
        # fig = plt.figure(figsize=(8, 8))
        # for i in range(4):
        #     fig.add_subplot(1, 4, i + 1)
        #     plt.imshow(img[i])
        self.ax.set_title(f"fuel: {fuel}, money: {money}")
        self.im.set_data(img)

        plt.pause(0.001)
        plt.draw()
