from gym.envs.registration import register
import gym 
import itertools

def safe_register(id,entry_point,kwargs=None):
    if id not in gym.envs.registration.registry.env_specs:
        register(id,entry_point=entry_point,kwargs=kwargs)

VALUES = [1,3,5,10,15,20,30,40,50]
PARAMS = itertools.product(VALUES,VALUES)

ENVS = {"Tradeoff": "JsonTradeoffEnv"}

def multi_register(envs, params):
    for env, entry in envs.items():
        safe_register(id=env + "-v0", entry_point="sage.domains.gym_tradeoff.envs.tradeoff_env:" + entry)
        for k, n in params:
            safe_register(id=f"{env}-{k}-{n}-v0", entry_point="sage.domains.gym_tradeoff.envs.tradeoff_env:" + entry, kwargs={"k": k,"n":n})


multi_register(ENVS, PARAMS)
