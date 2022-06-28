from gym.envs.registration import register

REWARDS = {
     "v1": {"base": 0, "failed-action": 0, "drop-off": 1},
#     "v1": {"base": -1, "failed-action": -1, "drop-off": 20},
#     "v3": {"base": -0.05, "failed-action": -0.05, "drop-off": 1},
}

# ENVS = {
#     "original-taxi-": {"representation": "screen", "scenario": "original"},
#     "large-taxi-": {"representation": "screen", "scenario": "expanded"},
#     "multi-taxi-": {"representation": "screen", "scenario": "multi"},
#     "fuel-taxi-": {"representation": "screen", "scenario": "fuel"},
#     "predictable-taxi-": {"representation": "screen", "scenario": "predictable"},
#     "predictable5-taxi-": {"representation": "screen", "scenario": "predictable5"},
#     "predictable10-taxi-": {"representation": "screen", "scenario": "predictable10"},
#     "predictable15-taxi-": {"representation": "screen", "scenario": "predictable15"},
#     "mixed-taxi-": {"representation": "mixed", "scenario": "original"},
#     "large-mixed-taxi-": {"representation": "mixed", "scenario": "expanded"},
#     "multi-mixed-taxi-": {"representation": "mixed", "scenario": "multi"},
#     "predictable-mixed-taxi-": {"representation": "mixed", "scenario": "predictable"},
#     "predictable5-mixed-taxi-": {"representation": "mixed", "scenario": "predictable5"},
#     "predictable10-mixed-taxi-": {
#         "representation": "mixed",
#         "scenario": "predictable10",
#     },
#     "predictable15-mixed-taxi-": {
#         "representation": "mixed",
#         "scenario": "predictable15",
#     },
#     "fuel-mixed-taxi-": {"representation": "mixed", "scenario": "fuel"},
#     "original-mlp-taxi-": {"representation": "mlp", "scenario": "original"},
#     "both-taxi-": {"representation": "both", "scenario": "original"},
#     "one-hot-taxi-": {"representation": "one-hot", "scenario": "original"},
# }


ENVS = {
    "city-taxi-": {"representation": "graph", "scenario": "city", "mask":True},
    "city-taxi-unmasked-": {"representation": "graph", "scenario": "city", "mask":False},
}

LEGACY_ENVS = {"city-taxi-image-": "BoxTaxiEnv"}

def multi_register(envs, rewards):
    for k, v in envs.items():
        register(id=k + "v0", entry_point="sage.domains.gym_taxi.envs:" + v)
        for i, r in rewards.items():
            register(id=k + i, entry_point="sage.domains.gym_taxi.envs:" + v, kwargs={"rewards": r})


def multi_register_graph(envs, rewards):
    for k, v in envs.items():
        register(id=k + "v0", entry_point="sage.domains.gym_taxi.envs:GraphTaxiEnv", kwargs=v.copy())
        for i, r in rewards.items():
            v["rewards"] = r.copy()
            register(id=k + i, entry_point="sage.domains.gym_taxi.envs:GraphTaxiEnv", kwargs=v.copy())


register(id="discrete-taxi-v0", entry_point="sage.domains.gym_taxi.envs:DiscreteTaxiEnv")

multi_register_graph(ENVS, REWARDS)
multi_register(LEGACY_ENVS,REWARDS)