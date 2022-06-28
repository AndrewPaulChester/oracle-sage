from gym.envs.registration import register

ENVS = {
    "small-room-": {"env_config": {"env":"small_room", "planning":False}},
    "small-room-unmasked-":{"env_config": {"env":"small_room", "planning":True}},
    "small-room-random-": {"env_config": {"env":"small_room_random", "planning":False}},
    "small-room-random-unmasked-":{"env_config": {"env":"small_room_random", "planning":True}},
    "big-room-": {"env_config": {"env":"big_room", "planning":False}},
    "big-room-unmasked-":{"env_config": {"env":"big_room", "planning":True}},
    "big-room-random-": {"env_config": {"env":"big_room_random", "planning":False}},
    "big-room-random-unmasked-":{"env_config": {"env":"big_room_random", "planning":True}},

    "lava-cross-unmasked-":{"env_config": {"env":"lava", "planning":True}},
    "lava-lev-unmasked-":{"env_config": {"env":"lava_lev", "planning":True}},
    "lava-lev-potion-unmasked-":{"env_config": {"env":"lava_lev_potion_inv", "planning":True}},
    "lava-lev-potion-pick-unmasked-":{"env_config": {"env":"lava_lev_potion_pick", "planning":True}},
    
    "wear-unmasked-":{"env_config": {"env":"mini_wear", "planning":True}},

    
    "hungry-unmasked-":{"env_config": {"env":"hungry", "planning":True,"penalty_time":-0.2,"penalty_step":-0.5}},
    "hungry-rooms-unmasked-":{"env_config": {"env":"hungry_rooms", "planning":True,"penalty_time":-0.2,"penalty_step":-0.5}},
    "hungry-fixed3-unmasked-":{"env_config": {"env":"hungry_fixed3", "planning":True,"penalty_time":-0.2,"penalty_step":-0.5}},
    "hungry-fixed5-unmasked-":{"env_config": {"env":"hungry_fixed5", "planning":True,"penalty_time":-0.2,"penalty_step":-0.5}},
    "hungry-danger5-unmasked-":{"env_config": {"env":"hungry_danger5", "planning":True,"penalty_step":-0.1,"fn_penalty_step":"always"}},
    "hungry-danger5-":{"env_config": {"env":"hungry_danger5", "planning":False,"penalty_step":-0.1,"fn_penalty_step":"always"}},
}


def multi_register_graph(envs):
    for k, v in envs.items():
        register(id=k + "v0", entry_point="sage.domains.gym_nle.envs:SAGENLEEnv", kwargs=v.copy())


multi_register_graph(ENVS)



# lava=skills_lava.MiniHackLC,
# lava_lev=skills_lava.MiniHackLCLevitate,
# lava_lev_potion_inv=skills_lava.MiniHackLCLevitatePotionInv,
# lava_lev_potion_pick=skills_lava.MiniHackLCLevitatePotionPickup,
# lava_lev_ring_inv=skills_lava.MiniHackLCLevitateRingInv,
# lava_lev_ring_pick=skills_lava.MiniHackLCLevitateRingPickup,


# mini_eat=skills_simple.MiniHackEat,
# mini_pray=skills_simple.MiniHackPray,
# mini_sink=skills_simple.MiniHackSink,
# mini_read=skills_simple.MiniHackRead,
# mini_zap=skills_simple.MiniHackZap,
# mini_puton=skills_simple.MiniHackPutOn,
# mini_wear=skills_simple.MiniHackWear,
# mini_wield=skills_simple.MiniHackWield,
# mini_locked=skills_simple.MiniHackLockedDoor,