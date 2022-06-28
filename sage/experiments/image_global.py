"""
Run DQN on grid world.
"""
import sys
import argparse
import torch as th

import gym
from sage.domains.boxworld import boxworld
from sage.domains.utils import spaces
import sage.domains.gym_tradeoff #this is required to register the environment
import sage.domains.gym_taxi
import sage.domains.gym_nle


from stable_baselines3 import A2C
from sage.agent.plan_feedback_a2c import PlanFeedback_A2C
from sage.agent.feedback_a2c import Feedback_A2C
from sage.forks.stable_baselines3.stable_baselines3.common.utils import get_linear_fn
from sage.forks.stable_baselines3.stable_baselines3.common.env_util import make_vec_env
from sage.agent.graph_policy import GNNPolicy
from sage.agent.graph_feedback_policy import GNNFeedbackPolicy
from sage.agent.graph_plan_feedback_policy import GNNPlanFeedbackPolicy
from sage.agent.tb_logging import TensorboardCallback
from sage.agent.async_vec_env import AsyncVecEnv

def run(variant):
    env = make_vec_env(variant['env_name'], n_envs=variant["num_processes"], seed=variant["seed"])

    model = A2C("CnnPolicy", env, verbose=variant["verbose"],supported_action_spaces=(spaces.BinaryAction,gym.spaces.Discrete,spaces.Autoregressive),**variant["algorithm_kwargs"])
        
    model.learn(total_timesteps=variant["num_env_steps"],log_interval=variant['log_interval'], callback=TensorboardCallback(variant["verbose"]))
    #model.save(variant["save_dir"])

def main(arglist):
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument(
        "--env-name", default="Boxworld-v0", help="Select the environment to run"
    )
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument(
        "--learning-rate", type=float, default=0.00025, help="learning rate"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=5,
        help="number of forward steps in A2C (default: 5)",
    )
    parser.add_argument(
        "--gnn-steps",
        type=int,
        default=5,
        help="number of message-passing steps in main gnn (default: 5)",
    )
    # parser.add_argument(
    #     "--eval-steps",
    #     type=int,
    #     default=2000,
    #     help="number of eval steps taken per epoch",
    # )
    parser.add_argument(
        "--algo", default="a2c", help="algorithm to use: a2c | ppo | acktr"
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-5,
        help="RMSprop optimizer epsilon (default: 1e-5)",
    )    
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor for rewards (default: 0.99)",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=1,
        help="gae lambda parameter (default: 1)",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.01,
        help="entropy term coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--value-loss-coef",
        type=float,
        default=0.5,
        help="value loss coefficient (default: 0.5)",
    )
    parser.add_argument(
        "--path-loss-coef",
        type=float,
        default=0.5,
        help="path value loss coefficient (default: 0.5)",
    )
    parser.add_argument(
        "--policy-coef",
        type=float,
        default=1,
        help="policy term coefficient (default: 1)",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="max norm of gradients (default: 0.5)",
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument(
        "--num-processes",
        type=int,
        default=16,
        help="how many training CPU processes to use (default: 16)",
    )
    parser.add_argument(
        "--num-env-steps",
        type=int,
        default=10000000,
        help="number of environment steps to train (default: 10e6)",
    )
    parser.add_argument(
        "--log-dir",
        default="./logs/",
        help="directory to save agent logs (default: ./logs/)",
    )
    parser.add_argument(
        "--save-dir",
        default="./trained_models/",
        help="directory to save agent logs (default: ./trained_models/)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--ortho-init", action="store_true", default=False, help="uses orthogonal initialisation"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=200,
        help="number of training steps per epoch (default: 200)",
    )
    parser.add_argument("--verbose", type=int, default=1, help="how much output will be logged: level 2 gives net weights, level 0 silent")

    parser.add_argument(
        "--tis-heuristic",
        type=float,
        default=None,
        help="Heuristic value to use for truncated importance sampling on policy loss. (default: No importance sampling used for policy loss)",
    )

    parser.add_argument(
        "--lr-decay", action="store_true", default=False, help="decay lr by a factor of 10 over the first half of training"
    )

    args = parser.parse_args(arglist)
    if args.lr_decay:
        learning_rate = get_linear_fn(
            args.learning_rate, args.learning_rate/10, 0.5
        )
    else:
        learning_rate=args.learning_rate

    variant = dict(
        algorithm="A2C",
        version="normal",
        env_name=args.env_name,
        seed=args.seed,
        num_env_steps=args.num_env_steps,
        num_processes=args.num_processes,
        log_interval=args.log_interval,
        verbose=args.verbose,
        save_dir=args.save_dir,
        algorithm_kwargs=dict(
            gamma=args.gamma,
            tensorboard_log=args.log_dir,
            n_steps=args.num_steps,
            gae_lambda=args.gae_lambda,
            vf_coef=args.value_loss_coef,
            pvf_coef=args.path_loss_coef,
            ent_coef=args.entropy_coef,
            policy_coef=args.policy_coef,
            learning_rate=learning_rate,
            rms_prop_eps=args.eps,
            max_grad_norm=args.max_grad_norm,
            tis_heuristic= args.tis_heuristic,
            policy_kwargs={"optimizer_class":th.optim.AdamW,"optimizer_kwargs":{"weight_decay":0.0001}
                            ,"ortho_init":args.ortho_init,
                            "features_extractor_kwargs": {}},
        ),
        # policy_kwargs=dict(
        # ),
    )
    # optionally set the GPU (default=False)
    run(variant)

if __name__ == "__main__":
    # try:
    #     mp.set_start_method("spawn")
    # except RuntimeError:
    #     pass
    main(sys.argv[1:])

# guild run merged:train-ppo env-name="rooms-craft-v2" train-loops=10 entropy-coef=0.01 symbolic="three-tier-pretrained" action-space=rooms eval-steps=0 learning-rate=0.001 seed=2
