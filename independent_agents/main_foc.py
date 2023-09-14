import numpy as np
from gym.spaces import Dict, Discrete, Box
import argparse
import os
import ray
import json
from independent_agents.MultiAgentahtad_env_foc import MultiAgentahtadEnv_FOC
from ray.tune.registry import register_env
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="tf",
    help="The DL framework specifier.")
parser.add_argument(
    "--stop-iters",
    type=int,
    default=2e9,
    help="Number of iterations to train.")
parser.add_argument(
    "--stop-timesteps",
    type=int,
    default=40e6,
    help="Number of timesteps to train.")
parser.add_argument(
    "--stop-reward",
    type=float,
    default=4000,
    help="Reward at which we stop training.")
parser.add_argument(
    "--step-cost",
    type=float,
    default=-1.0
)
parser.add_argument(
    "--actor-neurons",
    nargs='+',
    type=int,
    default=[64,64]
)
parser.add_argument(
    "--critic-neurons",
    nargs='+',
    type=int,
    default=[64,64]
)
parser.add_argument(
    "--n-processes",
    type=int,
    default=10
)
parser.add_argument(
    "--message",
    type=str,
    default="ten_processes"
)
parser.add_argument(
    "--exp-folder",
    type=str,
    default="ten_processes"
)


class FillInActions(DefaultCallbacks):
    """Fills in the opponent actions info in the training batches."""

    def on_postprocess_trajectory(self, worker, episode, agent_id, policy_id,
                                  policies, postprocessed_batch,
                                  original_batches, **kwargs):
        to_update = postprocessed_batch[SampleBatch.CUR_OBS]
        other_id = 1 if agent_id == 0 else 0
        action_encoder = ModelCatalog.get_preprocessor_for_space(Discrete(n_processes+1))

        _, opponent_batch = original_batches[other_id]
        opponent_actions = np.array([
            action_encoder.transform(a)
            for a in opponent_batch[SampleBatch.ACTIONS]
        ])
        to_update[:, :n_processes+1] = opponent_actions


def central_critic_observer(agent_obs, **kw):

    new_obs = {
        0: {
            "own_obs": agent_obs[0],
            "opponent_obs": agent_obs[1],
            "opponent_action": 0,  
        },
        1: {
            "own_obs": agent_obs[1],
            "opponent_obs": agent_obs[0],
            "opponent_action": 0,  
        },
    }
    return new_obs


class CentralizedCritic(TFModelV2):

    def __init__(self, obs_space, action_space, num_outputs, model_config,name):
        super(CentralizedCritic, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        action_shape=model_config['custom_model_config']['n_processes']+(action_space.n-1)*model_config['custom_model_config']['share_action']
        self.action_model = FullyConnectedNetwork(Box(low=0.0,high=1.0,shape=(action_shape,),dtype=np.float32), action_space, num_outputs, model_config, name + "_action")

        obs = tf.keras.layers.Input(shape=(int(np.product(obs_space.shape)),), name="obs")
        central_vf_dense_1 = tf.keras.layers.Dense(model_config['custom_model_config']['critic_neurons'][0], activation=tf.nn.tanh, name="c_vf_dense")(obs)
        central_vf_dense_3 = tf.keras.layers.Dense(model_config['custom_model_config']['critic_neurons'][1], activation=tf.nn.tanh, name="c_vf_dense_1")(central_vf_dense_1)
        central_vf_out = tf.keras.layers.Dense(1, activation=None, name=name + "_vf")(central_vf_dense_3)

        self.value_model = tf.keras.Model(inputs=obs, outputs=central_vf_out)

    def forward(self, input_dict, state, seq_lens):
        self._value_out = self.value_model({"obs": input_dict["obs_flat"]}, state, seq_lens)
        return self.action_model({"obs": input_dict["obs"]["own_obs"]}, state, seq_lens)

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

if __name__ == "__main__":
    ray.init(local_mode=True, num_cpus=12)
    env_checkpoint = r''
    transfer_learning = False
    transfer_timesteps = 60e6
    args = parser.parse_args()
    step_cost = args.step_cost
    actor_neurons = args.actor_neurons
    critic_neurons = args.critic_neurons
    n_processes = args.n_processes
    message = args.message
    print(f'Multi-agent AHTAD run:\n'
          f'message-{message}\n'
          f'timesteps-{args.stop_timesteps}\n'
          f'processes-{n_processes}\n'
          f'step_cost-{step_cost}\n'
          f'actor_neurons-{actor_neurons}\n'
          f'critic_neurons-{critic_neurons}\n'
          )
    num_agents=2
    add_actions_to_obs = True
    coupled_reward = True
    num_workers = 10
    sigma = 1.5
    sigma_env = 1.5
    timesteps = 3000000
    win_reward = +500.0
    lose_reward = -30.0
    horizon_punishment = 300
    punishment = -1000.0

    ModelCatalog.register_custom_model("centralized_critic_model", CentralizedCritic)

    custom_message=message
    algorithm='PPO'
    save_path_inference = os.path.join(os.getcwd(),
                                       'experiments','experimental_results',f'{n_processes}_processes','multi_agent',
                                       f'{n_processes}_{step_cost}',
                                       f'sigma_{sigma_env}')
    if not os.path.isdir(save_path_inference):
        os.makedirs(save_path_inference)

    def env_creator(_):
        return MultiAgentahtadEnv_FOC(n_processes=n_processes,
                                  sigma=sigma_env,
                                  win_reward=win_reward,
                                  lose_reward=lose_reward,
                                  step_cost=step_cost,
                                  punishment=punishment,
                                  num_agents=num_agents,
                                  save_path_inference=save_path_inference,
                                  add_actions_to_obs=add_actions_to_obs,
                                  coupled_reward=coupled_reward,
                                  horizon_punishment=horizon_punishment
                                  )

    env_name = "MultiAgentahtad_Env_foc"
    register_env(env_name, env_creator)
    a_space = Discrete(n_processes+1)
    obs_space = Box(low=0.0,high=1.0,shape=(n_processes+(num_agents-1)*(n_processes)*add_actions_to_obs,),dtype=np.float32)
    action_space = a_space
    observer_space = Dict({
        "own_obs":obs_space,
        "opponent_obs": obs_space,
        "opponent_action": a_space,
    })

    config_ppo = {
        "explore": True,
        "no_done_at_end": False,
        "num_cpus_per_worker":int(10/num_workers),
        "num_cpus_for_driver":2,
        "env": "MultiAgentahtad_Env_foc",
        "log_level":'INFO',
        "env_config": {
            "num_agents": num_agents,
            "n_processes":n_processes,
            "sigma" : sigma_env,
            "win_reward" :win_reward,
            "lose_reward" : lose_reward,
            "step_cost" :step_cost,
            "punishment": punishment,
            "save_path_inference":save_path_inference,
            "mode": "training",
            "add_actions_to_obs":add_actions_to_obs,
            "coupled_reward":coupled_reward,
            "horizon_punishment":horizon_punishment
        },
        "batch_mode": "complete_episodes",
        "callbacks": FillInActions,
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_workers": 1,#num_workers,
        "lr": 1e-3,
        "lambda": 0.95,
        "lr_schedule": [[0,1e-3],
                        [10000000,1e-4],
                        [25000000,1e-4],
                        [32000000,1e-5],
                        [40000000,1e-6],
                        ],
        "multiagent": {
            "policies": {
                "pol1": (None, observer_space, action_space, {"framework": args.framework,}),
            },
            "policy_mapping_fn": (
                lambda aid, **kwargs: "pol1"),
            "observation_fn": central_critic_observer,
        },
        "vf_loss_coeff": 1e-7,
        "entropy_coeff": 0.0,
        "kl_coeff": 0.2,
        "kl_target": 0.01,
        "clip_param": 0.2,
        "model": {
            "custom_model": "centralized_critic_model",
            "custom_model_config":{
                "share_action": add_actions_to_obs,
                "n_processes": n_processes,
                "critic_neurons": critic_neurons,
                "vf_share_layers": False,
            },
            "fcnet_hiddens": actor_neurons,
            "fcnet_activation":"tanh",
            "use_lstm": False,
            "lstm_use_prev_action": False,
            "lstm_use_prev_reward": False,
            "zero_mean": False,
            "vf_share_layers": False,
            "use_attention": False,
            "attention_num_transformer_units": 1,
            "attention_dim": 64,
            "attention_num_heads": 1,
            "attention_head_dim": 32,
            "attention_memory_inference": 50,
            "attention_memory_training": 50,
            "attention_position_wise_mlp_dim": 32,
            "attention_init_gru_gate_bias": 2.0,
        },
        "framework": args.framework,
    }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps + transfer_timesteps if transfer_learning else args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    exp_name = args.exp_folder
    exp_path = os.path.join(os.getcwd(), os.path.split(save_path_inference)[0])
    if not os.path.isdir(exp_path):
        os.makedirs(exp_path)
    results = tune.run("PPO", config=config_ppo, stop=stop, verbose=0, name=exp_name, local_dir=exp_path, log_to_file=True,
                       checkpoint_at_end=True, checkpoint_score_attr="episode_reward_mean",reuse_actors=False,
                       keep_checkpoints_num=4, checkpoint_freq=1, restore = env_checkpoint if transfer_learning else None)
    best_trial = results.get_best_trial(metric="episode_reward_mean", mode="max", scope="all")
    best_checkpoint = results.get_best_checkpoint(best_trial, metric="episode_reward_mean", mode='max')
    print(best_checkpoint)
    best_checpoint_log = dict(best_checkpoint=best_checkpoint, step_cost=step_cost, n_processes=n_processes)
    with open(os.path.join(results._get_trial_paths()[0], 'best_checkpoint.json'), 'w') as f:
        json.dump(best_checpoint_log, f, indent=1)
    if args.as_test:
        check_learning_achieved(results, args.stop_reward)
    print(f'is ray initialized - {ray.is_initialized()}')
    ray.shutdown()
    print(f'is ray initialized (shutdown)  - {ray.is_initialized()}')