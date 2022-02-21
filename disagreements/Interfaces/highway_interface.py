import cv2
import json
import numpy as np
from copy import deepcopy
from os.path import join
from pathlib import Path

import gym

from disagreements.Interfaces.abstract_interface import AbstractInterface, \
    AbstractDisagreementTrace
from rl_agents.agents.common.exploration.abstract import exploration_factory
from rl_agents.agents.common.factory import agent_factory
from rl_agents.trainer.evaluation import Evaluation
from highway_env.envs.highway_env import HighwayEnv

ACTION_DICT = {0: 'LANE_LEFT', 1: 'IDLE', 2: 'LANE_RIGHT', 3: 'FASTER', 4: 'SLOWER'}


class HighwayInterface(AbstractInterface):
    def __init__(self, config, output_dir, num_episodes, seed=0):
        super().__init__(config, output_dir)
        self.num_episodes = num_episodes
        self.seed = seed

    def initiate(self):
        config_file, output_dir = self.config, self.output_dir
        f = open(config_file)
        config = json.load(f)
        env_config, agent_config, = config['env'], config['agent']
        env = gym.make(env_config["env_id"])
        env.seed(self.seed)
        agent = agent_factory(env, agent_config)
        env_config.update({"simulation_frequency": 15, "policy_frequency": 5, })
        env.configure(env_config)
        env.define_spaces()
        agent.exploration_policy = exploration_factory({'method': 'Greedy'}, env.action_space)
        evaluation = MyEvaluation(env, agent, display_env=False)
        agent_path = Path(join(config['agent_path'], 'checkpoint-final.tar'))
        evaluation.load_agent_model(agent_path)
        return env, agent, evaluation

    def get_state_action_values(self, agent, state):
        return agent.get_state_action_values(state)

    def get_state_from_obs(self, agent, obs, params=None):
        return obs

    def get_next_action(self, agent, obs, state):
        return agent.act(state)

    def get_features(self, env):
        return {"position": deepcopy(env.road.vehicles[0].destination)}

    def pre_disagreement(self, env):
        return deepcopy(env)

    def post_disagreement(self, agent1, agent2, pre_params=None):
        env = pre_params
        agent2.previous_state = agent1.previous_state
        return env

    def disagreement_trace(self, e, trajectory_length, agent_ratio, params=None):
        return HighwayDisagreementTrace(e, trajectory_length, agent_ratio)

    def da_states_functionality(self, trace, params=None):
        trace.a2_max_q_val = max(max(params), trace.a2_max_q_val)
        trace.a2_min_q_val = min(min(params), trace.a2_min_q_val)

    def update_trace(self, trace, agent, t, states, scores):
        a2_s_a_values = [x.action_values for x in states]
        a1_values_for_a2_states = [
            agent.interface.get_state_action_values(agent, x.state) for x in states]
        trace.a2_s_a_values.append(a2_s_a_values)
        trace.a2_trajectories.append(states)
        trace.a2_rewards.append(scores)
        trace.disagreement_indexes.append(t)
        trace.a1_values_for_a2_states.append(a1_values_for_a2_states)


class HighwayDisagreementTrace(AbstractDisagreementTrace):
    def __init__(self, episode, trajectory_length, agent_ratio=1):
        super().__init__(episode, trajectory_length, agent_ratio)
        self.a1_trajectory_indexes = []
        self.a2_rewards = []
        self.a1_max_q_val = 0
        self.a2_max_q_val = 0
        self.a1_min_q_val = float('inf')
        self.a2_min_q_val = float('inf')
        self.a1_s_a_values = []
        self.a2_s_a_values = []
        self.a2_values_for_a1_states = []
        self.a1_values_for_a2_states = []

    def update(self, state_object, obs, a, r, done, infos, params=None):
        a1_s_a_values, a2_values_for_a1_states = params[0], params[1]
        self.obs.append(obs)
        self.rewards.append(r)
        self.dones.append(done)
        self.infos.append(infos)
        self.actions.append(a)
        self.reward_sum += r
        self.states.append(state_object)
        self.length += 1
        self.a1_s_a_values.append(a1_s_a_values)
        self.a2_values_for_a1_states.append(a2_values_for_a1_states)
        self.a1_max_q_val = max(max(a1_s_a_values), self.a1_max_q_val)
        self.a2_max_q_val = max(max(a2_values_for_a1_states), self.a2_max_q_val)
        self.a1_min_q_val = min(min(a1_s_a_values), self.a1_min_q_val)
        self.a2_min_q_val = min(min(a2_values_for_a1_states), self.a2_min_q_val)


class MyEvaluation(Evaluation):
    def __init__(self, env, agent, output_dir='../agents', num_episodes=1000, display_env=False):
        self.OUTPUT_FOLDER = output_dir
        super(MyEvaluation, self).__init__(env, agent, num_episodes=num_episodes,
                                           display_env=display_env)


agent_position = [164, 66]  # static in this domain


def highway_mark_frames(t, d, a1_frames, a2_frames, relative_idx):
    mark_a1_trajectory(a1_frames, agent_position, color=0)
    da_index = t.trajectory_length // 2 - 1
    relative_positions = get_relative_position(t, d, da_index, relative_idx)
    mark_a2_trajectory(a1_frames, da_index, relative_positions, agent_position, a2_frames,
                       color=255)

    mark_frames(a1_frames, a2_frames, da_index, agent_position)
    return a1_frames, a2_frames


def mark_a1_trajectory(a1_frames, agent_position, color=255):
    for i in range(len(a1_frames)):
        a1_frames[i] = mark_agent(a1_frames[i], position=agent_position, color=color)


def mark_a2_trajectory(a1_frames, da_index, relative_positions, ref_position, a2_frames,
                       color=255):
    for i in range(len(relative_positions)):
        a1_frames[da_index + i + 1] = mark_trajectory_step(a1_frames[da_index + i + 1],
                                                           relative_positions[i], ref_position,
                                                           a2_frames[da_index + i + 1],
                                                           color=color)


def mark_trajectory_step(img, rel_pos, ref_pos, temp_img, color=255, thickness=-1):
    img2 = img.copy()
    add_x, add_y = int(rel_pos[1] * 5), int(rel_pos[0] * 10)
    top_left = (ref_pos[0] + add_y, ref_pos[1] + add_x)
    bottom_right = (ref_pos[0] + 30 + add_y, ref_pos[1] + 15 + add_x)
    cv2.rectangle(img2, top_left, bottom_right, color, thickness)
    cv2.rectangle(img2, (ref_pos[0] + add_y + 4, ref_pos[1] + add_x + 4),
                  (ref_pos[0] + 30 + add_y - 4, ref_pos[1] + 15 + add_x - 4), (43, 165, 0, 255),
                  thickness)
    """for testing"""
    # plt.imshow(img2)
    # plt.show()
    # plt.imshow(temp_img)
    # plt.show()
    return img2


def get_relative_position(trace, trajectory, da_index, relative_idx):
    a1_obs = np.array([trace.states[x].features["position"] for x in
                       trajectory.a1_states[da_index + 1:]])
    a2_obs = np.array([x.features["position"] for x in
                       trace.a2_trajectories[trajectory.trajectory_index]][relative_idx + 1:])
    if len(a1_obs) != len(a2_obs):
        a2_obs = np.array(list(a2_obs) + [a2_obs[-1] for _ in range(len(a1_obs) - len(a2_obs))])
    mark_rel_cords = np.around(a2_obs - a1_obs, 3)
    return mark_rel_cords


def mark_agent(img, position=None, color=255, thickness=2):
    assert position, 'Error - No position provided for marking agent'
    img2 = img.copy()
    top_left = (position[0], position[1])
    bottom_right = (position[0] + 30, position[1] + 15)
    cv2.rectangle(img2, top_left, bottom_right, color, thickness)
    return img2


def mark_frames(a1_frames, a2_frames, da_index, mark_position):
    if mark_position:
        """mark disagreement state"""
        a1_frames[da_index] = mark_agent(a1_frames[da_index], position=mark_position)
        a2_frames[da_index] = a1_frames[da_index]
        """mark chosen action"""
        a1_frames[da_index + 1] = mark_agent(a1_frames[da_index + 1],
                                             position=mark_position)
        a2_frames[da_index + 1] = mark_agent(a2_frames[da_index + 1],
                                             position=mark_position, color=0)
