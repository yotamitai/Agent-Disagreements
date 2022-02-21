from copy import deepcopy

import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from PIL import ImageFont, ImageDraw, Image

from disagreements.common.utils import make_clean_dirs, save_image, create_video
from disagreements.logging_info import log
from disagreements.get_trajectories import trajectory_importance_max_min


def get_trajectories(trace):
    """for each trajectory of agent 2 - find corresponding trajectory of agent 1"""
    for i, a2_traj in enumerate(trace.a2_trajectories):
        start_idx, end_idx = a2_traj[0].id[1], a2_traj[-1].id[1]
        a1_traj = trace.states[start_idx:end_idx + 1]
        a1_traj_q_values = [x.action_values for x in a1_traj]
        a2_traj_q_values = [x.action_values for x in a2_traj]
        a1_traj_indexes = [x.id[1] for x in a1_traj]
        a2_traj_indexes = list(range(start_idx, end_idx + 1))
        dt = DisagreementTrajectory(trace.disagreement_indexes[i], a1_traj_indexes,
                                    a2_traj_indexes, trace.trajectory_length, trace.episode, i,
                                    a1_traj_q_values, a2_traj_q_values,
                                    trace.a1_values_for_a2_states[i],
                                    trace.a2_values_for_a1_states[start_idx:end_idx + 1],
                                    trace.agent_ratio)
        trace.a1_trajectory_indexes.append(a1_traj_indexes)
        trace.disagreement_trajectories.append(dt)


def get_frames(trace, s1_indexes, s2_indexes, s2_traj, mark_position=None):
    a1_frames = [trace.states[x].image for x in s1_indexes]
    a2_frames = [trace.a2_trajectories[s2_traj][x - min(s2_indexes)].image for x in s2_indexes]
    assert len(a1_frames) == trace.trajectory_length, 'Error in highlight frame length'
    assert len(a2_frames) == trace.trajectory_length, 'Error in highlight frame length'
    da_index = trace.trajectory_length // 2 - 1
    if mark_position:
        """mark disagreement state"""
        a1_frames[da_index] = mark_agent(a1_frames[da_index], text='Disagreement',
                                         position=mark_position)
        a2_frames[da_index] = a1_frames[da_index]
    return a1_frames, a2_frames


class State(object):
    def __init__(self, idx, episode, obs, state, action_values, img, features, **kwargs):
        self.observation = obs
        self.image = img
        self.state = state
        self.action_values = action_values
        self.features = features
        self.kwargs = kwargs
        self.id = (episode, idx)

    def plot_image(self):
        plt.imshow(self.image)
        plt.show()

    def save_image(self, path, name):
        imageio.imwrite(path + '/' + name + '.png', self.image)


class DisagreementTrajectory(object):
    def __init__(self, da_index, a1_states, a2_states, horizon, episode, i, a1_s_a_values,
                 a2_s_a_values, a1_values_for_a2_states, a2_values_for_a1_states, agent_ratio):
        self.a1_states = a1_states
        self.a2_states = a2_states
        self.episode = episode
        self.trajectory_index = i
        self.horizon = horizon
        self.da_index = da_index
        self.disagreement_score = None
        self.importance = None
        self.state_importance_list = []
        self.agent_ratio = agent_ratio
        self.a1_s_a_values = a1_s_a_values
        self.a2_s_a_values = a2_s_a_values
        self.a1_values_for_a2_states = a1_values_for_a2_states
        self.a2_values_for_a1_states = a2_values_for_a1_states
        self.importance_funcs = {
            "max_min": trajectory_importance_max_min,
            "max_avg": trajectory_importance_max_min,
            "avg": trajectory_importance_max_min,
            "avg_delta": trajectory_importance_max_min,
        }

    def calculate_state_disagreement_extent(self, importance):
        self.state_importance = importance
        da_idx = self.da_index
        traj_da_idx = self.a1_states.index(da_idx)
        s1_vals, s2_vals = self.a1_s_a_values[traj_da_idx], self.a2_s_a_values[traj_da_idx]
        if importance == 'sb':
            return self.second_best_confidence(s1_vals, s2_vals)
        elif importance == 'bety':
            return self.better_than_you_confidence(s1_vals, s2_vals)

    def calculate_trajectory_importance(self, trace, i, importance):
        """calculate trajectory score"""
        s_i, e_i = self.a1_states[0], self.a1_states[-1]
        self.trajectory_importance = importance
        rel_idx = e_i - s_i
        if importance == "last_state":
            s1, s2 = trace.states[e_i], trace.a2_trajectories[i][rel_idx]
            return self.trajectory_importance_last_state(s1, s2, rel_idx)
        else:
            return self.get_trajectory_importance(importance, rel_idx)

    def get_trajectory_importance(self, importance, end):
        """state values"""
        s1_a1_vals = self.a1_s_a_values
        s1_a2_vals = self.a2_values_for_a1_states
        s2_a1_vals = self.a1_values_for_a2_states[:end + 1]
        s2_a2_vals = self.a2_s_a_values[:end + 1]
        """calculate value of all individual states in both trajectories,
         as ranked by both agents"""
        traj1_states_importance, traj2_states_importance = [], []
        for i in range(len(s1_a1_vals)):
            traj1_states_importance.append(self.get_state_value(s1_a1_vals[i], s1_a2_vals[i]))
            traj2_states_importance.append(self.get_state_value(s2_a1_vals[i], s2_a2_vals[i]))
        """calculate score of trajectories"""
        traj1_score = self.importance_funcs[importance](traj1_states_importance)
        traj2_score = self.importance_funcs[importance](traj2_states_importance)
        """return the difference between them. bigger == greater disagreement"""
        return abs(traj1_score - traj2_score)

    def trajectory_importance_last_state(self, s1, s2, idx):
        if s1.image.tolist() == s2.image.tolist(): return 0
        """state values"""
        s1_a1_vals = self.a1_s_a_values[-1]
        s1_a2_vals = self.a2_values_for_a1_states[-1]
        s2_a1_vals = self.a1_values_for_a2_states[idx]
        s2_a2_vals = self.a2_s_a_values[idx]
        """the value of the state is defined by the best available action from it"""
        s1_score = max(s1_a1_vals) * self.agent_ratio + max(s1_a2_vals)
        s2_score = max(s2_a1_vals) * self.agent_ratio + max(s2_a2_vals)
        return abs(s1_score - s2_score)

    def second_best_confidence(self, a1_vals, a2_vals):
        """compare best action to second-best action"""
        sorted_1 = sorted(a1_vals, reverse=True)
        sorted_2 = sorted(a2_vals, reverse=True)
        a1_diff = sorted_1[0] - sorted_1[1] * self.agent_ratio
        a2_diff = sorted_2[0] - sorted_2[1]
        return a1_diff + a2_diff

    def better_than_you_confidence(self, a1_vals, a2_vals):
        a1_diff = (max(a1_vals) - a1_vals[np.argmax(a2_vals)]) * self.agent_ratio
        a2_diff = max(a2_vals) - a2_vals[np.argmax(a1_vals)]
        return a1_diff + a2_diff

    def get_state_value(self, a1_vals, a2_vals):
        """
        the value of the state is defined by the best available action from it, as this is
        calculated by estimated future returns
        """
        return max(a1_vals) * self.agent_ratio + max(a2_vals)

    def normalize_q_values(self, a1_max, a1_min, a2_max, a2_min):
        self.a1_s_a_values = (np.array(self.a1_s_a_values) - a1_min) / (a1_max - a1_min)
        self.a2_s_a_values = (np.array(self.a2_s_a_values) - a2_min) / (a2_max - a2_min)
        self.a1_values_for_a2_states = (np.array(self.a1_values_for_a2_states) - a1_min) / (
                a1_max - a1_min)
        self.a2_values_for_a1_states = (np.array(self.a2_values_for_a1_states) - a2_min) / (
                a2_max - a2_min)


def disagreement(timestep, trace, env2, a1, a2, obs, s):
    trajectory_states, trajectory_scores = \
        disagreement_states(trace, env2, a2, timestep, obs, s)
    a1.interface.update_trace(trace, a1, timestep, trajectory_states, trajectory_scores )


def save_disagreements(a1_DAs, a2_DAs, output_dir, fps):
    highlight_frames_dir = join(output_dir, "highlight_frames")
    video_dir = join(output_dir, "videos")
    make_clean_dirs(video_dir)
    make_clean_dirs(join(video_dir, 'temp'))
    make_clean_dirs(highlight_frames_dir)
    dir = join(video_dir, 'temp')

    height, width, layers = a1_DAs[0][0].shape
    size = (width, height)
    trajectory_length = len(a1_DAs[0])
    da_idx = trajectory_length // 2
    for hl_i in range(len(a1_DAs)):
        for img_i in range(len(a1_DAs[hl_i])):
            save_image(highlight_frames_dir, "a1_DA{}_Frame{}".format(str(hl_i), str(img_i)),
                       a1_DAs[hl_i][img_i])
            save_image(highlight_frames_dir, "a2_DA{}_Frame{}".format(str(hl_i), str(img_i)),
                       a2_DAs[hl_i][img_i])

        """up to disagreement"""
        create_video('together' + str(hl_i), highlight_frames_dir, dir, "a1_DA" + str(hl_i), size,
                     da_idx, fps, add_pause=[0, 4])
        """from disagreement"""
        name1, name2 = "a1_DA" + str(hl_i), "a2_DA" + str(hl_i)
        create_video(name1, highlight_frames_dir, dir, name1, size,
                     trajectory_length, fps, start=da_idx, add_pause=[7, 0])
        create_video(name2, highlight_frames_dir, dir, name2, size,
                     trajectory_length, fps, start=da_idx, add_pause=[7, 0])
    return video_dir


# def get_pre_disagreement_states(t, horizon, states):
#     start = t - (horizon // 2) + 1
#     pre_disagreement_states = []
#     if start < 0:
#         pre_disagreement_states = [states[0] for _ in range(abs(start))]
#         start = 0
#     pre_disagreement_states = pre_disagreement_states + states[start:]
#     return pre_disagreement_states


def disagreement_states(trace, env, agent, timestep, obs, s):
    horizon, da_rewards = env.args.horizon, []
    start = timestep - (horizon // 2) + 1
    if start < 0: start = 0
    trajectory_states = trace.states[start:]
    da_state = deepcopy(trajectory_states[-1])
    da_state.action_values = agent.interface.get_state_action_values(agent, s)
    trajectory_states[-1] = da_state
    done = False
    next_timestep = timestep + 1
    for step in range(next_timestep, next_timestep + (horizon // 2)):
        if done: break
        a = agent.interface.get_next_action(agent, obs, s)
        obs, r, done, info = env.step(a)
        s = agent.interface.get_state_from_obs(agent, obs)
        s_a_values = agent.interface.get_state_action_values(agent, s)
        frame = env.render(mode='rgb_array')
        features = agent.interface.get_features(env)
        state_obj = State(step, trace.episode, obs, s, s_a_values, frame, features)
        trajectory_states.append(state_obj)
        da_rewards.append(r)
        agent.interface.da_states_functionality(trace, params=s_a_values)
    return trajectory_states, da_rewards


def get_top_k_disagreements(traces, args):
    """obtain the N-most important trajectories"""
    top_k_diverse_trajectories, discarded_context = [], []
    """get all trajectories"""
    all_trajectories = []
    for trace in traces:
        all_trajectories += [t for t in trace.disagreement_trajectories]
    sorted_trajectories = sorted(all_trajectories, key=lambda x: x.importance, reverse=True)
    """select trajectories"""
    seen_indexes = {i: [] for i in range(len(traces))}
    for d in sorted_trajectories:
        t_indexes = d.a1_states
        intersecting_indexes = set(seen_indexes[d.episode]).intersection(set(t_indexes))
        if len(intersecting_indexes) > args.similarity_limit:
            discarded_context.append(d)
            continue
        seen_indexes[d.episode] += t_indexes
        top_k_diverse_trajectories.append(d)
        if len(top_k_diverse_trajectories) == args.n_disagreements:
            break

    if not len(top_k_diverse_trajectories) == args.n_disagreements:
        top_k_diverse_trajectories += discarded_context
    top_k_diverse_trajectories = top_k_diverse_trajectories[:args.n_disagreements]

    log(f'Chosen disagreements:')
    for d in top_k_diverse_trajectories:
        log(f'Name: ({d.episode},{d.da_index})')

    return top_k_diverse_trajectories


def make_same_length(trajectories, horizon, traces):
    """make all trajectories the same length"""
    for d in trajectories:
        if len(d.a1_states) < horizon:
            """insert to start of video"""
            da_traj_idx = d.a1_states.index(d.da_index)
            for _ in range((horizon // 2) - da_traj_idx - 1):
                d.a1_states.insert(0, d.a1_states[0])
                d.a2_states.insert(0, d.a1_states[0])
            """insert to end of video"""
            while len(d.a1_states) < horizon:
                last_idx = d.a1_states[-1]
                if last_idx < len(traces[d.episode].states) - 1:
                    last_idx += 1
                    d.a1_states.append(last_idx)
                else:
                    d.a1_states.append(last_idx)

        for _ in range(horizon - len(d.a2_states)):
            d.a2_states.append(d.a2_states[-1])
    return trajectories


def mark_agent(img, action=None, text=None, position=None, color=255, thickness=2):
    assert position, 'Error - No position provided for marking agent'
    img2 = img.copy()
    top_left = (position[0], position[1])
    bottom_right = (position[0] + 30, position[1] + 15)
    cv2.rectangle(img2, top_left, bottom_right, color, thickness)

    """add action text"""
    if action or text:
        font = ImageFont.truetype('Roboto-Regular.ttf', 20)
        text = text or f'Chosen action: {ACTION_DICT[action]}'
        image = Image.fromarray(img2, 'RGB')
        draw = ImageDraw.Draw(image)
        draw.text((40, 40), text, (255, 255, 255), font=font)
        img_array = np.asarray(image)
        return img_array

    return img2
