import argparse
import random
from os.path import abspath
from os.path import join

from numpy import argmax

from disagreements.disagreement import save_disagreements, get_top_k_disagreements, disagreement, \
    State, make_same_length, get_trajectories, get_frames
from disagreements.get_trajectories import rank_trajectories
from disagreements.logging_info import log, get_logging
from disagreements.get_agent import get_agent
from disagreements.common.utils import load_traces, save_traces
from copy import deepcopy

from disagreements.merge_and_fade import merge_and_fade


def online_comparison(args):
    """Compare two agents running online, search for disagreements"""
    """get agents and environments"""
    env1, a1, evaluation1 = get_agent(args, args.a1_path)
    env2, a2, evaluation2 = get_agent(args, args.a2_path)
    env1.args = env2.args = args

    """agent assessment"""
    agent_ratio = 1

    """Run"""
    traces = []
    for e in range(args.num_episodes):
        log(f'Running Episode number: {e}', args.verbose)
        trace = a1.interface.disagreement_trace(e, args.horizon, agent_ratio)
        """initial state"""
        obs, _ = env1.reset(), env2.reset()
        assert obs.tolist() == _.tolist(), f'Nonidentical environment'
        t, r, done, infos = 0, 0, False, {}
        s = a1.interface.get_state_from_obs(a1, obs, [r, done])
        a1.previous_state = a2.previous_state = obs  # required
        a1_s_a_values = a1.interface.get_state_action_values(a1, s)
        a2_s_a_values = a2.interface.get_state_action_values(a2, s)
        a1_a = a1.interface.get_next_action(a1, obs, s) if not done else None
        a2_a = a2.interface.get_next_action(a2, obs, s) if not done else None
        state_id, frame = (e, t), env1.render(mode='rgb_array')
        features = a1.interface.get_features(env1)
        state_obj = State(t, e, obs, s, a1_s_a_values, frame, features)
        trace.update(state_obj, obs, a1, r, done, infos, params=[a1_s_a_values, a2_s_a_values])

        # for _ in range(20):  # TODO remove
        while not done:

            """check for disagreement"""
            if a1_a != a2_a:
                log(f'\tDisagreement at step {t}', args.verbose)
                pre_vars = a2.interface.pre_disagreement(env1)
                disagreement(t, trace, env2, a1, a2, obs, s)
                """return agent 2 environment to the disagreement state"""
                env2 = a2.interface.post_disagreement(a1, a2, pre_vars)
            """Transition both agent's based on agent 1 action"""
            t += 1
            obs, r, done, info = env1.step(a1_a)
            _ = env2.step(a1_a)  # dont need returned values
            assert obs.tolist() == _[0].tolist(), f'Nonidentical environment transition'

            s = a1.interface.get_state_from_obs(a1, obs, [r, done])
            a1_s_a_values = a1.interface.get_state_action_values(a1, s)
            a2_s_a_values = a2.interface.get_state_action_values(a2, s)
            a1_a = a1.interface.get_next_action(a1, obs, s) if not done else None
            a2_a = a2.interface.get_next_action(a2, obs, s) if not done else None
            state_id, frame = (e, t), env1.render(mode='rgb_array')
            features = a1.interface.get_features(env1)
            state_obj = State(t, e, obs, s, a1_s_a_values, frame, features)
            trace.update(state_obj, obs, a1, r, done, infos,
                         params=[a1_s_a_values, a2_s_a_values])

        """end of episode"""
        traces.append(trace)

    """close environments"""
    env1.close()
    env2.close()
    if evaluation1:
        evaluation1.close()
        evaluation2.close()
    return traces


def compare_agents(args):
    name = get_logging(args)
    traces = load_traces(args.traces_path) if args.traces_path else online_comparison(args)
    log(f'Obtained traces', args.verbose)

    """get trajectories"""
    [get_trajectories(trace) for trace in traces]
    log(f'Obtained trajectories', args.verbose)

    """save traces"""
    save_traces(traces, args.output_dir)
    log(f'Saved traces', args.verbose)

    """rank disagreement trajectories by importance measures"""
    rank_trajectories(traces, args.importance)

    """top k diverse disagreements"""
    disagreements = get_top_k_disagreements(traces, args)
    if not disagreements:
        log(f'No disagreements found', args.verbose)
        return
    log(f'Obtained {len(disagreements)} disagreements', args.verbose)

    """make all trajectories the same length"""
    disagreements = make_same_length(disagreements, args.horizon, traces)

    """randomize order"""
    if args.randomized: random.shuffle(disagreements)

    """get frames and mark disagreement frame"""
    a1_disagreement_frames, a2_disagreement_frames = [], []
    for d in disagreements:
        t = traces[d.episode]
        relative_idx = d.da_index - d.a1_states[0]
        a1_frames, a2_frames = get_frames(t, d.a1_states, d.a2_states, d.trajectory_index)

        if args.env == "Highway":
            from disagreements.Interfaces.highway_interface import highway_mark_frames
            a1_frames, a2_frames = highway_mark_frames(t,d,a1_frames, a2_frames, relative_idx)


        a1_disagreement_frames.append(a1_frames)
        a2_disagreement_frames.append(a2_frames)

    """save disagreement frames"""
    video_dir = save_disagreements(a1_disagreement_frames, a2_disagreement_frames,
                                   args.output_dir, args.fps)
    log(f'Disagreements saved', args.verbose)

    """generate video"""
    fade_duration = 2
    fade_out_frame = args.horizon - fade_duration + 11  # +11 from pause in save_disagreements
    merge_and_fade(video_dir, args.n_disagreements, fade_out_frame, name)
    log(f'DAs Video Generated', args.verbose)

    """ writes results to files"""
    log(f'\nResults written to:\n\t\'{args.output_dir}\'', args.verbose)
