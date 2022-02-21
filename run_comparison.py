import argparse

from disagreements.compare_agents import compare_agents

configuration_dict = {
    "Highway" : {
        "FastRight": "disagreements/configs/Highway/FastRight.json",
        "SocialDistance": "disagreements/configs/Highway/SocialDistance.json",
        "ClearLane": "disagreements/configs/Highway/ClearLane.json",
    },

}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL Agent Comparisons')
    parser.add_argument('-env', '--env', help='environment name', default="Highway")
    parser.add_argument('-a1', '--a1_name', help='agent name', type=str, default="FastRight")
    parser.add_argument('-a2', '--a2_name', help='agent name', type=str, default="SocialDistance")
    parser.add_argument('-n', '--num_episodes', help='number of episodes to run', type=int,
                        default=3)
    parser.add_argument('-fps', '--fps', help='summary video fps', type=int, default=5)
    parser.add_argument('-l', '--horizon', help='number of frames to show per highlight',
                        type=int, default=10)
    parser.add_argument('-sb', '--show_score_bar', help='score bar', type=bool, default=False)
    parser.add_argument('-rand', '--randomized', help='randomize order of summary trajectories',
                        type=bool, default=True)
    parser.add_argument('-k', '--n_disagreements', help='# of disagreements in the summary',
                        type=int, default=5)
    parser.add_argument('-overlaplim', '--similarity_limit', help='# overlaping',
                        type=int, default=3)
    parser.add_argument('-imp', '--importance',
                        help='importance method', default='last_state')
    parser.add_argument('-v', '--verbose', help='print information to the console', default=True)
    parser.add_argument('-se', '--seed', help='environment seed', default=0)
    parser.add_argument('-res', '--results_dir', help='results directory', default='results')
    parser.add_argument('-tr', '--traces_path', help='path to traces file if exists',
                        default=None)
    args = parser.parse_args()

    args.a1_path = configuration_dict[args.env][args.a1_name]
    args.a2_path = configuration_dict[args.env][args.a2_name]

    """RUN"""
    compare_agents(args)