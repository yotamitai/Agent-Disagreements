# from disagreements.Interfaces.frogger_interface import FroggerInterface
# from disagreements.Interfaces.gym_interface import GymInterface
from disagreements.Interfaces.highway_interface import HighwayInterface


def get_agent(args, config_path):
    """Implement here for specific agent and environment loading scheme"""
    if args.env == "Highway":
        interface = HighwayInterface(config_path, args.output_dir, args.num_episodes)

    env, agent, evaluation = interface.initiate()
    agent.interface = interface
    env.seed(0)
    return env, agent, evaluation