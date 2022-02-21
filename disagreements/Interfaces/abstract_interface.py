

class AbstractInterface(object):
    def __init__(self, config, output_dir):
        self.output_dir = output_dir
        self.config = config

    def initiate(self):
        return

    def get_state_action_values(self, agent, state):
        return

    def get_state_from_obs(self, agent, obs, params=None):
        return

    def get_next_action(self, agent, obs, state):
        return

    def get_features(self, env):
        return

    def pre_disagreement(self, env):
        return

    def post_disagreement(self, agent1, agent2, pre_params=None):
        return

    def disagreement_trace(self, e, trajectory_length, agent_rati, params=None):
        return

    def da_states_functionality(self, trace, params=None):
        return

    def update_trace(self, trace, agent, t, states, scores):
        return

class AbstractDisagreementTrace(object):
    def __init__(self, episode, horizon, agent_ratio):
        self.episode = episode
        self.trajectory_length = horizon
        self.agent_ratio = agent_ratio
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.infos = []
        self.reward_sum = 0
        self.length = 0
        self.states = []
        self.a2_trajectories = []
        self.a1_trajectory_indexes = []
        self.disagreement_indexes = []
        self.disagreement_trajectories = []

    def update(self, state_object, obs, a, r, done, infos, params=None):
        self.obs.append(obs)
        self.rewards.append(r)
        self.dones.append(done)
        self.infos.append(infos)
        self.actions.append(a)
        self.reward_sum += r
        self.states.append(state_object)
        self.length += 1
