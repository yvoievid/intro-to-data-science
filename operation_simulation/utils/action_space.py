import gymnasium as gym
from gymnasium import Space

class MultiAgentActionSpace(Space):
    def __init__(self, agents_action_space):
        for x in agents_action_space:
            assert isinstance(x, gym.Space)

        super(MultiAgentActionSpace, self).__init__(agents_action_space)
        self._agents_action_space = agents_action_space

    def sample(self):
        """ samples action for each agent from uniform distribution"""
        return [agent_action_space.sample() for agent_action_space in self._agents_action_space]

    @property
    def action_spaces(self):
        return self._agents_action_space