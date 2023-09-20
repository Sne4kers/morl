import copy

class TestCase:
    def __init__(self, args, rewards):
        self.tc = copy.deepcopy(args)
        self.rewards = copy.deepcopy(rewards)

    def __str__(self):
        return str(self.tc)