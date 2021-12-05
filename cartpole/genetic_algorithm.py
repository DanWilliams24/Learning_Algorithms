import math
class GA_Model:
    def __init__(self) -> None:
        self.max_reward = 60
        pass


    def calc_fitness(reward):
        if(reward > 60):
            return 60
        else:
            return (1/(61-reward))

    def calc_fitness(self, reward):
        return math.tanh(reward-(self.max_reward-4.15))