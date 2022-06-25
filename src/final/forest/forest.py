import numpy as np

class ForestManagement:
    def __init__(self,max_age,seed=42):
        self.max_age = max_age
        self.action_size = 2
        self.rand_generator = np.random.RandomState(seed)
        self.prob_fire = 0.2
        self.age = 0

    def reset(self):
        self.age = 0
        return self.age/self.max_age

    def step(self, action):
        if action == 1: 
            # cut action
            reward = self.age
            self.age = 0            
        else:
            # wait action
            reward = 0
            if self.rand_generator.rand() < self.prob_fire:
                self.age = 0
            else:
                if self.age >= self.max_age:
                    self.age = self.max_age
                else:
                    self.age += 1

        return self.age/self.max_age, reward/self.max_age


if __name__ == "__main__":
    forest = ForestManagement(max_age=10)
    for i in range(30):
        action = np.random.choice([0,1],p=[0.8,0.2])
        print(action,forest.step(action))