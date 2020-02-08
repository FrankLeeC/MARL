# -*- coding:utf-8 -*-

from matplotlib import pyplot as plt
import random
import numpy as np

class Player:

    def __init__(self, q, strategy=[0.5, 0.5], learn=True):
        self.q = [0 for _ in range(len(strategy))]
        self.strategy = strategy
        self.equilibrium_strategy = [1/len(self.strategy) for _ in range(len(self.strategy))]
        self.turns = 0
        self.gamma = 0.9
        self.delta_win = 0.0
        self.delta_lose = 2 * self.delta_win
        self.learn = learn

    def action(self):
        '''
        执行动作
        '''
        if not self.learn:
            self.current_action = self.policy_action()
            return self.current_action
        j = 0
        if np.random.binomial(1, self.epsilon()) == 1:
            np.random.randint(0, len(self.strategy))
        else:
            j = np.argmax(self.q)
        self.current_action = j
        return self.current_action
    
    def policy_action(self):
        r = random.random()
        s = 0.0
        i = 0
        while True:
            s += self.strategy[i]
            if s >= r:
                break
            i += 1
        return i

    def learning_rate(self):
        return 1/(10+0.00001*self.turns)

    def epsilon(self):
        return 0.5/(1+0.0001*self.turns)

    def get_strategy(self, i):
        return self.strategy[i]

    def update(self, reward):
        '''
        更新
        '''
        if self.learn:
            self.update_value(reward)
            self.update_strategy()
            self.udpate_equilibrium_strategy()
            self.turns += 1

    def update_value(self, reward):
        self.q[self.current_action] = (1-self.learning_rate()) * self.q[self.current_action] + self.learning_rate() * (reward + self.gamma * np.max(self.q))

    def get_delta(self):
        self.delta_win = 1.0 / (1000 + self.turns)
        self.delta_lose = 2.0 * self.delta_win
        if np.sum(np.dot(self.strategy, self.q)) > np.sum(np.dot(self.equilibrium_strategy, self.q)):
            return self.delta_win
        return self.delta_lose

    def update_strategy(self):
        best_action = np.argmax(self.q)
        s = 0.0
        idx = 0
        for i in range(len(self.strategy)):
            if i != best_action:
                a = min(self.strategy[i], self.get_delta()/(len(self.strategy)-1))
                self.strategy[i] += -a
                s += a
            else:
                idx = i
        self.strategy[idx] += s

    def udpate_equilibrium_strategy(self):
        self.equilibrium_strategy[self.current_action] += 1/(self.turns+1) * (self.strategy[self.current_action] - self.equilibrium_strategy[self.current_action])
            
    
class PHC:

    def __init__(self, name, player1, player2, count):
        self.name = name
        self.p1 = player1
        self.p2 = player2
        self.iteration = count

    def run(self):
        x = []
        y1, y2 = [], []
        for i in range(self.iteration):
            s1, s2 = self.p1.get_strategy(0), self.p2.get_strategy(0)
            x.append(i)
            y1.append(s1)
            y2.append(s2)
            # self.draw(x, y1, y2)
            a1, a2 = self.p1.action(), self.p2.action()
            r1, r2 = self.reward(a1, a2)
            self.p1.update(r1)
            self.p2.update(r2)
        self.draw(x, y1, y2)
        self.close()

    def reward(self, a1, a2):
        if a1 + a2 == 0 or a1 + a2 == 2:
            return 1, -1
        return -1, 1
    
    def draw(self, x, y1, y2):
        # plt.clf()
        plt.plot(x, y1, color='blue', label='p1')
        plt.plot(x, y2, color='red', label='p2')
        plt.legend()
        # plt.pause(0.1)

    def close(self):
        plt.title(self.name)
        plt.savefig('./' + self.name + '.png')
        plt.close()

def ALearningBFix():
    p1 = Player(q=[0.0, 0.0], strategy=[0.2, 0.8], learn=True)
    p2 = Player(q=[0.0, 0.0], strategy=[1.0, 0.0], learn=False)
    phc = PHC('A_learn_B_fix', p1, p2, 10000)
    phc.run()

def ALearningBFix1():
    p1 = Player(q=[0.0, 0.0], strategy=[0.2, 0.8], learn=True)
    p2 = Player(q=[0.0, 0.0], strategy=[0.2, 0.8], learn=False)
    phc = PHC('A_learn_B_fix1', p1, p2, 10000)
    phc.run()

if __name__ == "__main__":
    ALearningBFix()
    ALearningBFix1()
