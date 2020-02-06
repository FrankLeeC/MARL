# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import random

'''
matching-pennies
两个玩家格子决定自己硬币的正反面
1.如果都是正面 p1 得 3 分
2.如果都是反面 p1 的 1 分
3.其余情况 p2 得 2 分

p1矩阵 r
| r11, r12 |
| r21, r22 |

V1(alpha, beta) = alpha * beta * r11 + 
                 alpha * (1 - beta) * r12 +
                 (1 - alpha) * beta * r21 +
                 (1 - alpha) * (1 - beta) * r22
u1 = r11 - r12 - r21 + r22

V1(alpha, beta) = u1 * alpha * beta + alpha * (r12 - r22) + beta * (r21 - r22) + r22
d(V1)/d(alpha) = beta * u1 +(r12 - r22)
alpha += eta * factor * d(V1)/d(alpha)
factor = factor_min   if win
factor = factor_max   if loss

p2矩阵 c
| c11, c12 |
| c21, c22 |

V2(alpha, beta) = alpha * beta * c11 + 
                 alpha * (1 - beta) * c12 +
                 (1 - alpha) * beta * c21 +
                 (1 - alpha) * (1 - beta) * c22
u2 = c11 - c12 - c21 + c22

V2(alpha, beta) = u2 * alpha * beta + alpha * (c12 - c22) + beta * (c21 - c22) + c22
d(V2)/d(beta) = alpha * u2 + (c21 - c22)
beta += eta * factor * d(V2)/d(beta)
factor = factor_min   if win
factor = factor_max   if loss
'''


'''
两个玩家(p1, p2)，每个玩家两个动作(a1, a2), (b1, b2)
alpha p1执行动作a1的概率
beta  p2执行动作b1的概率
'''

class MatchingPennies:

    def __init__(self):
        self.init_p1()
        self.init_p2()
        self.eta = 0.1
        self.factor_max = 0.8
        self.factor_min = 0.2
        self.init_draw()

    def init_p1(self):
        self.r11 = 3.0
        self.r12 = -2.0
        self.r21 = -2.0
        self.r22 = 1.0
        self.u1 = self.r11 - self.r12 - self.r21 + self.r22
        self.alpha = random.random()
        self.alpha_best = 3/8
    
    def init_p2(self):
        self.c11 = -3.0
        self.c12 = 2.0
        self.c21 = 2.0
        self.c22 = -1.0
        self.u2 = self.c11 - self.c12 - self.c21 + self.c22
        self.beta = random.random()
        self.beta_best = 3/8
    
    def init_draw(self):
        self.x = []
        self.d1 = []
        self.d2 = []
        self.i = 1
        plt.ion()

    def v1_best(self):
        return self.u1 * self.alpha_best * self.beta_best + self.alpha_best * (self.r12 - self.r22) + self.beta_best * (self.r21 - self.r22) + self.r22

    def v1(self):
        return self.u1 * self.alpha * self.beta + self.alpha * (self.r12 - self.r22) + self.beta * (self.r21 - self.r22) + self.r22

    def v2_best(self):
        return self.u2 * self.alpha_best * self.beta_best + self.alpha_best * (self.c12 - self.c22) + self.beta_best * (self.c21 - self.c22) + self.c22

    def v2(self):
        return self.u2 * self.alpha * self.beta + self.alpha * (self.c12 - self.c22) + self.beta * (self.c21 - self.c22) + self.c22

    def draw(self):
        plt.clf()
        self.x.append(self.i)
        self.d1.append(self.alpha)
        self.d2.append(self.beta)
        self.i += 1
        plt.plot(self.x, self.d1, color='blue', label='p1')
        plt.plot(self.x, self.d2, color='red', label='p2')
        plt.legend()
        plt.pause(0.1)

    def close(self):
        plt.savefig('./matching_pennies.png')
        plt.ioff()

    def match(self):
        factor1, factor2 = self.factor_max, self.factor_max
        if self.v1() > self.v1_best():
            factor1 = self.factor_min
        if self.v2() > self.v2_best():
            factor2 = self.factor_min
        return factor1, factor2

    def update(self):
        epsilon = 1e-6
        while True:
            self.draw()
            factor1, factor2 = self.match()
            va = self.eta * factor1 * (self.beta * self.u1 + (self.r12 - self.r22))
            vb = self.eta * factor2 * (self.alpha * self.u2 + (self.c21 - self.c22))
            v = abs(va) + abs(vb)
            if v <= epsilon:
                break
            self.alpha += va
            self.beta += vb

    def output_strategy(self):
        print(self.alpha, self.beta)

    def run(self):
        self.output_strategy()
        self.update()
        self.close()
        self.output_strategy()
        print('count: ', self.i)

if __name__ == "__main__":
    mp = MatchingPennies()
    mp.run()
    