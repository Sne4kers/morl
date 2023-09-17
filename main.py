from objective_list import Objective_list
from objective import Objective
from qtable import QTable
from testcase import TestCase
import random
import copy
import logging
from datetime import datetime

# reward functions for each of the objectives
def sat_0(a, b):
    if a > 1000:
        return 0
    else:
        return -abs(a - 1001)/10000

def sat_1(a, b):
    if a <= 1000:
        return 0
    else:
        return -abs(a - 1000)/10000

def sat_2(a, b):
    if a > 1000:
        return -1
    if b < 666:
        return 0
    else:
        return -abs(b - 665)/10000

def sat_3(a, b):
    if a > 1000:
        return -1
    if b >= 666:
        return 0
    else:
        return -abs(666 - b)/10000

# logger initialization
def get_logger():
    logger = logging.getLogger()
    now = datetime.now()
    log_file = str(now) + '_transfuser.log'
    logging.basicConfig(filename=log_file,
                        format='%(asctime)s %(message)s')
    logger.setLevel(logging.DEBUG)
    logger.info("Started")
    return logger

class Morlot:
    def __init__(self):
        self.epsilon = 1.0
        self.actions = list(range(4))
        self.obj_list = Objective_list()
        self.total_objs = 4
        self.action_count = 0
        self.number_of_actions = 10**7

        self.a = 10
        self.b = 10

        for i in range(self.total_objs):
            self.obj_list.add_to_list(Objective(i, self.actions))

    def choose_action(self, observation, rewards):
        self.action_count = self.action_count + 1
        if not rewards:
            return random.randint(0, len(self.actions)-1)
        uncovered_list = self.obj_list.get_all_uncovered()
        rewards_of_interests = []
        for i in range(len(rewards)):
            if i in uncovered_list:
                rewards_of_interests.append(rewards[i])
        max_index = rewards_of_interests.index(max(rewards_of_interests))
        q_table_index = uncovered_list[max_index]

        return self.obj_list.choose_action(q_table_index, observation, self.epsilon)

    def observe(self):
        return (self.a_n, self.b_n)

    def perform(self, action):
        # all types of mutation, basically increment and decrement operations for a and b
        if action == 0:
            self.a_n += 1
        elif action == 1:
            self.a_n -= 1
        elif action == 2:
            self.b_n += 1
        elif action == 3:
            self.b_n -= 1

        return (
            [
            sat_0(self.a_n, self.b_n),
            sat_1(self.a_n, self.b_n),
            sat_2(self.a_n, self.b_n),
            sat_3(self.a_n, self.b_n)
            ]
        , (self.a_n, self.b_n))

    def morlot(self, iterations):
        satisfied_obj = set()
        for i in range(iterations):
            self.epsilon = 1
            rewards = []
            tc = TestCase()
            self.a_n = 10
            self.b_n = 10
            actions_taken = 0

            stopping_condition = False
            logger = get_logger()
            while not stopping_condition:
                # epsilon decay
                if self.epsilon > 0.1:
                    self.epsilon *= 0.998
                else:
                    self.epsilon = 0.1
                
                # morlot section of tehe algorithm
                s = self.observe()
                a = self.choose_action(s, rewards)
                rewards, next_state = self.perform(a)
                self.obj_list.learn(s, a, rewards, next_state)

                actions_taken += 1

                if actions_taken > self.number_of_actions:
                    stopping_condition = True

                print(self.a_n, self.b_n, self.epsilon)

                # check if covered any objectives and log if so
                for index in range(self.total_objs):
                    if rewards[index] == 0 and (index not in satisfied_obj):
                        satisfied_obj.add(index)
                        self.obj_list.remove_from_uncovered(index)
                        if tc.get_reward() > -1:  # one test case satisfying multiple objectives
                            tc = copy.deepcopy(tc)
                        tc.update_reward(rewards[index])
                        logger.info("TEST CASE A = " + str(self.a_n) + " B = " + str(self.b_n) + " COVERS OBJECTIVE " + str(index))

alg = Morlot()
alg.morlot(10)
