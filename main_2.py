from objective_list import Objective_list
from objective import Objective
from qtable import QTable
from testcase import TestCase
import random
import copy
import logging
from datetime import datetime
from collections import Counter
import string

# Get all ASCII symbols
all_ascii_symbols = string.ascii_letters + string.digits + string.punctuation

# reward functions for each of the objectives
def sat_2(a, b, s):
    if a < 1000:
        return 0
    else:
        return -abs(a - 999)/10000

def sat_3(a, b, s):
    if a >= 1000:
        return 0
    else:
        return -abs(a - 1000)/10000

def sat_2_1(a, b, s):
    if a < 1000:
        if len(s) != 0:
            return 0
        else:
            return -1
    else:
        return -2

def sat_2_2(a, b, s):
    if a < 1000:
        if len(s) != 0:
            min_diff = 256
            for character in s:
                if character == '1':
                    return 0
                min_diff = min(min_diff, abs(ord('1') - ord(character)))
            return -min_diff/256
        else:
            return -2
    else:
        return -3

def sat_3_1(a, b, s):
    if a >= 1000:
        if len(s) != 0:
            return 0
        else:
            return -1
    else:
        return -2

def sat_3_2(a, b, s):
    if a >= 1000:
        if len(s) != 0:
            min_diff = 256
            for character in s:
                if ord(character) - ord('0') <= 9:
                    return 0
                min_diff = min(min_diff, min(abs(ord(character) - ord('0')), abs(ord(character) - ord('9'))))
            return -min_diff/256
        else:
            return -2
    else:
        return -3

def sat_4(count, b):
    if count > b:
        return 0
    else:
        return -abs(b + 1 -count)/10000

def sat_5(count, b):    
    if count <= b:
        return 0
    else:
        return -abs(b-count)/10000

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
        self.actions = list(range(2+2+3))
        self.obj_list = Objective_list()
        self.total_objs = 8
        self.action_count = 0
        self.number_of_actions = 10**7

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
        elif action == 4:
            # Choose a random character from the list of ASCII symbols
            self.s_n = self.s_n + random.choice(all_ascii_symbols)
        elif action == 5:
            self.s_n = self.s_n[:-1]
        elif action == 6:
            if len(self.s_n) != 0:
                index_to_replace = random.randint(0, len(self.s_n) - 1)
                self.s_n = self.s_n[:index_to_replace] + chr(random.randint(32, 126)) + self.s_n[index_to_replace+1:]

        counter = Counter(self.s_n)
        if self.a_n >= 1000:
            count = counter['0'] + counter['1'] + counter['2'] + counter['3'] + counter['4'] + counter['5'] + counter['6'] + counter['7'] + counter['8'] + counter['9']
        else: # a < 1000
            count = counter['1']

        return (
            [
            sat_2(self.a_n, self.b_n, self.s_n),
            sat_2_1(self.a_n, self.b_n, self.s_n),
            sat_2_2(self.a_n, self.b_n, self.s_n),
            sat_3(self.a_n, self.b_n, self.s_n),
            sat_3_1(self.a_n, self.b_n, self.s_n),
            sat_3_2(self.a_n, self.b_n, self.s_n),
            sat_4(count, self.b_n),
            sat_5(count, self.b_n)
            ]
        , (self.a_n, self.b_n, self.s_n))

    def morlot(self, iterations):
        satisfied_obj = set()
        for i in range(iterations):
            self.epsilon = 1
            rewards = []
            tc = TestCase()
            self.a_n = 10
            self.b_n = 10
            self.s_n = ""
            actions_taken = 0

            stopping_condition = False
            logger = get_logger()
            while not stopping_condition:
                # epsilon decay
                if self.epsilon > 0.1:
                    self.epsilon -= 0.0001
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

                if actions_taken % 1000 == 0:
                    print(self.a_n, self.b_n, repr(self.s_n), self.epsilon)

                # check if covered any objectives and log if so
                for index in range(self.total_objs):
                    if rewards[index] == 0 and (index not in satisfied_obj):
                        satisfied_obj.add(index)
                        self.obj_list.remove_from_uncovered(index)
                        if tc.get_reward() > -1:  # one test case satisfying multiple objectives
                            tc = copy.deepcopy(tc)
                        tc.update_reward(rewards[index])
                        objective_number = index
                        match index:
                            case 0:
                                objective_str = "2"
                            case 1:
                                objective_str = "2.1"
                            case 2:
                                objective_str = "2.2"
                            case 3:
                                objective_str = "3"
                            case 4:
                                objective_str = "3.1"
                            case 5:
                                objective_str = "3.2"
                            case 6:
                                objective_str = "4"
                            case 7:
                                objective_str = "5"
                        logger.info("TEST CASE A = " + str(self.a_n) + " B = " + str(self.b_n) + " S = " + repr(self.s_n) + " COVERS OBJECTIVE " + objective_str)

alg = Morlot()
alg.morlot(10)
