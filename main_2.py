from objective_list import Objective_list
from objective import Objective
from qtable import QTable
from testcase import TestCase
import random
import copy
import logging
import numpy as np
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
    if count == b:
        return 0
    else:
        return -abs(b + 1 -count)/10000

def sat_5(count, b):    
    if count != b:
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
        self.covered_objective_set_ids = set()
        self.epsilon = 1.0
        self.new_test_case_prob = 0.1
        self.actions = list(range(2+2+3))
        self.obj_list = Objective_list()
        self.population = []
        self.total_objs = 8
        self.action_count = 0
        self.number_of_actions = 10**7

        t = self.new_random_testcase()
        print(t.rewards)
        self.run(t)
        print(t.rewards)

        for i in range(self.total_objs):
            self.obj_list.add_to_list(Objective(i, self.actions))
            self.population.append(t)

        self.logger = get_logger()

        for index in range(self.total_objs):
            if t.rewards[index] == 0 and (index not in self.covered_objective_set_ids):
                self.covered_objective_set_ids.add(index)
                self.obj_list.remove_from_uncovered(index)

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
                self.logger.info("TEST CASE " + str(t) + " COVERS OBJECTIVE " + objective_str)

    def choose_from_population(self):
        random_value = np.random.randint(low=0, high=len(self.obj_list.uncovered_objective_list))
        id_of_uncovered_obj = self.obj_list.uncovered_objective_list[random_value].obj_no
        return self.population[id_of_uncovered_obj]

    def choose_action(self, observation):
        self.action_count = self.action_count + 1


        uncovered_list = self.obj_list.get_all_uncovered()
        rewards_of_interests = []
        for i in range(len(observation.rewards)):
            if i in uncovered_list:
                rewards_of_interests.append(observation.rewards[i])
        max_index = rewards_of_interests.index(max(rewards_of_interests))
        q_table_index = uncovered_list[max_index]

        return self.obj_list.choose_action(q_table_index, (observation.tc[0], observation.tc[1], observation.tc[2]), self.epsilon)

    def new_random_testcase(self):
        str_len = random.randint(0, 64)
        s = ""
        for i in range(str_len):
            # Choose a random character from the list of ASCII symbols
            s = s + random.choice(all_ascii_symbols)
        return TestCase([random.randint(-10000000, 10000000), random.randint(-10000000, 10000000), s], [0 for i in range(self.total_objs)])

    def run(self, testcase):
        counter = Counter(testcase.tc[2])
        if testcase.tc[0] >= 1000:
            count = counter['0'] + counter['1'] + counter['2'] + counter['3'] + counter['4'] + counter['5'] + counter['6'] + counter['7'] + counter['8'] + counter['9']
        else: # a < 1000
            count = counter['1']

        testcase.rewards = [
            sat_2(testcase.tc[0], testcase.tc[1], testcase.tc[2]),
            sat_2_1(testcase.tc[0], testcase.tc[1], testcase.tc[2]),
            sat_2_2(testcase.tc[0], testcase.tc[1], testcase.tc[2]),
            sat_3(testcase.tc[0], testcase.tc[1], testcase.tc[2]),
            sat_3_2(testcase.tc[0], testcase.tc[1], testcase.tc[2]),
            sat_3_1(testcase.tc[0], testcase.tc[1], testcase.tc[2]),
            sat_4(count, testcase.tc[1]),
            sat_5(count, testcase.tc[1])
            ]

        return testcase

    def perform_action_on_testcase(self, action, testcase):
        # all types of mutation, basically increment and decrement operations for a and b
        new_actions_set = copy.deepcopy(testcase.tc)
        if action == 0:
            new_actions_set[0] += 1
        elif action == 1:
            new_actions_set[0] -= 1
        elif action == 2:
            new_actions_set[1] += 1
        elif action == 3:
            new_actions_set[1] -= 1
        elif action == 4:
            # Choose a random character from the list of ASCII symbols
            new_actions_set[2] = new_actions_set[2] + random.choice(all_ascii_symbols)
        elif action == 5:
            new_actions_set[2] = new_actions_set[2][:-1]
        elif action == 6:
            if len(new_actions_set[2]) != 0:
                index_to_replace = random.randint(0, len(new_actions_set[2]) - 1)
                new_actions_set[2] = new_actions_set[2][:index_to_replace] + chr(random.randint(32, 126)) + new_actions_set[2][index_to_replace+1:]

        new_test_case = TestCase(new_actions_set, [])
        self.run(new_test_case)
        return new_test_case

    def morlot(self):
        self.epsilon = 1
        rewards = []

        actions_taken = 0

        stopping_condition = False
        
        while len(self.obj_list.uncovered_objective_list) != 0 and not stopping_condition:
            # epsilon decay
            if self.epsilon > 0.1:
                self.epsilon -= 0.0001
            else:
                self.epsilon = 0.1
            
            # morlot section of tehe algorithm
            random_value = np.random.uniform()
            if random_value > self.new_test_case_prob:
                new_test_case = self.new_random_testcase()
                self.run(new_test_case)
            else:
                s = self.choose_from_population()
                a = self.choose_action(s)
                new_test_case = self.perform_action_on_testcase(a, s)
                self.obj_list.learn((s.tc[0], s.tc[1], s.tc[2]), a, new_test_case.rewards, (new_test_case.tc[0], new_test_case.tc[1], new_test_case.tc[2]))

            actions_taken += 1

            if actions_taken > self.number_of_actions:
                stopping_condition = True

            if actions_taken % 1000 == 0:
                print(new_test_case)
            
            # check if covered any objectives and log if so
            for index in range(self.total_objs):
                if new_test_case.rewards[index] == 0 and (index not in self.covered_objective_set_ids):
                    self.covered_objective_set_ids.add(index)
                    self.obj_list.remove_from_uncovered(index)

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
                    self.logger.info("TEST CASE " + str(new_test_case) + " COVERS OBJECTIVE " + objective_str)
                elif new_test_case.rewards[index] > self.population[index].rewards[index]:
                    self.population[index] = new_test_case
            

alg = Morlot()
alg.morlot()
