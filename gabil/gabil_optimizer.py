#-*-encoding: utf-8 -*-
if __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from geneticoptimizer import GeneticOptimizer

import csv
import random
from time import time

# revisar
# - darle penalizacion en el fitness a las soluciones que tengan muchos 1's (son muy generales)
# - penalizar largo del arreglo (cantidad de reglas; favorecer conjuntos de reglas mas pequenos)
# - una solución no debe tener c1&c2 => p y c1&c2 => -p como reglas

class GabilOptimizer(GeneticOptimizer):
    def __init__(self, count_rules):
        self.count_rules = count_rules
        self.classes = [
            ['a', 'b'],
            [25, 50, 70, 81], # x<=25 -> 0 | 25<x<=50 -> 1 | ...
            [8, 16, 24, 30],
            ['u', 'y', 'l', 't'],
            ['g', 'p', 'gg'],
            ['c','d','cc','i','j','k','m','r','q','w','x','e','aa','ff'],
            ['v', 'h', 'bb', 'j', 'n', 'z', 'dd', 'ff', 'o'],
            [8, 14, 23, 30],
            ['t', 'f'],
            ['t', 'f'],
            [20, 40, 55, 70],
            ['t', 'f'],
            ['g', 'p', 's'],
            [50, 100, 200, 500, 1000, 1300, 1600, 2100],
            [20, 50, 100, 300, 1000, 2000, 5000, 10000, 20000, 110000],
            ['+', '-']
        ]
        self.rule_length = sum(len(c) for c in self.classes)
        self.continuous_indexes = {1,2,7,10,13,14}
        self.discrete_indexes = {0,3,4,5,6,8,9,11,12,15}

    def discretize_example(self, example):
        """
        :param example: example given for classification
        :return: example with all continuous fields discretized according to
                 the mapping described in self.classes

        DocTests:
        >>> GabilOptimizer(3).discretize_example(['b',56.75,12.25,'u','g','m','v',1.25,'t','t',04,'t','g',00200,0,'+'])
        ['b', 70, 16, 'u', 'g', 'm', 'v', 8, 't', 't', 20, 't', 'g', 200, 20, '+']
        """
        for i in self.continuous_indexes:
            for max_value in self.classes[i]:
                if float(example[i]) < max_value:
                    example[i] = max_value
                    break
        return example

    def remove_na(self, example):
        """
        :param example: example given for classification
        :return: curated example without NA's

        DocTests:
        >>> '?' in GabilOptimizer(3).remove_na(['b',34.83,4,'u','g','d','bb',12.5,'t','f',0,'t','g','?',0,'-'])
        False
        """
        for i in xrange(len(example)):
            if example[i] == '?':
                example[i] = random.choice(self.classes[i])

        assert '?' not in example
        return example

    def encode(self, decoded):
        """
        :param decoded: decoded example as a list of values or classes
        :return: encoded example as a list of 1's and 0's
        
        DocTests:
        >>> GabilOptimizer(3).encode(['b', 50, 8, 'u', 'g', 'c', 'h', 14, 'f', 'f', 20, 't', 'g', 50, 20, '-'])
        [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        """
        encoded = []

        for i in xrange(len(self.classes)):
            encoded += [1 if decoded[i] == class_k else 0 
                        for class_k in self.classes[i]]

        assert len(encoded) == sum((len(i) for i in self.classes))
        return encoded

    def decode(self, encoded):
        """
        :param encoded: encoded example as a list of 1's and 0's
        :return: decoded example as a list of values or classes

        DocTests:
        >>> GabilOptimizer(3).decode([0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        ['b', 50, 8, 'u', 'g', 'c', 'h', 14, 'f', 'f', 20, 't', 'g', 50, 20, '-']
        """
        decoded = []
        l = 0

        for i in xrange(len(self.classes)):
            slice_i = encoded[l : l + len(self.classes[i])]
            class_index = slice_i.index(1)
            decoded.append(self.classes[i][(class_index)])
            l += len(self.classes[i])

        assert len(decoded) == len(self.classes)
        return decoded

    def get_continuous_minmax(self):
        """
        Calculate min and max of columns with continuous values, this will help
        us construct discrete classes for these columns
        """
        mini = {1:1000000,2:1000000,7:1000000,10:1000000,13:1000000,14:1000000}
        maxi = {1:-1000000,2:-1000000,7:-1000000,10:-1000000,13:-1000000,
                14:-1000000}

        for e in self.examples:
            for i in [1,2,7,10,13,14]:
                try:
                    mini[i] = min(mini[i], float(e[i]))
                    maxi[i] = max(maxi[i], float(e[i]))
                except: pass

        return (mini, maxi)

    def load_input(self, file_path, training_percent):
        input_file = open(file_path, "r")
        csvreader = csv.reader(input_file)
        self.examples = [line for line in csvreader]
        train_len = int(training_percent * len(self.examples))
        # training dataset
        self.train_dataset = list([line for line in self.examples[:train_len]])
        self.train_dataset = [self.remove_na(e) for e in self.train_dataset]
        self.train_dataset = [self.discretize_example(e) 
                              for e in self.train_dataset]
        self.encoded_train = [self.encode(e) for e in self.train_dataset]
        # testing dataset
        self.test_dataset  = list([line for line in self.examples[train_len:]])
        self.test_dataset = [self.remove_na(e) for e in self.test_dataset]
        self.test_dataset = [self.discretize_example(e) 
                             for e in self.test_dataset]
        self.encoded_test = [self.encode(e) for e in self.test_dataset]
        #self.decoded_examples = [self.decode(e) for e in self.encoded_examples]

    @staticmethod
    def new_rule():
        """
        :return: randomly created list of 1's and 0's, where the last two bits
                 can't be equal

        DocTests:
        >>> len(GabilOptimizer.new_rule())
        77
        >>> sum(GabilOptimizer.new_rule()[75:])
        1
        """
        def shuffled(l):
            random.shuffle(l)
            return l

        return [random.choice([0,1]) for _ in xrange(75)] + shuffled([0,1])

    def individual(self):
        """
        representation example:
        [   
            [1,1,1,1,0,0,0,0],
            [0,0,0,0,1,0,1,0],
            [0,0,1,1,1,0,0,1]
        ]
        meaning by row:
            Feature1: arr[0:4]
            Feature2: arr[4:7]
            Class: arr[7]
        meaning of full matrix:
            Object with Feature1 in [1,2,3,4] => Class 0
            Object with Feature2 in [1,3] => Class 0
            Object with Feature1 in [3,4] and Feature2 in [1] => Class 1

        DocTests:
        """
        #return [self.new_rule() for _ in xrange(random.randint(3, self.count_rules))]
        return [random.choice(self.encoded_train) 
                for _ in xrange(random.randint(1, self.count_rules))]

    def mutate(self, mutate_prob, parents):
        for solution in parents:
            if mutate_prob > random.random():
                rule = random.randint(0, len(solution)-1)
                bit = random.randint(0, len(solution[rule])-1)
                solution[rule][bit] = 0 if solution[rule][bit] == 1 else 1
            #if mutate_prob > random.random():
            #    solution.append(self.new_rule())
            #if mutate_prob > random.random():
            #    if len(solution) > 1:
            #        solution.pop(random.randint(0, len(solution)-1))

    def flatten(self, l):
        """
        :param l: list to flatten
        :return: flattened list

        DocTests:
        >>> GabilOptimizer(3).flatten([[1, 1, 1], [0, 0, 0], [1, 0, 1]])
        [1, 1, 1, 0, 0, 0, 1, 0, 1]
        """
        return [i for sublist in l for i in sublist]

    def build(self, l, offset):
        """
        >>> GabilOptimizer(3).build([1, 1, 1, 0, 0, 0, 1, 0, 1], 3)
        [[1, 1, 1], [0, 0, 0], [1, 0, 1]]
        >>> GabilOptimizer(3).build([1, 1, 1, 0, 0, 0, 1, 0], 2)
        [[1, 1], [1, 0], [0, 0], [1, 0]]
        """
        builded = []
        i = 0
        #print len(l)

        while i < len(l):
            #print i
            builded.append(l[i : i + offset])
            i += offset

        return builded

    def locate(self, pos):
        """
        >>> GabilOptimizer(3).locate(3, 77)
        {'attribute': 1, 'offset': 1}
        >>> GabilOptimizer(3).locate(77, 77)
        {'attribute': 0, 'offset': 0}
        >>> GabilOptimizer(3).locate(155, 77)
        {'attribute': 0, 'offset': 1}
        >>> GabilOptimizer(3).locate(144, 77)
        {'attribute': 14, 'offset': 2}
        >>> GabilOptimizer(3).locate(77*8+1, 77)
        {'attribute': 0, 'offset': 1}
        >>> GabilOptimizer(3).locate(77*100+24, 77)
        {'attribute': 5, 'offset': 7}
        """
        while True:
            acc = 0

            for i in xrange(len(self.classes)):
                if acc + len(self.classes[i]) > pos:
                    return {'attribute': i, 'offset': pos - acc}
                acc += len(self.classes[i])

            pos -= self.rule_length

    def mix(self, parent1, parent2):
        def position(i, attribute, offset):
            return i * self.rule_length +\
                   sum(len(c) for c in self.classes[:attribute-1]) +\
                   offset

        flat1 = self.flatten(parent1)
        flat2 = self.flatten(parent2)

        # calculate swap points for parent1
        pos1 = random.randint(0, len(parent1)-1)
        pos2 = random.randint(0, len(parent1)-1)

        if pos1 > pos2:
            pos1, pos2 = pos2, pos1

        location1 = self.locate(pos1)
        location2 = self.locate(pos2)

        # calculate swap points for parent2 (based on parent1's locations)
        count_classes = len(parent2) / self.rule_length
        class1 = random.randint(0, count_classes)
        class2 = random.randint(0, count_classes)

        pos3 = position(class1, **location1)
        pos4 = position(class2, **location2)

        if pos3 > pos4:
            pos3, pos4 = pos3, pos4

        # create the child by swapping gene segments from parent1
        child = list(flat1)
        child[pos1:pos2] = flat2[pos3:pos4]
        child = self.build(child, self.rule_length)
        return child

    @staticmethod
    def matches(example, rule):
        for i in xrange(len(example)-2):
            if example[i] == 1 and rule[i] == 0:
                return False
        return True

    @staticmethod
    def count_correct(examples, solution):
        count = 0

        for example in examples:
            for rule in solution:
                if GabilOptimizer.matches(example, rule):
                    if example[len(example)-1] == rule[len(rule)-1]:
                        count += 1
                        break
                    else:
                        count -= 1
        return count

    def fitness(self, solution):
        fit = (GabilOptimizer.count_correct(self.train_dataset, solution) /
              len(self.encoded_train)) ** 2
        return fit

    @staticmethod
    def precision(classifier, dataset):
        count = 0
        print len(dataset), len(dataset[0])
        for example in dataset:
            print example
            if example[len(example)-2:] == GabilOptimizer.classify(classifier, example):
                count += 1
        return float(count) / len(dataset)

    @staticmethod
    def classify(classifier, example):
        for rule in classifier:
            if GabilOptimizer.matches(example, rule):
                return rule[len(rule)-2:]


if __name__ == '__main__':
    go = GabilOptimizer(5)
    go.load_input("credit-screening/crx.data", training_percent=0.85)
    best = (float('-inf'), [])
    solution = go.runGA(best=best, iterations=200, pop_count=10, target=10000.0,
                        mutate_prob=0.01, reverse=True)
    #go.find_optimal(iterations=100, pop_count=10, target=10000.0,
    #                mutate_prob=0.1, reverse=True)
    #print solution[1]
    print solution[0]
    print len(solution[1])

    print GabilOptimizer.precision(solution[1], go.encoded_test)

# TODO: flatten all rulesets from the beggining and use iterators to 
# yield lists of 77 characters every time
