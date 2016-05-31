#-*-encoding: utf-8 -*-
if __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from geneticoptimizer import GeneticOptimizer
    from parents_mixins import ParentsRouletteSelectionMixin,\
        ParentsRandomSelectionMixin
    from survivors_mixins import SurvivorsRouletteSelectionMixin,\
        SurvivorsTruncatedSelectionMixin

import csv
import random
import pdb
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

    def fill_na(self, example):
        """
        :param example: example given for classification
        :return: curated example without NA's

        DocTests:
        >>> '?' in GabilOptimizer(3).fill_na(['b',34.83,4,'u','g','d','bb',12.5,'t','f',0,'t','g','?',0,'-'])
        False
        """
        for i in xrange(len(example)):
            if example[i] == '?':
                example[i] = random.choice(self.classes[i])

        assert '?' not in example
        return example

    def remove_na(self, dataset, symbol='?'):
        """
        :return: cleaned dataset without cases with NA's ('?')

        >>> GabilOptimizer(3).remove_na([['?',1],[1,1],['?',2]])
        [[1, 1]]
        >>> GabilOptimizer(3).remove_na([['?',1],['NA',1],['?',2]], 'NA')
        [['?', 1], ['?', 2]]
        """
        return [i for i in dataset if symbol not in i]

    def encode(self, decoded):
        """
        :param decoded: decoded example as a list of values or classes
        :return: encoded example as a list of 1's and 0's
        
        DocTests:
        >>> GabilOptimizer(3).encode(['b', 50, 8, 'u', 'g', 'c', 'h', 14, 'f', 'f', 20, 't', 'g', 50, 20, '-'])
        [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        >>> sum(GabilOptimizer(3).encode(['b', 50, 8, 'u', 'g', 'c', 'h', 14, 'f', 'f', 20, 't', 'g', 50, 20, '-']))
        16
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
        self.examples = self.remove_na(self.examples)
        self.examples = [self.discretize_example(e) for e in self.examples]
        random.shuffle(self.examples)
        train_len = int(training_percent * len(self.examples))
        # training dataset
        self.train_dataset = list([line for line in self.examples[:train_len]])
        self.encoded_train = [self.encode(e) for e in self.train_dataset]
        # testing dataset
        self.test_dataset  = list([line for line in self.examples[train_len:]])
        self.encoded_test = [self.encode(e) for e in self.test_dataset]
        input_file.close()

    def toBooleanRule(self, rule):
        matrix = [[] for c in self.classes]
        acc = 0

        for i in xrange(len(self.classes)):
            for j in xrange(len(self.classes[i])):
                if rule[acc] == 1:
                    matrix[i].append('F%d = %s' % (i, str(self.classes[i][j])))
                acc += 1
            matrix[i] = ' or '.join(matrix[i])
        matrix = [i for i in matrix if len(i) > 0]
        conditions = '(' + ') and ('.join(matrix[:-1])
        output = matrix[-1]
        #print matrix
        #print conditions + ') => ' + output
        #exit(0)
        #assert '+' in output or '-' in output
        return conditions + ') => ' + output


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
        return [random.choice([random.choice(self.encoded_train), self.new_rule()])
                for _ in xrange(random.randint(1, self.count_rules))]

    def mutate(self, mutate_prob, parents):
        for solution in parents:
            if mutate_prob > random.random():
                rule = random.randint(0, len(solution['individual'])-1)
                bit = random.randint(0, len(solution['individual'][rule])-1)
                solution['individual'][rule][bit] = \
                    0 if solution['individual'][rule][bit] == 1 else 1

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
        # print "BUILD: %d" % len(l)

        while i < len(l):
            #print i
            builded.append(l[i : i + offset])
            i += offset

        return builded

    def mix(self, parent1, parent2):
        # print "\nMIX"
        flat1 = self.flatten(parent1)
        flat2 = self.flatten(parent2)
        # print "len(flat) %d %d " % (len(flat1), len(flat2))
        # calculate swap points for parent1
        pos1, pos2 = 0, 0

        while pos1 == pos2:
            pos1 = random.randint(0, len(flat1)-1)
            pos2 = random.randint(0, len(flat1)-1)

        if pos1 > pos2:
            pos1, pos2 = pos2, pos1

        location1 = pos1 % self.rule_length
        location2 = pos2 % self.rule_length

        # print "location %d %d" % (location1, location2)

        # calculate swap points for parent2 (based on parent1's locations)
        rule1 = random.randint(0, len(parent2)-1)
        rule2 = random.randint(0, len(parent2)-1)
        
        if location1 > location2:
            if len(parent2) == 1:
                child = list(flat1)
                child[location2:location1+1] = flat2[location2:location1+1]
                assert len(child) % self.rule_length == 0
                return self.build(child, self.rule_length)
            else:
                while rule1 == rule2:
                    rule1 = random.randint(0, len(parent2)-1)
                    rule2 = random.randint(0, len(parent2)-1)

        if rule1 > rule2:
            rule1, rule2 = rule2, rule1
        # print "rule %d %d" % (rule1, rule2)
        pos3 = (rule1 * self.rule_length) + location1
        pos4 = (rule2 * self.rule_length) + location2

        # print "pos %d %d %d %d" % (pos1, pos2, pos3, pos4)
        # create the child by swapping gene segments from parent1
        child = list(flat1)
        child[pos1:pos2+1] = flat2[pos3:pos4+1]

        # print "len(flat) %d %d %d" % (len(flat1), len(flat2), len(child))
        assert len(child) % self.rule_length == 0
        child = self.build(child, self.rule_length)
        if len(child) > 100: return parent1
        return child

    def matches(self, example, rule):
        """
        >>> GabilOptimizer(3).matches([1,1,1,1], [0,0,1,1])
        False
        >>> GabilOptimizer(3).matches([1,1,1,0], [1,1,0,0])
        True
        >>> GabilOptimizer(3).matches([1,1,1,1], [1,1,1,0])
        True
        >>> GabilOptimizer(3).matches([0,0,1,1], [0,0,0,0])
        True
        """
        # print "example\n" + str(example)
        # print "rule\n" + str(rule)
        for i in xrange(len(example)-2):
            if example[i] == 1 and rule[i] == 0:
                return False
        return True

    def fitness(self, solution):
        def count_correct(examples, solution):
            count = 0

            for example in examples:
                for rule in solution:
                    if self.matches(example, rule):
                        if example[len(example)-1] == rule[len(rule)-1]:
                            count += 1
                        else:
                            count -= 1
                        break
            return count

        correct = count_correct(self.encoded_train, solution)
        if len(solution) > 30:
            length_penalty = len(solution) / 1000.0
        else:
            length_penalty = 0
        correct -= length_penalty*correct
        correct_percent_sq = (correct / float(len(self.encoded_train))) ** 2
        return correct_percent_sq

    def avg_precision(self, classifier, class_length, iterations):
        """
        Measure the precision of a classifier with shuffled datasets to get an
        average
        :param classifier: trained model
        :param class_length: length of the class attribute in an example
        :param iterations: number of precisions to calculate
        :return: averaged precision
        """
        acc = 0.0

        for i in xrange(iterations):
            self.load_input("credit-screening/crx.data", training_percent=0.85)
            acc += self.precision(classifier, self.encoded_test, class_length)

        return acc / iterations

if __name__ == '__main__':
    import cPickle
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', action='store', dest='input_filename',
                        required=True,
                        help='While training: file with training data | While'+\
                             ' testing: file with the saved classifier')
    parser.add_argument('--output', action='store', dest='output_filename', 
                        help='File to save the produced classifier')
    parser.add_argument('--action', action='store', dest='action',
                        required=True, help="'train' or 'test'")
    parser.add_argument('--count_rules', action='store', dest='count_rules',
                        type=int, default=15, 
                        help='Max number of rules on initial population')
    parser.add_argument('--iterations', action='store', dest='iterations',
                        type=int, default=1000,
                        help='Number of iterations to perform')
    parser.add_argument('--pop_count', action='store', dest='pop_count',
                        type=int, default=50,
                        help='Total number of individuals in the population')
    parser.add_argument('--mutate_prob', action='store', dest='mutate_prob',
                        type=float, default=0.01,
                        help='Probability in [0..1] to mutate an individual')
    parser.add_argument('--diversity_prob', action='store', type=float,
                        dest='diversity_prob', default=0.05,
                        help='Probability in [0..1] to save an ' + \
                             'unfitted individual')
    parser.add_argument('--retain_percent', action='store', type=float,
                        dest='retain_percent', default=0.2,
                        help='Probability in [0..1] that a parent will live '+\
                             'another generation (opposite of crossover rate)')
    parser.add_argument('--parents', action='store', dest='parents',
                        default='roullette',
                        help='Parents selection method: random, roullette')
    parser.add_argument('--survivors', action='store', dest='survivors',
                        default='truncated',
                        help='Survivors selection method: ' + \
                             'truncated, roullette')
    args = parser.parse_args()

    if args.action == 'train':
        go = GabilOptimizer(args.count_rules)
        go.__class__ = \
            type('Classifier', 
                 (eval('Parents%sSelectionMixin' % args.parents.capitalize()),
                  eval('Survivors%sSelectionMixin'%args.survivors.capitalize()),
                  GabilOptimizer), {})
        go.load_input(args.input_filename, training_percent=0.85)
        solution = go.runGA(iterations=args.iterations, 
                            pop_count=args.pop_count,
                            target=10000.0,
                            mutate_prob=args.mutate_prob,
                            retain_percent=args.retain_percent,
                            diversity_prob=args.diversity_prob,
                            reverse=True)
        classifier = solution['individual']
        if args.output_filename:
            output_file = open(args.output_filename, 'w')
            cPickle.dump(classifier, output_file)
        # go.find_optimal(iterations=400, pop_count=20, target=10000.0,
        #                mutate_prob=0.1, reverse=True)
        # print solution[1]
        print "fitness = %f" % solution['fitness']
        print "len = %d" % len(solution['individual'])

    elif args.action == 'test':
        go = GabilOptimizer(args.count_rules)
        go.load_input("credit-screening/crx.data", training_percent=0.85)
        input_file = open(args.input_filename, 'r')
        classifier = cPickle.load(input_file)
        print go.avg_precision(classifier, class_length=2, iterations=100)

    elif args.action == 'humanify':
        go = GabilOptimizer(args.count_rules)
        input_file = open(args.input_filename, 'r')
        classifier = cPickle.load(input_file)
        for rule in classifier:
            if rule[-1] == 1 or rule[-2] == 1:
                print go.toBooleanRule(rule)
        # print [go.toBooleanRule(rule)
        #       for rule in classifier if rule[-1] == 1 or rule[-2] == 1]

# TODO: flatten all rulesets from the beginning and use iterators to
# yield lists of 77 characters every time
