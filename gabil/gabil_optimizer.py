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
# - penalizar largo del arreglo (cantidad de reglas; favorecer conjuntos de reglas mas pequenos de reglas)
# - una solución no debe tener c1&c2 => p y c1&c2 => -p como reglas


# match(regla, ej):
#   cada '1' del ejemplo es '1' en la regla

class GabilOptimizer(GeneticOptimizer):
    def __init__(self, count_rules):
        self.count_rules = count_rules
        self.examples = []
        self.encoded_examples = []
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
            encoded += [1 if decoded[i] == class_k else 0 for class_k in self.classes[i]]

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
        """
        l = range(self.count_rules)
        return l

    def fitness(self, solution): # porcentaje correctos al cuadrado
        def count_correct(rules_list, examples_list):
            return sum(1 for rule in rules_list 
                         for example in encoded_examples
                         if example == rule)
            # correctos(hip, ejms):
            # for e in ejms:
            #   for regla in hip:
            #       if match(regla, e):  # probar: si hace match pero la clase está errada => correctos--
            #           correctos++ 
            #           break

        return count_correct(solution, self.encoded_examples)

    def mutate(self, mutate_prob, parents):
        pass

    def mix(self, parent1, parent2):
        pass

    def get_continuous_minmax(self):
        """
        Calculate min and max of columns with continuous values, this will help
        us construct discrete classes for these columns
        """
        mini = {1:1000000,2:1000000,7:1000000,10:1000000,13:1000000,14:1000000} 
        maxi = {1:-1000000,2:-1000000,7:-1000000,10:-1000000,13:-1000000,14:-1000000}

        for e in self.examples:
            for i in [1,2,7,10,13,14]:
                try:
                    mini[i] = min(mini[i], float(e[i]))
                    maxi[i] = max(maxi[i], float(e[i]))
                except: pass

        return (mini, maxi)

    def load_input(self, file_path):
        input_file = open(file_path, "r")
        csvreader = csv.reader(input_file)
        self.examples = [line for line in csvreader]
        self.curated_examples = [self.remove_na(e) for e in self.examples]
        self.discretized_examples = [self.discretize_example(e) 
                                     for e in self.curated_examples]
        self.encoded_examples = [self.encode(e) for e in self.curated_examples]
        self.decoded_examples = [self.decode(e) for e in self.encoded_examples]
        assert self.curated_examples == self.decoded_examples


if __name__ == '__main__':
    go = GabilOptimizer(3)
    go.load_input("credit-screening/crx.data")
    # print go.encoded_examples