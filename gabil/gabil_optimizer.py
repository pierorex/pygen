if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from geneticoptimizer import GeneticOptimizer

import csv
from random import random, randint, shuffle
from timeit import timeit

# revisar
# - darle penalización en el fitness a las soluciones que tengan muchos 1's (son muy generales)
# - penalizar largo del arreglo (cantidad de reglas; favorecer conjuntos de reglas mas pequeños de reglas)
# - una solución no debe tener c1&c2 => p y c1&c2 => -p como reglas


# match(regla, ej):
#   cada '1' del ejemplo es '1' en la regla

class GabilOptimizer(GeneticOptimizer):
    def __init__(self, count_rules):
        self.count_rules = count_rules
        self.examples = []
        self.encoded_examples = []
        self.classes = [
            {'a':0, 'b':1},
            {},
            {},
            {'u':0, 'y':1, 'l':2, 't':3},
            {'g':0, 'p':1, 'gg':2},
            {'c':0, 'd':1, 'cc':2, 'i':3, 'j':4, 'k':5, 'm':6, 'r':7,
             'q':8, 'w':9, 'x':10, 'e':11, 'aa':12, 'ff':13},
            {'v':0, 'h':1, 'bb':2, 'j':3, 'n':4, 'z':5, 'dd':6, 'ff':7, 'o':8},
            {},
            {'t':0, 'f':1},
            {'t':0, 'f':1},
            {},
            {'t':0, 'f':1},
            {'g':0, 'p':1, 's':2},
            {},
            {},
            {'+':0, '-':1}
        ]

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

    def encode(self, example):

        return 

    def decode(self, bitstring):
        return boolean_rules

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
                    mini[i], maxi[i] = min(mini[i], float(e[i])), max(maxi[i], float(e[i]))
                except: pass

        return (mini, maxi)

    def read_input(self, file_path):
        input_file = open(file_path, "r")
        csvreader = csv.reader(input_file)
        self.examples = [line for line in csvreader]
        self.encoded_examples = [self.encode(example) for example in self.examples]


if __name__ == '__main__':
    go = GabilOptimizer(3)
    go.read_input("gabil/credit-screening/crx.data")
    mini, maxi = go.get_continuous_minmax()
    print mini
    print maxi
    # print go.encoded_examples
