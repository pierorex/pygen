usage: gabil_optimizer.py [-h] --input INPUT_FILENAME
                          [--output OUTPUT_FILENAME] --action ACTION
                          [--count_rules COUNT_RULES]
                          [--iterations ITERATIONS] [--pop_count POP_COUNT]
                          [--mutate_prob MUTATE_PROB]
                          [--diversity_prob DIVERSITY_PROB]
                          [--retain_percent RETAIN_PERCENT]
                          [--parents PARENTS] [--survivors SURVIVORS]

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT_FILENAME
                        While training: file with training data | While
                        testing: file with the saved classifier
  --output OUTPUT_FILENAME
                        File to save the produced classifier
  --action ACTION       train or test
  --count_rules COUNT_RULES
                        Max number of rules on initial population
  --iterations ITERATIONS
                        Number of iterations to perform
  --pop_count POP_COUNT
                        Total number of individuals in the population
  --mutate_prob MUTATE_PROB
                        Probability in [0..1] to mutate an individual
  --diversity_prob DIVERSITY_PROB
                        Probability in [0..1] to save an unfitted individual
  --retain_percent RETAIN_PERCENT
                        Probability in [0..1] that a parent will live another
                        generation (opposite of crossover rate)
  --parents PARENTS     Parents selection method: random, roullette
  --survivors SURVIVORS
                        Survivors selection method: truncated, roullette

Example to train:
pypy gabil_optimizer.py --action train --pop_count 10 --mutate_prob 0.2 --iterations 100 --input credit-screening/crx.data --output classifier

Example to test:
pypy gabil_optimizer.py --action test --input classifier
