# **Pygen**
## **A Python Framework implementing the Genetic Algorithm**

The main class, **GeneticOptimizer**, should be specialized by overriding some 
methods:

- **_ _init_ _(self)** 
    - Add parameters to your instances that may be needed for calculations
- **individual(self)**
    - Define a representation for a single solution
- **fitness(self, x)**
    - Penalize an individual x, according to some rules. High fitness means
    it is a worse solution
- **mutate(self, mutate_prob, parents)**
    - Change an individual in some way to allow it to explore new conditions
- **mix(self, parent1, parent2)**
    - Define how two individuals produce a new one

After that, normal usage would go like this:
```python
MyOptimizer(n=12, p=2).runGA(iterations=500,
                             pop_count=100,
                             target=0.0,
                             mutate_prob=0.1,
                             retain=0.2,
                             diversity_prob=0.05)
```
This will output a tuple (rank, sol), where rank is the fitness of the best
found solution and sol is that solution (an array of chromosomes).
The framework will handle all the process involving evolution, promoting
genetic diversity, working on new children, etc.