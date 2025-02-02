import math
import operator
import random
import itertools

import numpy as np
from sklearn.linear_model import LinearRegression
from deap import base, creator, tools

def get_grid_searcher(size):
    max_weight = size
    granularity = 40
    step = max_weight / granularity 

    def grid_search(evaluator):
        best_weight = []
        best_accs = None
        all_results = list()
        for combination in itertools.product(range(granularity + 1), repeat=size):
            if sum(combination) == granularity:
                weight = [x * step for x in combination]
                accs = evaluator(weight)
                all_results.append((weight, accs[0]))
                if not best_accs or accs > best_accs:
                    best_accs = accs
                    best_weight = weight
                print(weight, accs[0])
        print(f'{best_weight} achieved accuracies of {best_accs}')
        return best_weight, '', all_results
    return grid_search

def linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return list(model.coef_), ''

def create_stats_and_logbook():
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    return stats, logbook

class DEOptimizer():
    def __init__(self, size):
        self.POPSIZE, self.NUMGEN, self.CXPB, self.DW = 40, 30, 0.8, 1.5
        creator.create("WeightedFitness", base.Fitness, weights=(1.0, 0.00001))
        creator.create("Agent", list, fitness=creator.WeightedFitness)

        self.toolbox = base.Toolbox()
        self.toolbox.register("SingleWeight", random.random)
        self.toolbox.register("individual", tools.initRepeat, creator.Agent, self.toolbox.SingleWeight, size)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

    def updateAgent(self, agent_index, pop):
        agent = pop[agent_index]
        size = len(agent)
        original_agent = self.toolbox.clone(agent)
        a, b, c = random.sample(pop[:agent_index] + pop[agent_index + 1:], 3)
        reference_agent = [a[i] + self.DW * (b[i] - c[i]) for i in range(size)]
        R = random.randint(0, size - 1)
        for i in range(size):
            if i == R or random.random() < self.CXPB:
                agent[i] = max(reference_agent[i], 0.001)
        new_fitness = self.toolbox.evaluate(agent)
        if tuple(agent.fitness.values) > new_fitness:
            agent[:] = original_agent[:]
        else:
            agent.fitness.values = new_fitness
        
    def run_de(self, evaluator):
        self.toolbox.register("evaluate", evaluator)
        pop = self.toolbox.population(n=self.POPSIZE)
        for ind in pop:
            ind.fitness.values = self.toolbox.evaluate(ind)

        stats, logbook = create_stats_and_logbook()
        best = None
        best_over_time = list()

        for g in range(self.NUMGEN):
            for i in range(self.POPSIZE):
                self.updateAgent(i, pop)
            current_best = tools.selBest(pop, 1)[0]
            best_over_time.append(current_best)
            if not best or tuple(best.fitness.values) < tuple(current_best.fitness.values):
                best = current_best
            logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
            print(logbook.stream)

        log = f'Population: {self.POPSIZE}\tGenerations: {self.NUMGEN}\tCrossover Prob: {self.CXPB}\tDifferential Weight: {self.DW}\n' + str(logbook)
        return best, log, best_over_time

def get_de_optimizer(size):
    optimizer = DEOptimizer(size)
    return optimizer.run_de
