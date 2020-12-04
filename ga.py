import numpy as np 

class GeneticAlgorithm: 

   
    def fitness_value(self,solution : np.ndarray):  
        return self.fitness_func(solution)


    def __init__(self,solutions : np.ndarray ,num_parents_for_mating=2,generations=10000,fitness_func=None):
        self.solutions_per_population = solutions.shape[0] 
        self.genes_per_solution = solutions.shape[1]
        self.generations = generations 
        self.num_parents_for_mating = num_parents_for_mating  
        self.fitness_vls = np.zeros(shape=self.solutions_per_population)
        self.solutions = solutions 
        self.fitness_func = fitness_func
    
    def sortByBestFitness(self): 
        for i  in np.arange(self.fitness_vls.shape[0]):
            for j in np.arange(self.fitness_vls.shape[0]):
                a = self.fitness_vls[i]
                b = self.fitness_vls[j]
                if a > b:
                   self.fitness_vls[i] = b
                   self.fitness_vls[j] = a 
                   sol_tmp = self.solutions[i].copy()
                   self.solutions[i] = self.solutions[j].copy()
                   self.solutions[j] = sol_tmp
    def crossover(self,parents,offspring_shape):      
       offspring = np.empty(offspring_shape)
       middle_idx = offspring_shape[1]//2
       for i in np.arange(offspring.shape[0]):
          parent1_idx = i % parents.shape[0]
          parent2_idx = (i + 1) % parents.shape[0]
          offspring[i,0:middle_idx] = parents[parent1_idx,0:middle_idx]
          offspring[i,middle_idx:] = parents[parent2_idx,middle_idx:]
       return offspring
    def mutation(self,offspring):
        for i in np.arange(offspring.shape[0]): 
            mut_idx = np.random.randint(low=0,high=offspring.shape[1])
            offspring[i][mut_idx] += np.random.uniform(-1,1,1)[0] 
    def update_fitness_vls(self):
        for j in np.arange(self.solutions.shape[0]):  
            sol = self.solutions[j]
            self.fitness_vls[j] = self.fitness_value(sol)         
    
    def try_update_solutions(self,offspring):
        for i in np.arange(offspring.shape[0]):
            sol = offspring[i]
            fitness_vl = self.fitness_value(sol)
            for j in np.arange(self.fitness_vls.shape[0]):
                if fitness_vl > self.fitness_vls[j]:
                    self.solutions[j] = sol
                    self.fitness_vls[j] = fitness_vl
                    break
    def start(self):
        i = 0 
        while (i < self.generations): 
           self.update_fitness_vls() 
           self.sortByBestFitness()
           parents_for_mating = self.solutions[0:self.num_parents_for_mating] #best individuals
           offspring = self.crossover(parents_for_mating, parents_for_mating.shape)
           self.mutation(offspring)
           self.try_update_solutions(offspring)
           i+=1 

if __name__ == '__main__': 
    initial_solutions = np.random.uniform(low=-4.0,high=4.0,size=(8,4))
    func = lambda sol: ((5 * sol[0] + 5 * sol[1] + 5 * sol[2]) - sol[3])
    ga = GeneticAlgorithm(
        initial_solutions, 
        num_parents_for_mating=2,
        generations=1000,
        fitness_func=func 
    )
    print('initial_solutions')
    print(initial_solutions)
    print('....')        
    ga.start()
    print(ga.solutions)
    print('...')
    print(ga.fitness_vls)
    print('____')
    print( ga.fitness_value( ga.solutions[0]))