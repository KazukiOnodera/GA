# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 20:26:37 2016

@author: Kazuki
"""

from operator import itemgetter
import numpy as np
import time
from joblib import Parallel, delayed

#==============================================================================
# set your seed
#np.random.seed(71)

#==============================================================================

def get_fitness1(gtype):
    return sum(gtype)

def get_fitness2(gtype):
    return sum([g%3 for g in gtype])

def get_fitness3(gtype):
    return sum([-g for g in gtype])

class GA:
    def __init__(self, threshold, feval=get_fitness1, init_gtype=None, random=True,
                 population=100, e_rate=0.1,  p_mutate=0.1, step_rate=0.1,
                 p_cross=0.9, generation=10, selection=1, tournament_size=5,
                 is_print=True, maximize=True, seed=0, 
                 n_jobs=1):
        """
        parameters:
        -----------
        threshold : list, gene range
        population       : int, population
        e_rate    : float(0~1), elite rate
        p_mutate  : float(0~1), mutation rate. reciprocal of len(THRESHOLD) is better
        p_cross   : float(0~1), cross rate
        
        """
        self.start = time.time()
        self.threshold = threshold
        self.population = population
        self.e_rate = e_rate
        self.glength = len(threshold)
        self.p_mutate = p_mutate
        self.p_cross = p_cross
        self.generation = generation
        self.selection = selection
        self.tournament_size = tournament_size
        self.feval = feval
        self.is_print = is_print
        self.maximize = maximize
        self.init_gtype = init_gtype
        self.random = random
        self.step_rate = step_rate
        self.n_jobs = n_jobs
        self.max_f = 0.0
        self.avg_f = 0.0
        self.min_f = 0.0
        self.genes = [None] * self.population
        self._mk_genes_()
        np.random.seed(seed)
        
    def __getitem__(self, index):
        return self.genes[index]
        
    def _mk_genes_(self):
        for i in range(self.population):
            self.genes[i] = Genes(self.glength, self.threshold, self.init_gtype)
            
    def _kill_genes_(self):
        # kill duplicated genes
        uniq_list = []
        for i in range(self.population):
            while str(self[i].gtype) in uniq_list:
                if self.random:
                    self[i].mk_random_gtype(self.threshold)
                else:
                    self[i].mk_random_gtype(self.threshold)
#                    self[i].mk_step_gtype(self.step_rate)
            uniq_list.append(str(self[i].gtype))
        
    def multi(self, p):
        if self[p].fitness == 0:
            return self.feval(self[p].gtype)
        else:
             return self[p].fitness
    
    def _calc_f_(self):
        tmp_fitness = []
        tmp_fitness_rank = []
        order_list = []
        ret_ptr = []
        avg = 0.0
        # get f
        if self.n_jobs!=1:
            callback = Parallel(n_jobs=self.n_jobs)( [delayed(self.multi)(i) for i in range(self.population)] )
            for i in range(self.population):
                self[i].fitness = callback[i]
        else:
            for i in range(self.population):
                if self[i].fitness == 0:
                    self[i].fitness = self.feval(self[i].gtype)
        # sort by f
        for i in range(self.population):
            tmp_fitness.append(self[i].fitness)
            avg += self[i].fitness
        avg = avg/self.population
        self.avg_f = avg
        for i, e in enumerate(tmp_fitness):
            tmp_fitness_rank.append((i,e))
        tmp_fitness_rank = sorted(tmp_fitness_rank, key=itemgetter(1), 
                                  reverse=self.maximize)  #True is descending
        for i in range(self.population):
            order_list.append(tmp_fitness_rank[i][0])
        for i in order_list:
            ret_ptr.append(self[i])
        self.genes = ret_ptr
        self.max_f = self[0].fitness
        self.min_f = self[self.population-1].fitness
        
    def _print_f_(self):
        for i in range(self.population):
            print(self[i].fitness,self[i].gtype)
        
    def _generate_population_(self):
        num_of_elite = int(self.population*self.e_rate) # define elite
        generated = num_of_elite
        f_list = []
        for i in range(self.population):
            f_list.append([i,self[i].fitness,self[i].gtype])#[i,f,gtype]
        self.f_list = f_list
        # if num of remains is odd num, generate one gene by mutation
        if( (self.population - generated)%2 == 1):
            self[generated].mutate(self.p_mutate, self.threshold)
            generated += 1
        # cross or mutate
        while (generated < self.population):
            # cross
            if(np.random.uniform() < self.p_cross):
                self._cross_gene_(generated)
                generated += 2
            # mutate
            else:
                #Mutant 1
                self[generated].mutate(self.p_mutate, self.threshold)
                generated += 1
                #Mutant 2
                self[generated].mutate(self.p_mutate, self.threshold)
                generated += 1
        
    def _cross_gene_(self, generated):
        if self.selection == 1:
            parent1 = self._select_parent_tournament_()
            parent2 = self._select_parent_tournament_()
        else:
            raise Exception("invalid number on SELECTION_METHOD\n")
        self._cross_gtype_(parent1, parent2, generated)
        
    def _select_parent_tournament_(self):
        max_selected = self.f_list[np.random.randint(0, self.population)]
        for i in range(self.tournament_size):
            tmp_parent = self.f_list[np.random.randint(0, self.population)]
            if self.maximize is True:
                if max_selected[1] < tmp_parent[1]:
                    max_selected = tmp_parent
                elif max_selected[1] > tmp_parent[1]:
                    max_selected = tmp_parent
        return max_selected
    
    def _cross_gtype_(self, p1, p2, generated):
        cross_point = np.random.randint(1, self.glength) #1~GLENGTH-1
        self[generated].gtype = p1[2][:cross_point]+p2[2][cross_point:]
        self[generated].mutate(self.p_mutate, self.threshold)
        self[generated+1].gtype = p2[2][:cross_point]+p1[2][cross_point:]
        self[generated+1].mutate(self.p_mutate, self.threshold)
    
    def fit(self):
        for i in range(self.generation):
            self._kill_genes_()
            self._calc_f_()
            
            if self.is_print:
                self._print_f_()
            
            if i != self.generation-1:
                self._generate_population_()
            
            elapsed_time = time.time() - self.start
            print('best fitness:{0}  elapsed_time:{1}'.format(self.max_f, elapsed_time))
        return
        
class Genes:
    def __init__(self, glength, threshold, init_gtype=None):
        self.fitness = 0.0
        if init_gtype is None:
            self.gtype = [0]*glength
            self.mk_random_gtype(threshold)
        else:
            self.gtype = list(init_gtype)
        
    def __len__(self):
        return len(self.gtype)
    
    def __getitem__(self, key):
        return self.gtype[key]
    
    def mk_random_gtype(self, threshold):
        for i in range(len(self.gtype)):
            if threshold[i]['type']==int:
                self.gtype[i] = np.random.randint(threshold[i]['min'], threshold[i]['max']+1)
            elif threshold[i]['type']==float:
                self.gtype[i] = np.random.uniform(threshold[i]['min'], threshold[i]['max'])
            else:
                raise Exception('Invalid type', threshold[i]['type'])
    
#    def mk_step_gtype(self, step_rate):
#        for i in range(len(self.gtype)):
#            self.gtype[i] = self.gtype[i] * (1+np.random.uniform(-step_rate, step_rate))
        
    def mutate(self, p_mutate, threshold):
        for i in range(len(self.gtype)):
            if np.random.uniform() < p_mutate:
                if threshold[i]['type']==int:
                    self.gtype[i] = np.random.randint(threshold[i]['min'], threshold[i]['max']+1)
                elif threshold[i]['type']==float:
                    self.gtype[i] = np.random.uniform(threshold[i]['min'], threshold[i]['max'])
        self.fitness = 0.0

#==============================================================================
# main
if __name__ == "__main__":
    # initialize
    THRESHOLD = [None] * 10
    THRESHOLD[0] =  {'min':0, 'max':10, 'type':int}   #
    THRESHOLD[1] =  {'min':0, 'max':10, 'type':int}   #
    THRESHOLD[2] =  {'min':0, 'max':10, 'type':int}   #
    THRESHOLD[3] =  {'min':0, 'max':10, 'type':int}   #
    THRESHOLD[4] =  {'min':0, 'max':10, 'type':int}   #
    THRESHOLD[5] =  {'min':0, 'max':10, 'type':int}   #
    THRESHOLD[6] =  {'min':0, 'max':10, 'type':int}   #
    THRESHOLD[7] =  {'min':0, 'max':10, 'type':int}   #
    THRESHOLD[8] =  {'min':0, 'max':10, 'type':int}   #
    THRESHOLD[9] =  {'min':0, 'max':10, 'type':int}   #
    
    ga = GA(THRESHOLD, generation=20, maximize=True, is_print=False, n_jobs=-1)
    ga.fit()
    
    print(ga.genes[0].gtype)
    
    
