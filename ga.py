# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 20:26:37 2016

@author: Kazuki
"""

from operator import itemgetter
import numpy as np
import time
# calc time
start = time.time()
#==============================================================================
# set your seed
np.random.seed(71)
# GA parameter
POP = 100              #num of population
ELITE_RATE = 0.1      #rate of elite
GLENGTH = 10
P_MUTATE = 0.8         #prob of mutation. reciprocal of len(THRESHOLD) is better 
                       #total mutate are about POP*len(THRESHOLD)*P_MUTATE
P_CROSS = 0.9            #rate of cross
GENERATION = 10         #num of generations
SELECTION_METHOD = 1    #1 is tournament. so far, only tournament
TOURNAMENT_SIZE = 5	    #num of tournament. only effective in tournament


THRESHOLD = [None] * GLENGTH
THRESHOLD[0] =  {'min':0, 'max':10}   #
THRESHOLD[1] =  {'min':0, 'max':10}   #
THRESHOLD[2] =  {'min':0, 'max':10}   #
THRESHOLD[3] =  {'min':0, 'max':10}   #
THRESHOLD[4] =  {'min':0, 'max':10}   #
THRESHOLD[5] =  {'min':0, 'max':10}   #
THRESHOLD[6] =  {'min':0, 'max':10}   #
THRESHOLD[7] =  {'min':0, 'max':10}   #
THRESHOLD[8] =  {'min':0, 'max':10}   #
THRESHOLD[9] =  {'min':0, 'max':10}   #
#==============================================================================
class Pop:
    def __init__(self):
        self.max_f = 0.0
        self.avg_f = 0.0
        self.min_f = 0.0
        self.genes = [None] * POP
        self.mk_genes()
        
    def mk_genes(self):
        for i in range(POP):
            self.genes[i] = Gene()
            
    def kill_genes(self):
        # kill duplicated genes
        uniq_list = []
        for i in range(POP):
            while str(self.genes[i].gtype) in uniq_list:
                self.genes[i].mk_random_gtype()
            uniq_list.append(str(self.genes[i].gtype))
        
    def calc_f(self):
        tmp_fitness = []
        tmp_fitness_rank = []
        order_list = []
        ret_ptr = []
        avg = 0.0
        # get f
        for i in range(POP):
            if self.genes[i].f == 0:
                self.genes[i].get_f()
        # sort by f
        for i in range(POP):
            tmp_fitness.append(self.genes[i].f)
            avg += self.genes[i].f
        avg = avg/POP
        self.avg_f = avg
        for i, e in enumerate(tmp_fitness):
            tmp_fitness_rank.append((i,e))
        tmp_fitness_rank = sorted(tmp_fitness_rank, key=itemgetter(1), reverse=True)  #True is dec
        for i in range(POP):
            order_list.append(tmp_fitness_rank[i][0])
        for i in order_list:
            ret_ptr.append(self.genes[i])
        self.genes = ret_ptr
        self.max_f = self.genes[0].f
        self.min_f = self.genes[POP-1].f
        
    def print_f(self):
        for i in range(POP):
            print self.genes[i].f,self.genes[i].gtype
        
    def generate_population(self):
        num_of_elite = int(POP*ELITE_RATE) # define elite
        generated = num_of_elite
        f_list = []
        for i in range(POP):
            f_list.append([i,self.genes[i].f,self.genes[i].gtype])#[i,f,gtype]
        self.f_list = f_list
        # if num of remains is odd num, generate one gene by mutation
        if( (POP - generated)%2 == 1):
            self.genes[generated].mutate()
            self.genes[generated].f = 0.0
            generated += 1
        # cross or mutate
        while (generated < POP):
            # cross
            if(np.random.uniform() < P_CROSS):
                self.cross_gene(generated)
                generated += 2
            # mutate
            else:
                #Mutant 1
                self.genes[generated].mutate()
                self.genes[generated].f = 0.0
                generated += 1
                #Mutant 2
                self.genes[generated].mutate()
                self.genes[generated].f = 0.0
                generated += 1
        
    def cross_gene(self,generated):
        if SELECTION_METHOD == 1:
            parent1 = self.select_parent_tournament()
            parent2 = self.select_parent_tournament()
        else:
            raise Exception("invalid number on SELECTION_METHOD\n")
        self.cross_gtype(parent1,parent2,generated)
        
    def select_parent_tournament(self):
        max_selected = self.f_list[np.random.randint(0, POP)]
        for i in range(TOURNAMENT_SIZE):
            tmp_parent = self.f_list[np.random.randint(0, POP)]
            if max_selected[1] < tmp_parent[1]:
                max_selected = tmp_parent
        return max_selected
    
    def cross_gtype(self,p1,p2,generated):
        cross_point = np.random.randint(1,GLENGTH) #1~GLENGTH-1
        self.genes[generated].gtype = p1[2][:cross_point]+p2[2][cross_point:]
        self.genes[generated].f = 0.0
        self.genes[generated+1].gtype = p2[2][:cross_point]+p1[2][cross_point:]
        self.genes[generated+1].f = 0.0
    
class Gene:
    def __init__(self):
        self.gtype = [0]*GLENGTH
        self.f = 0.0
        self.mk_random_gtype()
        
    def mk_random_gtype(self):
        for i in range(GLENGTH):
            self.gtype[i] = np.random.randint(THRESHOLD[i]['min'], 
            THRESHOLD[i]['max']+1)
    
    def get_f(self):
        self.f = sum(self.gtype)
        
    def mutate(self):
        for i in range(GLENGTH):
            if np.random.uniform() < P_MUTATE:
                self.gtype[i] = np.random.randint(THRESHOLD[i]['min'], 
                THRESHOLD[i]['max']+1)
#==============================================================================
# initialize
pop = Pop()

for i in range(GENERATION):
    pop.kill_genes()
    pop.calc_f()
    pop.print_f()
    pop.generate_population()
    
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time))
    print (elapsed_time/60), "min"
    
# main
if __name__ == "__main__":

    print pop.genes[0].gtype
