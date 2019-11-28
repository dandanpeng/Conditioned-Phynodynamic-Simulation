#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 11:40:44 2018

@author: pengdandan
"""
import random
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from scipy.special import comb
import sys  
sys.setrecursionlimit(1000000)

#########----Gillespie-----#############
class Reaction:
    def __init__(self,rate,left_num,right_num):
        self.rate = rate
        #assert len(left_num) == len(right_num)
        self.left_num = np.array(left_num) #number of reactants before reaction
        self.right_num = np.array(right_num) #number of reactants after reaction
        self.state_change = self.right_num - self.left_num
    def combine(self,n,s):
        return np.prod(comb(n,s))
    def propensity(self,n):
        return self.rate * self.combine(n,self.left_num)

class System:
    def __init__(self,num_elements):
        assert num_elements > 0
        self.num_elements = num_elements #number of reactant species
        self.reactions = [] #set of reactions
    def add_reaction(self,rate,left_num,right_num):
        #assert len(left_num) == self.num_elements
        #assert len(right_num) == self.num_elements
        self.reactions.append(Reaction(rate,left_num,right_num))
    def evolve(self,sampling_time,inits):  #evolution simulation
        self.t = [0] #time track
        self.n = [np.array(inits)] #number of every reactant
        self.birth_or_death = [] #reaction track
        while True:
            A = np.array([rec.propensity(self.n[-1]) for rec in self.reactions]) 
            A0 = A.sum()
            A /= A0
            mu = np.random.rand(1)
            while mu == 0:
                mu = np.random.rand(1)
            t0 = -np.log(mu)/A0
            if self.t[-1]+t0 > sampling_time:
                self.t.append(sampling_time)
                self.log_weight = np.log(0.3) - 0.3 * sum(np.diff(self.t) * np.asarray([i[0] for i in self.n]))
                break
            else:
                self.t.append((self.t[-1]+t0)[0])
                d = np.random.choice(self.reactions,p=A)
                #self.log_weight -= 0.1 * t0 * self.n[-1]
                self.birth_or_death.append(d.state_change[0])
                self.n.append(self.n[-1] + d.state_change)
       
num_elements = 1
system = System(num_elements)

system.add_reaction(1.0,[1],[2])
system.add_reaction(0.01,[1],[0])
system.evolve(5,[1])
x = system.t
y = system.n
y.append(y[-1])
#system.birth_or_death
#list(np.asarray(system.t[1:]) - np.asarray(system.t[:-1]))
#plt.clf()
plt.plot(x,y)
#plt.plot(x,y2,'r--') 
#plt.xlim(0, x[-1]+1)


#########----BirthDeathTree-----#############
class Node:
    def __init__(self,parent = 'null',children = [],time = 0,state = 'alive'):
        self.parent = parent
        self.children = children
        self.time = time
        self.state = state
    def __str__(self):
        if self.children != []:
            return ' (%(child1)s, %(child2)s):%(selftime)f' % {"child1":self.children[0], "child2":self.children[1], "selftime":self.time} 
        if self.state == 'sample' and self.children == []:
            return 'sample:%(time)f' %{"time": self.time}
            #return ':%(time)f' %{"time": self.time}
        else:
            return ':%(time)f' %{"time": self.time}
        
class Population:
    def __init__(self,root = Node('null',[],0)):
        self.set = [root]
        self.root = root
    def addchild(self,Node1,Node2):
        self.set.append(Node1)
        self.set.append(Node2)
    def delnode(self,Node):
         self.set.remove(Node)
    def grow(self,interval):
        for i in self.set:
            i.time += interval
    def propagate(self,birthdeath,timepoint):
         intervals = list(np.diff(np.asarray(timepoint)))
         if birthdeath != []:
             for i in range(len(birthdeath)):
                  if birthdeath[i] == 1:
                      self.grow(intervals[i])
                      parent = random.choice(self.set)
                      child1 = Node(parent,[],0,'alive')
                      child2 = Node(parent,[],0,'alive')                 
                      parent.children = [child1,child2]
                      self.delnode(parent)                 
                      self.addchild(child1,child2)
                  else:
                      self.grow(intervals[i])
                      death = random.choice(self.set)
                      death.state = 'dead'
                      self.delnode(death)                
         else:
                self = self
         return(self) 
    def sampling(self,timepoint):
        if self.set != []:
            intervals = list(np.diff(np.asarray(timepoint)))
            self.grow(intervals[-1])
            sample = random.choice(self.set)
            sample.state = 'sample'
            self.delnode(sample)
        else:
            self.set = self.set
        return(self)


#########----Particle Filtering-----#############
nb_particles = 1000

tree = []
for i in range(nb_particles):
    tree.append(copy.deepcopy(Population())) #initialize particles
log_weight = np.repeat(0.0,nb_particles) #initialize weight

sampling_time = [1,3,5]
sampling_time.insert(0,0)
sampling_time_interval = list(np.diff(sampling_time))

def hist(indice):
    labels, values = zip(*Counter(indice).items())
    indexes = np.arange(len(labels))
    width = 1

    plt.bar(indexes, values, width)
    plt.xticks(indexes + width * 0.5, labels)
    plt.show()

for j in sampling_time_interval:
    for nb in range(nb_particles):
        system = System(1)
        system.add_reaction(1.0,[1],[2])
        system.add_reaction(0.2,[1],[0])
        system.evolve(j,[len(tree[nb].set)])
        tree[nb] = tree[nb].propagate(system.birth_or_death,system.t)
        if len(tree[nb].set) > 0:  
            tree[nb] = tree[nb].sampling(system.t)
            log_weight[nb] = system.log_weight
        else:
            log_weight[nb] = float('-inf')
    ### normalize and scale ####
    weight = log_weight - max(log_weight)
    weight = list(np.exp(weight)/sum(np.exp(weight)))
    ## resampling ###
    indice = np.random.choice(range(nb_particles),nb_particles,p = weight)
    tree = [copy.deepcopy(tree[i]) for i in indice]
    #hist(indice)

#########----Comparison with MCMC-----#############
f = open('test1.nex','w')
f.write('#NEXUS\n')
f.write('BEGIN TREES;\n')
for i in range(nb_particles):
        f.write('     TREE tree'+str(i) + '='+str(tree[i].root)+';\n')
f.write('END;')
f.close()

treestat = pd.read_table('final.output')

tree_length = treestat.iloc[:,1]
tree_height = treestat.iloc[:,2]

plt.hist(tree_length,bins = 50)
plt.hist(tree_height,bins = 50)

mcmc_stat = pd.read_table('mcmc_output')
mcmc_length = mcmc_stat.iloc[:,1]
mcmc_height = mcmc_stat.iloc[:,2]

plt.hist(mcmc_length,bins = 50)
plt.hist(mcmc_height,bins = 50)

