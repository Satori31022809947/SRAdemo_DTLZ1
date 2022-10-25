from cgitb import reset
import numpy as np
import math

class problem():
    def __init__(self, M=3,K=5):
        self.M=M
        self.name = 'DTLZ1'
        self.Dim = M + K
        self.varTypes = np.array([0] * self.Dim) 
        self.lb = [0] * self.Dim
        self.ub = [1] * self.Dim
        self.lbin = [1] * self.Dim
        self.ubin = [1] * self.Dim

    def calc_f_individual(self, Vars):
        # target function
        XM = Vars[:, (self.M - 1):]
        g = 100 * (self.Dim - self.M + 1 + np.sum(((XM - 0.5) ** 2 - np.cos(20 * np.pi * (XM - 0.5))), 1,
                                                  keepdims=True))
        ones_metrix = np.ones((Vars.shape[0], 1))
        f = 0.5 * np.hstack([np.fliplr(np.cumprod(Vars[:, :self.M - 1], 1)), ones_metrix]) * np.hstack(
            [ones_metrix, 1 - Vars[:, range(self.M - 2, -1, -1)]]) * (1 + g)
        return f
    
    def init_population(self,N):
        res=np.random.rand(N,self.Dim)
        return res
    

def calc(population,N):
    SDE_Mat = np.zeros([N, N])
    IBEA_Mat = np.zeros([N, N])

    for i in range(N):
        for j in range(N):
            temp = np.max(np.vstack((population[i, :], population[j, :])), axis=0)
            if i == j:
                SDE_Mat[i, j] = np.inf
            else:
                SDE_Mat[i, j] = np.linalg.norm(population[i, :] - temp, ord=2, axis=0)
                IBEA_Mat[i, j] = np.max((population[i, :] - population[j, :]))
    IBEA_Mat=-IBEA_Mat / 0.05
    sdefitness = np.min(SDE_Mat, axis=1)
    ibeafitness = np.sum(-np.exp(IBEA_Mat), axis=0) + 1
    return sdefitness,ibeafitness

def random_sort(popprop,N,pc=0.5):
    I1,I2=calc(popprop,N)
    res=[]
    for i in range(N):
        res.append(i)
    res=np.array(res)
    
    for sweepCounter in range(N):
        flag=False
        for j in range(0,N-1):
            u=np.random.rand()
            if u<pc:
                if (I1[res[j]]<I1[res[j+1]]):
                    t=res[j]
                    res[j]=res[j+1]
                    res[j+1]=t
                    flag=True
            else :
                if (I2[res[j]]<I2[res[j+1]]):
                    t=res[j]
                    res[j]=res[j+1]
                    res[j+1]=t
                    flag=True
        if (not flag):
            break
    return res

def calc_f_population(problem,population):
    return problem.calc_f_individual(population)

def reproduce(p,q):
    t=np.random.random()
    res_=p*t+q*(1-t)
    res=[]
    id=math.floor(np.random.random()*p.shape[0])
    for i in res_:
        if (id==0):
            i+=np.random.normal()*0.3
            if (i>1):
                i=1
            if (i<0):
                i=0
        id-=1
        res.append(i)
    res=[res]
    return np.array(res)

def main_loop(problem,MaxGen,N):
    population=problem.init_population(N)
    popprop=calc_f_population(problem,population)
    t=0
    while t<MaxGen:
        for i in range(N):
            # generate new population
            x,y=math.floor(np.random.rand()*N),math.floor(np.random.rand()*N)
            population=np.append(population,reproduce(population[x],population[y]),axis=0)
        #calculate function in problem
        popprop=calc_f_population(problem,population)
        #random sort
        res=random_sort(popprop,2*N)
        #select top N
        n_population=[]
        for i in range(N):
            n_population.append(population[res[i]])
        population=np.array(n_population)
        
        #for debug
        if t%500==0:
            popprop=calc_f_population(problem,population)
            print(popprop)
        t=t+1
    return population

problem=problem();
res=main_loop(problem,3000,50)
popprop=calc_f_population(problem,res)
print(popprop)