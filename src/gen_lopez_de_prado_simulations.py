import pandas as pd
import numpy as np
import random
import os

INPUTS_PATH = os.path.join(os.getcwd(), "src", "data", "inputs")

def generateMonteCarloData(nObs, sLength, size0, size1, mu0, sigma0, sigma1F):
    # Time series of correlated variables
    #1) generate random uncorrelated data
    x = np.random.normal(mu0,sigma0,size=(nObs,size0)) # each row is a variable
    
    #2) create correlation between the variables
    cols = [random.randint(0,size0-1) for i in range(size1)]
    y = x[:,cols]+np.random.normal(0,sigma0*sigma1F,size=(nObs,len(cols)))
    x = np.append(x,y,axis=1)
    
    #3) add common random shock
    point = np.random.randint(sLength,nObs-1,size=2)
    x[np.ix_(point,[cols[0],size0])] = np.array([[-.5,-.5],[2,2]])

    #4) add specific random shock
    point = np.random.randint(sLength,nObs-1,size=2)
    x[point,cols[-1]] = np.array([-.5,2])
    return x, cols

def generateData(nObs, size0, size1, sigma1):
    # Time series of correlated variables
    #1) generating some uncorrelated data
    np.random.seed(seed=12345);random.seed(12345)
    x = np.random.normal(0,1,size=(nObs,size0)) # each row is a variable
    
    #2) creating correlation between the variables
    cols = [random.randint(0,size0-1) for i in range(size1)]
    y = x[:,cols]+np.random.normal(0,sigma1,size=(nObs,len(cols)))
    x = np.append(x,y,axis=1)
    x = pd.DataFrame(x,columns=range(1,x.shape[1]+1))

    return x, cols

if __name__ == "__main__":

    # 1) Generate correlated data
    nObs, size0, size1, sigma1 = 10000, 5, 5, .25
    x, cols = generateData(nObs,size0,size1,sigma1)
    
    x_df = pd.DataFrame(x)
    x_df.to_excel(os.path.join(INPUTS_PATH, "simulation-lopez-de-prado.xlsx"), index=False)

    # # 2) Generate Monte Carlo data
    # nObs, sLength, size0, size1, mu0, sigma0, sigma1F = 520, 260, 5, 5, 0, 1e-2, .25
    # x_mc, cols_mc = generateMonteCarloData(nObs, sLength, size0, size1, mu0, sigma0, sigma1F)