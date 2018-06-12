#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Adaptado de --> https://www.dropbox.com/s/slvfo2k8za9ocli/SelfOrgMaps.py?dl=0

http://www.computacaointeligente.com.br/algoritmos/mapas-auto-organizaveis-som/
"""
 
import numpy as np
from time import sleep, time
import matplotlib.pyplot as plt
 
class SOM:
    # Vetor m x n x dim, onde m e n são o grid de nodes e dim é a dimensão do peso
    wNodes = None
     
    alpha0 = None # Taxa de aprendizagem inicial
    sigma0 = None # Sigma inicial
    dataIn = None # Entrada
    grid = None   # Treliça da grade
     
    def __init__ (self, dataIn, grid=[10,10], alpha=0.1, sigma=None):
        dim = dataIn.shape[1]
        self.wNodes = np.random.uniform(-1,1,[grid[0], grid[1], dim])
        #self.wNodes = np.random.randn (grid[0], grid[1], dim)    
         
        self.alpha0 = alpha
        if (sigma is None):
            self.sigma0 = max(grid) / 2.0
        else:
            self.sigma0 = sigma
         
        self.dataIn = np.asarray(dataIn)
        self.grid = grid
         
         
    def train (self, maxIt=100, verbose=True, analysis=False, timeSleep = 0.5):
        nSamples = self.dataIn.shape[0]
        m = self.wNodes.shape[0]        
        n = self.wNodes.shape[1]        
     
     
        # A constante de tempo é computada apenas uma vez
        timeCte = (maxIt/np.log(self.sigma0))        
        if analysis:
            print ('cteTempo = '), timeCte
             
        timeInit = 0       
        timeEnd = 0
        for epc in range(maxIt):
            # Calculando constantes
            alpha = self.alpha0 * np.exp(-epc/timeCte)
            sigma = self.sigma0 * np.exp(-epc/timeCte)
             
            if verbose:
                print ('Epoch: ', epc, ' - Tempo esperado: ', (timeEnd-timeInit)*(maxIt-epc), ' seg')
                 
            timeInit = time()
 
            for k in range(nSamples):    
                 
                # Getting the winner node
                matDist = self.distance (self.dataIn[k,:], self.wNodes)
                posWin = self.getWinNodePos(matDist)                              
                 
                deltaW  = 0               
                h = 0   
                           
                 
                for i in range(m):
                    for j in range(n):      
                        # Calcula a distância entre dois nós
                        dNode = self.getDistanceNodes([i,j],posWin)                       
                         
                         
                        #if dNode <= sigma: 
                             
                        # Calcula a influência do nó vencedor
                        h = np.exp ((-dNode**2)/(2*sigma**2))
                         
                        # Atualiza pesos
                        deltaW = (alpha*h*(self.dataIn[k,:] - self.wNodes[i,j,:]))                       
                        self.wNodes[i,j,:] += deltaW
                             
                        if analysis:  
                            print ('Epoch = ', epc)
                            print ('Amostra = ', k)
                            print ('-------------------------------') 
                            print ('alpha = ', alpha) 
                            print ('sigma = ', sigma)                             
                            print ('h = ',  h) 
                            print ('-------------------------------') 
                            print ('Node Vencedor = [', posWin[0],', ',posWin[1],']') 
                            print ('Node Atual = [',i,', ',j,']') 
                            print ('dist. Nodes = ', dNode) 
                            print ('deltaW = ', deltaW  )                       
                            print ('wNode antes = ', self.wNodes[i,j,:]) 
                            print ('wNode depois = ', self.wNodes[i,j,:] + deltaW) 
                            print ( '\n'                       )
                            sleep(timeSleep) 
                             
            timeEnd = time()                       
         
 
    # Método que calcula a distância entre entradas e pesos entre a matriz 3D(dist. euclidiana)
    def distance (self,a,b):
        return np.sqrt(np.sum((a-b)**2,2,keepdims=True))        
 
    # Pega a distância entre dois nós no grid
    def getDistanceNodes (self,n1,n2):
        n1 = np.asarray(n1)
        n2 = np.asarray(n2)
        return np.sqrt(np.sum((n1-n2)**2))
         
    # Pega a posição do node vencedor
    def getWinNodePos (self,dists):
        arg = dists.argmin()
        m = dists.shape[0]
        return arg//m, arg%m
         
    # Pega o centroid da entrada
    def getCentroid (self, data):
        data = np.asarray(data)        
        N = data.shape[0]
        centroids = list()
         
        for k in range(N):
            matDist = self.distance (data[k,:], self.wNodes)
            centroids.append (self.getWinNodePos(matDist))
             
        return centroids
         
    # Salva e carrega nodes treinados
    def saveTrainedSOM (self, fileName='trainedSOM.csv'):
        np.savetxt(fileName, self.wNodes)
 
    def setTrainedSOM (self, fileName):
        self.wNodes = np.loadtxt(fileName)
 
 
 
# Entradas de cor e treino
colors = np.array(
     [[0., 0., 0.],
      [0., 0., 1.],
      [0., 0., 0.5],
      [0.1, 0.529, 1.0],
      [0.2, 0.4, 0.67],
      [0.3, 0.5, 1.0],
      [0., 1., 0.],
      [1., 0., 0.],
      [0., 0., 1.],
      [1., 0., 1.],
      [1., 0., 0.],
      [1., 0., 0.],
      [.33, .33, .33],
      [.5, .5, .5],
      [.66, .66, .66]])
       
colors2 = np.array(
     [[0., 0., 0.],
      [0., 0., 1.],     
      [1., 1., 0.],
      [1., 1., 1.],     
      [1., 0., 0.]])      
       
color_names = \
    ['black', 'blue', 'darkblue', 'skyblue',
     'greyblue', 'lilac', 'green', 'red',
     'cyan', 'violet', 'yellow', 'white',
     'darkgrey', 'mediumgrey', 'lightgrey']
 
s = SOM(colors,[20,30], alpha=0.3)
plt.imshow(s.wNodes)
 
s.train(maxIt=30)
 
plt.imshow(s.wNodes)
plt.show()
