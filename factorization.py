"""
Created on Fri Nov 1 
@author: mudasir

Mudasir Hanif Shaikh
Student, DSSE
CS Class of 2021
Habib University

CS 351 - Artificial Intelligence
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

"""This function takes actual and predicted ratings and compute total mean square error(mse) in observed ratings.
"""
def computeError(R, predR):
    
    """Calculate the MSE for predictions"""
    error = ((R - predR)**2).mean()
    
    return error


"""
This fucntion takes P (m*k) and Q(k*n) matrices along with user bias 
(U) and item bias (I) and returns predicted rating. 
where m = No of Users, n = No of items
"""
def getPredictedRatings(P,Q,U,I):

    """
    predict ratinng matrix R using the values P,Q,U,I
    """    
    R_ = np.zeros(shape = (P.shape[0], Q.shape[1]))
    for user in range(P.shape[0]):
        for item in range(Q.shape[1]): 
            #Reshapes P and Q into (1,K) vectors
            R_[user][item] = np.dot(P[user].reshape((1, P.shape[1])), Q[:,item].reshape((P.shape[1], 1))) + U[user] + I[item]
    return R_
    
    
"""
This fucntion runs gradient descent to minimze error in ratings by 
adjusting P, Q, U and I matrices based on gradients.
The functions returns a list of (iter,mse) tuple that lists mse in 
each iteration
"""
def runGradientDescent(R,P,Q,U,I,iterations,alpha):
   
    stats = []
    
    """
    Performs gradient descent and updates the stats list.
    Stats contains MSE for each iteration.
    """    
    for itr in range(iterations):
        R_ = getPredictedRatings(P,Q,U,I)
        for user in range(P.shape[0]):
            for item in range(Q.shape[1]): 
                if R[user][item]:
                    error = R[user][item] - R_[user][item]
                    #updating features and biases
                    P[user, :] += 2*alpha*error*Q[:, item]
                    Q[:, item] += 2*alpha*error*P[user, :]
                    U[user] -= alpha*error
                    I[item] -= alpha*error
        R_ = getPredictedRatings(P,Q,U,I)
        err = computeError(R, R_)
        stats.append((itr, err))
       
    """"
    finally returns (iter, mse) values in a list.
    """
    return stats
    
""" 
This method applies matrix factorization to predict unobserved values
in a rating matrix (R) using gradient descent. K is number of latent 
variables and alpha is the learning rate to be used in gradient decent
"""    

def matrixFactorization(R,k,iterations, alpha):

    """Your code to initialize P, Q, U and I matrices goes here. P and Q will
    be randomly initialized whereas U and I will be initialized as zeros. 
    Be careful about the dimension of these matrices
    """
    P = np.random.normal(size = (R.shape[0], k))
    Q = np.random.normal(size = (k, R.shape[1]))
    U = np.zeros((R.shape[0]))
    I = np.zeros((R.shape[1]))
    
    #Run gradient descent to minimize error
    stats = runGradientDescent(R,P,Q,U,I,iterations,alpha)
    
    print('P matrx:')
    print(P)
    print('Q matrix:')
    print(Q)
    print("User bias:")
    print(U)
    print("Item bias:")
    print(I)
    print("P x Q:")
    print(getPredictedRatings(P,Q,U,I))
    plotGraph(stats)
       
    
def plotGraph(stats):
    i = [i for i,e in stats]
    e = [e for i,e in stats]
    plt.plot(i,e)
    plt.xlabel("Iterations")
    plt.ylabel("Mean Square Error")
    plt.show()    
    
""""
User Item rating matrix given ratings of 5 users for 6 items.
Note: If you want, you can change the underlying data structure and can
work with standard python lists instead of np arrays. We may test with 
different matrices with varying dimensions and number of latent factors. 
Make sure your code works fine in those cases.
"""
R = np.array([
[5, 3, 0, 1, 4, 5],
[1, 0, 2, 0, 0, 0],
[3, 1, 0, 5, 1, 3],
[2, 0, 0, 0, 2, 0],
[0, 1, 5, 2, 0, 0],
])

k = 3
alpha = 0.01
iterations = 100

matrixFactorization(R,k,iterations, alpha)
