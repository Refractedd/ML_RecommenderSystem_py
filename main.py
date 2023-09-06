# Movie Recommender System
# 15 April 2023
# Authors:
# Benjamin Williams R11544055
# Austin ___
# ___ ___

# Import Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.linalg as lin #import svd
from icecream import ic

# FUNCTIONS
# You may add or remove functions according to the need of you code.

# Uncomment and modify to use a function if you need them

def init_UM(Nu, Nm, Nf):
    u = np.random.normal(0, .1, (Nu, Nf))
    m = np.random.normal(0, .1, (Nf, Nm))

    return u, m

# CODE to initialize U and M matrix

# return U and M matrix

# CODE to implement forward pass to get prediction
# Return all the required matrices and outputs
def get_prediction(U, M, Ri):
    return np.matmul(U, M) * Ri
# return prediction


def comp_grad(X, R, U, M, Y, lam):

    dEdU = np.matmul(U, M)
    dEdU = np.subtract(dEdU, X)
    dEdU = dEdU * R
    dEdU = np.matmul(dEdU, np.transpose(M))
    dEdU = dEdU + lam * U

    dEdM = np.matmul(U, M)
    dEdM = np.subtract(dEdM, X)
    dEdM = dEdM * R
    dEdM = np.matmul((np.transpose(U)), dEdM)
    dEdM = dEdM + lam * M
    return dEdU, dEdM

    # return np.transpose(M) * (np.dot((U * M) - X), R) + lam * U
# CODE to implement gradient computation
# Return all the required matrices and outputs
# Make sure to divide the first part by size of U and M otherwise use a very tiny learning rate.
# If not, then gradient may not converge and diverge to infinity crashing the training.
# Ask me if in doubt about implementation.

# return gradients for U and M


# Other useful functions

def comp_error(X, Xhat):
    return 0.5 * ((X - Xhat)**2)#0.5 * (np.square((np.subtract(X, Xhat))))

# def comp_error(X,Y,R,Rdiff):
#     XX = X[R==1]
#     YY = Y[R==1]
#
#     return sum(np.abs(XX-YY)<=Rdiff)/len(XX)

## Load Dataset

# Replace with your directory
hm_dir = r"C:\Users\ninja\Desktop\School_Senior\Adv Linear Algebra\Final Project"

# Replace with your filename (CSV)
fname = 'ratings_small.csv'

# Load data and prepare Input/Output matrices
full_file = os.path.join(hm_dir, fname)
data = np.array(pd.read_csv(full_file))

# Extract data to matrices

# You may change this to get more/less movies/users
Nu = 500  # Number of users
Nm = 1000  # Number of movies

X = np.zeros((Nu, Nm))  # initialize Blank rating matrix
R = np.zeros((Nu, Nm))  # initialize blank R matrix
Ri = np.nan * np.zeros((Nu, Nm))

for i in range(np.size(data, 0)):
    uu = np.int32(data[i, 0]) - 1
    mm = np.int32(data[i, 1]) - 1
    if ((uu < Nu) & (mm < Nm)):
        X[uu, mm] = data[i, 2]
        R[uu, mm] = 1
        Ri[uu, mm] = 1

#  Set up Rating matrices

Am = np.nanmean(X * Ri, 1, keepdims=True)  # Mean for normalization
idx = np.reshape(~np.isnan(Am), Nu)  # Identify users with no rating

# Remove users with no ratings
X = X[idx, :]
Ri = Ri[idx, :] # ratings with nan
R = R[idx, :]   # rating with 0
Am = Am[idx]    # mean

Nu = sum(idx)  # Modify the number of users

# Initialize U and M matrix

# Number of latent features. More the better but too much and it will crash due to time/memory limitations
Nf = 32

# YOUR CODE to initialize U and M matrix
# Make sure the matrix size are proper.
# U size => [Nu,Nf]
# M size => [No,Nh]

# Uncomment and modify code below.


# You may create a function or add code lines directly here.
# DO NOT CHANGE VARIABLE NAMES

U, M = init_UM(Nu, Nm, Nf)
# ratings = X.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)


# Training process
# You may run this module multiple times to keep on training and fine tuning your network.
# Usually 3 (hyper)parameters are modified with each fine-tuning run
# You may play around with different values for the 3 variables below.

# Regularization constant
lam = 1e-3
# Too small and it does not have any effect, too large and the accuracy will saturate to a lower value


# Learning rate
lr = 1e-3
# Start with 0.1 or 0.01 and then modify according to the error progression.
# If not normalized the gradient, use very small value like 1e-7 or lower.


# How many times you want to modify the U and M matrix?
MaxItr = 600  # maximum number of iterations

# initialize Accuracy plot
Acc = np.zeros((MaxItr, 1))

for i in range(MaxItr):

    # Your function or code to do forward propagation
    # Forward pass requires U and M matrixes.
    # It should output prediction (Xhat)
    # Do not change the variable name otherwise the code might not work
    # Uncomment the line below and put your own code or your function name
    # Xhat = get_prediction(U, M, Ri)
    # Xhat : Predicted output
    # U    : user latent matrix
    # M    : Movies latent matrix
    #ic(U.shape, M.shape, R.shape, X.shape)
    Xhat = get_prediction(U, M, Ri)
    Y = Xhat + Am
    #ic(comp_error(X, Xhat))
    Acc[i] = np.nanmean(comp_error(X, Xhat))
    #Acc[i] = comp_error(X, U, M, R)

    # Your function or code to do backpropagation
    # Backpropagation should output gradients corresponding to U and M
    # The input and output variable names are for reference.
    # Do not change the variable names otherwise the code might not work.
    # Uncomment the line below and put your own code or your function name

    # gradU,gradM = comp_grad(X,R,U,M,Y,lam)
    # gradU : Gradient for U
    # gradM : Gradient for M
    gradU, gradM = comp_grad(X, R, U, M, Y, lam)

    U = U - lr * gradU
    M = M - lr * gradM

    if (np.mod(i, 10) == 0):
        print(Acc[i])#Er[i])

plt.plot(Acc)
B_acc = np.nanmean(comp_error(X, Xhat))

#ic(B_acc)

print('Accuracy at the end of training = ', B_acc)
# Visualize the prediction accuracy
# Histogram to show the error distribution

# Your code to compute prediction for entire dataset (X)
# You may either use the function created or your own lines of code.
# Edit the lines below
# Xhat = get_prediction(U,M,R)
Xhat = np.matmul(U, M)

Y = Xhat + Am
ic(Y)
XX = X[R == 1]
YY = Y[R == 1]

fig1 = plt.hist(YY - XX, 8)
plt.title("Histogram of Prediction accuracy")
plt.savefig("Histogram of Prediction accuracy")
# plt.show()
plt.close()
# Visualize the original and reconstructed recommendation matrix

fig2 = plt.figure(figsize=(20, 15))
plt.title("X Data")
plt.imshow(X)
# plt.show()
plt.savefig("X_Mtrx")
plt.close()

fig3 = plt.figure(figsize=(20, 15))
plt.title("Normalized Data")
plt.imshow(Y)
# plt.show()
plt.savefig("Y_Mtrx")
# Visualize the U and M matrix
plt.close()

fig4 = plt.figure(figsize=(20, 5))
plt.title("U matrix")
plt.imshow(U.T)
#plt.show()
plt.savefig("U_Mtrx")
plt.close()

fig5 = plt.figure(figsize=(30, 2))
plt.title("M matrix")
plt.imshow(M)
plt.show()
plt.savefig("M_Mtrx")
plt.close()

# Suggest new movies to users: (Extremely simplified and quite different from the actual code used)

# User ID
UU = 50  # User number 50

# How many movies to suggest
NM = 5  # top 5

# Your code to compute prediction for entire dataset (X)
# You may either use the function created or your own lines of code.
# Edit the lines below

# Xhat = np.matmul(U, M)

Y = Xhat + Am

Yu = Y[UU - 1, :]
YUix = np.argsort(Yu)
print('Movie ID: ', YUix[-5:] + 1)
Yu[YUix[-5:]]