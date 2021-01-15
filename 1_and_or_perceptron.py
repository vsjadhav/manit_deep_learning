import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
X = np.array([[0,0],[0,1],[1,0],[1,1]])
# y = np.array([0,1,1,1]) # For OR use this y array
y = np.array([0,0,0,1]) # For AND use this y array

def step_function(x):
    return x>=0

def perceptron(aplha,bias):
    W = np.random.rand(2)
    l=[False]
    epochs=0
    while(all(l)!=True):
        l=[]
        for i in range(4):
            pred = step_function(W.dot(X[i]) + bias)
            W = W - (alpha*(pred-y[i]))*X[i]
            bias = bias - (alpha*(pred-y[i]))
        for i in range(4):
            l.append((step_function(W.dot(X[i]) + bias))==y[i])
        epochs +=1
    return (W,bias,epochs);

def test(x1,x2,weights,bias):
    ans1 =(weights[0]*x1 + weights[1]*x2 + bias)
    return step_function(ans1)


alpha = 0.1
bias = 0
ans = perceptron(alpha,bias)
weights = ans[0]
bias = ans[1]
epochs= ans[2]

# If you want to test use this
x1 = int(input("Enter x1: "))
x2 = int(input("Enter x2: "))
# Works only if x1 & x2 are binary
print(f"epochs taken: {epochs}")
print(test(x1,x2,weights,bias))

X1 = [0,0,1,1]
X2 = [0,1,0,1]
plt.plot(X1,X2,'ro')
pnt1 = (-bias/weights[0])
pnt2 = (-bias/weights[1])
point1 = [pnt1,0]
point2 = [0,pnt2]
# point1 = [0,pnt1]
# point2 = [pnt2,0]
plt.plot(point1,point2,'b-')
plt.show()