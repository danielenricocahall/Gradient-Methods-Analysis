'''
Created on Mar 12, 2019

@author: daniel
'''
import numpy as np
import matplotlib.pyplot as plt

eps = 1e-6
delta = 1e-6
MAX_ITER = 500
def f(x):
    return np.square(x) - 4*x

def df(x):
    return (f(x) - f(x - delta))/delta



def main():
    x_0 = -10;
    x_k = x_0;
    path = []
    path.append((x_0, f(x_0)))
    i = 0
    d_k = df(x_k)
    betas = []
    a = 0.1
    k = []
    while True:
        prev_grad = df(x_k)
        x_k = x_k - a * d_k;
                
        B_k = df(x_k)*df(x_k)/(prev_grad * prev_grad)
        
        d_k = df(x_k) - B_k * d_k;
            
        path.append((x_k, f(x_k)))
        betas.append(B_k)

        k.append(i)

        if abs(df(x_k)/df(x_0)) <= eps:
            break
        if i > MAX_ITER:
            break
        i = i+1
            
    print("It took " + str(i) + " iterations to converge!", flush = True)
    print("The optimal value occurs at " + str(x_k), " where the value of the function is " + str(f(x_k)))
    foo, bar = zip(*path)

    x = np.linspace(-20, 20, 200)
    plt.plot(x, f(x))
    plt.plot(foo, bar, label = 'f(x_k)')
    plt.plot(x_k, f(x_k), 'ro')
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    
    plt.figure()
    plt.plot(k, betas)
    plt.xlabel("k")
    plt.ylabel(r"$\beta_k$")
    
    plt.show()
    
    
    
    
    
if __name__=="__main__":
    main()
    exit()