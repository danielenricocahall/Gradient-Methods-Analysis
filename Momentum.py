'''
Created on Mar 12, 2019

@author: daniel
'''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os



eps = 1e-3
delta = 1e-6
MAX_ITER = 500
def f(x):
    #return np.square(x) - 4*x
    return np.square(x) / 10 - 2*np.sin(x)


def df(x):
    return (f(x) - f(x - delta))/delta

def main():
    x_0 = -15;
    a = 0.1;
    x_k = x_0;
    B = 0.9;
    path = []
    path.append((x_0, f(x_0)))
    i = 0
    d_k = df(x_k)
    while True:
        d_k = B * d_k + (1 - B) * df(x_k)
        x_k = x_k - a * d_k;
        path.append((x_k, f(x_k)))
        

        if abs(df(x_k)/df(x_0)) <= eps:
            break
        if i > MAX_ITER:
            break
        foo, bar = zip(*path)

        x = np.linspace(-20, 20, 200)
        plt.figure()
        plt.plot(x, f(x))
        plt.plot(foo, bar, label = 'f($x_k$)')
        plt.plot(x_k, f(x_k), 'ro')
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.savefig("Plot_Pictures/" + str(i) + ".png")
        i = i+1
            
    print("It took " + str(i) + " iterations to converge!", flush = True)
    print("The optimal value occurs at " + str(x_k), " where the value of the function is " + str(f(x_k)))

    os.system("ffmpeg -r 10 -i Plot_Pictures/%d.png -vcodec mpeg4 -y movie.mp4")

    
    
    
    
if __name__=="__main__":
    main()
    exit()