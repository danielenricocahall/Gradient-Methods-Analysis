'''
Created on Jul 7, 2019

@author: daniel
'''
import matplotlib.animation as animation
import numpy as np
from pylab import *


dpi = 100

def ani_frame():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    im = ax.imshow(rand(300,300),cmap='gray',interpolation='nearest')
    im.set_clim([0,1])


    tight_layout()
    eps = 1e-6
    delta = 1e-6
    MAX_ITER = 500
    def f(x):
        #return np.square(x) - 4*x
        return np.square(x) / 10 - 2*np.sin(x)


    def df(x):
        return (f(x) - f(x - delta))/delta


    def update_img(n):
        x_0 = -10;
        a = 0.1;
        x_k = x_0;
        B = 0.99;
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
        
        #im.set_data()

    #legend(loc=0)
    ani = animation.FuncAnimation(fig,update_img,300,interval=30)
    writer = animation.writers['ffmpeg'](fps=30)

    ani.save('demo.mp4',writer=writer,dpi=dpi)
    return ani

if __name__=="__main__":
    ani_frame();
    exit();