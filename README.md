# Gradient-Methods-Analysis
Analysis of Gradient Methods, including Conjugate Gradient, Momentum, RMSProp, and Adam (adaptive moments). While the test case is a simple 1-D function with several local minima, I believe the insights derived here are helpful for understanding how this methods apply in higher dimensional space.

# Usage

Setup a virtual environment with `pipenv install`. Once installed, in the `runner.py` module, you should be able to run simple 1-D optimizations by changing the function `f` and changing the optimizer type in the `Optimizer.get_optimizer` call. The initial value `x_0` and step size `a` can also be changed in this script. If you're using an optimizer outside of SGD, you can supply `B` as an argument to `opt.run`. In the case of Adam, you would need to supply `B_1` and `B_2`.

## Demo (GD + Momentum) ##
![alt text](https://github.com/danielenricocahall/Gradient-Methods-Analysis/blob/master/Demos/gd_momentum_demo.gif)

