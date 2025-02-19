import numpy as np

def hill_climb(f, x, step=0.01, max_iter=1000):
    for i in range(max_iter):
        x_step=max([x-step, x, x+step], key=f)
        if(f(x_step)<=f(x)):break
        x=x_step
    return x, f(x)

inp=input("Enter the Function: ")
f=lambda x: eval(inp, {"x":x, "np":np})
x=float(input("Enter the Starting point: "))
a, b=hill_climb(f,x)
print(f"Maxima at x = {a}, Value = {b}")
