import pandas as pd

def foo(x):
    x = x + 1
    z = 2
    x = z * 2
    return x

x = 1
x = foo(x)
type(x)
x = pd.DataFrame()