import  numpy as np

a=np.tile(4, (3, 1))

d=[[2,3],[4,5]]


g=np.array(d)
print(g.min(0))