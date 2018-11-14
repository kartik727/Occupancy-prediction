import numpy as np

abc = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

#print(abc[1][2])

abc = np.asarray(abc)

for a in abc.T:
    for b in a:
        b += 1

#print(abc)


defg = [1, 2, 3, 4, 5, 6, 7, 8, 9]
h = 3

for x in defg[1:3]:
    print(x)

ddd = range(3, 7)

print(ddd)