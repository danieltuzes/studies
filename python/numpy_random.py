"""numpy_random.py
    A small test file for the presentation."""

import numpy

instance1 = numpy.random.Generator(numpy.random.MT19937(1000))
instance2 = numpy.random.Generator(numpy.random.MT19937(1001))

instance1.random()  # 0.8610884206320808, modifies the state of the 1st instance
instance2.random()  # 0.7507719112585033, modifies the state of the 2st instance

P = 6  # or 6
T = 1000

pos = [[0] for i in range(P)]         # the position the particles
for i in range(1, T):                 # time evolution
    for j in range(P):
        if instance1.random() > 0.5:  # move up
            pos[j].append(pos[j][-1] + 1)
        else:                         # move down
            pos[j].append(pos[j][-1] - 1)

for i in range(T):
    for j in range(P):
        print(pos[j][i], end="\t")
    print("\n", end="")


# alternative version

for j in range(P):          # for every particle
    pos = []                # the position the particle
    prng_inst = numpy.random.Generator(numpy.random.MT19937(P))
    for i in range(1, T):   # time evolution
        if prng_inst.random() > 0.5:  # move up
            pos.append(pos[-1] + 1)
        else:                         # move down
            pos.append(pos[-1] - 1)
