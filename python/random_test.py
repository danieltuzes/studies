"""random_test.py
    This file provides 3 tests for a random number generator
    and checks for some basic quality property on the engine.
    It is not aimed to provide a deep analysis but to come
    up with some ideas that are easy to implement and interpret.

    # Test 1:
    Create two lists A and B as vectors with 1-1 million of random numbers
    generated with the prng with adjacent seeds and calculate
    the scalar product of the vectors as function of the shift of values in B,
    i.e. let C a vector for which
    C[i] = Σ_j A[j] * B[j-i]
    and then plot C[i] as a function of i in the range [-200:200].

    # Test 2:
    Create list of lists A[i], i=0:10, where each list has 100'000
    random numbers generated with the prng with adjacent seeds. Then calculate
    the list of correlation matrices C using a 1000-width window, i.e.
    C[k][i][j] = corr(A[i][k*1000:k*1000 + 1000], A[j][k*1000:k*1000 + 1000])
    for every i, j in 0:10 and for every k in 0:100.
    For each k, M[i][j] = C[k][i][j] is matrix with a size of 10 x 10.
    Calculate the eigenvalues of M, and denote the largest with λ(k)
    for a given k. Plotting λ(k) as a function of k should be a random noise
    with an amplitude in the order of magnitude of 1/sqrt(1000).

    # Test 3
    Generete 1'000'000 random numbers with the prng using seed 0 into A.
    Then check what is the longest sequence from the end of the array
    that can be found in the array. A[-2:-1] is a sequence of 1 number,
    this can be found at the end of A, and if the numbers are represented
    with 64 bit precision, there is a chance of P1 = 1 - (1-(1/2^64))^1'000'000
    that this number is repeated, P = 5 * 10^-14. Because this propability
    is very low, most propably even the last element cannot be found again,
    and there is no real need to check for a 2-long sequence."""

import numpy

HW1 = 200  # half-width of the region of interest in test 1, 200 is good enough

LENGTH2 = 2000  # the length of the random vector, aim to 100'000

LENGTH3 = 100000000  # the length of the random vector, aim to 100'000'000
LENGTH3b = 100  # the uniqueness of the last LENGTH3b numbers

TC = {3}  # test cases, can be 1, 2 or 3, or any combination

if __name__ == "__main__":

    if 1 in TC:  # Test 1
        corr = numpy.zeros((2*HW1), dtype=numpy.float64)
        for i in range(-HW1, HW1):
            vecA = numpy.ndarray((10**6), dtype=numpy.float64)
            prngA = numpy.random.Generator(numpy.random.MT19937(0))
            vecA = prngA.random(10**6)

            vecB = numpy.ndarray((10**6), dtype=numpy.float64)
            prngB = numpy.random.Generator(numpy.random.MT19937(1))
            vecB = prngB.random(10**6)

            corr[i] = numpy.dot(vecA, numpy.roll(vecB, i)) - 10**6/4

        with open("test1.dat", 'w') as ofile:
            for i in range(-HW1, HW1):
                print(i, corr[i], sep="\t", file=ofile)

    if 2 in TC:  # Test 2
        A = numpy.ndarray((10, LENGTH2), dtype=numpy.float64)
        for i in range(10):
            prng = numpy.random.Generator(numpy.random.MT19937(i))
            A[i] = prng.random(LENGTH2)

    if 3 in TC:  # Test 3
        A = numpy.ndarray(LENGTH3, dtype=numpy.float64)
        prng = numpy.random.Generator(numpy.random.MT19937(0))
        A = prng.random(LENGTH3)

        # turn off commenting to make the result false
        # A[LENGTH3-2] = A[LENGTH3-1]

        UNIQUE = True
        for i in range(LENGTH3-1, LENGTH3 - LENGTH3b, -1):
            UNIQUE &= numpy.where(A == A[i])[0][0] == i

        print("The last", LENGTH3b, "elements are all unique:", UNIQUE, sep=" ")
