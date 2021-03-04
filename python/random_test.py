"""random_test.py
    This file provides 3 tests for a random number generator
    and checks for some basic quality property on the engine.
    It is not aimed to provide a deep analysis but to come
    up with some ideas that are easy to implement and interpret.

    # Test 1:
    Create two arrays A and B as vectors with 1-1 million of random numbers
    generated with the prng with adjacent seeds. The average of such arrays
    tends to 0 with increasing size, but for a fixed length of 1 million,
    it will have a non-0 average. Substract the average then calculate
    the scalar product of the vectors as function of the shift of values in B,
    i.e. let C a vector for which
    C[i] = Σ_j A[j] * B[j-i]
    and then plot C[i] as a function of i in the range [-200:200].
        * This should be a random noise with values around 0. The fact
          that it is around 0 is a consequence of the operation
          subtracting the average.
        * Its cumulative distribution at C[0] must tend to be 0.5 showing
          that the same amount of numbers are smaller than 0 than larger
          than 0, i.e. the median tends to 0 if the average is 0.
        * Accumulating the values of C represent a non-binary
          random walk, and it should not tend to 0 with increasing
          length of A or C.

    # Test 2:
    Create list of lists A[i], i=0:10, where each list has L random numbers
    generated with the prng with adjacent seeds. The size of the array
    will increase and we check how a property scales with increasing size.
    Then calculate the correlation matrix C for every i, j in 0:10
    C[i][j] = pearson_correlation(A[i], A[j])
    C is a matrix with a size of 10 x 10. Calculate the eigenvalues of C.
    If the length L of A[i] is smaller than 10, it will be a degenrate matrix,
    and not more than L eigenvalue will be relevant, but if L>10, we have
    10 eigenvalues. The largest eigenvalues must tend to 1 with
    increasing L. Plot the eigenvalues as a function of increasing L and be
    sure it tends to 1.

    # Test 3
    Generete 1'000'000 random numbers with the prng using seed 0 into A.
    Then check what is the longest sequence from the end of the array
    that can be found in the array. A[-2:-1] is a sequence of 1 number,
    this can be found at the end of A, and if the numbers are represented
    with 64 bit precision, there is a chance of P1 = 1 - (1-(1/2^64))^1'000'000
    that this number is repeated, P = 5 * 10^-14. Because this propability
    is very low, most propably even the last element cannot be found again,
    and there is no real need to check for a 2-long sequence.
    I also implemented the test of the uniquness of the fist few numbers.

    # Test 4
    The average of random numbers generated from one seed tends to 0.5, as well
    as the average of the first random number from increasing seed values."""

import numpy

HW1 = 200            # half-width of the region of interest in test 1, 200 is good enough

PWR2 = 6            # the max length of the random vector, aim to 2 ** 20

LENGTH3 = 100000000  # the length of the random vector, aim to 100'000'000
LENGTH3B = 50        # the uniqueness of the last LENGTH3b numbers

PWR4 = 22            # The length of random numbers

TC = {4}  # test cases, can be 1, 2, 3, 4 or any combination of them

if __name__ == "__main__":

    if 1 in TC:  # Test 1

        vecA = numpy.ndarray((10**6), dtype=numpy.float64)
        prngA = numpy.random.Generator(numpy.random.MT19937(0))
        vecA = prngA.random(10**6)
        vecA = vecA - numpy.average(vecA)

        vecB = numpy.ndarray((10**6), dtype=numpy.float64)
        prngB = numpy.random.Generator(numpy.random.MT19937(1))
        vecB = prngB.random(10**6)
        vecB = vecB - numpy.average(vecA)

        corr = numpy.zeros((2*HW1), dtype=numpy.float64)
        for i in range(-HW1, HW1):
            corr[i] = numpy.dot(vecA, numpy.roll(vecB, i))

        with open("test1.dat", 'w') as ofile:
            for i in range(-HW1, HW1):
                print(i, corr[i], sep="\t", file=ofile)

    if 2 in TC:  # Test 2
        W = numpy.ndarray((PWR2+1, 10), dtype=numpy.float64)
        for power in range(0, PWR2):
            length = 2 ** (power+1)
            A = numpy.ndarray((10, length), dtype=numpy.float64)
            for i in range(10):
                prng = numpy.random.Generator(numpy.random.MT19937(i))
                A[i] = prng.random(length)

            C = numpy.ndarray((10, 10), dtype=numpy.float)
            C = numpy.corrcoef(A)
            w, v = numpy.linalg.eig(C)
            W[power] = numpy.real(w)

        with open("test2.dat", 'w') as ofile:
            print("# length of rnd vector",
                  *["eigval" + str(i) for i in range(10)], sep="\t", file=ofile)
            for power in range(0, PWR2):
                print(2**power, *W[power], sep="\t", file=ofile)

    if 3 in TC:  # Test 3
        A = numpy.ndarray(LENGTH3, dtype=numpy.float64)
        prng = numpy.random.Generator(numpy.random.MT19937(0))
        A = prng.random(LENGTH3)

        # turn off commenting to make the result false
        # A[LENGTH3-2] = A[LENGTH3-1]

        UNIQUE = True
        for i in range(LENGTH3B):
            UNIQUE &= numpy.where(A == A[i])[0][0] == i

        for i in range(LENGTH3-1, LENGTH3 - LENGTH3B, -1):
            UNIQUE &= numpy.where(A == A[i])[0][0] == i

        print("The first and last", LENGTH3B,
              "elements are all unique:", UNIQUE, sep=" ")

    if 4 in TC:  # Test 4

        # random numbers from single seed
        RND_SS = numpy.ndarray(2 ** PWR4, dtype=numpy.float64)
        prng = numpy.random.Generator(numpy.random.MT19937(0))
        RND_SS = prng.random(2 ** PWR4)

        # random numbers from multiple seeds
        RND_MS = numpy.ndarray(2 ** PWR4, dtype=numpy.float64)
        print("adj seed rnd number generation starts")
        for seed in range(2 ** PWR4):
            prng = numpy.random.Generator(numpy.random.MT19937(seed))
            rand_numb = prng.random()
            RND_MS[seed] = rand_numb

        print("adj seed rnd number generation finished")
        with open("test4.dat", "w") as ofile:
            print("# length of rnd vector", "from seed 0",
                  "1st rnd from adjacent seeds", sep="\t", file=ofile)
            for pwr_limit in range(1, PWR4):
                print(2**pwr_limit, numpy.average(RND_MS[0:2 ** pwr_limit]),
                      numpy.average(RND_SS[0:2 ** pwr_limit]), sep="\t", file=ofile)
