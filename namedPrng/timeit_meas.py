"""timeit_meas.py
    Compares the runtime of some small code snippets."""

from timeit import default_timer

from named_prng import NamedPrng, Distr, FStrat


ABC = "qwertzuiopasdfghjklyxcvbnm"
four_letters = [a+b+c+d for a in ABC for b in ABC for c in ABC for d in ABC]

# atoms is 456976 long
atoms = {key: val for val, key in enumerate(four_letters)}
words_subset = {"this", "that", "what", "mine", "your", "hers", "stop"}

mparticles = {"atoms": atoms}

mpurposes = ["random_walk", "fusion", "fission"]


mnprng = NamedPrng(mpurposes, mparticles)

start = default_timer()

# I don't really trust timeit


arr = mnprng.generate_r(
    (Distr.STN, (1, 3)),
    ["atoms", mpurposes[0], (0, 100)],
    (words_subset, FStrat.EXC))

# arr = mnprng.generate_r(
#     (Distr.STN, (1, 3)),
#     ["atoms", mpurposes[0], (0, 100)],
#     (words_subset, FStrat.INC))

stop = default_timer()
print(stop-start)
