"""timeit_meas.py
    Compares the runtime of some small code snippets."""

from timeit import default_timer

import named_prng


ABC = "qwertzuiopasdfghjklyxcvbnm"
four_letters = [a+b+c+d for a in ABC for b in ABC for c in ABC for d in ABC]

# atoms is 456976 long
atoms = {key: val for val, key in enumerate(four_letters)}
remove_atoms = {"this", "that", "what", "mine", "your", "hers", "stop"}

mparticles = {"atoms": atoms}

mpurposes = ["random_walk", "fusion", "fission"]


mnprng = named_prng.NamedPrng(mpurposes, mparticles)

start = default_timer()

# I don't really trust timeit

# arr = mnprng.generate_r(
#     ("normal", (1, 3)),
#     ["atoms", mpurposes[0], (0, 100)],
#     (remove_atoms, None))

# arr = mnprng.normal_r(
#     ["atoms", mpurposes[0], (0, 100)],
#     (remove_atoms, None),
#     (1, 3))

arr = mnprng.generate_r(
    ("normal", (1, 3)),
    ["atoms", mpurposes[0], (0, 100)],
    (None, remove_atoms))

# arr = mnprng.normal_r(
#     ["atoms", mpurposes[0], (0, 100)],
#     (None, remove_atoms),
#     (1, 3))

stop = default_timer()
print(stop-start)
