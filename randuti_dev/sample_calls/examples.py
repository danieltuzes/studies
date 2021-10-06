"""examples.py
Here we call named_prng and test it with some cases. This file proves:

- random numbers can be generated
    for a given particle set (aka particle type)
- the random numbers generated can be restricted by either
    excluding some particles, or only by allowing a set of them
- for the same particle type, random numbers can be generated
    for different purposes
- the random numbers can be read back in the same order

For a more detailed test, check the ``test_named_prng.py``.
"""

import os
from randuti import NamedPrng, FStrat, Distr

# quarks is a particle type with 6 different IDs as keys
# The order number will be important once we remove IDs.
quarks = {"up": 0, "down": 1, "charm": 2,
          "strange": 3, "top": 4, "bottom": 5}

# atoms is another particle type with 4 different IDs as keys.
# We won't use atoms btw.
atoms = {"H": 0, "He": 1, "Li": 2, "Be": 3}

# the dict of dict, all particles together,
# and we also know the name of their type
# their order is important for the prng instance,
# simulations are comparable realization-wise
# if the order is kept the same. Since py 3.6,
# dict is ordered, but sets are not

mparticles = {"quarks": quarks, "atoms": atoms}
uparticles = {"quarks": len(quarks), "atoms": len(atoms)}

# I will generate random numbers for a smaller set of IDs too
remove_quarks = {"charm", "strange"}
quarks_subset = quarks.copy()  # contains the subset of quarks
for rid in remove_quarks:
    quarks_subset.pop(rid)

# for what purposes will we need the random numbers
# prng instances are assigned to a particle type and purpose combination
# their order is important for the prng instance,
# simulations are comparable realization-wise
# if the order is kept the same. Since py 3.6,
# dict is ordered, but sets are not
mpurposes = ["random_walk", "radioactive_decay"]


def do_some_stuff(mnprng: NamedPrng) -> None:
    """Do some stuff with the NamedPrng.

    I will call it without and with _sourcefile defined to see
    if it gives the same result.
    """

    for realization_id in range(2):
        for purpose in mpurposes:
            print("realization_id =", realization_id, "purpose =", purpose)

            # generate random numbers for quarks
            mnprng.init_prngs(realization_id)
            random_for_quarks = mnprng.generate(
                Distr.UNI, ("quarks", purpose))

            for (key, value), rnd in zip(quarks.items(), random_for_quarks):
                print(key, value, rnd, sep="\t")

            # generate random numbers for a subset of quarks by excluding
            print(
                "--------  now for the restricted set of particles by excluding  --------")

            mnprng.init_prngs(realization_id)    # reset the prngs

            random_for_quarks_subset = mnprng.generate(Distr.UNI,
                                                       ("quarks", purpose),
                                                       id_filter=(remove_quarks, FStrat.EXC))

            for (key, value), rnd in zip(quarks_subset.items(), random_for_quarks_subset):
                print(key, value, rnd, sep="\t")

            # generate random numbers for a subset of quarks by including
            print(
                "--------  now for the restricted set of particles by including  --------")

            mnprng.init_prngs(realization_id)    # reset the prngs

            random_for_quarks_subset_in = mnprng.generate(Distr.UNI,
                                                          ("quarks", purpose),
                                                          id_filter=(quarks_subset, FStrat.INC))

            for (key, value), rnd in zip(quarks_subset.items(), random_for_quarks_subset_in):
                print(key, value, rnd, sep="\t")
            print(
                "========================================================================")

    print('--  now for all realizations in 1 go for the purpose "random_walk"  --')
    print(mnprng.generate_it(Distr.UNI, [
        "quarks", "random_walk", (0, 1, 2)]))


print("""
###################################################################
####    Generating random numbers with the Mersenne Twister    ####
###################################################################""")
Mnprng_gen = NamedPrng(
    mpurposes, mparticles, exim_settings=("tee.dat", "", True))
do_some_stuff(Mnprng_gen)
del Mnprng_gen

print("""
###################################################################
###########     Random numbers are read from a file     ###########
###################################################################""")
Mnprng_use = NamedPrng(
    mpurposes, mparticles, exim_settings=("", "tee.dat", True))
do_some_stuff(Mnprng_use)
Mnprng_use.export_particles()
del Mnprng_use

Mnprng_stu = NamedPrng(mpurposes)

print("\nLoad back the particles and generate random numbers for quarks for random_walk.")
print(Mnprng_stu.generate_it(Distr.UNI, [
    "quarks", "random_walk", (0, 1, 2)]))

os.remove("dict_of_particles.pickle")

print(f"\nPrint out {len(range(0,4))} times random number for quarks.",
      f"quarks are {len(quarks)} long")
ind_particles = NamedPrng(mpurposes, uparticles)
rnd_array = ind_particles.generate_it(
    Distr.UNI, ("quarks", "random_walk", range(0, 4)))
print(rnd_array)
