"""examples.py
    Here we call named_prng and test it with some cases. This file proves:
    
    - random numbers can be generated
      for a given particle set (aka particle type)
    - the random numbers generated can be restrcited by either
      excluding some particles, or only by allowing a set of them
    - for the same particle type, random numbers can be generated
      for different purposes
    - the random numbers can be read back in the same order"""

import named_prng

# quarks is a particle type with 6 different IDs as keys
# The order number will be important once we remove IDs.
quarks = {"up": 0, "down": 1, "charm": 2, "strange": 3, "top": 4, "bottom": 5}

# atoms is another particle type with 4 different IDs as keys.
# We wont't use atoms btw.
atoms = {"H": 0, "He": 1, "Li": 2, "Be": 3}

# the dict of dict, all particles together,
# and we also know the name of their type
# their order is important for the prng instance,
# simulations are comparable realization-wise
# if the order is kept the same. Since py 3.6,
# dict is ordered, but sets are not
mparticles = {"quarks": quarks, "atoms": atoms}

# I will generate random numbers for a smaller set of IDs too
remove_quarks = {"charm", "bottom"}
quarks_subset = quarks.copy()  # contains the subset of quarks
for rid in remove_quarks:
    quarks_subset.pop(rid)

# for what purposes will we need the random numbers
# prng instances are assigned to a particle type and purpose combination
# their order is important for the prng instance,
# simulations are comparable realization-wise
# if the order is kept the same. Since py 3.6,
# dict is ordered, but sets are not
mpurposes = {"random_walk": 0, "radioactive_decay": 1}


def do_some_stuff(mnprng: named_prng.NamedPrng) -> None:
    """Do some stuff with the NamedPrng.
        I will call it without and with _sourcefile defined to see
        if it gives the same result."""
    for realization_id in range(2):
        for purpose in mpurposes:
            print("realization_id =", realization_id, "purpose =", purpose)

            # generate random numbers for quarks
            mnprng.init_prngs(realization_id)
            random_for_quarks = mnprng.random("quarks", purpose)

            for key, rnd in zip(quarks, random_for_quarks):
                print(key, rnd)

            # generate random numbers for a subset of quarks by excluding
            print("------ now for the restricted set of particles by excluding ------")

            mnprng.init_prngs(realization_id)    # reset the prngs

            random_for_quarks_subset = mnprng.random(
                "quarks", purpose, id_filter=(remove_quarks, None))

            for key, rnd in zip(quarks_subset, random_for_quarks_subset):
                print(key, rnd)

            # generate random numbers for a subset of quarks by including
            print("------ now for the restricted set of particles by including ------")

            mnprng.init_prngs(realization_id)    # reset the prngs

            random_for_quarks_subset_in = mnprng.random(
                "quarks", purpose, id_filter=(None, quarks_subset))

            for key, rnd in zip(quarks_subset, random_for_quarks_subset_in):
                print(key, rnd)
            print("====================================================================")


print("""
###########################################################
##  Generating random numbers with the Mersenne Twister  ##
###########################################################""")
Mnprng_gen = named_prng.NamedPrng(
    mparticles, mpurposes, filenames=("tee.dat", None))
do_some_stuff(Mnprng_gen)

print("""
###########################################################
#########   Random numbers are read from a file   #########
###########################################################""")
Mnprng_use = named_prng.NamedPrng(
    mparticles, mpurposes, filenames=(None, "tee.dat"))
do_some_stuff(Mnprng_use)

print("All the random numbers in the tee file:")
print(named_prng.NamedPrng.get_rnds_from_file("tee.dat"))
