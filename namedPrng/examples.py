"""examples.py
    Here we call named_prng and test it with some cases."""

import named_prng

# quarks is a particle type with 6 different IDs as keys
# The order number will be important once we remove IDs.
quarks = {"up": 0, "down": 1, "charm": 2, "strange": 3, "top": 4, "bottom": 5}

# atoms is another particle type with 4 different IDs as keys.
# We wont't use them btw.
atoms = {"H": 0, "He": 1, "Li": 2, "Be": 3}

# the dict of dict, all particles together,
# and we also know the name of their type
mparticles = {"quarks": quarks, "atoms": atoms}

# I will generate random numbers for a smaller set of IDs too
remove_quarks = {"charm", "bottom"}
quarks_subset = quarks.copy()  # contains the subset of quarks
for rid in remove_quarks:
    quarks_subset.pop(rid)


def do_some_stuff(mnprng: named_prng.NamedPrng) -> None:
    """Do some stuff with the NamedPrng.
        I will call it without and with _sourcefile defined to see
        if it gives the same result."""
    for realization_id in range(2):
        print("realization_id = ", realization_id)

        # generate random numbers for quarks
        mnprng.init_prngs(realization_id)
        random_for_quarks = mnprng.random("quarks")

        for key, rnd in zip(quarks, random_for_quarks):
            print(key, rnd)

        # generate random numbers for a subset of quarks by excluding
        print("--------- now for the restricted set of particles by excluding ---------")

        mnprng.init_prngs(realization_id)    # reset the prngs

        random_for_quarks_subset = mnprng.random(
            "quarks", exclude_ids=remove_quarks)

        for key, rnd in zip(quarks_subset, random_for_quarks_subset):
            print(key, rnd)

        # generate random numbers for a subset of quarks by including
        print("--------- now for the restricted set of particles by including ---------")

        mnprng.init_prngs(realization_id)    # reset the prngs

        random_for_quarks_subset_in = mnprng.random(
            "quarks", include_ids=quarks_subset)

        for key, rnd in zip(quarks_subset, random_for_quarks_subset_in):
            print(key, rnd)
        print("=======================================================================")


Mnprng_gen = named_prng.NamedPrng(mparticles, teefilename="tee.dat")
do_some_stuff(Mnprng_gen)

Mnprng_use = named_prng.NamedPrng(mparticles, sourcefilename="tee.dat")
do_some_stuff(Mnprng_use)

print("All the random numbers in the tee file:")
print(named_prng.NamedPrng.get_rnds_from_file("tee.dat"))
