"""examples.py
    Here we call named_prng and test it with some cases."""

import named_prng

quarks = {"up": 0, "down": 1, "charm": 2, "strange": 3, "top": 4, "bottom": 5}
atoms = {"H": 0, "He": 1, "Li": 2, "Be": 3}

mparticles = {"quarks": quarks, "atoms": atoms}

Mnprng = named_prng.NamedPrng(mparticles)

for realization_id in range(3):
    print("realization_id = ", realization_id)
    # generate random numbers for quarks
    Mnprng.init_prngs(realization_id)
    random_for_quarks = Mnprng.normal("quarks")

    for key, rnd in zip(quarks, random_for_quarks):
        print(key, rnd)

    # generate random numbers for a subset of quarks
    remove_IDs = {"charm", "bottom"}

    quarks_constr = quarks.copy()  # contains the subset of quarks, just for printing
    for ID in remove_IDs:
        quarks_constr.pop(ID)
    print("------------------ now for the restricted set of particles ------------------")

    Mnprng.init_prngs(realization_id)    # reset the prngs

    random_for_quarks_constr = Mnprng.normal("quarks", exclude_ids=remove_IDs)

    for key, rnd in zip(quarks_constr, random_for_quarks_constr):
        print(key, rnd)
    print("=============================================================================")
