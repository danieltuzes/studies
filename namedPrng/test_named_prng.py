"""test_named_prng.py
    Tests the named_prng.py with pytest."""

import os
from _pytest.python_api import approx
import numpy
import pytest
import named_prng

quarks = {"up": 0, "down": 1, "charm": 2, "strange": 3, "top": 4, "bottom": 5}
atoms = {"H": 0, "He": 1, "Li": 2, "Be": 3}
barions = {"p": 0, "n": 1, "s0": 2, "s+": 3, "s-": 4, "xi0": 5, "xi-": 7}
mparticles = {"quarks": quarks, "atoms": atoms, "barions": barions}

remove_quarks = {"charm", "strange"}
quarks_subset = quarks.copy()  # contains the subset of quarks
for rid in remove_quarks:
    quarks_subset.pop(rid)

mpurposes = ["random_walk", "fusion", "fission"]


def test_uniform() -> None:
    """Tests the class if it generates the same sequence of random numbers
        as it generated in a controlled environment long time ago."""

    mnprng_gen = named_prng.NamedPrng(mpurposes, mparticles)
    mnprng_gen.init_prngs(0, ["quarks"], ["random_walk"])
    arr = mnprng_gen.random("quarks", "random_walk")
    orig_arr = numpy.array(
        [0.47932306384132817, 0.2864961735272108, 0.022216695186381585,
         0.5453879896772311, 0.0012294157898979918, 0.5108467779409858])

    assert pytest.approx(arr) == orig_arr


def test_normal() -> None:
    """Tests the first few random numbers from the seed 0
        if matches a previously generated sequence for normal distribution
        with 3 different mean and std."""

    param_pairs = [(mean, std) for mean in [-1, 0, 1] for std in [0.1, 1, 10]]

    ret_arrs = numpy.ndarray((9, 6), dtype=numpy.float64)
    for i, param_pair in enumerate(param_pairs):
        mnprng_gen = named_prng.NamedPrng(mpurposes, mparticles)
        mnprng_gen.init_prngs(0, ["quarks"], ["random_walk"])
        ret_arrs[i] = mnprng_gen.normal(
            "quarks", "random_walk", params=param_pair)
    orig_arr = numpy.array([[-1.0625151089e+00, -1.0596945414e+00, -9.8609030314e-01,
                             -1.0276130879e+00, -9.9928650089e-01, -1.0060799379e+00],
                            [-1.6251510894e+00, -1.5969454139e+00, -8.6090303136e-01,
                             -1.2761308794e+00, -9.9286500888e-01, -1.0607993786e+00],
                            [-7.2515108944e+00, -6.9694541388e+00,  3.9096968635e-01,
                             -3.7613087942e+00, -9.2865008877e-01, -1.6079937856e+00],
                            [-6.2515108944e-02, -5.9694541388e-02,  1.3909696864e-02,
                             -2.7613087942e-02,  7.1349911225e-04, -6.0799378562e-03],
                            [-6.2515108944e-01, -5.9694541388e-01,  1.3909696864e-01,
                             -2.7613087942e-01,  7.1349911225e-03, -6.0799378562e-02],
                            [-6.2515108944e+00, -5.9694541388e+00,  1.3909696864e+00,
                             -2.7613087942e+00,  7.1349911225e-02, -6.0799378562e-01],
                            [9.3748489106e-01,  9.4030545861e-01,  1.0139096969e+00,
                             9.7238691206e-01,  1.0007134991e+00,  9.9392006214e-01],
                            [3.7484891056e-01,  4.0305458612e-01,  1.1390969686e+00,
                             7.2386912058e-01,  1.0071349911e+00,  9.3920062144e-01],
                            [-5.2515108944e+00, -4.9694541388e+00,  2.3909696864e+00,
                             -1.7613087942e+00,  1.0713499112e+00,  3.9200621438e-01]])

    assert pytest.approx(ret_arrs) == orig_arr


def test_diff_or_identical_cases() -> None:
    """Tests if for different realization_id and for different
        purpose you get different random numbers,
        but for the same realization_id and same purpose
        you get the same random numbers."""
    realization_ids = [1, 2]
    for mparticle in mparticles:  # pylint: disable= too-many-nested-blocks
        for realization_id1 in realization_ids:
            for mpurpose1 in mpurposes:
                for realization_id2 in realization_ids:
                    for mpurpose2 in mpurposes:
                        mnprng_gen1 = named_prng.NamedPrng(
                            mpurposes, mparticles)
                        mnprng_gen2 = named_prng.NamedPrng(
                            mpurposes, mparticles)
                        mnprng_gen1.init_prngs(realization_id1)
                        mnprng_gen2.init_prngs(realization_id2)
                        arr1 = mnprng_gen1.random(mparticle, mpurpose1)
                        arr2 = mnprng_gen2.random(mparticle, mpurpose2)

                        if realization_id1 == realization_id2 and mpurpose1 == mpurpose2:
                            assert approx(arr1) == arr2
                        else:
                            assert approx(arr1) != arr2


def test_subset_exc_inc() -> None:
    """Tests if for a subset of particles you get the proper subset of random
        numbers and whether you get the same random numbers by exclusion and
        inclusion if the class is initialized for a realization_id range."""
    mnprng_gen_full = named_prng.NamedPrng(mpurposes, mparticles)
    mnprng_gen_sbs1 = named_prng.NamedPrng(mpurposes, mparticles)
    mnprng_gen_sbs2 = named_prng.NamedPrng(mpurposes, mparticles)
    seed_args = ("quarks", "random_walk", (0, 2))
    arr_full = mnprng_gen_full.generate_r("random", seed_args)

    arr_sbs1 = mnprng_gen_sbs1.generate_r(
        "random", seed_args, (remove_quarks, None))
    arr_sbs2 = mnprng_gen_sbs2.generate_r(
        "random", seed_args, (None, quarks_subset))

    # exclusion and inclusion is the same if the resulting set is the same
    assert arr_sbs1 == approx(arr_sbs2)

    # The subset random array of the original, full set's random array
    # is the same as
    # the full random array of the subset's random array.
    # Test it particle-wise (i.e. columnwise)
    for i, quark in enumerate(quarks_subset):
        assert arr_full[:, quarks_subset[quark]] == approx(
            arr_sbs1[:, i])


def test_same_case_after_pickle() -> None:
    """Exports particles and loads back, and tests if the generated
        random numbers for a subset is the same, for one
        realization_id for uniform and for a range for gaussian."""
    mnprng_save = named_prng.NamedPrng(mpurposes, mparticles, realization_id=0)
    arr_save = mnprng_save.random(
        "quarks", mpurposes[0], (None, remove_quarks))
    mparticles_fname = "test_same_case_after_pickle_random"
    mnprng_save.export_particles(mparticles_fname)
    del mnprng_save

    mnprng_load = named_prng.NamedPrng(
        mpurposes, mparticles_fname, realization_id=0)
    arr_load = mnprng_load.random(
        "quarks", mpurposes[0], (None, remove_quarks))
    del mnprng_load

    assert arr_save == approx(arr_load)

    if os.path.isfile(mparticles_fname):
        os.remove(mparticles_fname)

    mnprng_save = named_prng.NamedPrng(mpurposes, mparticles, realization_id=0)
    arr_save = mnprng_save.generate_r(
        ("normal", (1, 3)),
        ["quarks", mpurposes[0], (0, 2)],
        (None, remove_quarks))
    mparticles_fname = "test_same_case_after_pickle_normal"
    mnprng_save.export_particles(mparticles_fname)
    del mnprng_save

    mnprng_load = named_prng.NamedPrng(
        mpurposes, mparticles_fname, realization_id=0)
    arr_load = mnprng_load.generate_r(
        ("normal", (1, 3)),
        ["quarks", mpurposes[0], (0, 2)],
        (None, remove_quarks))

    assert arr_save == approx(arr_load)

    if os.path.isfile(mparticles_fname):
        os.remove(mparticles_fname)


def test_teefile() -> None:
    """Create a set of random numbers then reads back from the teefile and
        checks if it the same for a realization_id range using filters."""
    tee_fname = "teefile_test_named_prng.dat"
    if os.path.isfile(tee_fname):
        os.remove(tee_fname)

    mnprng_save = named_prng.NamedPrng(mpurposes, mparticles,
                                       filenames=(tee_fname, None))

    arr_save_r = mnprng_save.generate_r(
        "random",
        ["quarks", "random_walk", (0, 2)],
        id_filter=(remove_quarks, None))
    arr_save_n = mnprng_save.generate_r(
        ("normal", (1, 3)),
        ["quarks", "random_walk", (0, 2)],
        id_filter=(remove_quarks, None))
    del mnprng_save

    mnprng_load = named_prng.NamedPrng(mpurposes, mparticles,
                                       filenames=(None, tee_fname))

    arr_load_r = mnprng_load.generate_r(
        "random",
        ["quarks", "random_walk", (0, 2)],
        id_filter=(remove_quarks, None))
    arr_load_n = mnprng_load.generate_r(
        ("normal", (1, 3)),
        ["quarks", "random_walk", (0, 2)],
        id_filter=(remove_quarks, None))

    assert arr_load_r == approx(arr_save_r)
    assert arr_load_n == approx(arr_save_n)

    del mnprng_load
    if os.path.isfile(tee_fname):
        os.remove(tee_fname)
