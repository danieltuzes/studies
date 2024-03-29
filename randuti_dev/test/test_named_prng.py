
"""test_named_prng.py
Tests the named_prng.py with pytest.
"""

import os
import filecmp
import numpy
import pytest
from randuti import NamedPrng, FStrat, Distr


quarks = {"up": 0, "down": 1, "charm": 2, "strange": 3, "top": 4, "bottom": 5}
atoms = {"H": 0, "He": 1, "Li": 2, "Be": 3}
barions = {"p": 0, "n": 1, "s0": 2, "s+": 3, "s-": 4, "xi0": 5, "xi-": 7}
mparticles = {"quarks": quarks,  # my distinguishable particles
              "atoms": atoms,
              "barions": barions}

uparticles = {"quarks": len(quarks),  # indistinguishable particles
              "atoms": len(atoms),
              "barions": len(barions)}

remove_quarks = {"charm", "strange"}
quarks_subset = quarks.copy()  # contains the subset of quarks
for rid in remove_quarks:
    quarks_subset.pop(rid)

mpurposes = ["random_walk", "fusion", "fission"]


def test_uniform() -> None:
    """Tests if uniform distribution is the same as before.

    Tests the class if it generates the same sequence of random numbers
    as it generated in a controlled environment long time ago.
    """

    for particles_type in [uparticles, mparticles]:
        mnprng_gen = NamedPrng(mpurposes, particles_type)
        mnprng_gen.init_prngs(0, ["quarks"], ["random_walk"])
        arr = mnprng_gen.generate(Distr.UNI, ["quarks", "random_walk"])

        orig_arr = numpy.array(
            [0.47932306384132817, 0.2864961735272108, 0.022216695186381585,
             0.5453879896772311, 0.0012294157898979918, 0.5108467779409858])

        assert pytest.approx(arr) == orig_arr


def test_normal() -> None:
    """Tests if normal distribution is the same as before.

    Tests the first few random numbers from the seed 0
    if matches a previously generated sequence for normal distribution
    with 3 different mean and std.
    """

    param_pairs = [(mean, std)
                   for mean in [-1, 0, 1] for std in [0.1, 1, 10]]

    for particle_type in [mparticles, uparticles]:
        ret_arrs = numpy.ndarray((9, 6), dtype=numpy.float64)

        for i, param_pair in enumerate(param_pairs):
            mnprng_gen = NamedPrng(mpurposes, particle_type)

            mnprng_gen.init_prngs(0, ["quarks"], ["random_walk"])

            ret_arrs[i] = mnprng_gen.generate((Distr.STN, param_pair),
                                              ["quarks", "random_walk"])

        orig_arr = numpy.array(
            [[-1.0625151089e+00, -1.0596945414e+00, -9.8609030314e-01,
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
    """Tests if same or diff. numbers are generated for same or diff. cases.

    Tests if for different realization_id and for different
    purpose you get different random numbers,
    but for the same realization_id and same purpose
    you get the same random numbers.
    """

    realization_ids = [1, 2]
    # pylint: disable=too-many-nested-blocks
    for particle_type in [mparticles, uparticles]:
        for mparticle in particle_type:
            for realization_id1 in realization_ids:
                for mpurpose1 in mpurposes:
                    for realization_id2 in realization_ids:
                        for mpurpose2 in mpurposes:
                            mnprng_gen1 = NamedPrng(
                                mpurposes, mparticles)
                            mnprng_gen2 = NamedPrng(
                                mpurposes, mparticles)
                            mnprng_gen1.init_prngs(realization_id1)
                            mnprng_gen2.init_prngs(realization_id2)
                            arr1 = mnprng_gen1.generate(
                                Distr.UNI, [mparticle, mpurpose1])
                            arr2 = mnprng_gen2.generate(
                                Distr.UNI, [mparticle, mpurpose2])

                            if (realization_id1 == realization_id2
                                    and mpurpose1 == mpurpose2):
                                assert pytest.approx(arr1) == arr2
                            else:
                                assert pytest.approx(arr1) != arr2
    # pylint: enable=too-many-nested-blocks


def test_subset_exc_inc() -> None:
    """Tests exclusion and inclusion filtering.

    Tests if for a subset of particles you get the proper subset of random
    numbers and whether you get the same random numbers by exclusion and
    inclusion if the class is initialized for a realization_id range.
    """

    mnprng_gen_full = NamedPrng(mpurposes, mparticles)
    mnprng_gen_sbs1 = NamedPrng(mpurposes, mparticles)
    mnprng_gen_sbs2 = NamedPrng(mpurposes, mparticles)
    seed_args = ("quarks", "random_walk", range(0, 2))
    arr_full = mnprng_gen_full.generate_it(Distr.UNI, seed_args)

    arr_sbs1 = mnprng_gen_sbs1.generate_it(
        Distr.UNI, seed_args, (remove_quarks, FStrat.EXC))
    arr_sbs2 = mnprng_gen_sbs2.generate_it(
        Distr.UNI, seed_args, (quarks_subset, FStrat.INC))

    # exclusion and inclusion is the same if the resulting set is the same
    assert arr_sbs1 == pytest.approx(arr_sbs2)

    # The subset random array of the original, full set's random array
    # is the same as
    # the full random array of the subset's random array.
    # Test it particle-wise (i.e. columnwise)
    for i, quark in enumerate(quarks_subset):
        assert arr_full[:, quarks_subset[quark]] == pytest.approx(
            arr_sbs1[:, i])


def test_same_case_after_pickle() -> None:
    """Tests if pickle works as expected.

    Exports particles and loads back, and tests if the generated
    random numbers for a subset is the same, for one
    realization_id for uniform and for a range for gaussian.
    """

    # uniform, single realization ID

    mnprng_save = NamedPrng(mpurposes, mparticles)
    mnprng_save.init_prngs(realizations=0)

    generate_arg = (Distr.UNI,  # distribution
                    ["quarks", mpurposes[0]],  # particle type and purpose
                    (remove_quarks, FStrat.EXC))  # filtering

    arr_save = mnprng_save.generate(*generate_arg)
    mparticles_fname = "test_same_case_after_pickle_random"
    mnprng_save.export_particles(mparticles_fname)
    del mnprng_save

    mnprng_load = NamedPrng(mpurposes, mparticles_fname)
    mnprng_load.init_prngs(realizations=0)

    arr_load = mnprng_load.generate(*generate_arg)
    del mnprng_load

    assert arr_save == pytest.approx(arr_load)

    if os.path.isfile(mparticles_fname):
        os.remove(mparticles_fname)

    #
    # gaussian, realization ID range

    seed_args = ["quarks", mpurposes[0], range(0, 2)]

    # distinguishable particles, filtering

    mnprng_save = NamedPrng(mpurposes, mparticles)
    mnprng_save.init_prngs(realizations=0)

    marr_save = mnprng_save.generate_it(
        (Distr.STN, (1, 3)),
        seed_args,
        (None, remove_quarks))
    mparticles_fname = "test_same_case_after_pickle_normal"
    mnprng_save.export_particles(mparticles_fname)
    del mnprng_save

    mnprng_load = NamedPrng(mpurposes, mparticles_fname)
    mnprng_load.init_prngs(realizations=0)

    marr_load = mnprng_load.generate_it(
        (Distr.STN, (1, 3)),
        seed_args,
        (None, remove_quarks))

    assert marr_save == pytest.approx(marr_load)

    if os.path.isfile(mparticles_fname):
        os.remove(mparticles_fname)

    # indistinguishable particles, no filtering

    unprng_save = NamedPrng(mpurposes, uparticles)
    unprng_save.init_prngs(realizations=0)

    uarr_save = unprng_save.generate_it((Distr.STN, (1, 3)), seed_args)
    uparticles_fname = "test_same_case_after_pickle_normal_indi"
    unprng_save.export_particles(uparticles_fname)
    del unprng_save

    unprng_load = NamedPrng(mpurposes, uparticles_fname)
    unprng_load.init_prngs(realizations=0)

    uarr_load = unprng_load.generate_it((Distr.STN, (1, 3)), seed_args)

    assert uarr_save == pytest.approx(uarr_load)

    if os.path.isfile(uparticles_fname):
        os.remove(uparticles_fname)


def test_teefile() -> None:
    """Checks if teeing and reading back works.

    Create a set of random numbers then reads back from the teefile and
    checks if it the same for a realization_id range using filters
    including or excluding.
    While reading back, copies the random numbers to another teefile
    and checks if the copied file is the same.
    """

    for only_used in [False, True]:
        for strategy in [FStrat.INC, FStrat.EXC]:
            tee_fname = "teefile_test_named_prng.dat"

            mnprng_save = NamedPrng(mpurposes, mparticles,
                                    exim_settings=(tee_fname, "", only_used))
            seed_args = ["quarks", mpurposes[0], range(0, 2)]
            arr_save_r = mnprng_save.generate_it(
                Distr.UNI,
                seed_args,
                id_filter=(remove_quarks, strategy))
            arr_save_n = mnprng_save.generate_it(
                (Distr.STN, (1, 3)),
                seed_args,
                id_filter=(remove_quarks, strategy))
            del mnprng_save

            mnprng_load = NamedPrng(mpurposes,
                                    mparticles,
                                    exim_settings=("B"+tee_fname,
                                                   tee_fname,
                                                   only_used))

            arr_load_r = mnprng_load.generate_it(
                Distr.UNI,
                seed_args,
                id_filter=(remove_quarks, strategy))
            arr_load_n = mnprng_load.generate_it(
                (Distr.STN, (1, 3)),
                seed_args,
                id_filter=(remove_quarks, strategy))

            assert filecmp.cmp(tee_fname, "B" + tee_fname)

            del mnprng_load
            os.remove(tee_fname)
            os.remove("B" + tee_fname)

            assert arr_load_r == pytest.approx(arr_save_r)
            assert arr_load_n == pytest.approx(arr_save_n)


def test_generate_it_it() -> None:
    """Tests the iterable feature of the generate_it function.

    Generates random numbers for the IDs 0, 2, 4, 7, 8, 9 using 3 ways:

    - 1 by 1 by initializing and generating realization-wise and then joining,
    - by a list containing the values
    - by joining range(0,6,2) and range(7,10).
    """

    for particle_type in [mparticles, uparticles]:
        ids = [0, 2, 4, 7, 8, 9]

        mnprng1by1 = NamedPrng(mpurposes, particle_type)
        mnprng_list = NamedPrng(mpurposes, particle_type)
        mnprng_rang = NamedPrng(mpurposes, particle_type)

        p_type = "quarks"
        purpose = mpurposes[0]  # random_walk

        ret1by1 = numpy.ndarray(
            shape=(len(p_type), len(ids)), dtype=numpy.float64)
        retrang = numpy.ndarray(
            shape=(len(p_type), len(ids)), dtype=numpy.float64)

        for count, realization_id in enumerate(ids):
            mnprng1by1.init_prngs(realization_id)
            ret_col = mnprng1by1.generate(Distr.STN, [p_type, purpose])
            ret1by1[count] = ret_col

        retlist = mnprng_list.generate_it(Distr.STN, (p_type, purpose, ids))
        retrang[0:3] = mnprng_rang.generate_it(
            Distr.STN, (p_type, purpose, range(0, 6, 2)))
        retrang[3:6] = mnprng_rang.generate_it(
            Distr.STN, (p_type, purpose, range(7, 10)))

        assert pytest.approx(ret1by1) == retlist
        assert pytest.approx(retrang) == retlist


def test_generate_r_t() -> None:
    """Test the generate_r_t function.

    For a subset of particles, create random numbers 2 times using the time
    range [1,3) for realizations 4 and 5, by including one half of the
    particles by including, then for the same half by excluding to see
    the same case and see if we get the same numbers.

    Then generate for time range [0,3) and manually exclude t=0 to see if
    non-0 time_start behaves well.
    """
    for particle_type in [mparticles, uparticles]:
        ids = [4, 5]
        mnprng = NamedPrng(mpurposes, particle_type)
        if particle_type == mparticles:
            id_filter = (remove_quarks, FStrat.EXC)
        else:
            id_filter = (None, None)
        arr1 = mnprng.generate_r_t(Distr.UNI,
                                   ("quarks", mpurposes[0], ids),
                                   (1, 3),
                                   id_filter=id_filter)

        if particle_type == mparticles:
            id_filter = (quarks_subset, FStrat.INC)
        else:
            id_filter = (None, None)
        arr2 = mnprng.generate_r_t(Distr.UNI,
                                   ("quarks", mpurposes[0], ids),
                                   (1, 3),
                                   id_filter=id_filter)

        assert pytest.approx(arr1) == arr2

        arr3 = mnprng.generate_r_t(Distr.UNI,
                                   ("quarks", mpurposes[0], ids),
                                   (0, 3),
                                   id_filter=id_filter)

        assert pytest.approx(arr3[:, 1:3]) == arr2


def test_bad_filenames() -> None:
    """Checks if exim settings exceptions are handled properly."""
    wrong_filename = "very/bad:filename?here*"
    with pytest.raises(OSError):
        mnprng = NamedPrng(mpurposes,
                           wrong_filename)

    with pytest.raises(OSError):
        mnprng = NamedPrng(mpurposes,
                           mparticles,
                           (wrong_filename, None, False))

    with pytest.raises(OSError):
        mnprng = NamedPrng(mpurposes,
                           mparticles,
                           (None, wrong_filename, False))

    with pytest.raises(OSError):
        mnprng = NamedPrng(mpurposes, mparticles)
        mnprng.export_particles(wrong_filename)


def test_violate_seed_limits() -> None:
    """Check for too many list x ptype combination."""
    mlist = [str(i) for i in range(0, 10**6)]
    with pytest.raises(ValueError):
        mnprng = NamedPrng(mlist,  # pylint: disable=unused-variable
                           mparticles)


def test_unsupported_rnd_type() -> None:
    """Test for not supported random number type."""
    with pytest.raises(NotImplementedError):
        mnprng = NamedPrng(mpurposes,  # pylint: disable=unused-variable
                           mparticles)
        mnprng.generate_it(Distr.STU,
                           ["quarks", mpurposes[0], range(0, 2)])


def test_generate_same_3d() -> None:
    """Initializes engines in 3D and generates prns.

    Engines are initialized for realizations, ptype and purpose,
    and generates random number for the same setup and checks if
    the corresponding values are the same and the unrelated values
    are different.
    """
    id_filter = (remove_quarks, FStrat.EXC)

    mnprng = NamedPrng(mpurposes, mparticles)
    reals = [0, 1, 2]
    mnprng.init_prngs(reals, purposes=["random_walk"])

    out1_t0 = mnprng.generate(Distr.UNI,
                              ["quarks", "random_walk", reals],
                              id_filter)
    out1_t1 = mnprng.generate(Distr.UNI,
                              ["quarks", "random_walk", reals],
                              id_filter)

    mnprng2 = NamedPrng(mpurposes, mparticles)
    mnprng2.init_prngs(reals, purposes=["random_walk"])

    out2_t0 = mnprng2.generate(Distr.UNI,
                               ["quarks", "random_walk", reals],
                               id_filter)
    out2_t1 = mnprng2.generate(Distr.UNI,
                               ["quarks", "random_walk", reals],
                               id_filter)

    reals3 = [3, 4, 5]
    mnprng3 = NamedPrng(mpurposes, mparticles)
    mnprng3.init_prngs(reals3, purposes=["random_walk"])

    out3_t0 = mnprng3.generate(Distr.UNI,
                               ["quarks", "random_walk", reals3],
                               id_filter)
    out3_t1 = mnprng3.generate(Distr.UNI,
                               ["quarks", "random_walk", reals3],
                               id_filter)

    assert (out1_t0 == out2_t0).all()
    assert (out1_t1 == out2_t1).all()
    assert (out3_t0 != out2_t0).any()
    assert (out3_t1 != out2_t1).any()
    assert (out3_t0 != out2_t1).any()


def test_compare_generate_3d_generate_it() -> None:
    """Compares the 3d init+gen method with the generate_it."""
    id_filter = (remove_quarks, FStrat.EXC)
    real_range = [0, 2]

    mnprng1 = NamedPrng(mpurposes, mparticles)
    mnprng1.init_prngs(real_range, purposes=["random_walk"])

    mnprng2 = NamedPrng(mpurposes, mparticles)
    out2 = mnprng2.generate_r_t(Distr.UNI,
                                ["quarks", "random_walk", real_range],
                                (0, 3),
                                id_filter)

    for time in range(0, 3):
        out1 = mnprng1.generate(Distr.UNI,
                                ["quarks", "random_walk", real_range],
                                id_filter)
        for i, _ in enumerate(real_range):
            assert (out1[i] == out2[i][time]).all()


def test_erase() -> None:
    """Test if deleting the engines removes the engines."""
    mnprng = NamedPrng(mpurposes, mparticles)
    mnprng.init_prngs(0)
    mnprng.generate(Distr.UNI, ("quarks", "random_walk"))

    mnprng.clear_prngs()
    assert isinstance(
        mnprng._engines, dict)  # pylint: disable=protected-access
    assert len(mnprng._engines) == 0  # pylint: disable=protected-access

    with pytest.raises(IndexError):
        mnprng.generate(Distr.UNI, ("quarks", "random_walk"))


def test_seed_logic_get() -> None:
    """Test if the two seed logic values are returned."""
    mnprng = NamedPrng(mpurposes, mparticles)
    assert (100, 10, 0, 0) == mnprng.get_seed_logic()

    mnprng = NamedPrng(mpurposes, mparticles, seed_logic=(800, 3, 1, 2))
    assert (800, 3, 1, 2) == mnprng.get_seed_logic()


def test_seed_shift() -> None:
    """Test if the realization shift works.

    I don't know why anybody would use the seed shift,
    it has no physical meaning."""
    mnprng_1 = NamedPrng(mpurposes, mparticles)
    mnprng_1.init_prngs(10)
    result_1 = mnprng_1.generate(Distr.UNI, ("quarks", "random_walk"))

    mnprng_2 = NamedPrng(mpurposes, mparticles, seed_logic=(100, 10, 0, 10))
    mnprng_2.init_prngs(0)
    result_2 = mnprng_2.generate(Distr.UNI, ("quarks", "random_walk"))

    assert numpy.equal(result_1, result_2).all()


def test_range_inited_with_single() -> None:
    """What happens if multiple realizations are initialized but only for a single one is used."""
    mnprng = NamedPrng(mpurposes, mparticles)
    mnprng.init_prngs([1, 2], ["quarks"], ["random_walk"])
    rnds12_1 = mnprng.generate(Distr.UNI, ["quarks", "random_walk", 1])
    rnds12_2 = mnprng.generate(Distr.UNI, ["quarks", "random_walk", 2])

    mnprng.init_prngs(1, ["quarks"], ["random_walk"])
    rnds1_1 = mnprng.generate(Distr.UNI, ["quarks", "random_walk", 1])

    mnprng.init_prngs(2, ["quarks"], ["random_walk"])
    rnds2_2 = mnprng.generate(Distr.UNI, ["quarks", "random_walk", 2])

    assert numpy.equal(rnds12_1, rnds1_1).all()
    assert numpy.equal(rnds12_2, rnds2_2).all()


def test_trigger_impl_real_warning() -> None:
    """Trigger the logging warning for implicit realization values."""
    mnprng = NamedPrng(mpurposes, mparticles)
    mnprng.init_prngs([1, 2], ["quarks"], ["random_walk"])
    prng_explicit = mnprng.generate(Distr.UNI, ["quarks", "random_walk", 1])

    mnprng.init_prngs([1, 2], ["quarks"], ["random_walk"])
    prng_implicit = mnprng.generate(Distr.UNI, ["quarks", "random_walk"])

    assert numpy.equal(prng_explicit, prng_implicit).all()


def test_trigger_bare_str_ptype_purpose_warning() -> None:
    """Trigger the logging warning for implicit realization values."""
    mnprng = NamedPrng(mpurposes, mparticles)
    mnprng.init_prngs([1, 2], "quarks", "random_walk")
    prng_explicit = mnprng.generate(Distr.UNI, ["quarks", "random_walk", 1])

    mnprng.init_prngs([1, 2], ["quarks"], ["random_walk"])
    prng_implicit = mnprng.generate(Distr.UNI, ["quarks", "random_walk", 1])

    assert numpy.equal(prng_explicit, prng_implicit).all()


def test_independent_ptype_real_time() -> None:
    """Test if prngs are indeed generation order-independent."""
    mnprng = NamedPrng(mpurposes, mparticles)
    mnprng.init_prngs([1, 2], ["quarks", "atoms"], "random_walk")
    a_1 = mnprng.generate(Distr.UNI, ["quarks", "random_walk", 1])
    a_2 = mnprng.generate(Distr.UNI, ["quarks", "random_walk", 2])
    a_3 = mnprng.generate(Distr.UNI, ["atoms", "random_walk", 1])
    a_4 = mnprng.generate(Distr.UNI, ["atoms", "random_walk", 2])
    a_1_2 = mnprng.generate(Distr.UNI, ["quarks", "random_walk", 1])

    mnprng.init_prngs([1, 2], ["quarks", "atoms"], "random_walk")
    b_1 = mnprng.generate(Distr.UNI, ["atoms", "random_walk", 2])
    b_2 = mnprng.generate(Distr.UNI, ["quarks", "random_walk", 1])
    b_3 = mnprng.generate(Distr.UNI, ["atoms", "random_walk", 1])
    b_4 = mnprng.generate(Distr.UNI, ["quarks", "random_walk", 2])
    b_2_2 = mnprng.generate(Distr.UNI, ["quarks", "random_walk", 1])

    assert numpy.equal(a_1, b_2).all()
    assert numpy.equal(a_2, b_4).all()
    assert numpy.equal(a_3, b_3).all()
    assert numpy.equal(a_4, b_1).all()
    assert numpy.equal(a_1_2, b_2_2).all()
