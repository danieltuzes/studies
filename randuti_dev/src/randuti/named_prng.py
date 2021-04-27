"""named_prng.py
This file contains the implementation of the
named pseudo random number generator. Detailed documentation is available
in the README.md file too. Check out examples.py for examples!
"""

import pickle
import sys
from typing import Dict, Iterable, Tuple, List, Union
from enum import Enum, auto
import numpy


class FStrat(Enum):
    """Filtering strategy: include or exclude."""
    INC = auto()
    EXC = auto()


class Distr(Enum):
    """Distributions. UNI: uniform, STN: standard normal"""
    UNI = auto()
    STN = auto()


class NamedPrng:
    """Creates pseudo random numbers for entity types and purposes.

    Stores multiple prng instances and a seed-assignment logic for different
    entity types, purposes and realizations for Monte Carlo simulations.
    Entities can be physical, biological, financial or any type of entities and
    they are called particles throughout this class.

    Attributes
    -----------
    N_max: int
        Maximum number of particle type - purpose
        combination. Changing this value breaks realization-wise comparison
        possibility with older runs. The number of particle type - purpose
        combination must be smaller than this. The seed value has a jump of
        size N_max between different realizations.
    N_ptl: int
        Limit of particle types, the maximum number of particle types.
        Changing this value breaks realization-wise comparison
        possibility with older runs. From one purpose to another,
        for the same particle type, seed value jumps with this value.
        number of particle types * N_ptl =< N_max must hold.
    _particles: Dict[str,Dict[str,int]]
        stores the name of different particle types as key,
        and a dict containing the the particles IDs as keys,
        and the order number of particles as the values.
    _purposes: List[str]
        Random numbers can be generated for a particle type for different
        purposes, and instead of keeping the same set of particles
        in the _particles dictionary with different particle type name for
        different purposes, it is more economical memory-wise to store
        he set of particles in _particles only once and store the
        set of possible purposes in a separate dictionary.
    _engines: Dict[str, numpy.random.Generator]
        The prng instances of type Mersenne Twister. Seeds are generated
        with _seed_map based on the realization, particle type and purpose.
    _teefile: BinaryIO
        If set, all random numbers generated are copied to this file.
        The file is opened with the initializator and closed once
        the instance goes out of scope.
    _sourcefile: BinaryIO
        If set, numbers from this file is read instead of generating them
        with the numpy random generator (e.g. Mersenne Twister). It is
        the user's responsibility to make sure the file has enough random
        numbers and that these numbers have the required properties.
    _only_used: bool
            Modifies which numbers are written into _teefile or
            read from _sourcefile.

            - If set to False or None, and _teefile is used, all random
            numbers that are generated, are written into the file, even
            if they are filtered out. When _sourcefile is used, even
            those,which need to be filtered out, will be still read in,
            and filtered out later.
            - If set to True, and _teefile is used, only non-filtered
            random numbers will be written into _teefile.
            If _sourcefile is used, only the necessary random numbers
            will be read in.
    """

    N_max = 100  # maximum number of particle type - purpose combinations
    N_ptl = 10   # maximum number of particle types

    def __init__(self,
                 purposes: List[str],
                 particles: Union[str,
                                  Dict[str, Dict[str, int]],
                                  Dict[str, int]] = "dict_of_particles.pickle",
                 realization_id: int = None,
                 exim_settings: Tuple[str, str, bool] = (None, None, None)
                 ) -> None:
        """Parameters
        -----------------
        purposes: List[str]
            prng instances are assigned to particle types and purposes, i.e.
            for a particle types, where you have a dict of particle IDs, you
            can have multiple prngs associated to it.
        particles: Union[str,
                         Dict[str, Dict[str, int]],
                         Dict[str, int]] = "dict_of_particles.pickle"
            Create the physical, financial or whatever related entities
            (particles) that will be assigned random numbers to,
            together with their entity type (particle type).

            - Dict[str, Dict[str, int]]: each key is the name of
              the particle type, each value as a dict stores the particles. A
              particles consists of a unique name (str) and an order number.
              Order numbers of the particles must be non-overlapping and
              gapless.
            - str: a file name that stores the particles. This file can be
              generated by `export_particles`.
            - Dict[str, int]: different particle types can be created with
              different amount provided as ints. The particles cannot
              be distinguished due to the lack of their unique name, therefore
              filtering cannot be applied. Particles can be exported with
              `export_particles`.
        realization_id: int = None
            If you set it, your prngs will be initialized immediately and
            you can generate random numbers for a single realization_id.
            If you generate random numbers in realization_id ranges,
            this argument is useless.
        exim_settings: (teefilename, sourcefilename, only_used)
            Random number export and import settings.

            - teefilename: the random numbers generated are copied to the
            file called teefilename if not an empty string
            - if sourcefilename is defined and not empty string, random numbers
            will be sequentially read from the file called sourcefilename
            - only_used modifies which numbers are written into _teefile or
            read from _sourcefile.

                - If set to False or None, and _teefile is used, all random
                numbers that are generated, are written into the file, even
                if they are filtered out. When _sourcefile is used, even
                those,which need to be filtered out, will be still read in,
                and filtered out later.
                - If set to True, and _teefile is used, only non-filtered
                random numbers will be written into _teefile.
                If _sourcefile is used, only the necessary random numbers
                will be read in.
        """

        self._particles = self._constr_particles(purposes, particles)
        self._purposes = purposes
        self._chk_seed_limits()

        teefilename = exim_settings[0]

        # where to copy the generated prn-s
        if teefilename is None or teefilename == "":
            self._teefile = None
        else:
            try:
                self._teefile = open(teefilename, "ab")
            except OSError as err:
                print("Cannot initialize a NamedPrng instance,",
                      "because teefilename is set but OSError occurred",
                      "while opening the file for binary appending.",
                      "No _teefile will be used.",
                      "Error details:", err)
                self._teefile = None

        sourcefilename = exim_settings[1]

        # from where to read the random numbers
        if sourcefilename is None or sourcefilename == "":
            self._sourcefile = None
        else:
            try:
                self._sourcefile = open(sourcefilename, "rb")
            except OSError as err:
                print("Cannot initialize a NamedPrng instance,",
                      "because sourcefilename is set but OSError occurred",
                      "while opening the file for binary reading.",
                      "prng will be used instead of the _sourcefile.",
                      "Error details:", err)
                self._sourcefile = None

        self._engines = dict()   # the prng instances

        if realization_id is not None:
            self.init_prngs(realization_id)

        if len(exim_settings) > 2 and exim_settings[2]:
            self._only_used = True
        else:
            self._only_used = False

    def _constr_particles(self,
                          purposes: List[str],
                          particles: Union[str,
                                           Dict[str, Dict[str, int]],
                                           Dict[str, int]]
                          ) -> Union[Dict[str, Dict[str, int]],
                                     Dict[str, int]]:

        # create from file
        if isinstance(particles, str):
            try:
                with open(particles, "rb") as ifile:
                    particles = pickle.load(ifile)
            except OSError as err:
                message = ("Cannot open file " + particles +
                           " for unpickling to load particles, " +
                           "because an OSError occurred: " + str(err))
                sys.exit(message)

        # create from explicit dict
        return particles

    def _chk_seed_limits(self):
        """Checks if unique seed for each ptype and purpose can be ensured."""
        if (len(self._particles) > self.N_ptl or
                len(self._purposes) * self.N_ptl > self.N_max):
            message = ("The NamedPrng is fed with " +
                       str(self._getamount(self._particles)) +
                       " number of types, and with " +
                       str(len(self._purposes)) +
                       " number of purposes, while N_ptl = " +
                       str(self.N_ptl) +
                       " and N_max = " +
                       str(self.N_max) + ". Program terminates.")
            sys.exit(message)

    def _seed_map(self, realization: int, ptype: str, purpose: str) -> int:
        """Assigns a seed to a realization and particle type."""
        ptype_order = list(self._particles.keys()).index(ptype)
        seed = (realization * self.N_max +
                self._purposes.index(purpose) * self.N_ptl +
                ptype_order)
        return seed

    def init_prngs(self,
                   realization_id: int,
                   ptypes: List[str] = None,
                   purposes: List[str] = None) -> None:
        """Initialize prngs for a given setup.

        Modifies the state of the prngs based on the realization_id,
        and for all or a given list of ptypes
        and for all or a given list of purposes.
        If the purposes list is None, all possibilities are initialized.
        """

        if ptypes is None:
            ptypes = self._particles.keys()
        if purposes is None:
            purposes = self._purposes

        for ptype in ptypes:
            self._engines[ptype] = dict()
            for purpose in purposes:
                self._engines[ptype][purpose] = numpy.random.Generator(
                    numpy.random.MT19937(self._seed_map(realization_id,
                                                        ptype,
                                                        purpose)))

    def _exclude_ids(self,
                     arr: numpy.ndarray,
                     ptype: str,
                     exclude_ids: Iterable) -> numpy.ndarray:
        """Excludes the ids provided from the return array.

        Useful if you need to remove some elements from the array.
        Random numbers have been already generated and although you may
        exclude them, they affected the state of the prng instance!
        """

        # indices to exclude
        exclude_ind = [self._particles[ptype][mid] for mid in exclude_ids]

        # delete the indices
        return numpy.delete(arr, exclude_ind)

    def _include_ids(self,
                     arr: numpy.ndarray,
                     ptype: str,
                     include_ids: Iterable) -> numpy.ndarray:
        """Include only the ids provided in the return array.

        Random numbers have been already generated,
        and memory is allocated for them once already. Although you may
        exclude them, they affected the state of the prng instance!
        """

        keep_indices = numpy.ndarray(len(include_ids), dtype=numpy.uint32)
        for index, mid in enumerate(include_ids):
            keep_indices[index] = self._particles[ptype][mid]
        return arr[keep_indices]

    def _filter_ids(self,
                    id_filter: Tuple[Iterable, "FStrat"],
                    arr: numpy.ndarray,
                    ptype: str) -> numpy.array:
        """Include or excludes particles from the array."""

        if (id_filter[1] == FStrat.EXC and
                not (self._sourcefile is not None and self._only_used)):
            return self._exclude_ids(arr, ptype, id_filter[0])
        if (id_filter[1] == FStrat.INC and
                not (self._sourcefile is not None and self._only_used)):
            return self._include_ids(arr, ptype, id_filter[0])

        return arr

    def generate(self,
                 rnd_type: Union["Distr", Tuple["Distr", Tuple[float, float]]],
                 ptype: str,
                 purpose: str,
                 id_filter: Tuple[Iterable, "FStrat"] = (None, None)
                 ) -> numpy.ndarray:
        """Generates random numbers using the initialized PRNGs.

        If _sourcefile is not set, random numbers with a
        type and properties passed as rnd_type, for each of the particles
        with type ptype and for the specified purpose that matches the
        filter criterion id_filter. The prngs must be initialized before
        this call by providing the realization_id at initialization time
        or by calling: func: `init_prngs`.

        Parameters
        ------------
        rnd_type: Distr.UNI | Distr.STN | (Distr.STN, (loc, scale))

            - which be an enum Distr.UNI, which defines the uniform
                distribution on[0, 1)
            - or an enum Distr.STN, which defines a standard normal
                distribution with a mean 0 and std 1
            - or a tuple of  Distr.STN, (mean, std), e.g.
                (Distr.STN, (1, 3)) for a mean = 1 and std = 3.

        ptype: str
            For which particle type should the engine generate random numbers

        purpose: str
            For what purpose would you like to generate the random numbers.
            For different purpose, you get different set of random numbers,
            and prng instances are independent purpose-wise.

        id_filter: (ids, filtering strategy)
            Filters random numbers for a specific set of particles if particles
            were created with unique name and order number.

            - if filtering strategy is FStrat.INC or None,
            ids tells which particles
            IDs should be used for the random number generation.
            - If filtering strategy is FStrat.EXC, then ids tells for which
            particles the random numbers should be omitted from the return
            value. The order number of the random numbers are read from the
            value of the correcponding ID key of the particles.

        Returns
        -----------------------
        numpy.ndarray:
            shape(number of realizations, number of particles)
            It has as many rows as many realization_id are in the range of
            [realization_id_start, realization_id_end),
            and it has as many columns as many particles with type ptype
            can be found, and it has dtype = numpy.float64.

        Modifies the state of the prng instance associated with ptype
        and advances as many steps as many particles with ptype can
        be found, regardless the id_filter.

        If _sourcefile is set, reads in 64-bit floats from _sourcefile
        and does not modify the state of the prng instance.
        """

        amount = self._getamount(ptype)
        ret = numpy.ndarray(amount, dtype=numpy.float64)

        if self._sourcefile is None:
            if isinstance(rnd_type, Distr) and rnd_type == Distr.UNI:
                ret = self._engines[ptype][purpose].random(size=amount)
            elif isinstance(rnd_type, Distr) and rnd_type == Distr.STN:
                ret = self._engines[ptype][purpose].normal(size=amount)
            elif isinstance(rnd_type, tuple) and rnd_type[0] == Distr.STN:
                ret = self._engines[ptype][purpose].normal(
                    loc=rnd_type[1][0], scale=rnd_type[1][1], size=amount)
            else:
                sys.exit("Unsupported rnd_type " + str(rnd_type))
        else:
            if self._only_used:
                if id_filter[1] == FStrat.EXC:
                    amount -= len(id_filter[0])
                elif id_filter[1] == FStrat.INC:
                    amount = len(id_filter[0])
            ret = numpy.fromfile(
                self._sourcefile, dtype=numpy.float64, count=amount)
        # random numbers are already read in or generated

        # if tee is requested for every random numbers
        if not self._only_used:
            ret = self._tee(ret)

        # filter them if requested and not read in with _only_used
        ret = self._filter_ids(id_filter, ret, ptype)

        # if tee is requested only for filtered numbers
        if self._only_used:
            ret = self._tee(ret)  # copy the random numbers if needed

        return ret

    def generate_it(self,
                    rnd_type: Union["Distr",
                                    Tuple["Distr", Tuple[float, float]]],
                    seed_args: Tuple[str, str, Iterable],
                    id_filter: Tuple[Iterable, "FStrat"] = (None, None)
                    ) -> numpy.ndarray:
        """Generates random numbers for a range or list of realizations.

        If _sourcefile is not set, returns a 2D array of random numbers with
        a type and properties passed as rnd_type,
        for all realization ID and for each particle with type ptype for
        the specified purpose that matches the filter criterion id_filter.
        Automatically initialize the prngs. The behavior is identical to
        calling: func: `init_prngs` and: func: `normal` with the proper
        realization_id parameters.

        Parameters
        -----------------------
        rnd_type: Distr.UNI | Distr.STN | (Distr.STN, (loc, scale))

            - which be an enum Distr.UNI, which defines the uniform
                distribution on[0, 1)
            - or an enum Distr.STN, which defines a standard normal
                distribution with a mean 0 and std 1
            - or a tuple of  Distr.STN, (mean, std), e.g.
                (Distr.STN, (1, 3)) for a mean = 1 and std = 3.

        seed_args: (ptype, purpose, iterable(realization ids))
            Values that affect the seeds.

            seed_args[2] can be an iterable range, like
            range(min_id, max_id) or a list of ids.

        id_filter: (ids, filtering strategy)
            Filters random numbers for a specific set of particles if particles
            were created with unique name and order number.

            - if filtering strategy is FStrat.INC or None,
            ids tells which particles
            IDs should be used for the random number generation.
            - If filtering strategy is FStrat.EXC, then ids tells for which
            particles the random numbers should be omitted from the return
            value. The order number of the random numbers are read from the
            value of the correcponding ID key of the particles.

        params: (loc, scale) = (0, 1)
            The parameters passed to numpy's normal function,
            i.e. the loc and scale paramters defining the
            mean and the standard deviation.

        Returns
        -----------------------
        numpy.ndarray:
            shape(number of realizations, number of particles)
            It has as many rows as many realization_id are in the range of
            [realization_id_start, realization_id_end),
            and it has as many columns as many particles with type ptype
            can be found, and it has dtype = numpy.float64.

        Modifies the state of the prng instance associated with ptype
        and advances as many steps as many particles with ptype can
        be found, regardless the id_filter.

        If _sourcefile is set, reads in 64-bit floats from _sourcefile
        and does not modify the state of the prng instance.
        """

        ptype = seed_args[0]
        purpose = seed_args[1]

        tot_amount = self._getamount(ptype)
        sbs_amount = tot_amount  # the amount for the subset
        if id_filter[1] == FStrat.EXC:
            sbs_amount -= len(id_filter[0])
        elif id_filter[1] == FStrat.INC:
            sbs_amount = len(id_filter[0])

        ret = numpy.ndarray((len(seed_args[2]), sbs_amount),
                            dtype=numpy.float64)

        for count, realization_id in enumerate(seed_args[2]):
            self.init_prngs(realization_id, [ptype], [purpose])
            ret_col = self.generate(rnd_type, ptype, purpose, id_filter)
            ret[count] = ret_col

        return ret

    def _tee(self, arr: numpy.ndarray) -> numpy.ndarray:
        """Copies arr to _teefile if exists and returns the array."""
        if self._teefile is not None:
            arr.tofile(self._teefile)
        return arr

    def _getamount(self, ptype: str) -> int:
        """Tells how many particles exist with in one ptype."""

        if isinstance(self._particles[ptype], int):
            return self._particles[ptype]
        return len(self._particles[ptype])

    def export_particles(self,
                         filename: str = "dict_of_particles.pickle") -> None:
        """Exports the attribute _particles.

        _particles contain the particles, including the particle types,
        particle ID and their order number.
        Uses pickle to save the dictionary.
        """

        try:
            with open(filename, "wb") as ofile:
                pickle.dump(self._particles, ofile, 4)
        except OSError as err:
            print("Cannot export particles to", filename,
                  "because an OSError occurred:", err)
