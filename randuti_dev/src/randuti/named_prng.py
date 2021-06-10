"""Implementation of a pseudo random number generator container.

Detailed documentation is available in the README.md file.
Check out examples.py for examples!
"""

from enum import Enum, auto
import pickle
from typing import Dict, Iterable, Tuple, List, Union
import numpy

__version__ = "0.0.1"  # single source of truth


class FStrat(Enum):
    """Filtering strategy: include or exclude."""

    INC = auto()
    EXC = auto()


class Distr(Enum):
    """Distributions. UNI: uniform, STN: standard normal, STU: Student's t."""

    UNI = auto()
    STN = auto()
    STU = auto()


class NamedPrng:
    """Creates pseudo random numbers for entity types and purposes.

    Stores multiple prng instances and a seed-assignment logic for different
    entity types, purposes and realizations for Monte Carlo simulations.
    Entities can be physical, biological, financial or any type of entities and
    they are called particles throughout this class.

    Attributes
    ----------
    _seed_logic: Tuple[int,int]
        Consists of (_n_max, _n_ptl).
        Choose these values as large as you will later.
        Modifying these values may break seed order and therefore makes it
        impossible to compare the simulations with previous ones elementwise.

        - _n_max: int

          Maximum number of particle type - purpose
          combination. Changing this value breaks realization-wise comparison
          possibility with older runs. The number of particle type - purpose
          combination must be smaller than this. The seed value has a jump of
          size _n_max between different realizations.

        - _n_ptl: int

          Limit of particle types, the maximum number of particle types.
          Changing this value breaks realization-wise comparison
          possibility with older runs. From one purpose to another,
          for the same particle type, seed value jumps with this value.
          number of particle types * _n_ptl =< _n_max must hold.

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

    def __init__(self,
                 purposes: List[str],
                 particles: Union[str,
                                  Dict[str, Dict[str, int]],
                                  Dict[str, int]] = "dict_of_particles.pickle",
                 exim_settings: Tuple[str, str, bool] = (None, None, None),
                 seed_logic: Tuple[int, int] = (100, 10)
                 ) -> None:
        """Initialize the a class instance.

        Parameters
        ----------
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

        seed_logic: (_n_max, _n_ptl)
            Set the value of the maximum number of particle type - purpose
            combination and the value of the maximum number of particle types.
            Read more in the docstring of the class.

        Raises
        ------
        OSError
            If teefile cannot be opened for binary append or
            if sourcefilename cannot be opened for binary read.

        """
        self._seed_logic = seed_logic

        self._particles = _constr_particles(particles)
        self._purposes = purposes
        self._chk_seed_limits()

        teefilename = exim_settings[0]

        # where to copy the generated prn-s
        if teefilename is None or teefilename == "":
            self._teefile = None
        else:
            try:
                self._teefile = open(  # pylint: disable=consider-using-with
                    teefilename, "ab")

                # self._teefile will live after the try is executed
                # `with` would free up the resource
            except OSError as err:
                self._teefile = None
                note = ("Cannot initialize a NamedPrng instance,"
                        "because teefilename is set but OSError occurred"
                        "while opening the file for binary appending."
                        "No _teefile will be used."
                        "Error details:")
                raise OSError(note) from err

        sourcefilename = exim_settings[1]

        # from where to read the random numbers
        if sourcefilename is None or sourcefilename == "":
            self._sourcefile = None
        else:
            try:
                self._sourcefile = open(  # pylint: disable=consider-using-with
                    sourcefilename, "rb")
            except OSError as err:
                self._sourcefile = None
                note = ("Cannot initialize a NamedPrng instance,",
                        "because sourcefilename is set but OSError occurred",
                        "while opening the file for binary reading.",
                        "prng will be used instead of the _sourcefile.",
                        "Error details:")
                raise OSError(note) from err

        self._engines = dict()   # the prng instances

        if len(exim_settings) > 2 and exim_settings[2]:
            self._only_used = True
        else:
            self._only_used = False

    def _chk_seed_limits(self):
        """Check if unique seed for each ptype and purpose can be ensured."""
        if len(self._particles) > self._seed_logic[1] or \
           (len(self._purposes) * self._seed_logic[1]) > self._seed_logic[0]:

            note = ("The NamedPrng is fed with"
                    f" {len(self._particles)} number of types, and with"
                    f" {len(self._purposes)} number of purposes"
                    f" while _n_ptl = {self._seed_logic[1]}"
                    f" and _n_max = {self._seed_logic[0]}")
            raise ValueError(note)

    def _seed_map(self, realization: int, ptype: str, purpose: str) -> int:
        """Assign a seed to a realization and particle type."""
        ptype_order = list(self._particles.keys()).index(ptype)
        seed = (realization * self._seed_logic[0] +
                self._purposes.index(purpose) * self._seed_logic[1] +
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
        """Exclude the ids provided from the return array.

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
        """Generate random numbers using the initialized PRNGs.

        If _sourcefile is not set, random numbers with a
        type and properties passed as rnd_type, for each of the particles
        with type ptype and for the specified purpose that matches the
        filter criterion id_filter. The prngs must be initialized before
        this call by providing the realization_id at initialization time
        or by calling: func: `init_prngs`.

        Parameters
        ----------
        rnd_type : Union["Distr", Tuple["Distr", Tuple[float, float]]]
            The distribution type of the random numbers.

            - can be an enum Distr.UNI, which defines the uniform
                distribution on[0, 1)
            - or an enum Distr.STN, which defines a standard normal
                distribution with a mean 0 and std 1
            - or a tuple of  Distr.STN, (mean, std), e.g.
                (Distr.STN, (1, 3)) for a mean = 1 and std = 3.
        ptype : str
            For which particle type should the engine generate random numbers
        purpose : str
            For what purpose would you like to generate the random numbers.
            For different purpose, you get different set of random numbers,
            and prng instances are independent purpose-wise.
        id_filter : Tuple[Iterable,, optional
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
        -------
        numpy.ndarray:
            shape(number of realizations, number of particles)
            It has as many rows as many realization_id are in the range of
            [realization_id_start, realization_id_end),
            and it has as many columns as many particles with type ptype
            can be found, and it has dtype = numpy.float64.

        Notes
        -----
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
                raise NotImplementedError(f"Unsupported rnd_type {rnd_type}")
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
        """Generate random numbers for realizations x particles.

        If _sourcefile is not set, returns a 2D array of random numbers with
        a type and properties passed as rnd_type,
        for all realization ID and for each particle with type ptype for
        the specified purpose that matches the filter criterion id_filter.
        Automatically initialize the prngs. The behavior is identical to
        calling: func: `init_prngs` and: func: `generate` with the proper
        realization_id parameters.

        Parameters
        ----------
        rnd_type : Union["Distr", Tuple["Distr", Tuple[float, float]]]
            The distribution type of the random numbers.

            - can be an enum Distr.UNI, which defines the uniform
                distribution on[0, 1)
            - or an enum Distr.STN, which defines a standard normal
                distribution with a mean 0 and std 1
            - or a tuple of  Distr.STN, (mean, std), e.g.
                (Distr.STN, (1, 3)) for a mean = 1 and std = 3.
        seed_args: (ptype, purpose, iterable(realization ids))
            Values that affect the seeds. seed_args[2] can be an iterable
            range, like range(min_id, max_id) or a list of ids.
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
        -------
        numpy.ndarray:
            shape(number of realizations, number of particles)
            It has as many rows as many realization_id are in the range of
            [realization_id_start, realization_id_end),
            and it has as many columns as many particles with type ptype
            can be found, and it has dtype = numpy.float64.

        Notes
        -----
        Modifies the state of the prng instance associated with ptype
        and advances as many steps as many particles with ptype can
        be found, regardless the id_filter.

        If _sourcefile is set, reads in 64-bit floats from _sourcefile
        and does not modify the state of the prng instance.

        """
        ret = self.generate_r_t(rnd_type, seed_args, (0, 1), id_filter)
        return ret.reshape((ret.shape[0], ret.shape[2]))

    def generate_r_t(self,
                     rnd_type: Union["Distr",
                                     Tuple["Distr", Tuple[float, float]]],
                     seed_args: Tuple[str, str, Iterable],
                     time_range: Tuple[int, int],
                     id_filter: Tuple[Iterable, "FStrat"] = (None, None)
                     ) -> numpy.ndarray:
        """Generate random numbers for realizations X times x particles.

        If _sourcefile is not set, returns a 3D array of random numbers with
        a type and properties passed as rnd_type,
        for all realization ID, for each time step in the range time_range
        and for each particle with type ptype for
        the specified purpose that matches the filter criterion id_filter.
        Automatically initialize the prngs. The behavior is identical to
        calling: func: `init_prngs` and: func: `generate` with the proper
        realization_id parameters, repeated the `generate` enough times.

        Parameters
        ----------
        rnd_type : Union["Distr", Tuple["Distr", Tuple[float, float]]]
            The distribution type of the random numbers.

            - can be an enum Distr.UNI, which defines the uniform
                distribution on[0, 1)
            - or an enum Distr.STN, which defines a standard normal
                distribution with a mean 0 and std 1
            - or a tuple of  Distr.STN, (mean, std), e.g.
                (Distr.STN, (1, 3)) for a mean = 1 and std = 3.

        seed_args: (ptype, purpose, iterable(realization ids))
            Values that affect the seeds. seed_args[2] can be an iterable
            range, like range(min_id, max_id) or a list of ids.
        time_range: (t_start, t_end)
            For each int in the interval [t_start, t_end), random numbers
            are generated and stored in the returned array.
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
        -------
        numpy.ndarray:
            shape(number of realizations,
                  number of time steps,
                  number of particles)
            It has as many blocks as many realization_id are in the range of
            [realization_id_start, realization_id_end),
            as many rows as many times steps,
            and it has as many columns as many particles with type ptype
            can be found, and it has dtype = numpy.float64.

        Notes
        -----
        Modifies the state of the prng instance associated with ptype
        and advances as many steps as many particles with ptype can
        be found, regardless the id_filter.

        If _sourcefile is set, reads in 64-bit floats from _sourcefile
        and does not modify the state of the prng instance.

        """
        ptype, purpose, realizations = seed_args

        sbs_amount = self._getamount(ptype)  # the amount for the subset
        if id_filter[1] == FStrat.EXC:
            sbs_amount -= len(id_filter[0])
        elif id_filter[1] == FStrat.INC:
            sbs_amount = len(id_filter[0])

        ret = numpy.ndarray((len(realizations),
                             int(time_range[1])-int(time_range[0]),
                             sbs_amount),
                            dtype=numpy.float64)

        for r_count, realization_id in enumerate(realizations):
            self.init_prngs(realization_id, [ptype], [purpose])
            for time in range(0, int(time_range[1])):
                if time < int(time_range[0]):
                    # no need to filter, it takes time
                    ret_col = self.generate(rnd_type, ptype, purpose)
                    # no need to save the random numbers
                else:
                    t_count = time - time_range[0]
                    ret_col = self.generate(rnd_type,
                                            ptype,
                                            purpose,
                                            id_filter)
                    ret[r_count][t_count] = ret_col

        return ret

    def _tee(self, arr: numpy.ndarray) -> numpy.ndarray:
        """Copy arr to _teefile if exists and returns the array."""
        if self._teefile is not None:
            arr.tofile(self._teefile)
        return arr

    def _getamount(self, ptype: str) -> int:
        """Tell how many particles exist with in one ptype."""
        if isinstance(self._particles[ptype], int):
            return self._particles[ptype]
        return len(self._particles[ptype])

    def export_particles(self,
                         filename: str = "dict_of_particles.pickle") -> None:
        """Export the attribute _particles.

        _particles contain the particles, including the particle types,
        particle ID and their order number.
        Uses pickle to save the dictionary.
        """
        try:
            with open(filename, "wb") as ofile:
                pickle.dump(self._particles, ofile, 4)
        except OSError as err:
            note = ("Cannot export particles to"
                    + filename
                    + "because an OSError occurred:")
            raise OSError(note) from err

    def get_seed_logic(self) -> Tuple[int, int]:
        """Get the parameters defining the seed logic.

        Returns
        -------
        Tuple[int,int]:
            The (_n_max,_n_ptl) tuple telling the maximum number of
            particle type - purpose combination and the
            the maximum number of particle types.

        """
        return self._seed_logic


def _constr_particles(particles: Union[str,
                                       Dict[str, Dict[str, int]],
                                       Dict[str, int]]
                      ) -> Union[Dict[str, Dict[str, int]], Dict[str, int]]:
    """Construct particles from file or return the original argument."""
    # create from file
    if isinstance(particles, str):
        try:
            with open(particles, "rb") as ifile:
                particles = pickle.load(ifile)
        except OSError as err:
            note = ("Cannot open file "
                    + particles
                    + " for unpickling to load particles, "
                    + "because an OSError occurred:")
            raise OSError(note) from err

    # create from explicit dict
    return particles
