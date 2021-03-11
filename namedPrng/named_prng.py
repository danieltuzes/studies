"""named_prng.py
    This file contains the implementation of the
    named pseudo random number generator. Detailed documentation is available
    in the README.md file too. Check out examples.py for examples!"""

import sys
from typing import Dict, Iterable, Tuple
import numpy


class NamedPrng:
    """The implementation of the named_prng. Stores multiple prng instances,
        and a seed-assignment logic for different
        particle types, purposes and realizations.

        Attributes
        -----------
        N_max: int
            Maximum number of maximum number of particle type - purpose
            combination. Changing this value breaks realization-wise comparison
            possibility with older runs. The number of particle type - purpose
            combination must be smaller than this.
        _particles: Dict[str,Dict[str,int]]
            stores the name of different particle types as key,
            and a dict containing the the particles IDs as keys,
            and the order number of particles as the values.
        _ptype_ind: Dict[str,int]
            The order indices of the particle types.
        _purposes: Dict[str,int]
            Random numbers can be generated for a particle type for different
            purposes, and instead of keeping the same set of particles
            in the _particles dictionary with different particle type name for
            different purposes, it is more econonomical memory-wise to store
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
        Methods
        -----------
        _seed_map(self, realization: int, ptype: str) -> int:
        init_prngs(self, realization_id: int):
        random(self, ptype: str, exclude_ids=None) -> numpy.ndarray:
        _tee(arr: numpy.ndarray) -> numpy.ndarray:
        """

    N_max = 100  # maximum number of particle type - purpose combination

    def __init__(self,
                 particles: Dict[str, Dict[str, int]],
                 purposes: Dict[str, int],
                 realization_id: int = None,
                 filenames: Tuple[str, str] = (None, None)):
        """filenames is the tuple of (teefilename, sourcefilename)"""
        self._particles = particles
        if len(particles) * len(purposes) > self.N_max:  # sanity check
            sys.exit("The prng is fed with", len(particles), "number of types,",
                     "but only", self.N_max,
                     "different types are supported. Program terminates.")

        self._ptype_ind = {key: index for index, key in enumerate(particles)}

        self._purposes = purposes

        teefilename = filenames[0]
        if teefilename is None:  # where to copy the generated prn-s
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

        sourcefilename = filenames[1]
        if sourcefilename is None:  # from where to read the random numbers
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

    def _seed_map(self, realization: int, ptype: str, purpose: str) -> int:
        """Assigns a seed to a realization and particle type."""
        ptype_order = self._ptype_ind[ptype]
        seed = realization * self.N_max + \
            self._purposes[purpose] * len(self._ptype_ind) + \
            ptype_order
        return seed

    def init_prngs(self, realization_id: int):
        """Initialize all prngs for a given realization ID,
            for all the possible particle type and purpose combination,
            and deletes all previous prng instances stored in the object."""

        self._engines = dict()
        for ptype in self._particles:
            self._engines[ptype] = dict()
            for purpose in self._purposes:
                self._engines[ptype][purpose] = numpy.random.Generator(
                    numpy.random.MT19937(self._seed_map(realization_id, ptype, purpose)))

    def _exclude_ids(self, arr: numpy.ndarray, ptype: str, exclude_ids: Iterable):
        """Excludes the ids provided from the return array.
            Useful if you need to remove some elements from the array.
            Random numbers have been already generated and although you may
            exclude them, they affected the state of the prng instance!"""
        # indices to exclude
        exclude_ind = [self._particles[ptype][mid] for mid in exclude_ids]

        # delete the indices
        return numpy.delete(arr, exclude_ind)

    def _include_ids(self, arr: numpy.ndarray, ptype: str, include_ids: Iterable):
        """Include only the ids provided in the return array.
            Random numbers have been already generated,
            and memory is allocated for them once already. Although you may
            exclude them, they affected the state of the prng instance!"""
        keep_or_del = numpy.ones(arr.size, dtype=bool)
        for mid in include_ids:
            keep_or_del[self._particles[ptype][mid]] = False
        return numpy.delete(arr, keep_or_del)

    def random(self,
               ptype: str,
               purpose: str,
               id_filter: Tuple[Iterable, Iterable] = (None, None)) -> numpy.ndarray:
        """If _sourcefile is not set,
            returns random numbers with uniform distribution on[0, 1)
            for each particle with type ptype for the specified purpose
            that matches the filter criterion id_filter.

            id_filter is a tuple of (exclude_ids, include_ids).

            Modifies the state of the prng instance associated with ptype
            and advances as many steps as many particles with ptype can
            be found, regardless the id_filter.

            excludeIDs tells the IDs for which particles the random numbers
            should be omitted from the return value. The order number of the
            random numbers are read from the value of the correcponding ID key
            of the particles.

            include_ids tells which particles IDs should be used for the random
            number generation. Effective only if exclude_ids is None.

            If _sourcefile is set, reads in 64-bit floats from _sourcefile
            and does not modify the state of the prng instance."""

        amount = len(self._particles[ptype])
        ret = numpy.ndarray(amount, dtype=numpy.float64)

        if self._sourcefile is None:
            ret = self._engines[ptype][purpose].random(amount)
            ret = self._tee(ret)  # copy the random numbers if needed
        else:
            ret = numpy.fromfile(
                self._sourcefile, dtype=numpy.float64, count=amount)

        exclude_ids = id_filter[0]
        include_ids = id_filter[1]
        if exclude_ids is not None:
            ret = self._exclude_ids(ret, ptype, exclude_ids)
        elif include_ids is not None:
            ret = self._include_ids(ret, ptype, include_ids)
        return ret

    def normal(self,
               ptype: str,
               purpose: str,
               id_filter: Tuple[Iterable, Iterable] = (None, None),
               params: Tuple[float, float] = (0, 1)) -> numpy.ndarray:
        """If _sourcefile is not set,
            returns random numbers with a normal(aka Gaussian) distribution
            with the properties passed for each of the particles ptype
            and for the specified purpose
            that matches the filter criterion id_filter.

            id_filter is a tuple of (exclude_ids, include_ids).

            Modifies the state of the prng instance associated with ptype
            and advances as many steps as many particles with ptype can
            be found, regardless the id_filter.

            If _sourcefile is set, reads in 64-bit floats from _sourcefile
            and does not modify the state of the prng instance.

            Parameters
            ------------
            ptype: str
                For which particle should the engine generate random numbers
            purpose: str
                For what purpose would you like to generate the random numbers.
                For different purpose, you get different set of random numbers,
                and prng instances are independent purpose-wise.
            filter: Tuple[Iterable, Iterable] = (None, None),
                Filters the output based on exclusion xor include only method.

                - exclude_ids: Iterable = None
                    Tells the IDs for which particles the random numbers
                    should be omitted from the return value. The order number of the
                    random numbers are read from the value of the correcponding ID key
                    of the particles.
                - include_ids: Iterable = None,
                    Which particles IDs should be
                    used for the random number generation.
                    Effective only if exclude_ids is None.

            params: Tuple[float,float] = (0,1)
                The parameters passed to numpy's normal function,
                i.e. the loc and scale paramters defining the
                mean and the standard deviation."""

        amount = len(self._particles[ptype])
        ret = numpy.ndarray(amount, dtype=numpy.float64)

        if self._sourcefile is None:
            ret = self._engines[ptype][purpose].normal(
                loc=params[0], scale=params[1], size=amount)
            ret = self._tee(ret)  # copy the random numbers if needed
        else:
            ret = numpy.fromfile(
                self._sourcefile, dtype=numpy.float64, count=amount)

        exclude_ids = id_filter[0]
        include_ids = id_filter[1]
        if exclude_ids is not None:
            ret = self._exclude_ids(ret, ptype, exclude_ids)
        elif include_ids is not None:
            ret = self._include_ids(ret, ptype, include_ids)
        return ret

    def _tee(self, arr: numpy.ndarray) -> numpy.ndarray:
        """Copy the input array arr to _teefile if exists and return the array."""
        if self._teefile is not None:
            arr.tofile(self._teefile)
        return arr

    @classmethod
    def get_rnds_from_file(cls, teefilename: str) -> numpy.ndarray:
        """Returns 64-bit float values from a file
        where random numbers are supposed to be stored."""
        try:
            with open(teefilename, "rb") as ifile:
                ret = numpy.fromfile(ifile, dtype=numpy.float64)
                return ret
        except OSError as err:
            print("Cannot show the random numbers from the file,",
                  "because an OSError occurred:", err)
            return numpy.ndarray(0)
