"""named_prng.py
    This file contains the implementation of the
    named pseudo random number generator. Detailed documentation is available
    in the README.md file too."""

import sys
from typing import Dict, Iterable
import numpy


class NamedPrng:
    """The implementation of the named_prng. Stores multiple prng instances,
        and a seed-assignment logic for different particle types and IDs.

        Attributes
        -----------
        N_max: int
            Maximum number of particle types. Changing this value breaks
            realization-wise comparison possibility with older runs. The
            number of particle types must be smaller than this.
        _particles: Dict[str,Dict[str,int]]
            stores the name of different particle types as key,
            and a dict containing the the particles IDs as keys,
            and the order number of particles as the values.
        _ptype_ind: Dict[str,int]
            The order indices of the particle types.
        _teefile: str
        _engines: Dict[str, numpy.random.Generator]

        Methods
        -----------
        _seed_map(self, realization: int, ptype: str) -> int:
        init_prngs(self, realization_id: int):
        random(self, ptype: str, exclude_ids=None) -> numpy.ndarray:
        _tee(arr: numpy.ndarray) -> numpy.ndarray:
        """

    N_max = 100  # maximum number of particle types

    def __init__(self,
                 particles: Dict[str, Dict[str, int]],
                 teefile: str = None,
                 realization_id: int = None):

        self._particles = particles
        if len(particles) > self.N_max:
            sys.exit("The prng is fed with", len(particles), "number of types,",
                     "but only", self.N_max,
                     "different types are supported. Program terminates.")

        self._ptype_ind = {key: index for index, key in enumerate(particles)}
        self._teefile = teefile  # where to copy the generated prn-s
        self._engines = dict()   # the prng instances

        if realization_id is not None:
            self.init_prngs(realization_id)

    def _seed_map(self, realization: int, ptype: str) -> int:
        """Assigns a seed to a realization and particle type."""
        ptype_order = self._ptype_ind[ptype]
        seed = realization * self.N_max + ptype_order
        return seed

    def init_prngs(self, realization_id: int):
        """Initialize all prngs for a given realization ID
            and deletes all previous prng instances stored in the object."""

        self._engines = dict()
        for ptype in self._particles:
            self._engines[ptype] = numpy.random.Generator(
                numpy.random.MT19937(self._seed_map(realization_id, ptype)))

    def _exclude_ids(self, arr: numpy.ndarray, ptype: str, exclude_ids: Iterable):
        # indices to exclude
        exclude_ind = [self._particles[ptype][id] for id in exclude_ids]

        # delete the indices
        return numpy.delete(arr, exclude_ind)

    def random(self,
               ptype: str,
               exclude_ids: Iterable = None) -> numpy.ndarray:
        """Returns random numbers with uniform distribution on [0,1)
            for each particle with type ptype that does not have the
            ID listed in excludeIDs.

            Modifies the state of the prng instance associated with ptype
            and advances as many steps as many particles with ptype can
            be found, regardless the size of excludeIDs.

            excludeIDs tells the IDs for which particles the random numbers
            should be omitted from the return value. The order number of the
            random numbers are read from the value of the correcponding ID key
            of the particles."""
        ret = self._engines[ptype].random(len(self._particles[ptype]))
        ret = self._tee(ret)  # copy the random numbers if needed
        if exclude_ids is not None:
            ret = self._exclude_ids(ret, ptype, exclude_ids)
        return ret

    def normal(self, ptype: str,
               exclude_ids: Iterable = None,
               loc: float = 0,
               scale: float = 1) -> numpy.ndarray:
        """Returns random numbers with a normal (aka Gaussian) distribution
            with the properties passed for each of the particles ptype
            that does not have the ID listed in excludeIDs.

            Modifies the state of the prng instance associated with ptype
            and advances as many steps as many particles with ptype can
            be found, regardless the size of excludeIDs.

            excludeIDs tells the IDs for which particles the random numbers
            should be omitted from the return value. The order number of the
            random numbers are read from the value of the correcponding ID key
            of the particles."""
        ret = self._engines[ptype].normal(
            loc=loc, scale=scale, size=len(self._particles[ptype]))
        ret = self._tee(ret)  # copy the random numbers if needed
        if exclude_ids is not None:
            ret = self._exclude_ids(ret, ptype, exclude_ids)
        return ret

    def _tee(self, arr: numpy.ndarray) -> numpy.ndarray:
        """Copy the input array arr to _teefile if exists and return the array."""
        if self._teefile is not None:
            with open(self._teefile, "ab") as ofile:
                arr.tofile(ofile)
        return arr

    @classmethod
    def get_rnds_from_file(cls, teefile: str) -> numpy.ndarray:
        """Returns the value in a file where random numbers are copied."""
        with open(teefile, "rb") as ifile:
            ret = numpy.fromfile(ifile, dtype=numpy.float64)
        return ret
