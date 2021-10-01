# randuti

Random number utilities package that contains the `named_prng` module only.

- [randuti](#randuti)
  - [Usage](#usage)
  - [The structure design of the Monte Carlo simulation](#the-structure-design-of-the-monte-carlo-simulation)
  - [Implementation of the prng container](#implementation-of-the-prng-container)
    - [The dictionary of dictionary containing the particle IDs](#the-dictionary-of-dictionary-containing-the-particle-ids)
    - [tee: copy the stream of random numbers to a file](#tee-copy-the-stream-of-random-numbers-to-a-file)

Many Monte Carlo simulations share similar patterns in their design. Although one can assign pseudo random numbers (prns) from arbitrarily initialized and used prn generators (prngs) to the different realizations, entities and to their different properties, to be efficient with the prn generation and be sparing with the seeds (also to save initialization time), some good design ideas need to be followed. This library offers one possibility that is believed to help to achieve these goals.

## Usage

This library implements a python API interface only, i.e. it has to be imported into python and the relevant function calls have to be called. The python source code contains detailed docstrings from which a documentation with sphinx is generated. Please browse the documentation for further details.

## The structure design of the Monte Carlo simulation

The investigated system is started from an initial state and developed in time according to the rules of the system, which can be nondeterministic, e.g. random-walk. At one or more later points, some properties $A_1, A_2, \ldots, A_n$ of the system is analyzed and exported. The behavior of the system from the beginning till the last investigated point is called a realization. In a new realization, the system is set back to the initial condition, which can be the same or different than the previous initial state, and the system is evolved again, properties are analyzed and exported if necessary. After performing $N$ realizations, the statistical properties of $A_i$ are analyzed. The different realizations are independent from each other and one CPU core should be responsible to execute only 1 at a time.

Depending on the nature of the initial conditions, evolving equations and properties investigated, according to the design principle of this library, one must group the entities within this simulations.
  > The entities are also called particles using the analogy with physics. These particles can be gas particles if an ideal gas is simulated, where gas particles do random walk in the volume available for them.

A group of entity must also share the same level and kind of statistical freedom, i.e. how many of its properties requires random number assignment and how often. The properties representing the different stochastic freedoms are called purpose.
  > Using the ideal gas analogy, let's suppose it is a mixture of He gas with different isotopes, $^4He$ and $^6He$. The first is stable but the latter is radioactive with a half life of $~1s$. The particles of $^4He$ form a group of particles, which do random walk only, but the group of $^6He$ particles do radioactive decay too. $^4He$ has 1 statistical freedom, where the latter has 2, therefore they form 2 different groups. In this example, the 2 purposes may be called "random walk" and "radioactive decay".

When this library is used, the user need to define the group of particles (or at least their number) and the list of purposes. For every realization, particle group and purpose, a unique seed and a dedicated prng engine can be assigned. In each random number generation steps (for a realization, particle group and purpose) as many random numbers are generated as many particles can be found within that particle group. Then the user can assign these random numbers to each particles. A filtering logic (after generation) can be also applied if only a subset of the particles needs random numbers.

## Implementation of the prng container

In the algorithm-based prng, I assign a prng instance to each particle type $n_i$. For some $i$, the number of particles will change from one week to another, some will remain the same, but even the number of particle types can change slightly.

To address these properties, we first need to assess how many particle types we need maximum in the upcoming months/years, denote this with $N_{\max}$, and in the actual runs, denote the number of particle types with $N$, where $N \le {N_{\max }}$ holds. We will fix ${N_{\max }}$, and change it once necessary, but this will break backward compatibility: realization-level comparison of old and new runs won't be possible. Setting this value too high is a wasteful handling of seed values.

At the initialization of the class instance, we need to tell how many types of particles, and how many particles per each type we need. We can provide it with a dict of dict, containing all the IDs as keys for different particles, and all the type of particles as keys. The order of elements in dict is kept while adding further elements or removing elements.

### The dictionary of dictionary containing the particle IDs

The initializator (aka constructor) of the class expects a dict of dict `A`, where the keys of `A` are the type names of the particles $n_0$, $n_1$, ..., $n_i$, ... $n_N$. Each particle type can have different number of particles, and the particle IDs `ID` must be unique within the particle type, as they will serve as key for the dict `A["n_i"]`. Each key `ID` within a dictionary `A["n_i"]` must be associated with a unique order number provided as a value for that particle ID `ID` as key. Its uniqueness is not checked. The order number does not affect the order of random numbers returned, but determines which random numbers should removed in case some particles with type $n_i$ and keys ${I{D_{{x_1}}}}$, ${I{D_{{x_2}}}}$, ..., ${I{D_{{x_X}}}}$ should be removed.

### tee: copy the stream of random numbers to a file

The generated random numbers can be written into a file, referred to as teefile, which contains the random numbers in a binary representation with 64 bit precision.
