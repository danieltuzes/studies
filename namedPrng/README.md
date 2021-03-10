# namedPrng

- [namedPrng](#namedprng)
  - [Details of the model](#details-of-the-model)
  - [Implementation of the prng container](#implementation-of-the-prng-container)
    - [The dictionary of dictionary containing the particle IDs](#the-dictionary-of-dictionary-containing-the-particle-ids)
    - [tee: copy the stream of random numbers to a file](#tee-copy-the-stream-of-random-numbers-to-a-file)

In one of my professional tasks, I had to implement a prng container for Monte Carlo simulations. The purpose of this container is to improve prn generation by supporting multiple seeds, and instead of seeds, to support named seeds.

- The first step is to print out all the random numbers used by the model, which produced the result A, store them in a file,
- and then use this super container to feed the model with the same numbers. This time I should get the same result A.
- Then the file-based random number feed has to replaced by a prng algorithm, like Mersenne Twister. This time the model should produce different result B. I can write out the random numbers again, into a file, and use the file instead of the algorithm producing the same B result. So we can see how changing the random number engine from a basic math algorithm to file-based solution keeps the same result A.
- Then we change the random numbers, and get result B, but the result remains the same if we change the source of random numbers from a file to a math algorithm again.

## Details of the model

In these simulations, there is no nondeterministic time evolution, the prng is used to initialize the state only. The system consists of $N$ different types of particle, and each particle has a unique name.

To initialize the state of the system, we assign a random number from a Gaussian distribution to each particle. The system is then being evolved and a property $P$ is evaluated. Starting from different initial states, we get different $P$ results, $\left\{ {{P_1},{P_2}, \ldots ,{P_L}} \right\}$, and this set is further evaluated,

$${\Pi _L} = f\left( {\left\{ {{P_1},{P_2}, \ldots ,{P_L}} \right\}} \right)$$

During the run of the model, $L$ increases, ${\Pi _L}$ evolves and is being monitored, and once it meets a requirement, some $P_i$ values, $\{ {P_i}\quad i \in I\} $ for an index set $I$ must be evaluated. Instead of storing all the details of the particles throughout the evolution of $\Pi_L$, we just identify $I$ and recreate the systems corresponding to each element of $I$.

We are also interested how systems with different number of particles of type $n_i$ affects the evolution of $\Pi_L$.

## Implementation of the prng container

In the algorithm-based prng, I assign a prng instance to each particle type $n_i$. For some $i$, the number of particles will change from one week to another, some will remain the same, but even the number of particle types can change slightly.

To address these properties, we first need to assess how many particle types we need maximum in the upcoming months/years, denote this with $N_{\max}$, and in the actual runs, denote the number of particle types with $N$, where $N \le {N_{\max }}$ holds. We will fix ${N_{\max }}$, and change it once necessary, but this will break backward compatibility: realization-level comparison of old and new runs won't be possible. Setting this value too high is a wasteful handling of seed values.

At the initialization of the class instance, we need to tell how many types of particles, and how many particles per each type we need. We can provide it with a dict of dict, containing all the IDs as keys for different particles, and all the type of particles as keys. The order of elements in dict is kept while adding further elements or removing elements.

### The dictionary of dictionary containing the particle IDs

The initializator (aka constructor) of the class expects a dict of dict `A`, where the keys of `A` are the type names of the particles $n_0$, $n_1$, ..., $n_i$, ... $n_N$. Each particle type can have different number of particles, and the particle IDs `ID` must be unique within the particle type, as they will serve as key for the dict `A["n_i"]`. Each key `ID` within a dictionary `A["n_i"]` must be associated with a unique order number provided as a value for that particle ID `ID` as key. Its uniqueness is not checked. The order number does not affect the order of random numbers returned, but determines which random numbers should removed in case some particles with type $n_i$ and keys ${I{D_{{x_1}}}}$, ${I{D_{{x_2}}}}$, ..., ${I{D_{{x_X}}}}$ should be removed.

### tee: copy the stream of random numbers to a file

The generated random numbers can be written into a file, referred to as teefile, which contains the random numbers in a binary representation with 64 bit precision.
