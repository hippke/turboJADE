![Logo](https://github.com/hippke/turboJADE/blob/main/logo.png?raw=true)
## Fast Adaptive Differential Evolution in pure Python
Implements the algorithm by [Zhang & Sanderson (2009)](https://ieeexplore.ieee.org/document/5208221). Inspired by [PyFDE](https://pythonhosted.org/PyFDE/), which is a similar JADE implementation in Cython.

Performance: One million steps per second per core (Intel Core i7-1185G), if function evaluations are cheap.
- Just-in-time compilation with numba
- Fast random-number generation
- Minimal overhead due to limited scope


### Complete example
Define a function to maximize:
```
import numpy as np
from numba import jit

@jit  # Has to be numba jitted
def rastrigin(p):
    x, y = p[0], p[1]
    return -(  # invert as we wish to minimize
        20
        + (x**2 - 10 * np.cos(2 * np.pi * x))
        + (y**2 - 10 * np.cos(2 * np.pi * y))
    )
```
Run turboJADE:
```
from turboJADE import turboJADE
solver = turboJADE(
    func=rastrigin,
    limits=[(-5.12, 5.12), (-5.12, 5.12)],
    n_dim=2,        # Number of dimensions
    n_pop=100,      # Population size
    c=0.1,          # Self-adaption rate
    progress=True,  # Print tqdm progress bar
    converge=True   # Stop after 200 iterations without change
    )
best, trend, evals = solver.run(n_it=1_000)  # Run for 1000 iterations
```
Print results:
```
print("x, y = {:.2f}, {:.2f} (expected: 0.00, 0.00)".format(best[0], best[1]))
print("func = {:.2f}".format(rastrigin(best)))
print("Model evaluations:", evals)

x, y = 0.00, -0.00 (expected: 0.00, 0.00)
func = 0.00
Model evaluations: 10000
```
Visualize convergence:
```
import matplotlib.pyplot as plt
plt.plot(np.linspace(0, len(trend), len(trend)), trend)
plt.xlabel("Iteration number")
plt.ylabel("Function value")
plt.show()
```
![convergence](https://github.com/hippke/turboJADE/blob/main/converge.png?raw=true)


### Installation
For now, just download the ``turboJADE.py`` file. Open an [issue](https://github.com/hippke/turboJADE/issues) to request a pip installable package if that's useful.

Requirements: ``pip install numpy numba tqdm matplotlib``

### Attribution
Please cite [tbd](http) if you find this code useful in your research. The BibTeX entry for the paper is:

```
@ARTICLE{
}
```

