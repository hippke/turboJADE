![Logo](https://github.com/hippke/turboJADE/blob/main/logo.png?raw=true)
## Fast Adaptive Differential Evolution in pure Python

Text

### Complete example

Define a function to maximize (invert if you wish to minimize). Decorate with @jit:

```
import numpy as np
from numba import jit

@jit
def rastrigin(p):
    x, y = p[0], p[1]
    val = (
        20
        + (x**2 - 10 * np.cos(2 * np.pi * x))
        + (y**2 - 10 * np.cos(2 * np.pi * y))
    )
    return -val
```
Call turboJADE:
```
from turboJADE import turboJADE
solver = turboJADE(
    func=rastrigin,
    limits=[(-5.12, 5.12), (-5.12, 5.12)],
    n_dim=2,
    n_pop=100,
    c=0.1,
    progress=True,
    converge=False
    )
best, trend, evals = solver.run(n_it=1_000)
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

```
```


Performance: One million steps per second per core (Intel Core i7-1185G), if function evaluations are cheap.

Functionr requirements:
- Added decorator for numba jit
- Maximize (invert if you wish to minimize) 
