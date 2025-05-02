from typing import Callable

import numpy as np
import numpy.typing as npt

Vector = npt.NDArray[np.float64]
Matrix = npt.NDArray[np.float64]
ScalarFn = Callable[[Vector], float]
VectorFn = Callable[[Vector], Vector]
