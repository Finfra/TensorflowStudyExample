import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
xx=np.array([1,2,3])

plt.plot(xx)
plt.savefig("/home/dockeruser/df/d.png")
