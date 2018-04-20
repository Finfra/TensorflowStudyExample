import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
rv=stats.norm()
xx=np.linspace(-8,8,100)
pdf=rv.pdf(xx)


plt.plot(xx, pdf)
plt.savefig("/home/dockeruser/df/c.png")
