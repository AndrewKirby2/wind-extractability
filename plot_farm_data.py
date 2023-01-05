"""Plot zeta for DS8 for different wind farm sizes
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn

#load farm data for different sizes
for farm_diameter in [10,15,20,25,30]:
    zeta = np.load(f'data/zeta_DS8_{farm_diameter}.npy')
    plt.plot(zeta, label=str(farm_diameter)+' km')
plt.legend()
plt.ylabel(r'$\zeta$')
plt.xlabel('Time (h)')

plt.savefig('plots/zeta_DS8.png')