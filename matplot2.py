import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import *



file = pd.read_csv('column_2C_weka.csv')
print(file)
plt.title('Biomechanical features of orthopedic patients',fontdict={'size':15})
plt.hist2d(file.pelvic_incidence,file.pelvic_tilt_numeric,norm=PowerNorm(gamma=1. / 2.))
plt.xlabel('pelvic incidence')
plt.ylabel('pelvic tilt numeric')
plt.colorbar()

plt.show()
