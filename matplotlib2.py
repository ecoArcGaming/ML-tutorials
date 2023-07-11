import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file = pd.read_csv('gas_prices.csv')
print(file)
x = file['Year']
y = file.Australia
#plt.plot(x,y,'g.-',label='Australia')
#plt.plot(x,file.Japan,'r.-',label='Japan')

for country in file:
    if country != 'Year':
        plt.plot(x,file[country],linewidth=2,marker='.',label=country)

plt.xticks(np.arange(1990,2009,2))
plt.yticks(np.arange(0,8,0.5))
plt.title('Gas Prices by Country')
plt.ylabel('Gas Prices (USD / Gallon)')
plt.legend()
plt.show()