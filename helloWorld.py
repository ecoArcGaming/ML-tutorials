import numpy 
import matplotlib.pyplot as plt
import pandas as pd
plt.figure(figsize=(5,4),dpi=100)
x = numpy.arange(0,5,0.5)
plt.plot(x[:5], x[:5]**2,label='line1', color='blue',linewidth=3, linestyle='--')
plt.plot(x, [0,1,2,3,4,5,6,7,8,9],label='line2', linewidth=3, linestyle='dashdot')
plt.plot(x[4:],x[4:]**2, color='blue',linewidth=3,linestyle='solid')
plt.title('First Graph')
plt.xlabel('x axis')
plt.xticks(range(0,6))

print(x)

labels=['A','B','C']
values=[10,20,30]
bars = plt.bar(labels,values)
bars[1].set_hatch('/')
plt.legend()
plt.show()

