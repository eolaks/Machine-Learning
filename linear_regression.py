import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score 

# generate data 
x = np.array([1,3,10,16,26,36])
y = np.array([42,50,75,100,150,200])

#determine the slope-m and intecept-c
m, c = np.polyfit(x, y, 1)

print('Slope ', m)
print('Intercept ', c)

# predict value of y1, if x1 = 12
x1 = 12
y1 = m*x1 + c
print("Predicted value of y1 when x is 12 : ", y1)

# plot the scatter plot of x and y
plt.scatter(x,y, color = "b", marker = "o")

# plot the linear regression of x and y
plt.plot(x, m*x + c, color = 'g')

plt.xlabel('X = Axis')
plt.ylabel('Y-Axis')


print(r2_score(y, m*x + c))

plt.show()
