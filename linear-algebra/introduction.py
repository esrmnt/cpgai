import numpy as np
import matplotlib.pyplot as plt

# Set up the figure and axis
v = np.array([1,2])
A = np.array([[1,-3],[2,1]])
v2 = A@v

fig, ax = plt.subplots()
ax.axis([-7,5,-5,5])
# Move left y-axis and bottim x-axis to centre, passing through (0,0)
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
# Eliminate upper and right axes
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.arrow(0, 0, v[0], v[1], head_width=0.2, head_length=0.2, fc='black', ec='black')
ax.arrow(0, 0, v2[0], v2[1], head_width=0.2, head_length=0.2, fc='black', ec='black')


print("v =", v)
print("A @ v =", v2)

plt.grid()
plt.show()