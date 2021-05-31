import matplotlib.pyplot as plt
import numpy as np

plt.rcdefaults()
fig, ax = plt.subplots()

# Example data
people = ('A', 'B', 'C', 'D')
y_pos = np.arange(len(people))
performance = np.array([0.024, 0.882, 0.005, 0.213])

ax.barh(y_pos, performance, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Probability')
ax.grid(True)

plt.show()
