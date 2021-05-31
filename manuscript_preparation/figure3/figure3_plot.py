import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pd_data = pd.read_csv('./figure3_data.txt', sep='\t')

case1_avg = pd_data['Case 1'].mean()
case2_avg = pd_data['Case 2'].mean()
case3_avg = pd_data['Case 3'].mean()
case4_avg = pd_data['Case 4'].mean()
print(case1_avg, case2_avg, case3_avg, case4_avg)

case1_std = pd_data['Case 1'].std()
case2_std = pd_data['Case 2'].std()
case3_std = pd_data['Case 3'].std()
case4_std = pd_data['Case 4'].std()
print(case1_std, case2_std, case3_std, case4_std)

avg = np.array([case1_avg, case2_avg, case3_avg, case4_avg])
std = np.array([case1_std, case2_std, case3_std, case4_std])

plt.errorbar(np.array([1,2,3,4]), avg, std, linestyle='None', marker='*')
plt.ylim(0, 0.6)

plt.show()

