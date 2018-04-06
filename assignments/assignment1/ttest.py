# Use a paired difference t-test to see if surgery leads to weight change.

from scipy import stats

before = [67.2, 67.4, 71.5, 77.6, 86.0, 89.1, 59.5, 81.9, 105.5]
after = [63.6, 64.2, 72.4, 65.6, 83.7, 82.2, 59.2, 71.5, 103.0]

t_stat, p_val = stats.ttest_rel(before, after)

print("t-stat: ", t_stat)
print("p-value: ", p_val)
