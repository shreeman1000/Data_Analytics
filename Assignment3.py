import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

df = pd.read_csv('Data_GeneSpring.txt', sep = '\t')

# male non-smoker, male smoker, female non-smoker, female smoker
mns = [[1, 0, 1, 0]]
ms = [[1, 0, 0, 1]]
fns = [[0, 1, 1, 0]]
fs = [[0, 1, 0, 1]]
xgender = np.matrix(mns*12 + ms*12 + fns*12 + fs*12)

mns = [[1, 0, 0, 0]]
ms = [[0, 1, 0, 0]]
fns = [[0, 0, 1, 0]]
fs = [[0, 0, 0, 1]]
pgender = np.matrix(mns*12 + ms*12 + fns*12 + fs*12)

I = np.identity(48)
rank_xgender = np.linalg.matrix_rank(xgender)
rank_pgender = np.linalg.matrix_rank(pgender)

scale_factor = (48 - rank_pgender) / (rank_pgender - rank_xgender)

num = np.linalg.pinv(np.matmul(xgender.T, xgender))
num = np.matmul(xgender, num)
num = np.matmul(num, xgender.T)
num = I - num

den = np.linalg.pinv(np.matmul(pgender.T, pgender))
den = np.matmul(pgender, den)
den = np.matmul(den, pgender.T)
den = I - den


F_stat = []
data = df.iloc[:, 1:49].values
for row_vector in data:
    a = np.matmul((np.matmul(row_vector.T, num)), row_vector)
    b = np.matmul(np.matmul(row_vector.T, den), row_vector)
    f = (a / b - 1) * scale_factor
    F_stat.append(f[0,0])

    
p = 1 - stats.f.cdf(F_stat, rank_pgender - rank_xgender, 48 - rank_pgender)
plt.hist(p, bins=25, edgecolor='white')
plt.xlabel('P-values')
plt.ylabel('Frequency')
plt.title('Histogram of P-values')
plt.show()