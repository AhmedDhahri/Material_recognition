import numpy as np
import pandas as pd

data = np.genfromtxt("dataset.csv", delimiter=',')


occurrences = pd.Series(data[1:, 3]).value_counts()

print(occurrences)