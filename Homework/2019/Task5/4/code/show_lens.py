import pickle
import numpy as np
import matplotlib.pyplot as plt
lens_dir = 'bins/lens.pkl'

with open(lens_dir,'rb') as f:
    lens=pickle.load(f)

plt.figure()
plt.bar([i for i in range(len(lens))],lens)
plt.show()