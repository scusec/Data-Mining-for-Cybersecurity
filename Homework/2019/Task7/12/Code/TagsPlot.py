
# coding: utf-8

# In[1]:


# coding = utf-8
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


Location = "../data/dataszy.txt"
df = pd.read_csv(Location,sep="\t", header=None,names = ["fsize"])


# In[6]:


df.head()


# In[7]:


XX = df[0:140]


# In[19]:


YY = df[140:]
print len(YY)


# In[29]:


import pylab as pl

pl.figure(figsize=(16,12))
x = np.linspace(0,140,140)
y = np.linspace(0,len(YY),len(YY))
pl.plot(x,XX,'.',color='red', label='webshell')
pl.plot(y,YY,'.',color='blue', label='edupage')

pl.legend()
pl.title('total num of tags and total type of tags')
pl.show()


# In[34]:


import pylab as pl

pl.figure(figsize=(4,8))
x = np.linspace(0,100,100)
pl.plot(x,XX[:100],'.',color='red', label='webshell')
pl.plot(x,YY[-100:],'.',color='blue', label='edupage')

pl.legend()
pl.title('total num of tags and total type of tags')
pl.show()

