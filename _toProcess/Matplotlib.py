
# coding: utf-8

# In[1]:


# 주의 현재 셀을 실행 시키면 아래 셀 실행 안됨

import matplotlib.pyplot as plt
x = range(100)
y = [ i*i for i in x]
plt.plot(x,y)
plt.show()


# In[1]:


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.plot([1,2,3])
plt.savefig('myfig')


# In[2]:



from IPython.display import Image 
Image(filename='myfig.png')


# In[3]:


from PIL import Image
pil_im = Image.open('myfig.png')
print pil_im
dir(pil_im)

