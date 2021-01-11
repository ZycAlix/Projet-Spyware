#!/usr/bin/env python
# coding: utf-8

# In[2]:


import time
import schedule
from PIL import ImageGrab
import imageio
 
count = time.strftime("%Y-%m-%d_%H-%M-%S",time.localtime())
counter = count.split('-')[-1]
tmp_counter = 0
 
def screen():
	im =ImageGrab.grab()
	im.save("D:\pic/"+str(count[:-2])+str(tmp_counter)+'.png','PNG')
	
schedule.every(1).seconds.do(screen)
while True:
	schedule.run_pending()
	print(time.ctime())
	count = time.strftime("%Y-%m-%d_%H-%M-%S",time.localtime())
	counter = count.split('-')[-1]
	print ("counter  is  %s " % counter)
	tmp_counter =tmp_counter+1
	
	
	time.sleep(1)


# In[ ]:




