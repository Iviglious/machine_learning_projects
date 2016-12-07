
# coding: utf-8

# # Team YSI - Titanic: Machine Learning from Disaster

# ## Compare outputs

# In[1]:

#########################################################################
#
# Titanic: Machine Learning from Disaster
#
# Python script for comparison of the output/results files
#
# Amendment date             Amended by            Description
# 07/12/2016                 Ivaylo Shalev         Initial version.
#
#
#########################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:

# Reading the output data (A vs B)
out_a_df = pd.read_csv('output/myfirstdtree_formated.csv', header=0)      # Load the train file into a dataframe
out_b_df = pd.read_csv('output/ysi_titanic_prediction.csv', header=0)        # Load the test file into a dataframe

print out_a_df.head()
print out_b_df.head()


# In[3]:

# Compare both outputs
compare_array = out_a_df.values == out_b_df.values
#print compare_array
print 100*compare_array[:,1].sum().astype(float)/compare_array[:,1].size


# In[ ]:



