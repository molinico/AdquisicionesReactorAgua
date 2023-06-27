# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:28:01 2023

@author: Descargas
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

data = pd.read_excel('sin-agua-solo-dbd.xls')
o3 = np.array(data['CH1 Conc'].values)
t = np.linspace(10,1950,num=195)

data2 = pd.read_excel('con-agua-solo-dbd-17kv.xls')
o3_2 = np.array(data2['CH1 Conc'].values)
t_2 = np.linspace(10,1780,num=178)

data3 = pd.read_excel('con-agua-dbdycc-17kv.xls')
o3_3 = np.array(data3['CH1 Conc'].values)
t3 = np.linspace(10,1730,num=173)

data4 = pd.read_excel('con agua dbd y cc 17kv bien.xls')
o3_4 = np.array(data4['CH1 Conc'].values)
t4 = np.linspace(10,1600,num=160)

data5 = pd.read_excel('E5 con agua dbd+cc 17kv.xls')
o3_5 = np.array(data5['CH1 Conc'].values)
t5 = np.linspace(10,1490,num=149)

data6 = pd.read_excel('E6 Y E5 co agua dbd+cc 17kv.xls')
o3_6 = np.array(data6['CH1 Conc'].values)
t6 = np.linspace(10,1410,num=141)


data7 = pd.read_excel('E5 E6 E3 dbd y cc 17kv.xls')
o3_7= np.array(data7['CH1 Conc'].values)
t7 = np.linspace(10,1880,num=188)

data8 = pd.read_excel('E4,5,6,3 dbd+cc 17kv.xls')
o3_8= np.array(data8['CH1 Conc'].values)
t8 = np.linspace(10,1690,num=169)


plt.figure()
plt.title('Concentración de Ozono en distintas etapas')
# plt.plot(t,o3, label='DBD sola sin agua')
plt.plot(t_2,o3_2, label='DBD con agua')
# plt.plot(t3,o3_3, label='DBD+CC con agua mal')
plt.plot(t4,o3_4, label='DBD+CC con agua')
# plt.plot(t5,o3_5, label='DBD+CC con agua E5')
plt.plot(t6,o3_6, label='DBD+CC con agua E5+E6')
plt.plot(t7,o3_7, label='DBD+CC con agua E5+E6+E3')
plt.plot(t8,o3_8, label='DBD+CC con agua E5+E6+E3+E4')
plt.xlabel('Tiempo [s]')
plt.ylabel('Concentración O3 [ppm]')
plt.legend()