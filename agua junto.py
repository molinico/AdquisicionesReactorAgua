import matplotlib.pyplot as plt
import numpy as np
import math
import time
import scipy.signal
from scipy.optimize import curve_fit
import scipy.stats as stats
import pandas as pd
import os
from  scipy.stats import chi2_contingency
#from statsmodels.stats.weightstats import DescrStatsW
import glob
from scipy.signal import butter, filtfilt

#ej de path=r'C:\Users\Sergio\Desktop\L6y7\18-04-23'
path='C:/Users/Nicolás Molina/Desktop/L6-7/18-04-23bis'

#get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('matplotlib', 'qt5')

#%%
#ARREGLO LOS CSV
carpeta = path

# Iterar sobre cada archivo CSV en la carpeta
for archivo in os.listdir(carpeta):
    if archivo.endswith('.csv'):
        ruta_archivo = os.path.join(carpeta, archivo)
        # Leer el contenido del archivo
        with open(ruta_archivo, 'r') as f:
            contenido = f.read()
        # Reemplazar las apariciones de ",,," por ",0,0,"
        contenido_modificado = contenido.replace(',,,', ',0,0,')
        # Escribir el contenido modificado en el archivo
        with open(ruta_archivo, 'w') as f:
            f.write(contenido_modificado)


#%%
os.chdir (path)

files=glob.glob('*.csv')

files=files[:1]  #SI QUIERO ANALIZAR SOLO UNOS POCOS O UNO

per0=0   #0 para primer periodo , 1000 para segundo

ampV=18000 #voltaje puesto en el generador aprox
frecusada=8000  #aprox
 

#CON 1 SE PLOTEAN LAS COSAS CON 0 NO
graficoscrudos=1
graficoajuste=1
graficofiltro=1
graficoaplan=1
graficosfinal=1

############################################
fig, (ax1, ax2) = plt.subplots(2, 1)

plt.close("all")
potencias=[]
j=1
for file in files:
    med = np.loadtxt(file, delimiter=',', skiprows=16, unpack=True)
    t=med[3]
    Vdbd=med[4]
    Vstr=med[8]
    Istr=Vstr/50 *1000
    print(str(j)+"-archivo="+file)

    if graficoscrudos==1:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        # Graficar los datos en el primer subplot
        ax1.plot(t, Vdbd, color='blue', label='Vdbd')
        ax1.grid()
        ax1.set_title("Voltaje de alimentación")
        ax1.set_xlabel('tiempos (s)')
        ax1.set_ylabel('V')
        ax1.legend()

        # Graficar los datos en el segundo subplot
        ax2.plot(t, Istr, color='red', label='Istr')
        ax2.grid()
        ax2.set_title("Corriente de streamers")
        ax2.set_xlabel('tiempo (s)')
        ax2.set_ylabel('I [mA]')
        ax2.legend()

        fig.suptitle("archivo="+file)
    
        plt.show()

    def sin(x,T,a,b):
        y=a*np.sin(2*np.pi/T*x)+b
        return y

    init_vals=[1/frecusada,ampV,0]



    popt, pcov = curve_fit(sin, t, Vdbd,absolute_sigma=True,p0=init_vals)       #,p0=init_vals
    perr = np.sqrt(np.diag(pcov))

    valorT= popt[0]
    valora=popt[1]
    valorb=popt[2]
    err_T = perr[0]
    #print("periodo=",valorT)
    #print("frecuencia=",1/valorT)
    #print("len",len(t))
    #print("tiempo capturado en pantalla=",t[-1]-t[0])
    longper=valorT*10000000
    longper=round(longper)
    #print(longper)

    #print("valores inversa",valora1,valorb1)
    bins=np.linspace(t[0],t[-1],len(t))

    ajuste= sin(bins,valorT,valora,valorb)
    if graficoajuste==1:
        plt.figure()
        plt.grid()
        plt.title("Tensión de entrada"+file)
        plt.plot(t,Vdbd,label="datos crudos")
        plt.plot(bins,ajuste,label="ajuste seno")
        plt.xlabel("tiempo (s)")
        plt.ylabel("I (mA)")
        plt.legend()

    frecuencia=1/valorT


    signal=Istr

    # Definir la frecuencia de corte del filtro
    cutoff = 100

    # Crear el filtro Butterworth de orden elevado
    order = 5
    nyquist = 0.5 * frecuencia
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high')

    # Aplicar el filtro a la señal
    filtered_signal = filtfilt(b, a, signal)

    Istr_plana=filtered_signal
    if graficofiltro==1:
        # Graficar la señal original y la señal filtrada
        plt.figure()
        plt.grid()
        plt.title("Aplanamiento de señal"+file)
        plt.plot(t, signal, label='Señal original')
        plt.plot(t, filtered_signal, label='Señal filtrada')
        plt.xlabel("tiempo (s)")
        plt.ylabel("I [mA]")
        plt.legend()
        plt.show()

    p = 1 #valor umbral

    # Recorrer la lista y actualizar los valores menores a 'p' a cero
    for i in range(len(Istr_plana)):
        if Istr_plana[i] < p:
            Istr_plana[i] = 0
            
    if graficoaplan==1:
        plt.figure()
        plt.grid()
        plt.title("Aplanamiento total de señal"+file)
        plt.plot(t, Istr_plana, label='Señal filtrada')
        plt.xlabel("tiempo (s)")
        plt.ylabel("I [mA]")
        plt.legend()
        plt.show()

    #os.chdir (r'C:\Users\Sergio\Desktop\labo5\difractiva\codigos\espectroscopia')
    #os.chdir (r'C:\Users\Nicolás Molina\Desktop\L6-7\13-4-23')
    #el len de los archivos es 2489
    # [500:500+longper]

    if graficosfinal==1:
        fig, (ax1, ax2) = plt.subplots(2, 1)


        # Graficar los datos en el primer subplot
        ax1.plot(t, Vdbd, color='grey', label='Vdbd')
        ax1.plot(t[per0:per0+longper], Vdbd[per0:per0+longper], color='blue', label='Vdbd para P')
        ax1.grid()
        ax1.set_title("Voltaje de alimentación")
        ax1.set_xlabel('tiempos (s)')
        ax1.set_ylabel('V')
        ax1.legend()

        # Graficar los datos en el segundo subplot
        ax2.plot(t, Istr_plana, color='grey', label='Istr')
        ax2.plot(t[per0:per0+longper], Istr_plana[per0:per0+longper], color='red', label='Istr para P')
        ax2.grid()
        ax2.set_title("Corriente de streamers")
        ax2.set_xlabel('tiempo (s)')
        ax2.set_ylabel('I [mA]')
        ax2.legend()

        fig.suptitle("archivo="+file)

        plt.show()
    Vpot=Vdbd[per0:per0+longper]
    Ipot=Istr_plana[per0:per0+longper]/1000 #paso la corriente a A
    N=len(Ipot)
    potencia=np.mean(Vpot*Ipot/N)
    potencias.append(potencia)
    print("potencia=",potencia, "W")
    j=j+1
    

    
    
desviacion_estandar = stats.tstd(potencias)

print("--------FINAL--------")
if per0==1000:
    print("segundo periodo")
else:
    print("primer periodo")
#print("array potencias=",potencias)
print("potencia media de todo=",np.mean(potencias),"+-",desviacion_estandar,"W")
#print("desviaciòn estandar",desviacion_estandar)
print("maximo de potencias",max(potencias),"W","//numero de archivo",potencias.index(max(potencias))+1)
print("minimo de potencias",min(potencias),"W","//numero de archivo",potencias.index(min(potencias))+1)

