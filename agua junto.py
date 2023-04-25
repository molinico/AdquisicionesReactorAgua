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

n_archivo=30


#files=files[n_archivo-1:n_archivo]  #SI QUIERO ANALIZAR SOLO UNOS POCOS O UNO
   

ampV=18000 #voltaje puesto en el generador aprox
ampI=3/1000
frecusada=8000  #aprox
 

#CON 1 SE PLOTEAN LAS COSAS CON 0 NO

graficoscrudos=0
graficoajuste=1
graficofiltro=1
graficoaplan=1
graficosfinal=1

if len(files) > 1:
    graficoscrudos=0
    graficoajuste=0
    graficofiltro=0
    graficoaplan=0
    graficosfinal=0

############################################
fig, (ax1, ax2) = plt.subplots(2, 1)

plt.close("all")


pers=[0,1000]   #0 para primer periodo , 1000 para segundo


potencias=[]
j=1
for file in files:
    med = np.loadtxt(file, delimiter=',', skiprows=16, unpack=True)
    t=med[3]
    Vdbd=med[4]
    Vstr=med[8]
    Istr=Vstr/50
    #print(str(j)+"-archivo="+file)

    if graficoscrudos==1:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        # Graficar los datos en el primer subplot
        ax1.plot(t*1000, Vdbd/1000, color='blue', label='$V_{dbd}$')
        ax1.grid()
        ax1.set_title("Voltaje de alimentación")
        ax1.set_xlabel('tiempos [ms]')
        ax1.set_ylabel('V [kV]')
        ax1.legend()

        # Graficar los datos en el segundo subplot
        ax2.plot(t*1000, Istr*1000, color='red', label='$I_{str}$')
        ax2.grid()
        ax2.set_title("Corriente de streamers")
        ax2.set_xlabel('tiempo [ms]')
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


    popt, pcov = curve_fit(sin, t, Istr,absolute_sigma=True,p0=init_vals)       #,p0=init_vals
    perr = np.sqrt(np.diag(pcov))

    
    valorbIstr=popt[2]
    

    bins=np.linspace(t[0],t[-1],len(t))

    ajuste= sin(bins,valorT,valora,valorb)
    """
    if graficoajuste==1:
        plt.figure()
        plt.grid()
        plt.title("Tensión de entrada"+file)
        plt.plot(t*1000,Vdbd/1000,label="Datos crudos V")
        plt.plot(bins*1000,ajuste/1000,label="Ajuste seno")
        plt.xlabel("tiempo [ms]")
        plt.ylabel("V [kV]")
        plt.legend()
    
    frecuencia=1/valorT
    """

    signal=Istr
    from scipy.signal import savgol_filter

    window_length = 201
    polyorder = 3
    seno_filtrado = savgol_filter(Istr, window_length, polyorder)


    def sin2(x,T,a,b):
        y=a*np.sin(2*np.pi/T*x+b)+valorbIstr
        return y

    init_valsIstr=[1/frecusada,ampI,0]
    popt, pcov = curve_fit(sin2, t, seno_filtrado,absolute_sigma=True,p0=init_valsIstr)       #,p0=init_valsIstr
    perr = np.sqrt(np.diag(pcov))

    valorTf= popt[0]
    valoraf=popt[1]
    valorbf=popt[2]
    err_Tf = perr[0]

    Istr_ajuste=sin2(bins,valorTf,valoraf,valorbf)

    Istr_plana=Istr-Istr_ajuste

    longper=valorTf*10000000
    longper=round(longper)
    

    #Istr_plana=filtered_signal
    if graficofiltro==1:

        plt.figure()
        plt.title("savgol")
        plt.plot(t*1000, Istr*1000, label='Señal original')
        plt.plot(t*1000, seno_filtrado*1000, label='Señal filtrada')
        plt.xlabel('Tiempo [ms]')
        plt.ylabel('Amplitud(mA)')
        plt.grid(True)
        plt.legend(loc='best')
        plt.show()

        plt.figure()
        plt.grid()
        plt.title("Savgol + ajuste")
        plt.plot(t*1000,seno_filtrado*1000,label="Solo filtro")
        plt.plot(t*1000,Istr_ajuste*1000,label="Ajuste al filtro")
        plt.xlabel('Tiempo [ms]')
        plt.ylabel('Amplitud(mA)')
        plt.grid(True)
        plt.legend(loc='best')
        plt.show()

        # Graficar la señal original y la señal filtrada
        plt.figure()
        plt.grid()
        plt.title("Aplanamiento de señal"+file)
        plt.plot(t*1000, signal*1000, label='Señal original')
        plt.plot(t*1000, Istr_plana*1000, label='Señal filtrada')
        plt.xlabel("tiempo [ms]")
        plt.ylabel("I [mA]")
        plt.legend()
        plt.show()

    p = 0.001 #valor umbral

    # Recorrer la lista y actualizar los valores menores a 'p' a cero
    for i in range(len(Istr_plana)):
        if Istr_plana[i] < p:
            Istr_plana[i] = 0
            
    if graficoaplan==1:
        plt.figure()
        plt.grid()
        plt.title("Aplanamiento total de señal"+file)
        plt.plot(t*1000, Istr_plana*1000, label='Señal filtrada')
        plt.xlabel("tiempo [ms]")
        plt.ylabel("I [mA]")
        plt.legend()
        plt.show()


    for per0 in pers:
        if graficosfinal==1:
            fig, (ax1, ax2) = plt.subplots(2, 1)


            # Graficar los datos en el primer subplot
            ax1.plot(t*1000, Vdbd/1000, color='grey', label='$V_{dbd}$')
            ax1.plot(t[per0:per0+longper]*1000, Vdbd[per0:per0+longper]/1000, color='blue', label='$V_{dbd}$ para P')
            ax1.grid()
            ax1.set_title("Voltaje de alimentación")
            ax1.set_xlabel('tiempos [ms]')
            ax1.set_ylabel('V [kV]')
            ax1.legend()

            # Graficar los datos en el segundo subplot
            ax2.plot(t*1000, Istr_plana*1000, color='grey', label='$I_{str}$')
            ax2.plot(t[per0:per0+longper]*1000, Istr_plana[per0:per0+longper]*1000, color='red', label='$I_{str}$ para P')
            ax2.grid()
            ax2.set_title("Corriente de streamers")
            ax2.set_xlabel('tiempo [ms]')
            ax2.set_ylabel('I [mA]')
            ax2.legend()

            fig.suptitle("archivo="+file+"/periodo="+str(per0))

            plt.show()

        
        Vpot=Vdbd[per0:per0+longper]
        Ipot=Istr_plana[per0:per0+longper] #paso la corriente a A
        N=len(Ipot)
        potencia=np.mean(Vpot*Ipot)
        potencias.append(potencia)
        #print("potencia=",potencia, "W")
        j=j+1
        

        
        
desviacion_estandar = stats.tstd(potencias)
    
def numarchmax(x):
    num=potencias.index(max(x))+1
    if num % 2 == 0:
        print("maximo en segundo periodo")
        return num/2 
    else:
        print("maximo en primer periodo")
        return (num-1)/2
    
def numarchmin(x):
    num=potencias.index(min(x))+1
    if num % 2 == 0:
        print("minimo en segundo periodo")
        return num/2
    else:
        print("minimo en primer periodo")
        return (num-1)/2
    
print(path)
if len(files)> 5:
    print("cantidad de archivos=",len(files))
    print("-------")

print("POTENCIA MEDIA de todo=",np.mean(potencias),"+-",desviacion_estandar,"W")
print("-----")
print("MAXIMO de potencias",max(potencias),"W","//numero de archivo",round(numarchmax(potencias)))
print("MINIMO de potencias",min(potencias),"W","//numero de archivo",round(numarchmin(potencias)))


