# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 08:54:18 2023

@author: osole
"""

#Import of basic packages that will be used

import matplotlib.pyplot as plt
import numpy as np
import os
from astropy.io.ascii import read

# Select working directory, where all the M....dat files are stored
wdpath = 'C:/Users/osole/OneDrive/Documentos/1_Astro/Estructura_evolucion/Entregable_2'
os.chdir(wdpath)
#%%
#Fix some preference for plotting 
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10, 8)
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14
matplotlib.rcParams['axes.titlesize'] = 20
matplotlib.rcParams['figure.titlesize'] = 16
matplotlib.rcParams['axes.labelsize'] = 16
matplotlib.rcParams['legend.fontsize'] = 14
matplotlib.rcParams['lines.linewidth'] = 2

matplotlib.rcParams['xtick.major.size'] = 8
matplotlib.rcParams['ytick.major.size'] = 8
matplotlib.rcParams['xtick.minor.size'] = 4
matplotlib.rcParams['ytick.minor.size'] = 4
matplotlib.rcParams['ytick.minor.visible'] = True  
matplotlib.rcParams['xtick.minor.visible'] = True   

matplotlib.rcParams['axes.edgecolor'] = 'black'   
matplotlib.rcParams['axes.linewidth'] = 2

#%% Class creation
# This code will create a class that will read and store model data and have specific functions
# to plot and visualize the data

class stellar_models:
    def __init__(self, path, printeo = True, dictionary = 0):
        # path: the path to the directory that stores the data files
        # printeo = True: It will print which file is it importing
        # dictionary variable can be inserted if the dictionary with the data files is already created
        
        print('Loading Stellar Models class')
        self.path = path
        # if the dictionary is not given it will create it from scratch
        # it will store all tables, and to access the table of M=0.8 one has to write
        # dictionary['M0p8z14V0']
        if dictionary == 0: 
            self.names = []
            for root, dirs, files in os.walk(path): # extraction of names of different files
                if printeo == True:
                    print(f"Current directory: {root}")
                for file in files:
                    if f"  {file}"[-4:] == '.dat':
                        self.names.append(f"{file}"[:-4]) # the name is all but the .dat of the end
            # creation of dictionary:
            self.df = {} 
            if printeo == True:
                print('Data loaded from files:')
            # saving of values into dictionary
            for i in range(len(self.names)):
                self.df[self.names[i]] = read(self.names[i]+'.dat', data_start=2)
                if printeo == True:
                    print(' '+self.names[i]+'.dat')
        # if the dictionary is given
        if dictionary != 0:
            self.df = dictionary
            self.names = list(self.df.keys())
                    
        self.n = len(self.names) #total number of models
        self.fignum = 0
        # save a list of the different variables included in each model
        self.varnames = self.df[self.names[0]].colnames
        # now the code identifies the surface and central variables to allow selection
        # of variable through input in the next function
        self.surf_names = ["surf" in self.varnames[i] for i in range(len(self.varnames))]
        self.cen_names = ["cen" in self.varnames[i] for i in range(len(self.varnames))]
        self.other_names = [f or g for f, g in zip(self.surf_names, self.cen_names)]
        
        # Now it will extract the different masses of each model
        self.masses = np.zeros(self.n)
        for i in range(self.n):
            if i<10:
                self.masses[i] = round(self.df[self.names[i]].columns['mass'][0],0)
            if i == 10 or i == 12:
                self.masses[i] = round(self.df[self.names[i]].columns['mass'][0],1)
            if i == 11:
                self.masses[i] = round(self.df[self.names[i]].columns['mass'][0],2)
        
        # Once the masses are extracted, model names will be ordered by mass
        self.srt_vals = sorted(zip(self.masses, self.names))
        self.masses, self.names = zip(*self.srt_vals)
        
        # The next part of the initialization corresponds to the detection of the different 
        # periods in the evolution of the star:
            
        #TAMS detection, time at which central Hydrogen is fully combusted
        self.tams_index = np.zeros(self.n)
        self.tams_times = np.zeros(self.n)
        self.zams_times = np.zeros(self.n)
        for i in range(self.n):
            self.tams_index[i] = int(np.where(self.df[self.names[i]]['1H_cen']<0.001)[0][0])
            self.tams_times[i] = self.df[self.names[i]]['time'][int(self.tams_index[i])]
            self.zams_times[i] = self.df[self.names[i]]['time'][0]
        #Hertzsprung's gap time, minimum lg(L) for M=4 at least
        self.Hgap_index = np.zeros(self.n)
        for i in range(self.n):
            self.Hgap_index[i] = int(np.where(self.df[self.names[i]]['lg(L)']==min(self.df[self.names[i]]['lg(L)']))[0][0])
        
        #End of RGB phase: core He burning begins, for M=4, by looking at central mass fraction of He vs time:
        self.RGB_time = 1.569E8 #when core burning of He starts
        self.RGB_index = np.argmin(np.abs(self.df[self.names[6]]['time']-self.RGB_time))
        
        # End of BL, beginning of AGB: He core burning ends. For M=4
        self.AGB_index = int(np.where(self.df[self.names[6]]['4He_cen']<0.001)[0][0])
        self.AGB_times = self.df[self.names[6]]['time'][int(self.AGB_index)]
        
        
# a function will be defined to perform all desired plots in the task. It is designed to either 
# be specified the x-y values of the plot or to allow the user to select from the different variables
# that are included in the tables
    def ploteo(self,  ms = 'none', x_var='none', log_x = False, y_var='none', t_tams=False, colormap = True,
               HR = False, Z_TAMS = False, zams = False, T_rho = False, vlines = False, parts = False,
               text = False, title = 'Set Title', figsz = (10,8)):
        # ms = list of the different masses that are wanted in the plot
        # x_var = str, variable in the x axis of the plot
        # log_x = boolean, If one wants the x axis with logarithmic scale
        # y_var = list of strings, different lines that are wanted on the plot
        # colormap = boolean, True if one wants the points following a color sequence or not
        # t_tams, HR, Z_TAMS, zams, T_rho, vlines, parts, text = boolean, different cases that are seen below
        
        self.x_var = x_var
        self.y_var = y_var
        
        # If HR is true, HR diagram:
        if HR == True:
            self.x_var ='lg(Teff)'
            self.y_var = ['lg(L)']
        
        # if no x value is specified, the user can choose through the input
        if self.x_var == 'none':
            typ = input('Type of data you want in the x axis: \'time\', \'mass\', \'ab-omeg\', \'other\': ')
            if typ == 'time':
                self.x_var = 'time'
            if typ == 'mass':
                self.x_var = 'mass'
            if typ == 'ab-omeg':
                for i in range(len(self.varnames)):
                    if self.other_names[i]==True:
                        print(str(i) + ' | ' + self.varnames[i])
                oth_index = int(input('Choose variable (index): '))
                self.x_var = self.varnames[oth_index]
            if typ == 'other':
                for i in range(len(self.varnames)):
                    if self.other_names[i]==False:
                        if self.varnames[i]!='time' and self.varnames[i]!='mass' and self.varnames[i]!='line':
                            print(str(i) + ' | ' + self.varnames[i])
                oth_index = int(input('Choose variable (index): '))
                self.x_var = self.varnames[oth_index]
        
        # Selection of y variables. If no y_var is set, the user can decide how many lines do they 
        # want plotted and which
        self.n_y = len(self.y_var)
        if self.y_var== 'none':
            self.n_y = int(input('How many lines do you want plotted? (int): '))
            self.y_var = [None]*self.n_y
            for k in range(self.n_y):
                typ = input('Type of data you want on the y axis: \'time\', \'mass\', \'ab-omeg\', \'other\': ')
                if typ == 'time':
                    self.y_var[k] = 'time'
                if typ == 'mass':
                    self.y_var[k] = 'mass'
                if typ == 'ab-omeg':
                    for i in range(len(self.varnames)):
                        if self.other_names[i]==True:
                            print(str(i) + ' | ' + self.varnames[i])
                    oth_index = int(input('Choose variable (index): '))
                    self.y_var[k] = self.varnames[oth_index]
                if typ == 'other':
                    for i in range(len(self.varnames)):
                        if self.other_names[i]==False:
                            if self.varnames[i]!='time' and self.varnames[i]!='mass' and self.varnames[i]!='line':
                                print(str(i) + ' | ' + self.varnames[i])
                    oth_index = int(input('Choose variable (index): '))
                    self.y_var[k] = self.varnames[oth_index]
        

        # Selection of different masses to plot. If none given, they can be chosen from input
        self.ms = ms
        self.n_ms = len(ms)
        if ms == 'none':
            self.n_ms = int(input('Number of different Mass models to plot (int): '))
            self.ms = np.zeros(self.n_ms)
            print('The possible masses are: ', self.masses)
            for i in range(self.n_ms):
                self.ms[i] = float(input('Input the masses you want separately: '))
        if ms == 'all':
            self.n_ms = len(self.masses)
            self.ms = self.masses
        
        self.index_ms = np.zeros(self.n_ms)
        for i in range(self.n_ms):
            self.index_ms[i] = self.masses.index(self.ms[i])
            
        # Radius calculation for last exercise:
        if y_var[0] == 'R': #calculation of R using stefan boltzmann's law for M=4
            L=10**np.array(self.df[self.names[int(self.index_ms[0])]]['lg(L)'])
            Teff=10**np.array(self.df[self.names[int(self.index_ms[0])]]['lg(Teff)'])
            R_s = 6.957E8
            self.df[self.names[int(self.index_ms[0])]]['R'] = (L*3.828E26/ 
                                                               (4*np.pi*5.6704E-8*Teff**4))**(1/2)/R_s
            sm.varnames.append('R') #new column added to the table
        
        # Extract index of the x and y variables corresponding to our self.varnames
        # self.arg_x_var = next((index for index, value in enumerate(sm.varnames) if sm.varnames[index]==self.x_var), None)
        self.arg_x_var = self.varnames.index(self.x_var)
        self.arg_y_vars = np.zeros(self.n_y)
        for i in range(self.n_y):
            self.arg_y_vars[i] = self.varnames.index(self.y_var[i])
        labels = ['', 'Time [$yr$]', '$\log (M/M_\odot)$', '$\log (L/L_\odot)$', '$\log (T_{eff}) \ [K]$',
                  'Mass Fraction', 'Mass Fraction', 'Mass Fraction', 'Mass Fraction', 'Mass Fraction', 'Mass Fraction', 'Mass Fraction', 'Mass Fraction', 'Mass Fraction', 'Mass Fraction', 'Mass Fraction',
                  '', '', '',
                  '$\log(\\rho_c) \ [g/cm^3]$', '$\log(T_c)\ [K]$',
                  'Mass Fraction', 'Mass Fraction', 'Mass Fraction', 'Mass Fraction', 'Mass Fraction', 'Mass Fraction', 'Mass Fraction', 'Mass Fraction', 'Mass Fraction', 'Mass Fraction', 'Mass Fraction',
                  '', '', '', '', '', '', '', '', '', '', '', 'Radius $[R_\odot]$']
                  
        # Creation of figure
        self.fignum = self.fignum + 1
        plt.close(self.fignum)
        fig, ax = plt.subplots(num = self.fignum, figsize = figsz)
        
        if colormap == True:
            colors = plt.cm.inferno_r(np.linspace(0, 1, self.n_ms))

        
        # First plotting case: we want all points for each mass' variable plotted
        if zams == False and parts == False and t_tams == False:
            for i in range(self.n_ms):
                if self.n_y == 1: #if there is only one curve in the plot
                    # plotting with or without colormap
                    if colormap == True:
                        ax.plot(self.df[self.names[int(self.index_ms[i])]].columns[self.x_var], 
                                self.df[self.names[int(self.index_ms[i])]].columns[self.y_var[0]],
                                '-', color = colors[i], #plt.cm.inferno_r((i+1)/self.n_ms), 
                                label = str(self.ms[i])+' $M_{\\odot}$')
                    if colormap == False:
                        ax.plot(self.df[self.names[int(self.index_ms[i])]].columns[self.x_var], 
                                self.df[self.names[int(self.index_ms[i])]].columns[self.y_var[0]],
                                '-', label = str(self.ms[i])+' $M_{\\odot}$')
                    # if one wants to mark the points of the ZAMS and TAMS
                    if Z_TAMS == True:
                        # ZAMS values
                        if i==0:
                            x_zams = np.zeros(self.n_ms)
                            y_zams = np.zeros(self.n_ms)
                            x_tams = np.zeros(self.n_ms)
                            y_tams = np.zeros(self.n_ms)
                        x_zams[i] = self.df[self.names[int(self.index_ms[i])]].columns[self.x_var][0]
                        y_zams[i] = self.df[self.names[int(self.index_ms[i])]].columns[self.y_var[0]][0]
                        ax.plot(x_zams[i], y_zams[i],
                                'D', fillstyle = 'none', color = colors[i],
                                markersize=8, markeredgewidth=2, alpha = .5)#,label = str(self.ms[i])+' $M_{\\odot}$')
                        ax.plot(x_zams[i], y_zams[i], '+', color = 'gray', markersize=8, markeredgewidth=2)
                        # TAMS values
                        x_tams[i] = self.df[self.names[int(self.index_ms[i])]].columns[self.x_var][int(self.tams_index[int(self.index_ms[i])])]
                        y_tams[i] = self.df[self.names[int(self.index_ms[i])]].columns[self.y_var[0]][int(self.tams_index[int(self.index_ms[i])])]
                        ax.plot(x_tams[i], y_tams[i], 's', fillstyle='none', color = colors[i], 
                                markersize=8, markeredgewidth=2, alpha = .5)
                        ax.plot(x_tams[i], y_tams[i], '+', color = 'gray', markersize=8, markeredgewidth=3)
                        #thicker lines for MS part
                       
                        ax.plot(self.df[self.names[int(self.index_ms[i])]].columns[self.x_var][:int(self.tams_index[int(self.index_ms[i])])], 
                                self.df[self.names[int(self.index_ms[i])]].columns[self.y_var[0]][:int(self.tams_index[int(self.index_ms[i])])],
                                '-', color = colors[i], linewidth=4)
                        #perform only at the end of the loop
                        if i == range(self.n_ms)[-1]:
                            ax.plot(x_zams, y_zams, 'gray', linestyle = 'dashed', label = 'ZAMS')
                            ax.plot(x_tams, y_tams, 'gray', linestyle = 'dotted', label = 'TAMS')
                
                # if there are different variables to be plotted:
                else:
                    for k in range(self.n_y):
                        if colormap == True:
                            ax.plot(self.df[self.names[int(self.index_ms[i])]].columns[self.x_var], 
                                    self.df[self.names[int(self.index_ms[i])]].columns[self.y_var[k]],
                                    '-', color = colors[i],
                                    label = self.y_var[k]+' '+str(self.ms[i])+' $M_{\\odot}$')
                        if colormap == False:
                            if self.n_ms == 1:
                                ax.plot(self.df[self.names[int(self.index_ms[i])]].columns[self.x_var], 
                                        self.df[self.names[int(self.index_ms[i])]].columns[self.y_var[k]],
                                        '-', label = self.y_var[k])
                            else:
                                ax.plot(self.df[self.names[int(self.index_ms[i])]].columns[self.x_var], 
                                        self.df[self.names[int(self.index_ms[i])]].columns[self.y_var[k]],
                                        '-', label = self.y_var[k]+' '+str(self.ms[i])+' $M_{\\odot}$')
        
        # for the exercise 1, extract the time of the TAMS for all masses and obtain the linear fit
        if t_tams == True: 
            for i in range(self.n_ms):
                ax.plot(self.ms[i], self.tams_times[i],
                        'D', fillstyle = 'none', color = colors[i],
                        markersize=8, markeredgewidth=2)
            # Linear fit
            self.p = np.polyfit(np.log10(self.ms), np.log10(self.tams_times), 1)
            ax.plot(self.ms, (10**self.p[1])*(self.ms**self.p[0]),
                    '--', color = 'gray', label = self.y_var[0]+'=$10^{%.1f}$'%self.p[1]+'$\cdot$mass$^{%.1f}$'%self.p[0])

        
        if zams == True: #plot only zams values, the first for each mass
            x_values = np.zeros(self.n_ms)
            y_values = np.zeros(self.n_ms)
            for i in range(self.n_ms):
                if log_x == True:
                    x_values[i] = np.log10(self.df[self.names[int(self.index_ms[i])]].columns[self.x_var][0]) #take first value (zams, t=0)
                    y_values[i] = self.df[self.names[int(self.index_ms[i])]].columns[self.y_var[0]][0]
                    ax.plot(x_values[i], y_values[i],
                            'D', fillstyle = 'none', color = colors[i],
                            markersize=8, markeredgewidth=2)
            # Linear fit for stars in low, middle and high mass ranges
            self.p1 = np.polyfit(x_values[:4], y_values[:4], 1)
            ax.plot(x_values[:4], self.p1[0]*x_values[:4] + self.p1[1],
                    color = 'red', label = self.y_var[0]+'=%.1f'%self.p1[0]+'$\cdot$lg('+self.x_var+')+%.1f'%self.p1[1])
            self.p2 = np.polyfit(x_values[4:10], y_values[4:10], 1)
            ax.plot(x_values[4:10], self.p2[0]*x_values[4:10] + self.p2[1],
                    color = 'green', label = self.y_var[0]+'=%.1f'%self.p2[0]+'$\cdot$lg('+self.x_var+')+%.1f'%self.p2[1])
            self.p3 = np.polyfit(x_values[10:], y_values[10:], 1)
            ax.plot(x_values[10:], self.p3[0]*x_values[10:] + self.p3[1],
                    color = 'blue', label = self.y_var[0]+'=%.1f'%self.p3[0]+'$\cdot$lg('+self.x_var+')+%.1f'%self.p3[1])

        # Specification of different stages in stellar evolution
        if parts == True:
            ax.plot(self.df[self.names[int(self.index_ms[i])]].columns[self.x_var][:int(self.tams_index[int(self.index_ms[0])]+1)], 
                    self.df[self.names[int(self.index_ms[i])]].columns[self.y_var[0]][:int(self.tams_index[int(self.index_ms[0])]+1)],
                    '-', label = 'MS', color = 'green')
            ax.plot(self.df[self.names[int(self.index_ms[i])]].columns[self.x_var][int(self.tams_index[int(self.index_ms[0])]):int(self.Hgap_index[int(self.index_ms[0])]+1)], 
                    self.df[self.names[int(self.index_ms[i])]].columns[self.y_var[0]][int(self.tams_index[int(self.index_ms[0])]):int(self.Hgap_index[int(self.index_ms[0])]+1)],
                    '-', label = 'Hertzsprung gap', color = 'black')
            ax.plot(self.df[self.names[int(self.index_ms[i])]].columns[self.x_var][int(self.Hgap_index[int(self.index_ms[0])]):int(self.RGB_index)+1], 
                    self.df[self.names[int(self.index_ms[i])]].columns[self.y_var[0]][int(self.Hgap_index[int(self.index_ms[0])]):int(self.RGB_index)+1],
                    '-', label = 'RGB', color = 'red')
            ax.plot(self.df[self.names[int(self.index_ms[i])]].columns[self.x_var][int(self.RGB_index):int(self.AGB_index+1)], 
                    self.df[self.names[int(self.index_ms[i])]].columns[self.y_var[0]][int(self.RGB_index):int(self.AGB_index+1)],
                    '-', label = 'BL', color = 'blue')
            ax.plot(self.df[self.names[int(self.index_ms[i])]].columns[self.x_var][int(self.AGB_index):], 
                    self.df[self.names[int(self.index_ms[i])]].columns[self.y_var[0]][int(self.AGB_index):],
                    '-', label = 'AGB', color = 'orange')
            
            if self.x_var == 'time' and self.y_var[0] != '1H_cen':
                ax.set_xlim(1.25E8)
        
        # If the plot is log(rhoC) vs log(Tc), different regions can be specified
        if T_rho == True:
            slopes = [3/2, 3, 0, 3] #slopes for the case of equal pressures. 
            #[0] for ideal=degenerate, [1] for ideal = radiation and 
            #[2] for degenarte = relativistic degenerate, [3] relt deg = ideal
            cts = [3/2*np.log10(8314*2**(5/3)/(0.61*10**7))  -3,#      #same but with the constants
                   np.log10(4*5.6704E-8/(3*2.99E8*8314/0.61))  -3 -1,#    #-3 to convert kg/m^3 to g/m^3, #-1 because P_ideal=10*P_rad
                   3*np.log10(1.24E10/2**(4/3)*2**(5/3)/1E7)  -3, 
                   3*np.log10(8314/(0.61*1.24E10/2**(4/3)))   -3]# 
            x_values = np.linspace(plt.xlim()[0], plt.xlim()[1], 100)
            y_limits = [plt.ylim()[0], plt.ylim()[1]]
            
            if plt.xlim()[1]>8.75:
                x_values_first = np.linspace(plt.xlim()[0],8.751315, 100)
            else: x_values_first = x_values
            #plot borders
            ax.plot(x_values, x_values*slopes[1]+cts[1], '--', color = 'gray')
            ax.plot(x_values_first, x_values_first*slopes[0]+cts[0], '--', color = 'gray')
            ax.plot(x_values_first, x_values_first*slopes[2]+cts[2], '--', color = 'gray')
            #draw the different zones
            ax.fill_between(x_values, x_values*slopes[2]+cts[2], x_values*slopes[0]+cts[0], where=(x_values<8.751315),
                            color = 'red', alpha=0.2)
            ax.fill_between(x_values, plt.ylim()[0], x_values*slopes[1]+cts[1],
                            color = 'blue', alpha=0.2)
            ax.fill_between(x_values_first, x_values_first*slopes[2]+cts[2], 8,# where=(x_values<8.751315),
                            color = 'green', alpha=0.2)
            
            if plt.xlim()[1]>8.75:
                x_values_last = np.linspace(8.751315, x_values[-1], 100)
                ax.plot(x_values_last, x_values_last*slopes[3]+cts[3], '--', color = 'gray')
                ax.fill_between(x_values_last, x_values_last*slopes[3]+cts[3], 8,
                                color = 'green', alpha=0.2)
                ax.fill_between(x_values_last, x_values_last*slopes[1]+cts[1], x_values_last*slopes[3]+cts[3],
                                color = 'gray', alpha=0.4)
            ax.fill_between(x_values_first, x_values_first*slopes[1]+cts[1], x_values_first*slopes[0]+cts[0],
                            color = 'gray', alpha=0.4)
            #include text of region type
            if text == True:
                ax.text(7.25, 0.75, 'Ideal gas', rotation='horizontal', ha='center', va='center', color = 'black', fontsize=16)
                ax.text(7.5, 5.9, 'Classic degenerate gas', ha = 'center', va = 'center', color = 'red', fontsize = 16)
                ax.text(7.5, 7.3, 'Relativistic degenerate gas', ha = 'center', va = 'center', color = 'green', fontsize = 16)
                ax.text(8.6, 0.75, 'Radiation pressure', ha = 'center', va = 'center', color = 'blue', fontsize = 16)
            
            ax.set_xlim(x_values[0], x_values[-1])
            ax.set_ylim(y_limits[0], y_limits[-1])
        
        #speaks for itself
        if HR == True:
            ax.invert_xaxis()
                  
        if vlines == True: #addition of vertical lines at end of each stage of stellar evolution in abundance vs time plot
            ax.axvline(x=self.tams_times[int(self.index_ms[0])], linestyle='--', color = 'gray')
            ax.text(self.tams_times[int(self.index_ms[0])]-1E6, .5, 'End of MS', rotation='vertical', ha='center', va='center', color = 'gray', fontsize=14)
            #Hertzsprung gap
            ax.axvline(x=self.df[self.names[int(self.index_ms[0])]]['time'][int(self.Hgap_index[int(self.index_ms[0])])], linestyle='--', color = 'gray')
            ax.text(self.df[self.names[int(self.index_ms[0])]]['time'][int(self.Hgap_index[int(self.index_ms[0])])]-5E5, .5, 'End of Hertzsprung\'s gap', rotation='vertical', ha='center', va='center', color = 'gray', fontsize=14)
            #Hertzsprung gap
            ax.axvline(x=self.RGB_time, linestyle='--', color = 'gray')
            ax.text(self.RGB_time+8E5, .5, 'End of RGB', rotation='vertical', ha='center', va='center', color = 'gray', fontsize=14)
            #AGB phase: He burning stops in the core
            ax.axvline(x=self.AGB_times, linestyle='--', color = 'gray')
            ax.text(self.AGB_times-2E6, .5, 'End of BL', rotation='vertical', ha='center', va='center', color = 'gray', fontsize=14)
            
            ax.set_xlim(1.5E8,1.955E8)

        ax.legend()
        if text == True: 
            ax.legend(loc='upper right')
        
        if title != 'Set Title':
            ax.set_title(title)

        ax.set_xlabel(labels[self.arg_x_var])        
        ax.set_ylabel(labels[int(self.arg_y_vars[0])])
        
        ax.grid()

        return fig


sm = stellar_models('C:/Users/osole/OneDrive/Documentos/1_Astro/Estructura_evolucion/Entregable_2',
                    printeo = True)

'''
I recommend to test and explore the function ploteo created. With the class initialized (previus line)
one can just play around with it. By typing sm.ploteo(), the console will progressively ask
for the variables that one wants plotted.

Below are written the necessary lines to obtain all figures of the paper at once.
'''
#%% 1) central hydrogen and helium abundances
fig = sm.ploteo(ms = [5.0, 9.0], x_var = 'time', y_var = ['1H_cen', '4He_cen'], 
                title = 'Central H and He abundances evolution', colormap = False)

ax = plt.gca()
tams5M = sm.df[sm.names[int(sm.index_ms[0])]]['time'][int(sm.tams_index[int(sm.index_ms[0])])]
tams9M = sm.df[sm.names[int(sm.index_ms[1])]]['time'][int(sm.tams_index[int(sm.index_ms[1])])]
ax.axvline(tams5M, linestyle = '--', color = 'gray')
ax.axvline(tams9M, linestyle = '--', color = 'gray')
ax.text(tams5M-4E6, .5, 'End of MS for \n 5 $M_\odot$, t=%.1e'%tams5M, rotation='vertical', ha='center', va='center', color = 'gray', fontsize=14)
ax.text(tams9M+5E6, .8, 'End of MS for \n 9 $M_\odot$, t=%.1e'%tams9M, rotation='vertical', ha='center', va='center', color = 'gray', fontsize=14)
#%% Different MS durations
fig0 = sm.ploteo('all', x_var = 'mass', y_var = ['time'], t_tams = True, title = 'MS time duration depending on stellar mass')
ax=plt.gca()
ax.plot(5, tams5M, '+', color='green', label = '5 $M_\odot$', markersize = 12, markeredgewidth = 3)
ax.plot(9, tams9M, '+', color='red', label = '9 $M_\odot$', markersize = 12, markeredgewidth = 3)
ax.set_xlabel('Mass [$M_\odot$]')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()

#%% 2) a,b,c,d - Evolutionary tracks in an HR diagram with ZAMS and TAMS values
fig2 = sm.ploteo(ms = 'all', HR = True, Z_TAMS = True, title = 'Evolutionary tracks in the HR diagram')

#%% 2) e,f,g
fig3 = sm.ploteo(ms = 'all', x_var = 'mass', log_x = True, y_var = ['lg(L)'], zams = True, figsz = (8,8))
fig4 = sm.ploteo(ms = 'all', x_var = 'mass', log_x = True, y_var = ['lg(rhoc)'], zams = True, figsz = (8,8))
fig5 = sm.ploteo(ms = 'all', x_var = 'mass', log_x = True, y_var = ['lg(Tc)'], zams = True, figsz = (8,8))
#%% 3) log T log rho plane
fig6 = sm.ploteo(ms = [1.0, 3.0, 9.0, 40.0], x_var = 'lg(Tc)', y_var = ['lg(rhoc)'],
                 Z_TAMS = True, T_rho = True, text = True,
                 title = '$\log (\\rho_c)$ vs $\log (T_c)$ and \n equation of state governing each region')

#%% 4) a - central abuncances of H, He, C and O vs time
fig7 = sm.ploteo(ms = [4], x_var = 'time', y_var = ['1H_cen', '4He_cen', '12C_cen', '16O_cen'],
                 colormap = False, vlines = True, title = 'Central abundances evolution for M=$4M_\odot$')

fig77 = sm.ploteo(ms = [4], x_var = 'time', y_var = ['1H_surf', '4He_surf', '12C_surf', '16O_surf'],
                 colormap = False, vlines = True, title = 'Surface abundances evolution for M=$4M_\odot$')
#%% 4) b - evolutionary tracks with different parts
fig8 = sm.ploteo(ms = [4], HR = True, Z_TAMS = True, parts = True, title = 'HR diagram for the model with M=$4M_\odot$')
#%% 4) c - logT logrho plane
fig10 = sm.ploteo(ms = [4], x_var = 'lg(Tc)', y_var = ['lg(rhoc)'], Z_TAMS = True, T_rho = True, parts = True,
                  title = '$\log (\\rho_c)$ vs $\log (T_c)$ and \n equation of state governing each region')

#%% 4) d,e,f - R vs time 
fig = sm.ploteo(ms = [4], x_var = 'time', y_var = ['R'], parts = True, vlines = False,
                title = 'Evolution of the stellar radius')


#%% Latex table

#simple function used to export arrays as latex tables
def table(array, exp = False):
    string = ''
    for i in range(len(array)):
        if exp == False:
            string+='%.1f '%array[i]
        if exp == True:
            string+='%.1e '%array[i]
        if i!=range(len(array))[-1]:
            string+='& '
    print(string)

