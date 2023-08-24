import csv 
import numpy as np
import matplotlib.pyplot as plt


def Comparison(cg_or_cp,boo_or_sus):

    y1 = []
    y2 = []
    title = boo_or_sus + ' ' + cg_or_cp.upper() + ' Comparison'

    if cg_or_cp == 'cg':
        row_index_fspro = 2
        row_index_rasaero = 11
    if cg_or_cp == 'cp':
        row_index_fspro = 3
        row_index_rasaero = 12

    if boo_or_sus == 'booster':

        time = np.arange(0,13.79,0.01)

        booster_data_csv = open('Booster_StabilityData.csv', newline='')
        rasaero_booster_data_csv = open('RasaeroBooster_StabilityData.csv', newline='')
        heading = next(booster_data_csv)
        heading = next(rasaero_booster_data_csv)
        booster_data_reader = csv.reader(booster_data_csv)
        rasaero_booster_data_reader = csv.reader(rasaero_booster_data_csv)

        for row in booster_data_reader:
            y1.append(float(row[row_index_fspro]))
        for row in rasaero_booster_data_reader:
            y2.append(float(row[row_index_rasaero]))
    
        booster_data_csv.close()
        rasaero_booster_data_csv.close()

    if boo_or_sus == 'sustainer':

        time = np.arange(0,161.97,0.01)

        sustainer_data_csv = open('Sustainer_StabilityData.csv', newline='')
        sustainer_coast_data_csv = open('SustainerCoast_StabilityData.csv', newline='')
        rasaero_sustainer_data_csv = open('RasaeroSustainer_StabilityData.csv', newline='')
        heading = next(sustainer_data_csv)
        heading = next(sustainer_data_csv)
        heading = next(sustainer_coast_data_csv)
        heading = next(rasaero_sustainer_data_csv)
        sustainer_data_reader = csv.reader(sustainer_data_csv)
        sustainer_coast_data_reader = csv.reader(sustainer_coast_data_csv)
        rasaero_sustainer_data_reader = csv.reader(rasaero_sustainer_data_csv)

        for row in sustainer_coast_data_reader:
            y1.append(float(row[row_index_fspro]))
        for row in sustainer_data_reader:
            y1.append(float(row[row_index_fspro]))
        for row in rasaero_sustainer_data_reader:
            y2.append(float(row[row_index_rasaero]))

        sustainer_data_csv.close()
        sustainer_coast_data_csv.close()
        rasaero_sustainer_data_csv.close()
    

    plt.figure()
    plt.plot(time,y1,'r',label = 'FS PRO')
    plt.plot(time,y2,'b',label = 'Rasaero')
    plt.legend()
    plt.title(title)


Comparison('cg','booster')
Comparison('cp','booster')
Comparison('cg','sustainer')
Comparison('cp','sustainer')
plt.show()


