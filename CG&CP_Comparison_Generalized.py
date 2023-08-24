import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def Comparison(cg_or_cp,booster_or_sustainer, is_time_independent=True):

    if not is_time_independent and cg_or_cp == 'cg':
        print('Warning: cg is not computable given the mach number alone')
    
    time = 0
    time_index_booster = 0
    time_index_sustainer = 0
    y1 = []
    y2 = []

    df_rasaero = pd.read_csv('Flight Test.csv')
    df_fspro_booster = pd.read_csv('Booster_StabilityData.csv')
    df_fspro_sustainer_coast = pd.read_csv('SustainerCoast_StabilityData.csv')
    df_fspro_sustainer = pd.read_csv('Sustainer_StabilityData.csv')
    
    time_index_booster = df_fspro_booster.iloc[-1][0]
    time_index_sustainer = df_fspro_sustainer_coast.iloc[-1][0] + df_fspro_sustainer.iloc[-1][0]

    title = booster_or_sustainer + ' ' + cg_or_cp.upper() + ' Comparison'

    if cg_or_cp == 'cg':
        col_name_fspro = 'cg (in)'
        col_name_rasaero = 'CG (in)'
    if cg_or_cp == 'cp':
        col_name_fspro = 'cp (in)'
        col_name_rasaero = 'CP (in)'

    if is_time_independent:
        if booster_or_sustainer == 'booster':
            
            time = np.arange(0,time_index_booster+0.01,0.01)

            y1 = df_fspro_booster[col_name_fspro]
            y2 = df_rasaero.loc[df_rasaero['t'] <= time_index_booster+0.01, col_name_rasaero]

        if booster_or_sustainer == 'sustainer':

            time = np.arange(0,time_index_sustainer+0.01,0.01)
        
            y1_a = df_fspro_sustainer_coast[col_name_fspro]
            y1_b = df_fspro_sustainer[col_name_fspro]
            y1 = pd.DataFrame(pd.concat((y1_a, y1_b), ignore_index = True))
            y1 = y1.iloc[1: , :]
            y2 = df_rasaero.loc[(df_rasaero['t'] >= time_index_booster) & (df_rasaero['t'] <= (time_index_sustainer+time_index_booster)), col_name_rasaero]

        plt.figure()
        plt.plot(time,y1,'r',label = 'FS PRO')
        plt.plot(time,y2,'b',label = 'Rasaero')

        error = np.asarray(y1) - np.asarray(y2)
        error_abs = np.absolute(error)

        print('Average ' + booster_or_sustainer + ' ' + cg_or_cp + ' error: ' + str(np.average(error_abs)))
        
    else:
        if booster_or_sustainer == 'booster':
            
            mach_fspro = df_fspro_booster['Mach Number']
            mach_thresh = df_fspro_booster['Mach Number'].max()
            mach_rasaero = df_rasaero.loc[(df_rasaero['Stage'] == 'B') & (df_rasaero['Mach Number'] <= mach_thresh), 'Mach Number']

            y1 = df_fspro_booster[col_name_fspro]
            y2 = df_rasaero.loc[(df_rasaero['Stage'] == 'B') & (df_rasaero['Mach Number'] <= mach_thresh), col_name_rasaero]

        if booster_or_sustainer == 'sustainer':

            mach_fspro = pd.DataFrame(pd.concat((df_fspro_sustainer_coast['Mach Number'], df_fspro_sustainer['Mach Number']), ignore_index = True))
            mach_fspro = mach_fspro.iloc[1: , :]
            mach_thresh = mach_fspro['Mach Number'].max()
            mach_rasaero = df_rasaero.loc[(df_rasaero['Stage'] == 'S') & (df_rasaero['Mach Number'] <= mach_thresh), 'Mach Number']

            y1_a = df_fspro_sustainer_coast[col_name_fspro]
            y1_b = df_fspro_sustainer[col_name_fspro]
            y1 = pd.DataFrame(pd.concat((y1_a, y1_b), ignore_index = True))
            y1 = y1.iloc[1: , :]
            y2 = df_rasaero.loc[(df_rasaero['Stage'] == 'S') & (df_rasaero['Mach Number'] <= mach_thresh), col_name_rasaero]

        plt.figure()
        plt.plot(mach_fspro,y1,'r',label = 'FS PRO')
        plt.plot(mach_rasaero,y2,'b',label = 'Rasaero')

    plt.legend()
    plt.title(title)


Comparison('cg','booster')
Comparison('cp','booster')
Comparison('cp','booster', is_time_independent=False)
Comparison('cg','sustainer')
Comparison('cp','sustainer')
Comparison('cp','sustainer', is_time_independent=False)
plt.show()


