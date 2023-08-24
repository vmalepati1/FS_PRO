"""
Translator
Dyllon Preston

All measures assumed to be in metric

"""
import numpy as np
from scipy import integrate
import csv
import pandas as pd
import os

df = list(csv.reader(open('weights.csv')))

def spliceBySection(section):
    return spliceBySection(section, 9)

def spliceBySection(section, columnToSearch = 9):
    minPos = 1000
    maxPos = 0
    masses = []
    coms = []
    indices = []
    finLoc = []
    propLoc = -1
    nozzLoc = -1
    for i in range(1, len(df)):
        pos = float(df[i][7])
        localCoM = float(df[i][6])
        length = float(df[i][5])
        mass = float(df[i][4])
        simulationSection = df[i][columnToSearch]
        globalCoM = pos + localCoM
        if simulationSection != "a" and simulationSection != "" and simulationSection != "x" and int(simulationSection) == section:
            if pos < minPos:
                minPos = pos
            if pos + length > maxPos:
                maxPos = pos + length
            masses.append(mass)
            coms.append(globalCoM)
            indices.append(i)
            if df[i][1] == "Fins":
                finLoc.insert(0, pos)
        elif simulationSection == "a":
            if df[i][1] == "Propellant":
                propLoc = pos + localCoM
                nozzLoc = float(df[i - 1][7]) + float(df[i - 1][5])
    
    # Takes out length that is shared with a section higher up on the rocket
    for i in range(1, len(df)):
        pos = float(df[i][7])
        localCoM = float(df[i][6])
        length = float(df[i][5])
        mass = float(df[i][4])
        simulationSection = df[i][columnToSearch]
        # If the piece is not in this section, see if the piece overlaps the current section
        if simulationSection != "a" and simulationSection != "" and simulationSection != "x" and int(simulationSection) < section:
            # If the piece impedes on the current section, the current section is shortened from the top
            if pos + length > minPos:
                minPos = pos + length
        elif simulationSection == "x":
            if pos > minPos and pos < maxPos:
                maxPos = pos
            elif pos < minPos and pos + length > minPos:
                minPos = pos + length
    com = 0
    mass = sum(masses)
    for i in range(0, len(coms)):
        com += coms[i] * masses[i] / mass
    
    moi = 0
    for i in range(0, len(indices)):
        index = indices[i]
        pos = float(df[index][7])
        localCoM = float(df[index][6])
        mass2 = float(df[index][4])
        moi += float(df[index][8]) - mass2 * (pos + localCoM) * (pos + localCoM) + mass2 * (pos + localCoM - com) * (pos + localCoM - com)
        #moi += float(df[index][8]) - mass2 * (pos + localCoM - com) * (pos + localCoM - com)
    return {"m": mass, "cg": com - minPos, "l": maxPos - minPos, "I_I": moi, "d_finsets": [com - loc for loc in finLoc],
            "d_nozzle": nozzLoc - com, "d_prop": propLoc - com, "d_nose": com - float(df[1][5]) }

def nosecone(diam):

    thickness=(1.0149*diam+0.0017123)-diam

    R = diam/2
    L = diam*6
    theta = lambda x, L: np.arccos(1-2*x/L)
    y = lambda x, theta, R, L: ((R*(theta(x, L)-np.sin(2*theta(x, L))/2)**0.5)/(np.pi**0.5))**2*np.pi
    y2 = lambda x, theta, R, L, thickness: ((R*(theta(x-thickness, L)-np.sin(2*theta(x-thickness, L))/2)**0.5)/(np.pi**0.5))**2*np.pi
    vol = integrate.quad(y,0,L, (theta, R, L))[0] 
    vol2 = integrate.quad(y2,thickness,L-thickness, (theta, R-thickness, L-thickness, thickness))[0] 

    m = (vol-vol2)*887.52
    cg = L*2/3
    I_I = m*cg**2
    I_Z = 1/2*m*(diam/2)**2

    return {"type": "nose", "m": m,"cg": cg,"l": L, "I_I": I_I, "I_Z": I_Z}

def body(diam, section):
    splice=spliceBySection(section)
    l=splice["l"]#m
    #body tube approximations
    odiam=1.0149*diam+0.0017123 #odiam meters, diam meters
    volume=((odiam/2)**2-(diam/2)**2)*np.pi*l #m3
    density=2581.107 #kg/m^3

    m=volume*density+splice["m"] #kg
    cg=(volume*density*splice["l"]/2+splice["m"]*splice["cg"])/m #m
    I_I=splice["I_I"]
    I_Z=1/2*m*(diam/2)**2
    return {"type": "body", "m": m,"cg": cg,"l": l, "I_I": I_I, "I_Z": I_Z}

def motor(diam, motor_params, fin_params):
    odia, dia, height, number, rho, fwd_grain_gap, grain_gap, aft_grain_gap, nozzle_length, mc_thickness = motor_params

    #length of propellant, and length of motor casing
    l_p=fwd_grain_gap+grain_gap*(number-1)+height*number+aft_grain_gap #propellant length, m
    l_mc=l_p+nozzle_length #total motor length, m
    
    #body tube approximations
    odiam=1.0149*diam+0.0017 #odiam meters, diam meters, linear interpolation
    volume=((odiam/2)**2-(diam/2)**2)*np.pi*l_mc #m3
    density=2581.107 #kg/m^3

    #motor casing approximations
    odia_mc=odia+0.01905+mc_thickness*2 #inner diameter plus a quarter inch, m
    diam_mc=odia+0.01905 #m
    volume_mc=((odia_mc/2)**2-(diam_mc/2)**2)*np.pi*l_mc #m3
    density_mc=4000 #kg/m3, approximated # ans = 9977.51782191244

    #calculate the relative location of each grain's center of mass based on grain spacing, m
    grain_locations = np.arange(start=fwd_grain_gap+height/2,step=height+grain_gap,stop=fwd_grain_gap+grain_gap*(number-1)+height*number) #array, m
    grain_mass =(np.pi*(odia/2)**2-np.pi*(dia/2)**2)*height*rho #kg

    #totals
    m=volume*density+volume_mc*density_mc+grain_mass*number #kg
    m_dry=volume*density+volume_mc*density_mc #kg
    m_p=grain_mass*number #kg
    m_motor=grain_mass*number+volume_mc*density_mc #mass of the motor without body casing, kg
    cg=(volume*density*l_mc/2+volume_mc*density_mc*l_mc/2+np.sum(grain_locations*grain_mass))/m #m
    cg_dry=(volume*density*l_mc/2+volume_mc*density_mc*l_mc/2)/m #m

    I_I=1/12*m_p*l_p**2+m_p*(cg-l_p/2)**2+1/12*(m-m_p)*l_mc**2+(m-m_p)*(cg-l_mc/2)**2+1/4*m_p*(diam/2)**2+1/4*(m-m_p)*(diam/2)**2
    I_Z=1/2*m*(odiam/2)**2
    I_I_dry=1/12*(m_dry)*l_mc**2+(m_dry)*(cg-l_mc/2)**2+1/4*(m_dry)*(diam/2)**2
    I_Z_dry=1/2*m_dry*(odiam/2)**2
   
    if fin_params != []: #if fins are present, recalcualte totals
        #calculate fin values
        area_density,fin_buffer,root,span,tip,sweep=fin_params
        fin_buffer = fin_buffer
        fin_area=(tip+root)/2*span
        fin_cg=l_mc-fin_buffer-(root-(tip**2+root**2+tip*root+root*sweep+2*tip*sweep)/(3*(tip+root)))
        fintip=l_mc-fin_buffer-root

        #new totals
        cg=(cg*m+fin_area*area_density*fin_cg)/(m+fin_area*area_density) #m
        cg_dry=(cg_dry*m_dry+fin_area*area_density*fin_cg)/(m_dry+fin_area*area_density) #m
        m=m+fin_area*area_density #kg
        m_dry=m_dry+fin_area*area_density #kg
    
    return {"type": "motor", "m": m,"cg": cg,"l": l_mc, "m_dry": m_dry, "cg_dry": cg_dry, "m_p": m_p, "m_motor": m_motor, "d_fintip": fintip, "I_I": I_I, "I_Z": I_Z, "I_I_dry": I_I_dry, "I_Z_dry": I_Z_dry}

def parachute(diam, diam_chute):
    #tabulated parachute data

    volume = {2: 0.0010295992, 3: 0.0014414061, 4: 0.00175030231, 5: 0.00267699078, 6: 0.00350076848, 7: 0.00602339311} #parachute in feet, volume in meters
    chute_mass = {2: 0.17009714, 3: 0.3203496113, 4: 0.51029142, 5: 0.66904875, 6: 0.89584493, 7: 1.13398} #parachute in feet, parachute mass in kg
    #resulting required tube length based on parachute diameter
    l=volume[diam_chute]/(np.pi*(diam/2)**2)

    #body tube approximations
    odiam=1.0149*diam+0.0017 #odiam meters, diam meters
    volume=((odiam/2)**2-(diam/2)**2)*np.pi*l #m3
    density=2581.107 #kg/m^3

    #totals
    m=volume*density+chute_mass[diam_chute] #kg
    cg=(volume*density*l/2+chute_mass[diam_chute]*l/2)/m #m

    I_I=1/12*m*l**2
    I_Z=1/2*m*(diam/2)**2

    return {"type": "parachute", "m": m,"cg": cg,"l": l, "I_I": I_I, "I_Z": I_Z}

def coupler(diam):
    l = np.max([0, diam-6*0.0254])

    odiam=1.0149*diam+0.0017123 #odiam meters, diam meters
    volume=((odiam/2)**2-(diam/2)**2)*np.pi*l #m3
    density=2000 #kg/m^3
    m=volume*density

    cg = l/2

    I_I=1/12*m*l**2
    I_Z=1/2*m*(diam/2)**2

    return {"type": "coupler", "m": m,"cg": cg,"l": l, "I_I": I_I, "I_Z": I_Z}


class Translator:
    def __init__(self, data_output = False, diam = 6, motor_1 = [], motor_2 = [], fin_params_1 = [], fin_params_2 = [], diam_chute_1 = 0, diam_chute_2 = 0, diam_chute_3 = 0):
        nose=nosecone(diam) #nose cone only
        nose_internals=body(diam,1) #nose cone internals
        nose_chute=parachute(diam, diam_chute_3) #nose cone parachute
        nose_coupler = coupler(diam) #nose coupler additional length
        sustainer_chute=parachute(diam, diam_chute_2) #sustainer parachute
        sustainer=body(diam,2) #everything in the sustainer until the motor
        sustainer_motor=motor(diam, motor_2, fin_params_2) #sustainer motor and fins
        staging_upper=body(diam,3) #staging upper
        staging_lower=body(diam,4) #staging lower
        booster_chute=parachute(diam, diam_chute_1) #booster parachute
        booster=body(diam,5) #everything in the booster until the motor
        booster_motor=motor(diam, motor_1, fin_params_1) #booster motor and fins
    
        #storing important measures for RASAERO
        self.booster_length = booster_motor["l"]
        self.booster_prop_weight = booster_motor["m_p"]
        self.booster_tot_weight = booster_motor["m_motor"]

        self.sustainer_length = sustainer_motor["l"]
        self.sustainer_prop_weight = sustainer_motor["m_p"]
        self.sustainer_tot_weight = sustainer_motor["m_motor"]

        #defining full rocket in order from top to bottom, with zeros denoting a staging seperation
        self.stack = np.array([nose, nose_internals, nose_chute, nose_coupler, 0, sustainer_chute, sustainer, sustainer_motor, staging_upper, 0, staging_lower, booster_chute, booster, booster_motor])

        #defining rocket stages
        #full stack
        self.event1 = [nose, nose_internals, nose_chute, nose_coupler, sustainer_chute, sustainer, sustainer_motor, staging_upper, staging_lower, booster_chute, booster, booster_motor]
        #sustainer coast
        self.event2 = [nose, nose_internals, nose_chute, nose_coupler, sustainer_chute, sustainer, sustainer_motor, staging_upper]
        #sustainer
        self.event3 = [nose, nose_internals, nose_chute, nose_coupler, sustainer_chute, sustainer, sustainer_motor, staging_upper]
        #booster recovery
        self.event4 = [staging_lower, booster_chute, booster, booster_motor]
        #sustainer recovery
        self.event5 = [sustainer_chute, sustainer, sustainer_motor, staging_upper]
        #nose recovery
        self.event6 = [nose, nose_internals, nose_chute, nose_coupler]

    
    def event(segment = [], wet = False): #calculate important measures for simulation event
        #find number of motors in this segment
        num_motors = len([x for x in segment if x["type"]=="motor"])

        #determining cg, m, and length for current rocket segment
        cg = 0
        l = 0
        m = 0
        I_I = 0
        I_Z = 0
        body_length = 0
        current_motor = num_motors
        
        # wet = True

        for x in segment:
            if x["type"] == "motor":
                if current_motor==1 and not wet:
                    # breakpoint()
                    cg=(cg*m+(l+x["cg_dry"])*x["m_dry"])/(m+x["m_dry"])
                    m+=x["m_dry"]
                    # print("last motor, m, l:", 1/12*x["m_dry"]*x["l"]**2+x["m_dry"]*(l+x["l"]/2)**2, x["m"], l)
                else:
                    # breakpoint()
                    cg=(cg*m+(l+x["cg"])*x["m"])/(m+x["m"])
                    m+=x["m"]
                    current_motor-=1
                    # print("not last motor, m, l:", 1/12*x["m"]*x["l"]**2+x["m"]*(l+x["l"]/2)**2, x["m"], l)
            else:
                cg=(cg*m+(l+x["cg"])*x["m"])/(m+x["m"])
                m+=x["m"]
            if x["type"] != "nose":
                body_length += x["l"]
            l+=x["l"]
        measures={"wet": wet, "m": float(m),"cg": cg,"body length": body_length}
 
        #determining important measures based off of total cg of segment
        d_finsets = []
        l = 0
        d_nozzle = 0
        d_prop = 0
        d_nose = 0
        current_motor = num_motors
        
        for x in segment:
            # print(x)
            #if this component is a motor
            if x["type"] == "motor":
                d_finsets.append(float(cg-(l+x["d_fintip"])))
                if current_motor==1 and not wet:
                    d_prop=abs(cg-(l+x["cg"]))
                    d_nozzle = abs(cg-(l+x["l"]))
                    I_I+=x["I_I_dry"]+x["m_dry"]*((l+x["cg_dry"])-cg)**2
                    I_Z+=x["I_Z"]
                else:
                    current_motor-=1
                    I_I+=x["I_I"]+x["m"]*((l+x["cg"])-cg)**2
                    I_Z+=x["I_Z"]
            else:
                I_I+=x["I_I"]+x["m"]*((l+x["cg"])-cg)**2
                I_Z+=x["I_Z"]
            l+=x["l"]
            #if this component is a nose cone, fine the nose distance
            if x["type"] == "nose":
                d_nose = abs(cg-l)

        measures["d_nozzle"]=float(d_nozzle)
        measures["d_prop"]=float(d_prop)
        measures["d_nose"]=float(d_nose)
        measures["d_finsets"]=d_finsets
        measures["l"]=l
        measures["I_I"]=float(I_I)
        measures["I_Z"]=float(I_Z)

        # print info for all components for debugging
        # for x in segment:
        #     print("Type: %s    Mass: %.3f    Length: %.3f   I_I: %.3f   " % (x["type"], x["m"], x["l"], x["I_I"]))
        return measures