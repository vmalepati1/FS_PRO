import pandas as pd
import numpy as np
import pygmo as pg
import matplotlib.pyplot as plt
import os

from rocketpy import Rocket, SolidMotor, Flight
from Translator import Translator
from AerodynamicNN import score, scoreNN
import csv
from datetime import datetime

class Udp:
    def __init__(
        self, 
        lower_bounds, 
        upper_bounds, 
        decision_dict,
        Env,
        num_int,
        num_constraints,
        identical_motors = False, 
        identical_fins = False, 
        post_process = False, 
        data_output = False
        ):

        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.decision_dict = decision_dict
        self.Env = Env
        self.num_int = num_int
        self.num_constraints = num_constraints
        self.identical_motors = identical_motors
        self.identical_fins = identical_fins
        self.post_process = post_process
        self.data_output = data_output




    def pratio_to_mach(self, pratio, gamma=1.4):
        """ Function to calculate Mach number from pressure ratio and gamma
            :param pratio: pressure ratio P/P0
            :param gamma: ratio of specific heats (cp/cv)
            :return mach: mach number
        """

        mach = np.sqrt((pratio ** (-(gamma - 1) / gamma) - 1) * (2 / (gamma - 1)))
        return mach


    def mach_to_tratio(self, mach, gamma=1.4):
        """ Function to calculate temperature ratio from Mach number and gamma
            :param mach: mach number
            :param gamma: ratio of specific heats (cp/cv)
        """

        tratio = (1 + (gamma - 1) / 2 * mach ** 2) ** (-1)
        return tratio


    def mach_to_aratio(self, mach, gamma=1.4):
        """ Function to calculate area ratio from Mach number and gamma)
            :param mach: mach number
            :param gamma: ratio of specific heats (cp/cv)
        """
        k = gamma

        aratio = ((k + 1) / 2) ** (-(k + 1) / (2 * (k - 1))) * (
                    (1 + (k - 1) / 2 * mach ** 2) ** ((k + 1) / (2 * (k - 1)))) / mach
        return aratio


    def prop_sim(self, MEOP, OD, ID, L, n_grain, P_a = 14.7):
        """ Function to simulate the performance of a solid rocket motor given geometric parameters
            :param A_t      :   throat area (in**2)
            :param OD       :   outer grain diameter (in)
            :param ID       :   inner grain diamter (in)
            :param L        :   grain length (in)
            :param n_grain  :   number of grains

            :return m_flux: max mass flux (slug/ft^2/s)
            :return m_pc: max chamber pressure (psi)
            :return M: propellant weight (lbs)
            :return m_F: max thrust (newtons)
            :return J: impulse (newton-seconds)
            :return n_grain: number of grains
            :return OD: outer grain diameter (in)
            :return ID: inner grain diameter (in)
            :return L: grain length (in)
            :return A_t: throat area (in**2)
            :return thrust_hist: thrust history array (seconds, newtons)
        """

        L = L / n_grain

        # Ambient conditions
        P_a

        # Propellant constants
        a = 0.027
        n = 0.3
        k = 1.2
        R = 2000
        T_0 = 4700
        rho_prop = 0.00176
        c_star = 4890

        # Motor geometry
        V = (np.pi*(OD/2)**2 - np.pi*(ID/2)**2)*L*n_grain
        W = V*rho_prop*32.2

        As = []
        Aps = []
        dr = 1/1000

        while ID < OD and L >= 0:
            A_end = (np.pi*(OD/2)**2 - np.pi*(ID/2)**2)*2
            A_port = 2*np.pi*(ID/2)*L

            As.append((A_end + A_port)*n_grain)
            Aps.append(np.pi*(ID/2)**2)

            ID += 2*dr
            L -= 2*dr

        As = np.array(As)
        Aps = np.array(Aps)

        # Calculate throat area, pressure, burn rate, and mass flow
        A_t = MEOP**(n - 1)*max(As)*c_star*a*rho_prop
        Pcs = ((As/A_t)*c_star*a*rho_prop)**(1/(1-n))
        Brs = a*Pcs**n
        mdots = Brs*As*rho_prop

        # Calculate time
        dts = dr/Brs
        ts = np.cumsum(dts)

        # Calculate exit conditions
        pratio = P_a/max(Pcs)
        M_e = self.pratio_to_mach(pratio, k)
        T_e = self.mach_to_tratio(M_e, k)*T_0
        v_e = M_e*np.sqrt(k*R*T_e)
        eps = self.mach_to_aratio(M_e, k)
        A_e = eps*A_t
        P_e = pratio*Pcs

        # Thrust and mass fluxes
        F = mdots*v_e + A_e*(P_e - P_a)
        flux = mdots/Aps

        # Values to return
        F *= 4.44822
        thrust_hist = np.column_stack((ts, F))
        return max(flux), max(Pcs), W, max(F), np.trapz(F, ts), A_t, thrust_hist










    # def prop_sim(self, A_t, odia, dia, length, number):
        
    #     a = 0.027 #coeff of pressure
    #     n = 0.3 #burning rate exponent (dimensionless)
    #     gamma = 1.2 #specific heat ratio (dimensionless)
    #     M = 2 #mach
    #     R = 2000 #gas constant
    #     T = 4700 #temp R
    #     c_star = 4890 #ft/sec
    #     rho_prop = 0.00176 #slug/in^3
    #     p_a = 14.7 #psi
    #     num = round(number)
    #     inhib = 0

    #     height = length/number

    #     w_grain = 0.5*(odia-dia)
    #     dia_vec = dia*np.ones(num)
    #     h_vec = height*np.ones(num)

    #     #isentropic relationships
    #     exp_r = 1.884
    #     A_exit = exp_r*A_t #exit area from expansion ratio
    #     t_e = T*((1+((gamma-1)/2)*M**2)**-1) #adjust temp

    #     #initialize
    #     Rad = 0.5*odia
    #     Pressure = np.array([])
    #     Thrust = np.array([])
    #     time = np.array([])
    #     Burn_Rate = np.array([])
    #     Mass_flow = np.array([])

    #     #find distance steps

    #     odia_vec = odia*np.ones(num)
    #     diff_vec = odia_vec - dia_vec
    #     max_dis = np.max(diff_vec)
    #     d_step = max_dis/2000
    #     d_step_vec = d_step*np.ones(1000)

    #     #mass flux calc

    #     A_port_start = (np.pi*(0.5*dia)**2)
    #     A_port = np.array([A_port_start])
    #     port_dia = np.array([dia])

    #     for q in range(0,d_step_vec.size-1):
    #         port_dia = np.append(port_dia, port_dia[q] + 2*(d_step))
    #         A_port = np.append(A_port, (np.pi*(0.5*port_dia[q+1])**2))

    #     Total_A_b_vec = np.array([])
    #     for y in range(0,d_step_vec.size):
    #         grains = np.array([])
    #         for w in range(0, num):
    #             if np.any((h_vec - 2*d_step)<=0):                     
    #                 h_vec = h_vec * 0
    #             dia = dia_vec[w]
    #             h = h_vec[w]
    #             rad = 0.5*dia
    #             a_inside = np.pi*dia*h
    #             a_circle = (np.pi*Rad**2)-(np.pi*rad**2)
    #             a_circle_total = a_circle*(2-inhib)
    #             A_grain = a_inside+a_circle_total
    #             diff = odia-dia
    #             if diff<=0:
    #                 A_grain = 0;
    #             grains = np.append(grains, A_grain)
    #         Burn_Area = sum(grains)
    #         Total_A_b_vec = np.append(Total_A_b_vec, Burn_Area)
    #         dia_vec = dia_vec + 2*d_step
    #         if inhib ==0:
    #             h_vec = h_vec - 2*d_step
        
    #     for c in Total_A_b_vec:
    #         A_b = c + 1
    #         K_n = A_b/A_t
    #         P_c = (K_n*c_star*a*rho_prop)**(1/(1-n)) #FIX THIS LINE, IT IS CAUSING US TO LOSE THRUST

    #         #Thrust
    #         #Calc m_dpt
    #         r_b = a*P_c**n
    #         m_dot = rho_prop*A_b*r_b
    #         #Calc exit pressure
    #         ratio_p = (1+((gamma-1)/2)*(M**2))**(-gamma/(gamma-1))
    #         p_exit = P_c*ratio_p
    #         #calc exit velocity
    #         v_e = M*(np.sqrt(gamma*R*t_e))
    #         F = (m_dot*v_e)+A_exit*(p_exit-p_a)

    #         if F < 0:
    #             F = 0

    #         Pressure = np.append(Pressure, P_c)
    #         Thrust = np.append(Thrust, F)
    #         Mass_flow = np.append(Mass_flow, m_dot)

    #         time_step = d_step/r_b
    #         if time.size == 0:
    #             time = np.append(time, time_step)
    #         else:
    #             time = np.append(time, time[time.size-1] + time_step)

    #     m_flux_vec = Mass_flow/A_port
    #     m_flux = np.max(m_flux_vec)*32.17
    #     p_max = max(Pressure)
    #     p_min = min(Pressure)
    #     if time.size == 0:
    #         tot_impulse = 0
    #     else:
    #         tot_impulse = np.trapz(Thrust, time)
    #         if np.iscomplex(tot_impulse):
    #             tot_impulse = 0
    #     thrust = np.column_stack((time, Thrust*4.448)).tolist()
    #     return m_flux, p_max, p_min, tot_impulse, thrust
    
    def event_sim(
        self, 
        A_t, 
        odia, 
        dia, 
        length, 
        number, 
        roots, 
        spans, 
        tips, 
        sweeps, 
        burnOut,
        diam,
        event, 
        CDOn,
        CDOff,
        Env,
        terminateOnApogee,
        trigger,
        chute = 0,
        initial = [],
        maxTime = 6000,
        thrust = [],
        recovery = False,
        ):

        # ensure input values are not numpy values
        length = float(length)
        number = int(number)
        odia = float(odia)
        dia = float(dia)
        A_t = float(A_t)
        height = length/number

        if thrust != []: # if motor is not empty, find burn out time
            burnOut = thrust[-1][0]
        
        if thrust == []: # if motor is emtpy, construct an empty motor
            motor = SolidMotor( #Empty Motor
            thrustSource="./Empty.eng",
            burnOut=0.02,
            grainNumber=1,
            grainSeparation=0.0015875,
            grainDensity=0.0001,
            grainOuterRadius=(odia*0.0254)/2,
            grainInitialInnerRadius=(dia*0.0254)/2,
            grainInitialHeight=(height*0.0254),
            nozzleRadius=0.113284/2,
            throatRadius=np.sqrt(A_t/np.pi)*0.0254,
            interpolationMethod='linear'
            )
        else: # if motor is not empty, construct motor based on inputs
            motor = SolidMotor(
            thrustSource=thrust,
            burnOut=burnOut,
            grainNumber=number,
            grainSeparation=0.0015875,
            grainDensity=1567.4113,
            grainOuterRadius=(odia*0.0254)/2,
            grainInitialInnerRadius=(dia*0.0254)/2,
            grainInitialHeight=(height*0.0254),
            nozzleRadius=0.113284/2,
            throatRadius=np.sqrt(A_t/np.pi)*0.0254,
            interpolationMethod='linear'
            )
        Event = Rocket( # create rocket object
            motor=motor,
            radius=diam/2*0.0254,
            mass=float(event["m"]),
            inertiaI=abs(float(event["I_I"])),
            inertiaZ=float(event["I_Z"]),
            distanceRocketNozzle=float(event["d_nozzle"]),
            distanceRocketPropellant=float(event["d_prop"]),
            powerOffDrag=CDOff,
            powerOnDrag=CDOn
            )

        # Event.centerOfMass
        #event["cg"]
        Event.setRailButtons([0.1, -0.2])

        if not recovery: # add aero surfaces 
            Event.addNose(length=diam*6*0.0254, kind="Von Karman", distanceToCM=event["d_nose"])

            for i in range(len(event["d_finsets"])):
                Event.addFins(4, span=spans[i]*0.0254, rootChord=roots[i]*0.0254, tipChord=tips[i]*0.0254, distanceToCM=event["d_finsets"][i])

        if chute != 0: # add parachute
            Event.addParachute('Main',
                CdS=.97*np.pi*(0.3048*chute/2)**2,
                trigger=trigger,
                samplingRate=105,
                lag=1.5,
                noise=(0, 8.3, 0.5))

        if initial != []: # simulate flight
            flight = Flight(rocket=Event, environment = Env, initialSolution = initial, timeOvershoot = True, terminateOnApogee = terminateOnApogee, maxTime = maxTime, verbose = False)
        else:
            flight = Flight(rocket=Event, environment = Env, inclination=90, heading=0, timeOvershoot = True, terminateOnApogee = terminateOnApogee, maxTime = maxTime, verbose = False)
        
        if chute == 0: # if the event is not a recovery event, run RocketPy's postProcess.
            flight.postProcess()

        return flight, Event, motor

    def fitness (self, x0):

        if self.post_process:
            x = x0
        else:
            x = self.lower_bounds.copy()
            for i, key in enumerate(x.keys()):
                x[key] = x0[i]





        if self.identical_motors:
            self.decision_dict['MEOP_2'] = x['MEOP_1']
            self.decision_dict['odia_2'] = x['odia_1']
            self.decision_dict['dia_2'] = x['dia_1']
            self.decision_dict['length_2'] = x['length_1']
            self.decision_dict['number_2'] = x['number_1']
        if self.identical_fins:
            self.decision_dict['root_2'] = x['root_1']
            self.decision_dict['span_2'] = x['span_1']
            self.decision_dict['tip_2'] = x['tip_1']
            self.decision_dict['sweep_2'] = x['sweep_1']

        viable_diams = [6, 7.76, 8.75, 11.67] 
        casing_thickness = [1, 1.135, 1.0, 1.5][int(x['diameter'])]

        # Casing thickness in inches = 4 * Max pressure in PSI * radius in inches / 30,000 psi


        x['diameter'] = viable_diams[int(x['diameter'])]
        self.decision_dict.update(x)
        x = self.decision_dict

        objective = {
            'apogee': 0
        }

        casing_thickness_1 = 4.5*x['MEOP_1']*(x['odia_1']/2+0.375)/30000
        casing_thickness_2 = 4.5*x['MEOP_2']*(x['odia_2']/2+0.375)/30000

        motor_constraints = {
            'diameter>odia': x['odia_1']-x['diameter'] + casing_thickness_1*2 + 0.75,
            'dia<odia': x['dia_1'] - x['odia_1'],
            'height<12': x['length_1']/x['number_1'] - 12,
            'A_t<A_dia': 0,
            'm_flux<1.2': 0,
            'burnOut<12': 0,
        }
        
        if not self.identical_motors:
            motor_constraints['diameter>odia_2'] = x['odia_2']-x['diameter'] + casing_thickness_2*2 + 0.75
            motor_constraints['dia_2<odia_2'] = x['dia_2'] - x['odia_2']
            motor_constraints['height_2<12'] = x['length_2']/x['number_2'] - 12
            motor_constraints['A_t_2<A_dia_2'] = 0
            motor_constraints['m_flux_2<1.2'] = 0
            motor_constraints['burnOut_2<12'] = 0

        fin_constraints = {
            'non_swept': x['tip_1'] + x['sweep_1'] - x['root_1'],
            'fin_cg': x['root_1'] - x['sweep_1'] - x['tip_1'],
        }

        # fin_constraints = {
        #     'non_swept': -1,
        #     'fin_cg': -1,
        # }

        if not self.identical_fins:
            fin_constraints['non_swept_2'] = x['tip_2'] + x['sweep_2'] - x['root_2']
            fin_constraints['fin_cg_2'] = x['root_2'] - x['sweep_2'] - x['tip_2']

        stability_constraints = {
            'boosterStart>3': 0,
            'outOfRailVelocity>100': 0,
            'sustainerStart>2': 0,
            'stableWhenSonic': 0,
            'maxQ>2.25': 0
        }

  
        objective.update(motor_constraints)
        objective.update(fin_constraints)
        objective.update(stability_constraints)


        objective["fin_cg"] = -1
        objective["fin_cg_2"] = -1

        if objective['dia<odia'] > 0:
            return list(objective.values())
        elif not self.identical_motors and (objective['dia<odia'] > 0 or objective['dia_2<odia_2'] > 0):
            return list(objective.values())

        """
        -------------------------------------------------------- Translator -------------------------------------------------------- 
        """
        
        # A_t, odia, dia, height, number, rho, fwd_grain_gap, grain_gap, aft_grain_gap, nozzle_length
        motor_1 = [x["odia_1"]*0.0254, x["dia_1"]*0.0254, x["length_1"]/int(x["number_1"])*0.0254, round(x["number_1"]), 0.00176*890575, 0.001778, 0.0015875, 0.0015875, 7.223*0.0254, casing_thickness_1*0.0254]
        
        motor_2 = [x["odia_2"]*0.0254, x["dia_2"]*0.0254, x["length_2"]/int(x["number_2"])*0.0254, round(x["number_2"]), 0.00176*890575, 0.001778, 0.0015875, 0.0015875, (7.223+1.22)*0.0254, casing_thickness_2*0.0254]
        #area density, fin_buffer, root, span, tip, sweep
        fin_params_1 = [28, 1.5*0.0254, x["root_1"]*0.0254, x["span_1"]*0.0254, x["tip_1"]*0.0254, x["sweep_1"]*0.0254] #1.5 inch fin buffer from base
        fin_params_2 = [28, 1.5*0.0254, x["root_2"]*0.0254, x["span_2"]*0.0254, x["tip_2"]*0.0254, x["sweep_2"]*0.0254]
        
        rocket = Translator(
            data_output = self.data_output, 
            diam = x["diameter"]*0.0254, 
            motor_1 = motor_1, 
            motor_2 = motor_2, 
            fin_params_1 = fin_params_1, 
            fin_params_2 = fin_params_2, 
            diam_chute_1 = x["diachute_1"], 
            diam_chute_2 = x["diachute_2"], 
            diam_chute_3 = x["diachute_3"]
        )
        
        event1 = Translator.event(rocket.event1, False) #booster
        event2 = Translator.event(rocket.event2, True) #sustainer coast, boolean set to true to include all propellant weight during coast
        event3 = Translator.event(rocket.event3, False) #sustainer
        event4 = Translator.event(rocket.event4, False) #booster recovery
        event5 = Translator.event(rocket.event5, False) #sustainer recovery
        event6 = Translator.event(rocket.event6, False) #nose recovery

        if self.post_process == True:



            event1_wet=Translator.event(rocket.event1, True)
            event2_wet=Translator.event(rocket.event2, True)
            event3_wet=Translator.event(rocket.event3, True)
            event4_wet=Translator.event(rocket.event4, True)
            event5_wet=Translator.event(rocket.event5, True)
            event6_wet=Translator.event(rocket.event6, True)

            print("RP Weights")
            rows =['booster', 'sustainer coast', 'sustainer', 'booster recovery', 'sustainer recovery', 'nose recovery']
            df_x=pd.DataFrame([list(event1.values()),list(event2.values()),list(event3.values()),list(event4.values()),list(event5.values()),list(event6.values())], columns=list(event1.keys()))
            df_x.insert(0,'Event', rows)
            print(df_x)
            
            print("Total Weights")
            df_x = pd.DataFrame([list(event1_wet.values()), list(event2_wet.values()), list(event3_wet.values()), list(event4_wet.values()), list(event5_wet.values()), list(event6_wet.values())], columns =list(event1.keys()))
            df_x.insert(0,'Event', rows)
            print(df_x)

            print("RASAERO Values")
            print("Sustainer CG (in): ", event2_wet["cg"]*39.3701)
            print("Sustainer Loaded Weight (lbs): ", event2_wet["m"]*2.20462)
            print("Ignition Delay (sec): ", x['ignitionDelay'])
            print("Combined CG (in): ", event1_wet["cg"]*39.3701)
            print("Total Loaded Weight (lbs): ", event1_wet["m"]*2.20462)
            print("nose cone (in):", x['diameter']*6)
            print("sustainer tube (in):", event5_wet["l"]*39.3701)
            print("booster tube (in):", event4_wet["l"]*39.3701)



        """
        -------------------------------------------------------------- First Motor Sim -------------------------------------------------------- 
        """


        m_flux_1, m_pc_1, W_1, m_F_1, J_1, A_t_1, thrust_1 = self.prop_sim(x["MEOP_1"], x["odia_1"], x["dia_1"], x["length_1"], int(x["number_1"])) #Motor 1
        x['A_t_1'] = A_t_1
 
        objective['A_t<A_dia'] = x['A_t_1']*1.5 - np.pi*((x['dia_1']/2)**2)
        objective['m_flux<1.2'] = m_flux_1 - 1.2
        objective['burnOut<12'] = thrust_1[-1][0] - 12

        if event1["m"] < 0 or event2["m"] < 0 or event1["m"]*30 >= np.max(thrust_1) or thrust_1[-1][0] <2:
            return list(objective.values())



        """
        -------------------------------------------------------------- Drag Coeff Neural Net -------------------------------------------------------- 
        """
        
        #Initialize list and values for drag approximation
        bs_power_on = []
        bs_power_off = []
        s_power_on = []
        s_power_off = []
        machs = np.arange(0, 6, 0.1)

        for mach in machs:
            cd_point = {"Body Diameter B+S": x["diameter"],
            "Body Length B": event4["body length"] * 39.3701, 
            "Body Length S": event5["body length"] * 39.3701, 
            "Root B": x["root_1"], 
            "Span B": x["span_1"], 
            "Sweep B": x["sweep_1"], 
            "Tip B": x["tip_1"], 
            "Root S": x["root_2"], 
            "Span S": x["span_2"], 
            "Sweep S": x["sweep_2"], 
            "Tip S": x["tip_2"], 
            "Mach Number": mach
        }
            cds = score(cd_point)
            bs_power_on.append(cds["Predicted B+S Power On"])
            bs_power_off.append(cds["Predicted B+S Power Off"])
            s_power_on.append(cds["Predicted S Power On"])
            s_power_off.append(cds["Predicted S Power Off"])

        #Construct drag coefficient lists for rocketpy
        bs_power_on = np.column_stack((machs, bs_power_on)).tolist()
        bs_power_off = np.column_stack((machs, bs_power_off)).tolist()
        s_power_on = np.column_stack((machs, s_power_on)).tolist()
        s_power_off = np.column_stack((machs, s_power_off)).tolist()


        """
        -------------------------------------------------------------- Simulation Run -------------------------------------------------------- 
        """

        def trigger(p, y):
            return True if y[5] < 0 else False

        
        def calculateStability(flight, rocket, event, times, type, csv_output_filename):
            staticMargin = []
            cg = []
            rows = []

      

            with open(csv_output_filename, 'w', newline='') as f:

                writer = csv.writer(f)

                writer.writerow(['t', 'Mach Number', 'cg (in)', 'cp (in)'])

                for t in times:
                    cd_point = {"Body Diameter B+S": x["diameter"],
                    "Body Length B": event4["body length"] * 39.3701, 
                    "Body Length S": event5["body length"] * 39.3701,  
                    "Root B": x["root_1"], 
                    "Span B": x["span_1"], 
                    "Sweep B": x["sweep_1"], 
                    "Tip B": x["tip_1"], 
                    "Root S": x["root_2"], 
                    "Span S": x["span_2"], 
                    "Sweep S": x["sweep_2"], 
                    "Tip S": x["tip_2"], 
                    "Mach Number": flight.MachNumber(t)
                    }

                    output = scoreNN(cd_point)

                    cg_n = (event["cg"] + rocket.centerOfMass(t))*39.3701
                    cp_n = output["Predicted CP B+S"]

                    cg.append(cg_n)

                    writer.writerow([t, flight.MachNumber(t), cg_n, cp_n])

                    if type.lower() == "booster":
                        staticMargin.append((output["Predicted CP B+S"] - (event["cg"] + rocket.centerOfMass(t))*39.3701)/x["diameter"])
                    elif type.lower() == "sustainer":
                        staticMargin.append((output["Predicted CP S"] - (event["cg"] + rocket.centerOfMass(t))*39.3701)/x["diameter"])


            return staticMargin
            



        booster, booster_rocket, motor1 = self.event_sim(
            thrust = thrust_1,
            A_t = x["A_t_1"],
            odia = x["odia_1"],
            dia = x["dia_1"],
            length = x["length_1"],
            number = x["number_1"],
            roots = [x["root_2"], x["root_1"]],
            spans = [x["span_2"], x["span_1"]],
            tips = [x["tip_2"], x["tip_1"]],
            sweeps = [x["sweep_2"], x["sweep_1"]],
            burnOut = thrust_1[-1][0],
            diam = x["diameter"],
            event = event1,
            CDOn = bs_power_on,
            CDOff = bs_power_off,
            Env = self.Env,
            trigger = trigger,
            maxTime = thrust_1[-1][0] + 2,
            terminateOnApogee = False
        )


        times = np.arange(0, booster.solution[-1][0],  0.01)

        boosterStability = calculateStability(booster, booster_rocket, event1, times, "booster", 'Booster_StabilityData.csv')

        times = [x[0] for x in booster.MachNumber if x[1] > 1]
        boosterStabilityMach1 = calculateStability(booster, booster_rocket, event1, times, "booster", 'UselessStabilityData')

        boosterStartStability = calculateStability(booster, booster_rocket, event1, [0.01], "booster", 'UselessStabilityData')[0]
        
        objective['boosterStart>3'] = 3 - boosterStartStability
        objective['outOfRailVelocity>100'] = 40 - booster.outOfRailVelocity * 3.28084


        if booster.solution[-1][6]<=0 or booster.apogee > 0:
            return list(objective.values())
        

        initial = booster.solution[-1].copy()
        initial[0] = 0

        sustainer_coast, sustainer_coast_rocket, _ = self.event_sim(
            A_t = x["A_t_1"],
            odia = x["odia_1"],
            dia = x["dia_1"],
            length = x["length_1"],
            number = x["number_1"],
            roots = [x["root_2"], x["root_1"]],
            spans = [x["span_2"], x["span_1"]],
            tips = [x["tip_2"], x["tip_1"]],
            sweeps = [x["sweep_2"], x["sweep_1"]],
            burnOut = 0.2,
            diam = x["diameter"],
            event = event2,
            CDOn = s_power_on,
            CDOff = s_power_off,
            Env = self.Env,
            initial = initial,
            trigger = trigger,
            maxTime = x["ignitionDelay"],
            terminateOnApogee = False
        )
        
        times = np.arange(0, sustainer_coast.solution[-1][0],  0.01)
        sustainer_coastStability = calculateStability(sustainer_coast, sustainer_coast_rocket, event2, times, "sustainer", 'SustainerCoast_StabilityData.csv')

        times = [x[0] for x in sustainer_coast.MachNumber if x[1] > 1]
        sustainer_coastStabilityMach1 = calculateStability(sustainer_coast, sustainer_coast_rocket, event2, times, "sustainer", 'UselessStabilityData')



        if sustainer_coast.solution[-1][6]<25:
            return list(objective.values())

        if not self.identical_motors:
            P_a = self.Env.pressure(sustainer_coast.solution[-1][3])/6894.76
            m_flux_2, m_pc_2, W_2, m_F_2, J_2, A_t_2, thrust_2 = self.prop_sim(x["MEOP_2"], x["odia_2"], x["dia_2"], x["length_2"], int(x["number_2"]), P_a) #Motor 2
            x['A_t_2'] = A_t_2

            objective['m_flux_2<1.2'] = m_flux_2 - 1.2
            objective['A_t_2<A_dia_2'] = x['A_t_2']*1.5 - np.pi*((x['dia_2']/2)**2)
            objective['burnOut_2<12'] = thrust_2[-1][0] - 12
        else:
            m_flux_2, m_pc_2, W_2, m_F_2, J_2, A_t_2, thrust_2 = (m_flux_1, m_pc_1, W_1, m_F_1, J_1, A_t_1, thrust_1)
            x['A_t_2'] = A_t_1



        if event3["m"] < 0 or event3["m"]*30 >= np.max(thrust_2) or thrust_2[-1][0] < 2:
            return list(objective.values())
        

        initial = sustainer_coast.solution[-1].copy()
        initial[0] = 0

        # if initial[6] < 0:
        #     return list(objective.values())

        sustainer, sustainer_rocket, motor2 = self.event_sim(
            thrust = thrust_2,
            A_t = x["A_t_2"],
            odia = x["odia_2"],
            dia = x["dia_2"],
            length = x["length_2"],
            number = x["number_2"],
            roots = [x["root_2"]],
            spans = [x["span_2"]],
            tips = [x["tip_2"]],
            sweeps = [x["sweep_2"]],
            burnOut = thrust_2[-1][0],
            diam = x["diameter"],
            event = event3,
            CDOn = s_power_on,
            CDOff = s_power_off,
            Env = self.Env,
            initial = initial,
            trigger = trigger,
            terminateOnApogee = True
        )

        times = np.arange(0, sustainer.solution[-1][0],  0.01)

        sustainerStability = calculateStability(sustainer, sustainer_rocket, event3, times, "sustainer", 'Sustainer_StabilityData.csv')

        times = [x[0] for x in sustainer.MachNumber if x[1] > 1]

        sustainerStabilityMach1 = calculateStability(sustainer, sustainer_rocket, event3, times, "sustainer", 'UselessStabilityData')

        sustainerStartStability = calculateStability(sustainer, sustainer_rocket, event3, [0.01], "sustainer", 'UselessStabilityData')[0]

        maxQStability = calculateStability(sustainer, sustainer_rocket, event3, [sustainer.maxDynamicPressureTime], "sustainer", 'UselessStabilityData')[0]



        objective['sustainerStart>2'] = 2 - sustainerStartStability
        objective['maxQ>2.25'] = 2.25 - boosterStartStability


        if initial[6] < 0:
            return list(objective.values())

        initial = [x for x in booster.solution if x[6] > 0][-1].copy()
        initial[0] = 0


        boosterRecovery, _, _ = self.event_sim(
            A_t = x["A_t_1"],
            odia = x["odia_1"],
            dia = x["dia_1"],
            length = x["length_1"],
            number = x["number_1"],
            roots = [x["root_1"]],
            spans = [x["span_1"]],
            tips = [x["tip_1"]],
            sweeps = [x["sweep_1"]],
            burnOut = 0.2,
            diam = x["diameter"],
            event = event4,
            CDOn = s_power_on,
            CDOff = s_power_off,
            Env = self.Env,
            initial = initial,
            trigger = trigger,
            chute = x["diachute_1"],
            terminateOnApogee = False,
            recovery = True
        )

        initial = [x for x in sustainer.solution if x[6] > 0][-1].copy()
        initial[0] = 0


        sustainerRecovery, _, _ = self.event_sim(
            A_t = x["A_t_2"],
            odia = x["odia_2"],
            dia = x["dia_2"],
            length = x["length_2"],
            number = x["number_2"],
            roots = [x["root_2"]],
            spans = [x["span_2"]],
            tips = [x["tip_2"]],
            sweeps = [x["sweep_2"]],
            burnOut = 0.2,
            diam = x["diameter"],
            event = event5,
            CDOn = s_power_on,
            CDOff = s_power_off,
            Env = self.Env,
            initial = initial,
            trigger = trigger,
            chute = x["diachute_2"],
            terminateOnApogee = False,
            recovery = True
        )


        noseRecovery, _, _ = self.event_sim(
            A_t = x["A_t_2"],
            odia = x["odia_2"],
            dia = x["dia_2"],
            length = x["length_2"],
            number = x["number_2"],
            roots = [x["root_2"]],
            spans = [x["span_2"]],
            tips = [x["tip_2"]],
            sweeps = [x["sweep_2"]],
            burnOut = 0.2,
            diam = x["diameter"],
            event = event6,
            CDOn = s_power_on,
            CDOff = s_power_off,
            Env = self.Env,
            initial = initial,
            trigger = trigger,
            chute = x["diachute_3"],
            terminateOnApogee = False,
            recovery = True
        )
        objective["apogee"] = -noseRecovery.apogee/1000

        objective['stableWhenSonic'] = 2 - (min(sustainerStabilityMach1 + boosterStabilityMach1 + sustainer_coastStabilityMach1))


        breakpoint()

        vector = pd.DataFrame(self.decision_dict, [0])
        fitness = pd.DataFrame(objective, [0])
        print(vector.round(4))
        print(fitness.round(4))
        print('\n')


        if self.post_process == True:

            i = 0.02
            f = open("FS_PRO_motor.eng", "w")
            f.write("Booster %.5f %.5f %s %.5f %.5f %s\n" %(x['diameter']*25.4, rocket.booster_length, "P", rocket.booster_prop_weight, rocket.booster_tot_weight, "FS_PRO"))
            while i < thrust_1[-1][0]:
                f.write("%.8f  %.6f\n" %(i, motor1.thrust(i)))
                i+=0.05
            f.write("%.8f  %.6f\n" %(i+0.01, 0))
            f.write(";\n")

            i = 0.02
            f.write("Sustainer %.5f %.5f %s %.5f %.5f %s\n" %(x['diameter']*25.4, rocket.sustainer_length, "P", rocket.sustainer_prop_weight, rocket.sustainer_tot_weight, "FS_PRO"))
            while i < thrust_2[-1][0]: #OMG WAS THIS IT?
                f.write("%.8f  %.6f\n" %(i, motor2.thrust(i)))
                i+=0.05
            f.write("%.8f  %.6f\n" %(i+0.01, 0))
            f.write(";\n")
            f.close()




            fig1 = plt.figure(figsize=(9, 9))
            ax1 = plt.subplot(111)
            ax1.plot(np.arange(0, booster.solution[-1][0], 0.01), boosterStability, linewidth='2', label="booster")
            ax1.plot(np.arange(0, sustainer_coast.solution[-1][0], 0.01) + booster.solution[-1][0], sustainer_coastStability, linewidth='2', label="sustainer coast")
            ax1.plot(np.arange(0, sustainer.solution[-1][0], 0.01) + sustainer_coast.solution[-1][0] + booster.solution[-1][0], sustainerStability, linewidth='2', label="sustainer")
            ax1.legend()
            ax1.set_ylim([0, 10])
            plt.show()

            from rasaero import mbsEditor

            mbsEditor(
                x, 
                event4["body length"]*39.3701, 
                event5["body length"]*39.3701, 
                event2_wet["m"]*2.20462,
                4.18,
                event2_wet["cg"]*39.3701,
                event1_wet["m"]*2.20462,
                2,
                event1_wet["cg"]*39.3701,
                x['ignitionDelay']
                )

        if self.data_output == True:
            
            vector.join(fitness)
            df_opt = vector.join(fitness)
            df_opt.insert(0,'is fit', [np.all(np.array(fitness) <= 0)])
            df_opt.insert(0,'process id', [os.getpid()])

            df_components = pd.DataFrame(rocket.event1[0], [0])
            for i in range(1, len(rocket.event1)):
                df_components = df_components.append(pd.DataFrame(rocket.event1[i], [0]))


            event1_wet=Translator.event(rocket.event1, True)
            event2_wet=Translator.event(rocket.event2, True)
            event3_wet=Translator.event(rocket.event3, True)
            event4_wet=Translator.event(rocket.event4, True)
            event5_wet=Translator.event(rocket.event5, True)
            event6_wet=Translator.event(rocket.event6, True)

            events = [event1, event2, event3, event4, event5, event6, event1_wet, event2_wet, event3_wet, event4_wet, event5_wet, event6_wet]
            df_events = pd.DataFrame([list(events[0].values())], columns=list(event1.keys()))

            for i in range(1, len(events)):
                rows =['booster', 'sustainer coast', 'sustainer', 'booster recovery', 'sustainer recovery', 'nose recovery', 'booster wet', 'sustainer coast wet', 'sustainer wet', 'booster recovery wet', 'sustainer recovery wet', 'nose recovery wet']
                df_events = df_events.append(pd.DataFrame([list(events[i].values())], columns=list(event1.keys())))
            df_events.insert(0,'Event', rows)

            if os.path.exists(os.path.join(os.getcwd(), "data output", "decision_fitness %d.csv" %os.getpid())):
                df_opt.to_csv("./data output/decision_fitness %d.csv" %os.getpid(), mode="a", index=False, header=False)
                df_components.to_csv("./data output/components %d.csv" %os.getpid(), mode="a", index=False, header=False)
                df_events.to_csv("./data output/events %d.csv" %os.getpid(), mode="a", index=False, header=False)
            else:
                df_opt.to_csv("./data output/decision_fitness %d.csv" %os.getpid(), mode="w", index=False, header=True)
                df_components.to_csv("./data output/components %d.csv" %os.getpid(), mode="w", index=False, header=True)
                df_events.to_csv("./data output/events %d.csv" %os.getpid(), mode="w", index=False, header=True)

        return list(objective.values())
    def get_bounds(self):
        return(list(self.lower_bounds.values()), list(self.upper_bounds.values()))
    def get_name(self):
        return "FS-PRO: Full Stack Parallel Rocket Optimizer"
    def get_nic(self):
        return self.num_constraints
    def get_nix(self):
        return self.num_int
    def gradient(self, x):
        return pg.estimate_gradient_h(lambda x: self.fitness(x), x)
        # return pg.estimate_gradient(lambda x: self.fitness(x), x)

        
