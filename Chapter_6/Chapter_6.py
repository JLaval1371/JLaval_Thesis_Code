import numpy as np
import math
import lmfit
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
import random
from datetime import datetime
from scipy.optimize import fsolve
from scipy.stats import f_oneway



def interpolate(x, y, method, T0, Tf, dt):
    x_new = np.arange(T0, Tf + dt, dt)
    f = interp1d(x.astype(float), y.astype(float), kind=method, fill_value = 'extrapolate')
    y_new = f(x_new)

    for i in range(len(y_new)):
        if y_new[i] < 0:
            y_new[i] = 0

    return y_new, f

def interpolate_frames(x, y, method, frametimes):
    x_new = frametimes
    f = interp1d(x.astype(float), y.astype(float), kind=method, fill_value = 'extrapolate')
    y_new = f(x_new)

    for i in range(len(y_new)):
        if y_new[i] < 0:
            y_new[i] = 0

    return y_new, f

def integrateTrapezium(time, values):
    """
    integrateTrapezium: use the trapezium rule to approximate an integral
    """
    if (len(time) < 1 or len(values) < 1):
        print("No data to integrate!")
        integral = []
    else:
        integral = np.zeros(len(values))
        for i in range(1, len(values)):
            integral[i] = 0.5 * (values[i] + values[i-1])
            #print(integral[i])

    return integral


def comp_ode_model1(u, C0, t, T0, p):
    K_1, k_2, vb = p
    du = np.zeros(1)

    ind = int(round((t - T0)/dt) -1)

    # dC_1 / dt 
    du = K_1 * C0[ind] - k_2 * u[0] 

    # u[0] = C_1

    return du

def comp_ode_model2(u, C0, t, T0, p):
    K_1, k_2, k_3, k_4, vb = p

    du = np.zeros(2)

    ind = int(round((t - T0)/dt))

    # dC_1 / dt 
    du[0] = K_1 * C0[ind] - (k_2 + k_3) * u[0] + k_4 * u[1]

    # dC_2 / dt
    du[1] = k_3 * u[0] - k_4 * u[1]

    # u[0] = C_1, u[1] = C_2

    return du


def RK4(func, C0, init, dt, T_f, T0, p):
    N_t = int(round((T_f - T0)/dt))# - 1
    f_ = lambda u, C0, t, T0, p: np.asarray(func(u, C0, t, T0, p))
    u = np.zeros((N_t + 1, len(init)))
    k1 = np.zeros((N_t + 1, len(init)))
    k2 = np.zeros((N_t + 1, len(init)))
    k3 = np.zeros((N_t + 1, len(init)))
    k4 = np.zeros((N_t + 1, len(init)))
    t = np.linspace(T0, T0 + N_t*dt, len(u))
    u[0] = init
    
    for n in range(N_t):
        k1[n] = dt * f_(u[n], C0, t[n], T0, p)
        k2[n] = dt * f_(u[n] + k1[n]/2.0, C0, t[n] + dt/2.0, T0, p)
        k3[n] = dt * f_(u[n] + k2[n]/2.0, C0, t[n] + dt/2.0, T0, p)
        k4[n] = dt * f_(u[n] + k3[n], C0, t[n] + dt, T0, p)
        u[n+1] = u[n] + (k1[n] + 2.0 * (k2[n] + k3[n]) + k4[n])/6.0
    
    return u, t


def RK4_disp(func, C0, C0_disp, init, dt, T_f, T0, p):
    N_t = int(round((T_f - T0)/dt))# - 1
    f_ = lambda u, C0, C0_disp, t, T0, p: np.asarray(func(u, C0, C0_disp, t, T0, p))
    u = np.zeros((N_t + 1, len(init)))
    k1 = np.zeros((N_t + 1, len(init)))
    k2 = np.zeros((N_t + 1, len(init)))
    k3 = np.zeros((N_t + 1, len(init)))
    k4 = np.zeros((N_t + 1, len(init)))
    t = np.linspace(T0, T0 + N_t*dt, len(u))
    u[0] = init
    
    for n in range(N_t):
        k1[n] = dt * f_(u[n], C0, C0_disp, t[n], T0, p)
        k2[n] = dt * f_(u[n] + k1[n]/2.0, C0, C0_disp, t[n] + dt/2.0, T0, p)
        k3[n] = dt * f_(u[n] + k2[n]/2.0, C0, C0_disp, t[n] + dt/2.0, T0, p)
        k4[n] = dt * f_(u[n] + k3[n], C0, C0_disp, t[n] + dt, T0, p)
        u[n+1] = u[n] + (k1[n] + 2.0 * (k2[n] + k3[n]) + k4[n])/6.0
    
    return u, t

def resid1_weighted(params, C0, data, y_time, y_dat, frame_lengths_m, framemidtimes_m, tracer):
    K_1 = params['K1'].value
    k_2 = params['k2'].value
    vb = params['vb'].value
    
    p = [K_1, k_2, vb]

    dt = 1/600
    T_f, T0, dt, time = time_vars(data, dt)
    
    u_out, t = RK4(comp_ode_model1, C0, init, dt, T_f, T0, p)

    model = (1 - vb) * u_out[:, 0] + vb * C0

    func = interp1d(np.array(t, dtype=float), np.array(model, dtype=float), kind='cubic', fill_value = 'extrapolate')
    model = func(np.array(y_time, dtype=float))     # This is the model fit refitted into the original 33 time points

    #result = integrateTrapezium(frame_lengths_m, y_dat)     # This is the approximate decay corrected PET data (TAC) to be used in the weighting calculating
    result = np.array(y_dat)
    # print(result)
    # print(y_dat)
    # print(y_dat - result)
    resids = model - np.array(y_dat, dtype=float)       # This is the plain residuals to be returned from the function after being multiplied by the weights, final five values are to replace any zero values in result

    if tracer == 'Rb82':
        decay_const = 0.55121048155860461981489631925104
        scale_factor = 0.002
    elif tracer == 'NH3':
        decay_const = 0.0695232879197537923186792498955
        scale_factor = 0.000003
    elif tracer == 'Water':
        decay_const = 0.34015041658021623807954727632776
        scale_factor = 0.5

    frame_dur = np.zeros(len(frame_lengths_m))
    exp_decay = np.zeros(len(frame_lengths_m))

    for i in range(len(frame_lengths_m)):
        frame_dur[i] = frame_lengths_m[i]
        exp_decay[i] = math.exp(- decay_const * framemidtimes_m[i])

        if result[i] == 0:
            result[i] = np.mean(resids[-5:])        # Maybe replace this value with an average of the last 5 residuals, 3 for degrado

    sigma_sq = scale_factor * (result / (frame_dur * exp_decay))
    #sigma_sq = 0.05 * (result / (frame_dur * exp_decay))            # Changed scale factor to 0.05 for comparison with results from PMOD using 0.05 as the scale factor
    weights = 1 / sigma_sq

    return (weights * resids)
    #return resids


def resid2_weighted(params, C0, data, y_time, y_dat, frame_lengths_m, framemidtimes_m, tracer):
    K_1 = params['K1'].value
    k_2 = params['k2'].value
    k_3 = params['k3'].value
    k_4 = params['k4'].value
    vb = params['vb'].value

    p = [K_1, k_2, k_3, k_4, vb]

    dt = 1/600
    T_f, T0, dt, time = time_vars(data, dt)

    u_out, t = RK4(comp_ode_model2, C0, init2, dt, T_f, T0, p)

    model = (1 - vb) * (u_out[:, 0] + u_out[:, 1]) + vb * C0

    func = interp1d(np.array(t, dtype=float), np.array(model, dtype=float), kind='cubic', fill_value = 'extrapolate')
    model = func(np.array(y_time, dtype=float))     # This is the model fit refitted into the original 33 time points

    #result = integrateTrapezium(frame_lengths_m, y_dat)     # This is the approximate decay corrected PET data (TAC) to be used in the weighting calculating
    result = np.array(y_dat)

    resids = model - np.array(y_dat, dtype=float)       # This is the plain residuals to be returned from the function after being multiplied by the weights, final five values are to replace any zero values in result

    if tracer == 'Rb82':
        decay_const = 0.55121048155860461981489631925104
        scale_factor = 0.002
    elif tracer == 'NH3':
        decay_const = 0.0695232879197537923186792498955
        scale_factor = 0.000003
    elif tracer == 'Water':
        decay_const = 0.34015041658021623807954727632776
        scale_factor = 0.5

    frame_dur = np.zeros(len(frame_lengths_m))
    exp_decay = np.zeros(len(frame_lengths_m))

    for i in range(len(frame_lengths_m)):
        frame_dur[i] = frame_lengths_m[i]
        exp_decay[i] = math.exp(- decay_const * framemidtimes_m[i])

        if result[i] == 0:
            result[i] = np.mean(resids[-5:])        # Maybe replace this value with an average of the last 5 residuals, 3 for degrado

    sigma_sq = scale_factor**2 * (result / (frame_dur * exp_decay))
    #sigma_sq = 0.05 * (result / (frame_dur * exp_decay))            # Changed scale factor to 0.05 for comparison with results from PMOD using 0.05 as the scale factor
    weights = 1 / sigma_sq

    return (weights * resids)
    #return resids



# Like I said you'll have to play around to work out a good fudge factor
def noise_gen(blood_curve, tissue_curve, tracer, state):
    if state == 'Rest':
        ind = 3
    elif state == 'Stress':
        ind = 4

    frametimes_m, frametimes_mid_m, frame_lengths_m, frametimes_s = frametimes(tracer)
    #tracer_decay_times = {'Rb82' : 0.55121048155860461981489631925104, 'NH3' : 0.0695232879197537923186792498955, 'Water' : 0.34015041658021623807954727632776}

    if tracer == 'Rb82':
        decay_time = 0.55121048155860461981489631925104
        ff = 0.002
    elif tracer == 'NH3':
        decay_time = 0.0695232879197537923186792498955
        ff = 0.000003
    elif tracer == 'Water':
        decay_time = 0.34015041658021623807954727632776
        ff = 0.5

    # Rebinning interpolating
    # tissue_curve here is a pandas dataframe which needed to be converted to an array, first one is the time, second is the tissue data
    func = interp1d(np.array(tissue_curve.iloc[:, 0], dtype=float), np.array(tissue_curve.iloc[:, ind], dtype=float), kind='cubic', fill_value = 'extrapolate')
    tissue = func(np.array(frametimes_mid_m, dtype = float))        # SHould this be the frame mid times?
    
    # Check this interpolation and then integrate
    noise_array = np.array([])

    # Define CV, delta_t = the frame time, decay = e^(-lambda*t) term, counts = blood curve activity (turns out this was blood, not tissue)
    for i in range(len(frametimes_mid_m)):
        delta_t = frame_lengths_m[i]
        decay = np.exp(-1 * decay_time * (frametimes_mid_m[i]))
        counts = blood_curve.iloc[i, 1]

        cv = np.sqrt(ff/(counts * delta_t * decay))
        rn = random.gauss(0, 1)
        noise = cv * rn
        noise_array = np.append(noise_array, [noise])

    # This is just to correct for any infinities or negative infinities, not sure you'll need this bit
    for i in range(len(noise_array)):
        if noise_array[i] == np.Inf or noise_array[i] == np.NINF:
            noise_array[i] = np.sqrt(counts)
            print('Inf found in noise array')


    # Creates a noisy tissue curve
    noisy_tissue = tissue * (1 + noise_array)

    for i in range(len(noisy_tissue)):
        if noisy_tissue[i] < 0:
            noisy_tissue[i] = 0

    print(np.absolute(noise_array)/tissue)
    print(np.mean(np.absolute(noise_array[1:])/tissue[1:])) 
    # plt.scatter(frametimes_mid_m, noise_array, color = 'b', s = 7)
    # plt.show()

    return noisy_tissue, tissue, noise_array 


# These are the frame times and frame lengths that I used for my data, yours will obviously be a bit different
def frametimes(tracer):
    if tracer == 'NH3':
        frametimes_s = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 150, 180, 210, 240, 360, 480, 600, 900])
        frametimes_m = frametimes_s / 60

        frametimes_mid_m = (frametimes_m[1:] + frametimes_m[:-1]) / 2
        
        frame_lengths_s = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 30, 30, 30, 30, 120, 120, 120, 300])
        frame_lengths_m = frame_lengths_s / 60
    
    elif tracer == 'Rb82':
        frametimes_s = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 120, 150, 180, 210, 240, 270, 300])
        frametimes_m = frametimes_s / 60

        frametimes_mid_m = (frametimes_m[1:] + frametimes_m[:-1]) / 2

        frame_lengths_s = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 30, 30, 30, 30, 30, 30, 30])
        frame_lengths_m = frame_lengths_s / 60

    elif tracer == 'Water':
        frametimes_s = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 70, 80, 90, 100, 110, 120, 140, 160, 180, 200, 220, 240])
        frametimes_m = frametimes_s / 60

        frametimes_mid_m = (frametimes_m[1:] + frametimes_m[:-1]) / 2
        
        frame_lengths_s = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 10, 10, 10, 10, 10, 10, 20, 20, 20, 20, 20, 20])
        frame_lengths_m = frame_lengths_s / 60

    return frametimes_m, frametimes_mid_m, frame_lengths_m, frametimes_s


def Rb82_aif(time):
    time = time*60
    a = 39218
    b = 1428

    c = a * np.power(time, 4)
    d = b + np.power(time, 5)

    return (c/d)/60


def NH3_aif(time):
    k = 1.72
    a = k**2 / (1 - (1 + k * time[-1]) * np.exp(-k * time[-1]))
    n = 0.26

    return n * a * time * np.exp(-k * time)


def water_expo(x, A, lambd, line, t_peak):
    p1, p3, p5 = A        
    p2, p4, p6 = lambd
    a, b = line

    result = np.array([])

    for i in x:
        if i < -(b/a):
            result = np.append(result, [0])
        elif i >= -(b/a) and i < t_peak:
            result = np.append(result, [a * i + b]) 
        elif i >= t_peak:
            result = np.append(result, [p1*np.exp(-p2*(i - t_peak)) + p3*np.exp(-p4*(i - t_peak)) + p5*np.exp(-p6*(i - t_peak))]) 
         
    return result

def water_resids(params_exp, t_peak, time, data):
    coeff = [params_exp['p1'].value, params_exp['p3'].value, params_exp['p5'].value]
    d_const = [params_exp['p2'].value, params_exp['p4'].value, params_exp['p6'].value]
    line = [params_exp['a'].value, params_exp['b'].value]

    model = water_expo(time, coeff, d_const, line, t_peak)

    #print(np.isnan(model - data).any())

    return model - data

def water_aif():
    water_data = pd.read_excel('O15_AIF.xlsx', sheet_name = 'O15_AIF', engine = 'openpyxl')
    water_data.columns = ['Time', 'Blood']

    t_peak = water_data.loc[water_data['Blood'].idxmax()]['Time']
    
    params = lmfit.Parameters()
    params.add('p1', 1)
    params.add('p2', 1)
    params.add('p3', 1)
    params.add('p4', 1)
    params.add('p5', 1)
    params.add('p6', 1)
    params.add('a', 1, min = 0)
    params.add('b', 1)

    fit = lmfit.minimize(water_resids, params, args = (t_peak, water_data.Time, water_data.Blood), method = 'leastsq', max_nfev = 2500)

    val1, val2, val3, val4, val5, val6, vala, valb = [fit.params['p1'].value, fit.params['p2'].value, fit.params['p3'].value, fit.params['p4'].value, fit.params['p5'].value, fit.params['p6'].value, fit.params['a'].value, fit.params['b'].value]
    print(f'Value of p1 is: {val1}')
    print(f'Value of p2 is: {val2}')
    print(f'Value of p3 is: {val3}')
    print(f'Value of p4 is: {val4}')
    print(f'Value of p5 is: {val5}')
    print(f'Value of p6 is: {val6}')
    print(f'Value of a is: {vala}')
    print(f'Value of b is: {valb}')

    fitted_blood = water_expo(water_data.Time, [val1, val3, val5], [val2, val4, val6], [vala, valb], t_peak)

    plt.scatter(water_data.Time, water_data.Blood, s = 7, label = 'Blood', color = 'r')
    plt.plot(water_data.Time, fitted_blood, label = 'Model', color = 'b')
    plt.title('Fitted Water Blood Curve')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Activity (Bq/cc)')
    plt.legend(loc = 7, fontsize = 'x-small')
    plt.savefig('Fitted_water_blood_curve')
    plt.close()
    #plt.show()

    return fitted_blood

def time_vars(data, dt):
    T_f = data.iloc[-1, 0]
    T0 = data.iloc[0, 0]
    time = np.arange(T0, T_f + dt, dt)

    return T_f, T0, dt, time

def time_vars_array(data, dt):
    T_f = data[-1]
    T0 = data[0]
    time = np.arange(T0, T_f + dt, dt)

    return T_f, T0, dt, time


def flow_func_rb82(F, K1):
    a = 0.77
    b = 0.63

    return  F * (1 - a * np.exp(-b/F)) - K1

def flow_func_NH3(F, K1):
    a = 0.607
    b = 1.25

    return  F * (1 - a * np.exp(-b/F)) - K1

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    
    return array[idx]

def add_flow(filename, foldername, tracer, state, vb_coeff):
    data = pd.read_excel(f'{filename}.xlsx', engine = 'openpyxl')

    f_values = np.array([])

    for i in range(len(data.K1)):
        k_1 = data.at[i, 'K1']
        if tracer == 'Rb82':
            f_root = fsolve(flow_func_rb82, k_1, args = (k_1))
        elif tracer == 'NH3':
            f_root = fsolve(flow_func_NH3, k_1, args = (k_1))

        f = f_root[0]
        #f = find_nearest(f_root, k_1)
        
        f_values = np.append(f_values, f)
    
    #print(f_values)
    data['Flow'] = f_values
    data['Extr_Frac'] = (data.K1 / data.Flow) * 100

    data.to_excel(f'{filename}_with_flow.xlsx')
    gens = len(data.Flow)

    plt.scatter(np.linspace(0, gens, num = gens), data.Flow, s = 5, label = 'F', color = 'r')
    plt.title(f'Flow values from fits of {gens} noise realizations for {tracer} ({state})')
    plt.xlabel('Number of Noise Realizations')
    plt.ylabel('F (mL/min/g)')
    plt.legend(loc = 7, fontsize = 'x-small')
    plt.savefig(f'{foldername}/{tracer}_Flow_values_{state}_vb_{vb_coeff}.png')
    plt.close()
    #plt.show()

    plt.boxplot(data.Flow, vert = True)
    plt.title(f'Flow values from fits of {gens} noise realizations for {tracer} ({state})')
    plt.ylabel('F (mL/min/g)')
    plt.savefig(f'{foldername}/{tracer}_Flow_values_{state}_vb_{vb_coeff}_box.png')
    plt.close()
    #plt.show()


def boxplots(filename, foldername, tracer):
    data_rest = pd.read_excel(f'{filename}.xlsx', sheet_name = f'{tracer}_Rest', engine = 'openpyxl')
    data_stress = pd.read_excel(f'{filename}.xlsx', sheet_name = f'{tracer}_Stress', engine = 'openpyxl')
    data_rest = data_rest.dropna(axis = 0)
    data_stress = data_stress.dropna(axis = 0)

    gens = len(data_rest.K1)
    states = ['Rest', 'Stress']


    if tracer == 'Rb82':
        K1_r = 0.47
        k2_r = 0.12
        vb_r = 0.48
        K1_s = 1.08
        k2_s = 0.21
        vb_s = 0.50
        f_root_rest = fsolve(flow_func_rb82, K1_r, args = (K1_r))
        flow_r = f_root_rest[0]
        f_root_stress = fsolve(flow_func_rb82, K1_s, args = (K1_s))
        flow_s = f_root_stress[0]

        ### K1
        fig, ax1 = plt.subplots()

        ax1.boxplot([data_rest.K1, np.NaN], labels = states, vert = True)
        ax1.axhline(y = K1_r, color = 'b', xmin = 0, xmax = 0.5)
        ax1.set_ylabel('K1 (mL/min/g)')
        ax1.set_xlabel('State')
       
        ax2 = ax1.twinx()
        ax2.boxplot([np.NaN , data_stress.K1], labels = states, vert = True)
        ax2.axhline(y = K1_s, color = 'r', xmin = 0.5, xmax = 1.0)
        ax2.set_ylabel('K1 (mL/min/g)')
        
        fig.suptitle(f'K1 values from fits of {gens} noise realizations for {tracer}')
        fig.tight_layout()
        plt.savefig(f'{foldername}/{tracer}_K1_boxplots')
        plt.close()
        #plt.show()


        ### K2
        fig, ax1 = plt.subplots()

        ax1.boxplot([data_rest.k2, np.NaN], labels = states, vert = True)
        ax1.axhline(y = k2_r, color = 'b', xmin = 0, xmax = 0.5)
        ax1.set_ylabel('k2 (1/min)')
        ax1.set_xlabel('State')
        
        ax2 = ax1.twinx()
        ax2.boxplot([np.NaN , data_stress.k2], labels = states, vert = True)
        ax2.axhline(y = k2_s, color = 'r', xmin = 0.5, xmax = 1.0)
        ax2.set_ylabel('k2 (1/min)')
        
        fig.suptitle(f'k2 values from fits of {gens} noise realizations for {tracer}')
        fig.tight_layout()
        plt.savefig(f'{foldername}/{tracer}_k2_boxplots')
        plt.close()
        #plt.show()

        ### VB
        fig, ax1 = plt.subplots()

        ax1.boxplot([data_rest.vb, np.NaN], labels = states, vert = True)
        ax1.axhline(y = vb_r, color = 'b', xmin = 0, xmax = 0.5)
        ax1.set_ylabel('vb')
        ax1.set_xlabel('State')
        
        ax2 = ax1.twinx()
        ax2.boxplot([np.NaN , data_stress.vb], labels = states, vert = True)
        ax2.axhline(y = vb_s, color = 'r', xmin = 0.5, xmax = 1.0)
        ax2.set_ylabel('vb')
        
        fig.suptitle(f'vb values from fits of {gens} noise realizations for {tracer}')
        fig.tight_layout()
        plt.savefig(f'{foldername}/{tracer}_vb_boxplots')
        plt.close()
        #plt.show()

        ### Flow
        fig, ax1 = plt.subplots()

        ax1.boxplot([data_rest.Flow, np.NaN], labels = states, vert = True)
        ax1.axhline(y = flow_r, color = 'b', xmin = 0, xmax = 0.5)
        ax1.set_ylabel('Flow (mL/min/g)')
        ax1.set_xlabel('State')
       
        ax2 = ax1.twinx()
        ax2.boxplot([np.NaN , data_stress.Flow], labels = states, vert = True)
        ax2.axhline(y = flow_s, color = 'r', xmin = 0.5, xmax = 1.0)
        ax2.set_ylabel('Flow (mL/min/g)')
        
        fig.suptitle(f'Flow values from fits of {gens} noise realizations for {tracer}')
        fig.tight_layout()
        plt.savefig(f'{foldername}/{tracer}_Flow_boxplots')
        plt.close()
        #plt.show()


    elif tracer == 'Water':
        K1_r = 0.87
        k2_r = 1.10
        vb_r = 0.29
        K1_s = 3.43
        k2_s = 3.76
        vb_s = 0.27
        flow_r = K1_r
        flow_s = K1_s

        ### K1
        fig, ax1 = plt.subplots()

        ax1.boxplot([data_rest.K1, np.NaN], labels = states, vert = True)
        ax1.axhline(y = K1_r, color = 'b', xmin = 0, xmax = 0.5)
        ax1.set_ylabel('K1 (mL/min/g)')
        ax1.set_xlabel('State')
       
        ax2 = ax1.twinx()
        ax2.boxplot([np.NaN , data_stress.K1], labels = states, vert = True)
        ax2.axhline(y = K1_s, color = 'r', xmin = 0.5, xmax = 1.0)
        ax2.set_ylabel('K1 (mL/min/g)')
        
        fig.suptitle(f'K1 values from fits of {gens} noise realizations for {tracer}')
        fig.tight_layout()
        plt.savefig(f'{foldername}/{tracer}_K1_boxplots')
        plt.close()
        #plt.show()


        ### K2
        fig, ax1 = plt.subplots()

        ax1.boxplot([data_rest.k2, np.NaN], labels = states, vert = True)
        ax1.axhline(y = k2_r, color = 'b', xmin = 0, xmax = 0.5)
        ax1.set_ylabel('k2 (1/min)')
        ax1.set_xlabel('State')
        
        ax2 = ax1.twinx()
        ax2.boxplot([np.NaN , data_stress.k2], labels = states, vert = True)
        ax2.axhline(y = k2_s, color = 'r', xmin = 0.5, xmax = 1.0)
        ax2.set_ylabel('k2 (1/min)')
        
        fig.suptitle(f'k2 values from fits of {gens} noise realizations for {tracer}')
        fig.tight_layout()
        plt.savefig(f'{foldername}/{tracer}_k2_boxplots')
        plt.close()
        #plt.show()

        ### VB
        fig, ax1 = plt.subplots()

        ax1.boxplot([data_rest.vb, np.NaN], labels = states, vert = True)
        ax1.axhline(y = vb_r, color = 'b', xmin = 0, xmax = 0.5)
        ax1.set_ylabel('vb')
        ax1.set_xlabel('State')
        
        ax2 = ax1.twinx()
        ax2.boxplot([np.NaN , data_stress.vb], labels = states, vert = True)
        ax2.axhline(y = vb_s, color = 'r', xmin = 0.5, xmax = 1.0)
        ax2.set_ylabel('vb')
        
        fig.suptitle(f'vb values from fits of {gens} noise realizations for {tracer}')
        fig.tight_layout()
        plt.savefig(f'{foldername}/{tracer}_vb_boxplots')
        plt.close()
        #plt.show()

        ### Flow
        fig, ax1 = plt.subplots()

        ax1.boxplot([data_rest.Flow, np.NaN], labels = states, vert = True)
        ax1.axhline(y = flow_r, color = 'b', xmin = 0, xmax = 0.5)
        ax1.set_ylabel('Flow (mL/min/g)')
        ax1.set_xlabel('State')
       
        ax2 = ax1.twinx()
        ax2.boxplot([np.NaN , data_stress.Flow], labels = states, vert = True)
        ax2.axhline(y = flow_s, color = 'r', xmin = 0.5, xmax = 1.0)
        ax2.set_ylabel('Flow (mL/min/g)')
        
        fig.suptitle(f'Flow values from fits of {gens} noise realizations for {tracer}')
        fig.tight_layout()
        plt.savefig(f'{foldername}/{tracer}_Flow_boxplots')
        plt.close()
        #plt.show()

    elif tracer == 'NH3':
        K1_r = 0.69
        k2_r = 0.23
        k3_r = 0.14
        vb_r = 0.38
        K1_s = 2.71
        k2_s = 0.89
        k3_s = 0.13
        vb_s = 0.28
        f_root_rest = fsolve(flow_func_NH3, K1_r, args = (K1_r))
        flow_r = f_root_rest[0]
        f_root_stress = fsolve(flow_func_NH3, K1_s, args = (K1_s))
        flow_s = f_root_stress[0]

        ### K1
        fig, ax1 = plt.subplots()

        ax1.boxplot([data_rest.K1, np.NaN], labels = states, vert = True)
        ax1.axhline(y = K1_r, color = 'b', xmin = 0, xmax = 0.5)
        ax1.set_ylabel('K1 (mL/min/g)')
        ax1.set_xlabel('State')
       
        ax2 = ax1.twinx()
        ax2.boxplot([np.NaN , data_stress.K1], labels = states, vert = True)
        ax2.axhline(y = K1_s, color = 'r', xmin = 0.5, xmax = 1.0)
        ax2.set_ylabel('K1 (mL/min/g)')
        
        fig.suptitle(f'K1 values from fits of {gens} noise realizations for {tracer}')
        fig.tight_layout()
        plt.savefig(f'{foldername}/{tracer}_K1_boxplots')
        plt.close()
        #plt.show()


        ### K2
        fig, ax1 = plt.subplots()

        ax1.boxplot([data_rest.k2, np.NaN], labels = states, vert = True)
        ax1.axhline(y = k2_r, color = 'b', xmin = 0, xmax = 0.5)
        ax1.set_ylabel('k2 (1/min)')
        ax1.set_xlabel('State')
        
        ax2 = ax1.twinx()
        ax2.boxplot([np.NaN , data_stress.k2], labels = states, vert = True)
        ax2.axhline(y = k2_s, color = 'r', xmin = 0.5, xmax = 1.0)
        ax2.set_ylabel('k2 (1/min)')
        
        fig.suptitle(f'k2 values from fits of {gens} noise realizations for {tracer}')
        fig.tight_layout()
        plt.savefig(f'{foldername}/{tracer}_k2_boxplots')
        plt.close()
        #plt.show()

        ### K3
        fig, ax1 = plt.subplots()

        ax1.boxplot([data_rest.k3, np.NaN], labels = states, vert = True)
        ax1.axhline(y = k3_r, color = 'b', xmin = 0, xmax = 0.5)
        ax1.set_ylabel('k3 (1/min)')
        ax1.set_xlabel('State')
        
        ax2 = ax1.twinx()
        ax2.boxplot([np.NaN , data_stress.k3], labels = states, vert = True)
        ax2.axhline(y = k3_s, color = 'r', xmin = 0.5, xmax = 1.0)
        ax2.set_ylabel('k3 (1/min)')
        
        fig.suptitle(f'k3 values from fits of {gens} noise realizations for {tracer}')
        fig.tight_layout()
        plt.savefig(f'{foldername}/{tracer}_k3_boxplots')
        plt.close()
        #plt.show()

        ### VB
        fig, ax1 = plt.subplots()

        ax1.boxplot([data_rest.vb, np.NaN], labels = states, vert = True)
        ax1.axhline(y = vb_r, color = 'b', xmin = 0, xmax = 0.5)
        ax1.set_ylabel('vb')
        ax1.set_xlabel('State')
        
        ax2 = ax1.twinx()
        ax2.boxplot([np.NaN , data_stress.vb], labels = states, vert = True)
        ax2.axhline(y = vb_s, color = 'r', xmin = 0.5, xmax = 1.0)
        ax2.set_ylabel('vb')
        
        fig.suptitle(f'vb values from fits of {gens} noise realizations for {tracer}')
        fig.tight_layout()
        plt.savefig(f'{foldername}/{tracer}_vb_boxplots')
        plt.close()
        #plt.show()

        ### Flow
        fig, ax1 = plt.subplots()

        ax1.boxplot([data_rest.Flow, np.NaN], labels = states, vert = True)
        ax1.axhline(y = flow_r, color = 'b', xmin = 0, xmax = 0.5)
        ax1.set_ylabel('Flow (mL/min/g)')
        ax1.set_xlabel('State')
       
        ax2 = ax1.twinx()
        ax2.boxplot([np.NaN , data_stress.Flow], labels = states, vert = True)
        ax2.axhline(y = flow_s, color = 'r', xmin = 0.5, xmax = 1.0)
        ax2.set_ylabel('Flow (mL/min/g)')
        
        fig.suptitle(f'Flow values from fits of {gens} noise realizations for {tracer}')
        fig.tight_layout()
        plt.savefig(f'{foldername}/{tracer}_Flow_boxplots')
        plt.close()
        #plt.show()

    print(f'Rest flow of {tracer} = {flow_r:.3f}')
    print(f'Stress flow of {tracer} = {flow_s:.3f}')
#water_blood = water_aif()

def boxplots_vb_coeff(filename, foldername, tracer, vb_coeff):
    data_rest = pd.read_excel(f'{filename}.xlsx', sheet_name = f'{tracer}_{vb_coeff}_Rest', engine = 'openpyxl')
    data_stress = pd.read_excel(f'{filename}.xlsx', sheet_name = f'{tracer}_{vb_coeff}_Stress', engine = 'openpyxl')
    data_rest = data_rest.dropna(axis = 0)
    data_stress = data_stress.dropna(axis = 0)

    gens = len(data_rest.K1)
    states = ['Rest', 'Stress']

    vb_coeff_value = vb_coeff
    vb_coeff_label = f'vb_coeff_{vb_coeff_value}'


    if tracer == 'Rb82':
        K1_r = 0.47
        k2_r = 0.12
        vb_r = 0.48
        K1_s = 1.08
        k2_s = 0.21
        vb_s = 0.50
        f_root_rest = fsolve(flow_func_rb82, K1_r, args = (K1_r))
        flow_r = f_root_rest[0]
        f_root_stress = fsolve(flow_func_rb82, K1_s, args = (K1_s))
        flow_s = f_root_stress[0]

        ### K1
        fig, ax1 = plt.subplots()

        ax1.boxplot([data_rest.K1, np.NaN], labels = states, vert = True)
        ax1.axhline(y = K1_r, color = 'b', xmin = 0, xmax = 0.5)
        ax1.set_ylim([0.4, 0.8])
        ax1.set_ylabel('K1 (mL/min/g)')
        ax1.set_xlabel('State')
       
        ax2 = ax1.twinx()
        ax2.boxplot([np.NaN , data_stress.K1], labels = states, vert = True)
        ax2.axhline(y = K1_s, color = 'r', xmin = 0.5, xmax = 1.0)
        ax2.set_ylim([1.0, 1.4])
        ax2.set_ylabel('K1 (mL/min/g)')
        
        fig.suptitle(f'K1 values from fits of {gens} noise realizations \n for {tracer} with vb_coeff = {vb_coeff_value}')
        fig.tight_layout()
        plt.savefig(f'{foldername}/{tracer}_K1_boxplots_{vb_coeff_label}.png')
        plt.close()
        #plt.show()


        ### K2
        fig, ax1 = plt.subplots()

        ax1.boxplot([data_rest.k2, np.NaN], labels = states, vert = True)
        ax1.axhline(y = k2_r, color = 'b', xmin = 0, xmax = 0.5)
        ax1.set_ylim([0.05, 0.55])
        ax1.set_ylabel('k2 (1/min)')
        ax1.set_xlabel('State')
        
        ax2 = ax1.twinx()
        ax2.boxplot([np.NaN , data_stress.k2], labels = states, vert = True)
        ax2.axhline(y = k2_s, color = 'r', xmin = 0.5, xmax = 1.0)
        ax2.set_ylim([0.1, 0.5])
        ax2.set_ylabel('k2 (1/min)')
        
        fig.suptitle(f'k2 values from fits of {gens} noise realizations \n for {tracer} with vb_coeff = {vb_coeff_value}')
        fig.tight_layout()
        plt.savefig(f'{foldername}/{tracer}_k2_boxplots_{vb_coeff_label}.png')
        plt.close()
        #plt.show()

        ### VB
        fig, ax1 = plt.subplots()

        ax1.boxplot([data_rest.vb, np.NaN], labels = states, vert = True)
        ax1.axhline(y = vb_r, color = 'b', xmin = 0, xmax = 0.5)
        ax1.set_ylim([0.0, 0.55])
        ax1.set_ylabel('vb')
        ax1.set_xlabel('State')
        
        ax2 = ax1.twinx()
        ax2.boxplot([np.NaN , data_stress.vb], labels = states, vert = True)
        ax2.axhline(y = vb_s, color = 'r', xmin = 0.5, xmax = 1.0)
        ax2.set_ylim([0.0, 0.55])
        ax2.set_ylabel('vb')
        
        fig.suptitle(f'vb values from fits of {gens} noise realizations \n for {tracer} with vb_coeff = {vb_coeff_value}')
        fig.tight_layout()
        plt.savefig(f'{foldername}/{tracer}_vb_boxplots_{vb_coeff_label}.png')
        plt.close()
        #plt.show()

        ### Flow
        # fig, ax1 = plt.subplots()

        # ax1.boxplot([data_rest.Flow, np.NaN], labels = states, vert = True)
        # ax1.axhline(y = flow_r, color = 'b', xmin = 0, xmax = 0.5)
        # ax1.set_ylabel('Flow (mL/min/g)')
        # ax1.set_xlabel('State')
       
        # ax2 = ax1.twinx()
        # ax2.boxplot([np.NaN , data_stress.Flow], labels = states, vert = True)
        # ax2.axhline(y = flow_s, color = 'r', xmin = 0.5, xmax = 1.0)
        # ax2.set_ylabel('Flow (mL/min/g)')
        
        # fig.suptitle(f'Flow values from fits of {gens} noise realizations \n for {tracer} with vb_coeff = {vb_coeff_value}')
        # fig.tight_layout()
        # plt.savefig(f'{foldername}/{tracer}_Flow_boxplots_{vb_coeff_label}.png')
        # plt.close()
        # #plt.show()


    elif tracer == 'Water':
        K1_r = 0.87
        k2_r = 1.10
        vb_r = 0.29
        K1_s = 3.43
        k2_s = 3.76
        vb_s = 0.27
        flow_r = K1_r
        flow_s = K1_s

        ### K1
        fig, ax1 = plt.subplots()

        ax1.boxplot([data_rest.K1, np.NaN], labels = states, vert = True)
        ax1.axhline(y = K1_r, color = 'b', xmin = 0, xmax = 0.5)
        ax1.set_ylabel('K1 (mL/min/g)')
        ax1.set_xlabel('State')
       
        ax2 = ax1.twinx()
        ax2.boxplot([np.NaN , data_stress.K1], labels = states, vert = True)
        ax2.axhline(y = K1_s, color = 'r', xmin = 0.5, xmax = 1.0)
        ax2.set_ylabel('K1 (mL/min/g)')
        
        fig.suptitle(f'K1 values from fits of {gens} noise realizations for {tracer}')
        fig.tight_layout()
        plt.savefig(f'{foldername}/{tracer}_K1_boxplots')
        plt.close()
        #plt.show()


        ### K2
        fig, ax1 = plt.subplots()

        ax1.boxplot([data_rest.k2, np.NaN], labels = states, vert = True)
        ax1.axhline(y = k2_r, color = 'b', xmin = 0, xmax = 0.5)
        ax1.set_ylabel('k2 (1/min)')
        ax1.set_xlabel('State')
        
        ax2 = ax1.twinx()
        ax2.boxplot([np.NaN , data_stress.k2], labels = states, vert = True)
        ax2.axhline(y = k2_s, color = 'r', xmin = 0.5, xmax = 1.0)
        ax2.set_ylabel('k2 (1/min)')
        
        fig.suptitle(f'k2 values from fits of {gens} noise realizations for {tracer}')
        fig.tight_layout()
        plt.savefig(f'{foldername}/{tracer}_k2_boxplots')
        plt.close()
        #plt.show()

        ### VB
        fig, ax1 = plt.subplots()

        ax1.boxplot([data_rest.vb, np.NaN], labels = states, vert = True)
        ax1.axhline(y = vb_r, color = 'b', xmin = 0, xmax = 0.5)
        ax1.set_ylabel('vb')
        ax1.set_xlabel('State')
        
        ax2 = ax1.twinx()
        ax2.boxplot([np.NaN , data_stress.vb], labels = states, vert = True)
        ax2.axhline(y = vb_s, color = 'r', xmin = 0.5, xmax = 1.0)
        ax2.set_ylabel('vb')
        
        fig.suptitle(f'vb values from fits of {gens} noise realizations for {tracer}')
        fig.tight_layout()
        plt.savefig(f'{foldername}/{tracer}_vb_boxplots')
        plt.close()
        #plt.show()

        ### Flow
        fig, ax1 = plt.subplots()

        ax1.boxplot([data_rest.Flow, np.NaN], labels = states, vert = True)
        ax1.axhline(y = flow_r, color = 'b', xmin = 0, xmax = 0.5)
        ax1.set_ylabel('Flow (mL/min/g)')
        ax1.set_xlabel('State')
       
        ax2 = ax1.twinx()
        ax2.boxplot([np.NaN , data_stress.Flow], labels = states, vert = True)
        ax2.axhline(y = flow_s, color = 'r', xmin = 0.5, xmax = 1.0)
        ax2.set_ylabel('Flow (mL/min/g)')
        
        fig.suptitle(f'Flow values from fits of {gens} noise realizations for {tracer}')
        fig.tight_layout()
        plt.savefig(f'{foldername}/{tracer}_Flow_boxplots')
        plt.close()
        #plt.show()

    elif tracer == 'NH3':
        K1_r = 0.69
        k2_r = 0.23
        k3_r = 0.14
        vb_r = 0.38
        K1_s = 2.71
        k2_s = 0.89
        k3_s = 0.13
        vb_s = 0.28
        f_root_rest = fsolve(flow_func_NH3, K1_r, args = (K1_r))
        flow_r = f_root_rest[0]
        f_root_stress = fsolve(flow_func_NH3, K1_s, args = (K1_s))
        flow_s = f_root_stress[0]

        ### K1
        fig, ax1 = plt.subplots()

        ax1.boxplot([data_rest.K1, np.NaN], labels = states, vert = True)
        ax1.axhline(y = K1_r, color = 'b', xmin = 0, xmax = 0.5)
        ax1.set_ylabel('K1 (mL/min/g)')
        ax1.set_xlabel('State')
       
        ax2 = ax1.twinx()
        ax2.boxplot([np.NaN , data_stress.K1], labels = states, vert = True)
        ax2.axhline(y = K1_s, color = 'r', xmin = 0.5, xmax = 1.0)
        ax2.set_ylabel('K1 (mL/min/g)')
        
        fig.suptitle(f'K1 values from fits of {gens} noise realizations for {tracer}')
        fig.tight_layout()
        plt.savefig(f'{foldername}/{tracer}_K1_boxplots')
        plt.close()
        #plt.show()


        ### K2
        fig, ax1 = plt.subplots()

        ax1.boxplot([data_rest.k2, np.NaN], labels = states, vert = True)
        ax1.axhline(y = k2_r, color = 'b', xmin = 0, xmax = 0.5)
        ax1.set_ylabel('k2 (1/min)')
        ax1.set_xlabel('State')
        
        ax2 = ax1.twinx()
        ax2.boxplot([np.NaN , data_stress.k2], labels = states, vert = True)
        ax2.axhline(y = k2_s, color = 'r', xmin = 0.5, xmax = 1.0)
        ax2.set_ylabel('k2 (1/min)')
        
        fig.suptitle(f'k2 values from fits of {gens} noise realizations for {tracer}')
        fig.tight_layout()
        plt.savefig(f'{foldername}/{tracer}_k2_boxplots')
        plt.close()
        #plt.show()

        ### K3
        fig, ax1 = plt.subplots()

        ax1.boxplot([data_rest.k3, np.NaN], labels = states, vert = True)
        ax1.axhline(y = k3_r, color = 'b', xmin = 0, xmax = 0.5)
        ax1.set_ylabel('k3 (1/min)')
        ax1.set_xlabel('State')
        
        ax2 = ax1.twinx()
        ax2.boxplot([np.NaN , data_stress.k3], labels = states, vert = True)
        ax2.axhline(y = k3_s, color = 'r', xmin = 0.5, xmax = 1.0)
        ax2.set_ylabel('k3 (1/min)')
        
        fig.suptitle(f'k3 values from fits of {gens} noise realizations for {tracer}')
        fig.tight_layout()
        plt.savefig(f'{foldername}/{tracer}_k3_boxplots')
        plt.close()
        #plt.show()

        ### VB
        fig, ax1 = plt.subplots()

        ax1.boxplot([data_rest.vb, np.NaN], labels = states, vert = True)
        ax1.axhline(y = vb_r, color = 'b', xmin = 0, xmax = 0.5)
        ax1.set_ylabel('vb')
        ax1.set_xlabel('State')
        
        ax2 = ax1.twinx()
        ax2.boxplot([np.NaN , data_stress.vb], labels = states, vert = True)
        ax2.axhline(y = vb_s, color = 'r', xmin = 0.5, xmax = 1.0)
        ax2.set_ylabel('vb')
        
        fig.suptitle(f'vb values from fits of {gens} noise realizations for {tracer}')
        fig.tight_layout()
        plt.savefig(f'{foldername}/{tracer}_vb_boxplots')
        plt.close()
        #plt.show()

        ### Flow
        fig, ax1 = plt.subplots()

        ax1.boxplot([data_rest.Flow, np.NaN], labels = states, vert = True)
        ax1.axhline(y = flow_r, color = 'b', xmin = 0, xmax = 0.5)
        ax1.set_ylabel('Flow (mL/min/g)')
        ax1.set_xlabel('State')
       
        ax2 = ax1.twinx()
        ax2.boxplot([np.NaN , data_stress.Flow], labels = states, vert = True)
        ax2.axhline(y = flow_s, color = 'r', xmin = 0.5, xmax = 1.0)
        ax2.set_ylabel('Flow (mL/min/g)')
        
        fig.suptitle(f'Flow values from fits of {gens} noise realizations for {tracer}')
        fig.tight_layout()
        plt.savefig(f'{foldername}/{tracer}_Flow_boxplots')
        plt.close()
        #plt.show()

    print(f'Rest flow of {tracer} = {flow_r:.3f}')
    print(f'Stress flow of {tracer} = {flow_s:.3f}')
    #water_blood = water_aif()

def comp_ode_model_blockage1(u, C0, C0_disp, t, T0, p):
    K_1, k_2, K_1_d, k_2_d, vb = p

    du = np.zeros(1)

    ind = int(round((t - T0)/dt))

    # dC_0_disp / dt 
    #du[0] = R * C0[ind] - K_1_d * C0_disp[ind] + k_2_d * u[0] 

    # dC_1 / dt
    du[0] = K_1 * C0[ind] + K_1_d * C0_disp[ind] - k_2 * u[0] - k_2_d * u[0]

    # u[0] = C_0_disp, u[1] = C_1

    return du

def comp_ode_model_blockage2(u, C0, t, T0, p):
    K_1, k_2, K_1_d, k_2_d, R, vb = p

    du = np.zeros(2)

    ind = int(round((t - T0)/dt))

    # dC_0_disp / dt 
    du[0] = R * C0[ind] - K_1_d * u[0] + k_2_d * u[1] - R * u[0]

    # dC_1 / dt
    du[1] = K_1 * C0[ind] + K_1_d * u[0] - k_2 * u[1] - k_2_d * u[1]

    # u[0] = C_0_disp, u[1] = C_1

    return du

def comp_ode_model_blockage_midstep(u, C0, t, T0, p):
    K_1, k_2, R, vb = p

    du = np.zeros(2)

    ind = int(round((t - T0)/dt))

    # dC_0_disp / dt 
    du[0] = R * C0[ind] - R * u[0]

    # dC_1 / dt
    du[1] = K_1 * C0[ind] - k_2 * u[1]

    # u[0] = C_0_disp, u[1] = C_1

    return du


def resid_blockage1(params, C0, C0_disp, data, y_time, y_dat, frame_lengths_m, framemidtimes_m, tracer):
    K_1 = params['K1'].value
    k_2 = params['k2'].value
    K_1_d = params['K1_d'].value
    k_2_d = params['k2_d'].value
    #R = params['R'].value
    vb = params['vb'].value
    
    p = [K_1, k_2, K_1_d, k_2_d, vb]

    dt = 1/600
    T_f, T0, dt, time = time_vars(data, dt)
    
    u_out, t = RK4_disp(comp_ode_model_blockage1, C0, C0_disp, init, dt, T_f, T0, p)

    model = (1 - vb) * u_out[:, 0] + vb * (0.5 * (C0 + C0_disp))

    func = interp1d(np.array(t, dtype=float), np.array(model, dtype=float), kind='cubic', fill_value = 'extrapolate')
    model = func(np.array(y_time, dtype=float))     # This is the model fit refitted into the original 33 time points

    #result = integrateTrapezium(frame_lengths_m, y_dat)     # This is the approximate decay corrected PET data (TAC) to be used in the weighting calculating
    result = np.array(y_dat)
    # print(result)
    # print(y_dat)
    # print(y_dat - result)
    resids = model - np.array(y_dat, dtype=float)       # This is the plain residuals to be returned from the function after being multiplied by the weights, final five values are to replace any zero values in result

    if tracer == 'Rb82':
        decay_const = 0.55121048155860461981489631925104
        scale_factor = 0.002
    elif tracer == 'NH3':
        decay_const = 0.0695232879197537923186792498955
        scale_factor = 0.000003
    elif tracer == 'Water':
        decay_const = 0.34015041658021623807954727632776
        scale_factor = 0.5

    frame_dur = np.zeros(len(frame_lengths_m))
    exp_decay = np.zeros(len(frame_lengths_m))

    for i in range(len(frame_lengths_m)):
        frame_dur[i] = frame_lengths_m[i]
        exp_decay[i] = math.exp(- decay_const * framemidtimes_m[i])

        if result[i] == 0:
            result[i] = np.mean(resids[-5:])        # Maybe replace this value with an average of the last 5 residuals, 3 for degrado

    sigma_sq = scale_factor * (result / (frame_dur * exp_decay))
    #sigma_sq = 0.05 * (result / (frame_dur * exp_decay))            # Changed scale factor to 0.05 for comparison with results from PMOD using 0.05 as the scale factor
    weights = 1 / sigma_sq

    return (weights * resids)
    #return resids

def resid_blockage2(params, C0, data, y_time, y_dat, frame_lengths_m, framemidtimes_m, tracer):
    K_1 = params['K1'].value
    k_2 = params['k2'].value
    K_1_d = params['K1_d'].value
    k_2_d = params['k2_d'].value
    R = params['R'].value
    vb = params['vb'].value
    
    p = [K_1, k_2, K_1_d, k_2_d, R, vb]

    dt = 1/600
    T_f, T0, dt, time = time_vars(data, dt)
    
    u_out, t = RK4(comp_ode_model_blockage2, C0, init2, dt, T_f, T0, p)

    model = (1 - vb) * u_out[:, 1] + vb * (0.5 * (C0 + u_out[:, 0]))

    func = interp1d(np.array(t, dtype=float), np.array(model, dtype=float), kind='cubic', fill_value = 'extrapolate')
    model = func(np.array(y_time, dtype=float))     # This is the model fit refitted into the original 33 time points

    #result = integrateTrapezium(frame_lengths_m, y_dat)     # This is the approximate decay corrected PET data (TAC) to be used in the weighting calculating
    result = np.array(y_dat)
    # print(result)
    # print(y_dat)
    # print(y_dat - result)
    resids = model - np.array(y_dat, dtype=float)       # This is the plain residuals to be returned from the function after being multiplied by the weights, final five values are to replace any zero values in result

    if tracer == 'Rb82':
        decay_const = 0.55121048155860461981489631925104
        scale_factor = 0.002
    elif tracer == 'NH3':
        decay_const = 0.0695232879197537923186792498955
        scale_factor = 0.000003
    elif tracer == 'Water':
        decay_const = 0.34015041658021623807954727632776
        scale_factor = 0.5

    frame_dur = np.zeros(len(frame_lengths_m))
    exp_decay = np.zeros(len(frame_lengths_m))

    for i in range(len(frame_lengths_m)):
        frame_dur[i] = frame_lengths_m[i]
        exp_decay[i] = math.exp(- decay_const * framemidtimes_m[i])

        if result[i] == 0:
            result[i] = np.mean(resids[-5:])        # Maybe replace this value with an average of the last 5 residuals, 3 for degrado

    sigma_sq = scale_factor * (result / (frame_dur * exp_decay))
    #sigma_sq = 0.05 * (result / (frame_dur * exp_decay))            # Changed scale factor to 0.05 for comparison with results from PMOD using 0.05 as the scale factor
    weights = 1 / sigma_sq

    return (weights * resids)
    #return resids

def comp_ode_model_disp(u, C0, t, T0, p):
    R = p[0]

    du = np.zeros(1)

    ind = int(round((t - T0)/dt))

    # dC_0_disp / dt 
    du = R * C0[ind] - R * u[0] 

    # u[0] = C_0_disp(t)

    return du


def generate_C0_disp(C0, data, tau):
    R = 1/tau

    p = [R]

    dt = 1/600
    T_f, T0, dt, time = time_vars(data, dt)

    init = [0.0]
    u_out, t = RK4(comp_ode_model_disp, C0, init, dt, T_f, T0, p)

    return u_out[:, 0]


def dispersion_convolution(time, blood, mode):
    tau = 0.1        # here, k = 1/tau where tau is the dispersion time constant
    k = 1 /tau
    disp = k * np.exp(-k * time)

    conv = np.convolve(blood, disp, mode = mode)
    #conv = fftconvolve(blood, disp, mode = mode)
    #conv = k * blood

    return conv

def ANOVA(filename, foldername, tracer, state):
    data_1 = pd.read_excel(f'{filename}.xlsx', sheet_name = f'{tracer}_0.1_{state}', engine = 'openpyxl')
    data_5 = pd.read_excel(f'{filename}.xlsx', sheet_name = f'{tracer}_0.5_{state}', engine = 'openpyxl')
    data_9 = pd.read_excel(f'{filename}.xlsx', sheet_name = f'{tracer}_0.9_{state}', engine = 'openpyxl')
    data_1 = data_1.dropna(axis = 0)
    data_5 = data_5.dropna(axis = 0)
    data_9 = data_9.dropna(axis = 0)

    data_1 = data_1.sample(n=10, axis = 0, random_state=1)
    data_5 = data_5.sample(n=10, axis = 0, random_state=1)
    data_9 = data_9.sample(n=10, axis = 0, random_state=1)

    gens = len(data_1.K1)
    group_labels = ['All groups', 'vb_coeff = 0.1', 'vb_coeff = 0.5', 'vb_coeff = 0.9']

    ### K1
    f_stat, pvalue = f_oneway(data_1.K1, data_5.K1, data_9.K1)
    print('For K1 across values of vb_coeff:')
    print(f'F statistic of {f_stat}')
    print(f'P-value of {pvalue} \n')

    fig, ax = plt.subplots()

    ax.boxplot([pd.concat([data_1.K1, data_5.K1, data_9.K1], axis = 0), data_1.K1, data_5.K1, data_9.K1], labels = group_labels, vert = True)
    ax.set_ylabel('K1 (mL/min/g)')
    ax.set_xlabel('vb_coeff')
    
    fig.suptitle(f'K1 values from fits of {gens} noise \n realizations for {tracer} ({state})')
    fig.tight_layout()
    plt.savefig(f'{foldername}/{tracer}_K1_boxplots_{state}.png')
    plt.close()
    #plt.show()


    # ### K2
    f_stat, pvalue = f_oneway(data_1.k2, data_5.k2, data_9.k2)
    print('For k2 across values of vb_coeff:')
    print(f'F statistic of {f_stat}')
    print(f'P-value of {pvalue} \n')

    fig, ax = plt.subplots()

    ax.boxplot([pd.concat([data_1.k2, data_5.k2, data_9.k2], axis = 0), data_1.k2, data_5.k2, data_9.k2], labels = group_labels, vert = True)
    ax.set_ylabel('k2 (1/min)')
    ax.set_xlabel('vb_coeff')
    
    fig.suptitle(f'k2 values from fits of {gens} noise \n realizations for {tracer} ({state})')
    fig.tight_layout()
    plt.savefig(f'{foldername}/{tracer}_k2_boxplots_{state}.png')
    plt.close()
    

    # ### VB
    f_stat, pvalue = f_oneway(data_1.vb, data_5.vb, data_9.vb)
    print('For vb across values of vb_coeff:')
    print(f'F statistic of {f_stat}')
    print(f'P-value of {pvalue} \n')

    fig, ax = plt.subplots()

    ax.boxplot([pd.concat([data_1.vb, data_5.vb, data_9.vb], axis = 0), data_1.vb, data_5.vb, data_9.vb], labels = group_labels, vert = True)
    ax.set_ylabel('vb')
    ax.set_xlabel('vb_coeff')
    
    fig.suptitle(f'vb values from fits of {gens} noise \n realizations for {tracer} ({state})')
    fig.tight_layout()
    plt.savefig(f'{foldername}/{tracer}_vb_boxplots_{state}.png')
    plt.close()
    # fig, ax1 = plt.subplots()

    # ax1.boxplot([data_rest.vb, np.NaN], labels = states, vert = True)
    # ax1.axhline(y = vb_r, color = 'b', xmin = 0, xmax = 0.5)
    # ax1.set_ylim([0.0, 0.55])
    # ax1.set_ylabel('vb')
    # ax1.set_xlabel('State')
    
    # ax2 = ax1.twinx()
    # ax2.boxplot([np.NaN , data_stress.vb], labels = states, vert = True)
    # ax2.axhline(y = vb_s, color = 'r', xmin = 0.5, xmax = 1.0)
    # ax2.set_ylim([0.0, 0.55])
    # ax2.set_ylabel('vb')
    
    # fig.suptitle(f'vb values from fits of {gens} noise realizations \n for {tracer} with vb_coeff = {vb_coeff_value}')
    # fig.tight_layout()
    # plt.savefig(f'{foldername}/{tracer}_vb_boxplots_{vb_coeff_label}.png')
    # plt.close()
    # #plt.show()


def IEEE_figure(folder):
    rb82_C0 = np.array(rb82_data_int.Blood)
    rb82_C0_disp = np.array(rb82_data_int.Blood_Disp)

    init = [0.0]
    init2 = [0.0, 0.0]

    p1 = [0.47, 0.12, 0.48]         # K1, k2, vb
    p2 = [1.08, 0.21, 0.50]
    k1_ratio = p1[0]/p2[0]
    k2_ratio = p1[1]/p2[1]

    p1_blockage = [p1[0], p1[1], k1_ratio * p1[0], k2_ratio * p1[1], 1/0.75, p1[2]]
    p2_blockage = [p2[0], p2[1], k1_ratio * p2[0], k2_ratio * p2[1], 1/0.75, p2[2]]

    x1, t1 = RK4(comp_ode_model1, rb82_C0, init, rb82_dt, rb82_T_f, rb82_T0, p1)
    x2, t2 = RK4(comp_ode_model1, rb82_C0, init, rb82_dt, rb82_T_f, rb82_T0, p2)

    rb82_data_int.Heart_Rest = (1-p1[-1]) * x1[:, 0] + p1[-1] * rb82_C0             # How does vb calculation work for this system?
    rb82_data_int.Heart_Stress = (1-p2[-1]) * x2[:, 0] + p2[-1] * rb82_C0

    x1, t1 = RK4(comp_ode_model_blockage2, rb82_C0, init2, rb82_dt, rb82_T_f, rb82_T0, p1_blockage)
    x2, t2 = RK4(comp_ode_model_blockage2, rb82_C0, init2, rb82_dt, rb82_T_f, rb82_T0, p2_blockage)

    rb82_data_int['Heart_Rest_01'] = (1 - p1[-1]) * x1[:, 1] + p1[-1] * (0.1 * rb82_C0 + (1 - 0.1) * x1[:, 0])
    rb82_data_int['Heart_Stress_01'] = (1 - p2[-1]) * x2[:, 1] + p2[-1] * (0.1 * rb82_C0 + (1 - 0.1) * x2[:, 0])

    rb82_data_int['Heart_Rest_05'] = (1 - p1[-1]) * x1[:, 1] + p1[-1] * (0.5 * rb82_C0 + (1 - 0.5) * x1[:, 0])
    rb82_data_int['Heart_Stress_05'] = (1 - p2[-1]) * x2[:, 1] + p2[-1] * (0.5 * rb82_C0 + (1 - 0.5) * x2[:, 0])

    rb82_data_int['Heart_Rest_09'] = (1 - p1[-1]) * x1[:, 1] + p1[-1] * (0.9 * rb82_C0 + (1 - 0.9) * x1[:, 0])
    rb82_data_int['Heart_Stress_09'] = (1 - p2[-1]) * x2[:, 1] + p2[-1] * (0.9 * rb82_C0 + (1 - 0.9) * x2[:, 0])
    
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (8, 8), constrained_layout = False, tight_layout = True)
    #fig.legend(loc = 7, fontsize = 'medium')

    # First row
    ax1.plot(rb82_data.Time, rb82_data.Blood, linestyle = '--', linewidth = 2, label = 'Blood', color = 'r')
    ax1.plot(rb82_data_int.Time, rb82_data_int.Heart_Rest, linewidth = 2, label = 'Tissue (Rest)', color = 'b')
    ax1.plot(rb82_data_int.Time, rb82_data_int.Heart_Stress, linewidth = 2, label = 'Tissue (Stress)', color = 'g')
    ax1.set_ylabel('Activity (kBq/cc)', fontsize = 'x-large')
    ax1.text(2.0, 85.0, 'Control', fontsize = 'xx-large', verticalalignment = 'top')
    ax1.tick_params(axis = 'both', which = 'both', labelsize = 'large')

    ax2.plot(rb82_data.Time, rb82_data.Blood, linestyle = '--', linewidth = 2, label = r'$C_0$', color = 'r')
    ax2.plot(rb82_data.Time, rb82_data.Blood_Disp, linestyle = '-.', linewidth = 2, label = r'$\'C_0$', color = 'm')
    ax2.plot(rb82_data_int.Time, rb82_data_int.Heart_Rest_01, linewidth = 2, label = r'$C_1$ (Rest)', color = 'b')
    ax2.plot(rb82_data_int.Time, rb82_data_int.Heart_Stress_01, linewidth = 2, label = r'$C_1$ (Stress)', color = 'g')
    ax2.legend(loc = 'lower left',  bbox_to_anchor = (0.48, 0.48), fontsize = 'large')
    ax2.text(2.0, 85.0, r'$\beta = 0.1$', fontsize = 'xx-large', verticalalignment = 'top')
    ax2.tick_params(axis = 'both', which = 'both', labelsize = 'large')

    # Second row
    ax3.plot(rb82_data.Time, rb82_data.Blood, linestyle = '--', linewidth = 2, label = 'Blood', color = 'r')
    ax3.plot(rb82_data.Time, rb82_data.Blood_Disp, linestyle = '-.', linewidth = 2, label = 'Blood (Disp)', color = 'm')
    ax3.plot(rb82_data_int.Time, rb82_data_int.Heart_Rest_05, linewidth = 2, label = 'Tissue (Rest)', color = 'b')
    ax3.plot(rb82_data_int.Time, rb82_data_int.Heart_Stress_05, linewidth = 2, label = 'Tissue (Stress)', color = 'g')
    ax3.set_ylabel('Activity (kBq/cc)', fontsize = 'x-large')
    ax3.set_xlabel('Time (minutes)', fontsize = 'x-large')
    ax3.text(2.0, 85.0, r'$\beta = 0.5$', fontsize = 'xx-large', verticalalignment = 'top')
    ax3.tick_params(axis = 'both', which = 'both', labelsize = 'large')

    ax4.plot(rb82_data.Time, rb82_data.Blood, linestyle = '--', linewidth = 2, label = 'Blood', color = 'r')
    ax4.plot(rb82_data.Time, rb82_data.Blood_Disp, linestyle = '-.', linewidth = 2, label = 'Blood (Disp)', color = 'm')
    ax4.plot(rb82_data_int.Time, rb82_data_int.Heart_Rest_09, linewidth = 2, label = 'Tissue (Rest)', color = 'b')
    ax4.plot(rb82_data_int.Time, rb82_data_int.Heart_Stress_09, linewidth = 2, label = 'Tissue (Stress)', color = 'g')
    ax4.set_xlabel('Time (minutes)', fontsize = 'x-large')
    ax4.text(2.0, 85.0, r'$\beta = 0.9$', fontsize = 'xx-large', verticalalignment = 'top')
    ax4.tick_params(axis = 'both', which = 'both', labelsize = 'large')

    plt.savefig(f'{folder}/Figure_2.png')
    plt.close()



# Obtain the frametimes for Rb82

rb82_frametimes_m, rb82_frametimes_mid_m, rb82_frame_lengths_m, rb82_frametimes_s = frametimes('Rb82')


# Calculate blood curve

rb82_blood = Rb82_aif(rb82_frametimes_mid_m)

# Create dataframes for Rb82 containing time and artificial blood data

rb82_data = pd.DataFrame(data = {'Time' : rb82_frametimes_mid_m, 'Blood' : rb82_blood, 'Blood_Disp' : np.zeros(len(rb82_blood))})
rb82_data = rb82_data.reset_index()
rb82_data = rb82_data.drop('index', axis=1)


# Create new dataframes of interpolated data for Rb82

date = datetime.now().strftime('%d-%m-%Y')

dt = 1/600
rb82_T_f, rb82_T0, rb82_dt, rb82_time = time_vars(rb82_data, dt)

rb82_data_int = pd.DataFrame(columns = ['Time', 'Blood', 'Blood_Disp', 'Heart_Rest', 'Heart_Stress', 'Heart_Rest_no_vb', 'Heart_Stress_no_vb'])
rb82_data_int.Time = rb82_time
rb82_data_int['Blood'], rb82_inter_func = interpolate(rb82_data.Time, rb82_data.Blood, 'cubic', rb82_T0, rb82_T_f, rb82_dt)



# Calculate the dispersed blood curve to represent the artery post-blockage

tau = 0.75    # Tau values between 0.05 to 0.5
rb82_data_int['Blood_Disp'] = generate_C0_disp(rb82_data_int['Blood'], rb82_data_int, tau)
rb82_data['Blood_Disp'], func1 = interpolate_frames(rb82_data_int.Time, rb82_data_int['Blood_Disp'], 'cubic', rb82_frametimes_mid_m)

# plt.plot(rb82_data_int['Time'], rb82_data_int['Blood'], label = 'Blood', color = 'r')
# plt.plot(rb82_data_int['Time'], rb82_data_int['Blood_Disp'], label = 'Blood (Dispersed)', color = 'b')
# plt.title(f'Blood Curves (Rb82), Tau = {tau}, R = {1/tau}')
# plt.xlabel('Time (minutes)')
# plt.ylabel('Activity (Bq/cc)')
# plt.legend(loc = 7, fontsize = 'x-small')
# #plt.savefig(f'Rb82_Blood_Curves_Tau_{tau}.png')
# #plt.close()
# plt.show()

# print(rb82_data_int['Blood'])
# print(rb82_data_int['Blood_Disp'])

# Generates graph showing all tau values for comparison
# tau_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
# blood_disp_values = []


# for tau in tau_values:
#     data = generate_C0_disp(rb82_data_int['Blood'], rb82_data_int, tau)
#     blood_disp_values.append(data)


# plt.plot(rb82_data_int['Time'], rb82_data_int['Blood'], label = 'Blood (Original)', color = 'r')
# plt.plot(rb82_data_int['Time'], blood_disp_values[0], label = 'Blood (Tau = 0.05)')
# plt.plot(rb82_data_int['Time'], blood_disp_values[1], label = 'Blood (Tau = 0.10)')
# plt.plot(rb82_data_int['Time'], blood_disp_values[2], label = 'Blood (Tau = 0.15)')
# plt.plot(rb82_data_int['Time'], blood_disp_values[3], label = 'Blood (Tau = 0.20)')
# plt.plot(rb82_data_int['Time'], blood_disp_values[4], label = 'Blood (Tau = 0.25)')
# plt.plot(rb82_data_int['Time'], blood_disp_values[5], label = 'Blood (Tau = 0.30)')
# plt.plot(rb82_data_int['Time'], blood_disp_values[6], label = 'Blood (Tau = 0.35)')
# plt.plot(rb82_data_int['Time'], blood_disp_values[7], label = 'Blood (Tau = 0.40)')
# plt.plot(rb82_data_int['Time'], blood_disp_values[8], label = 'Blood (Tau = 0.45)')
# plt.plot(rb82_data_int['Time'], blood_disp_values[9], label = 'Blood (Tau = 0.50)')
# plt.title(f'Blood Curves (Rb82)')
# plt.xlabel('Time (minutes)')
# plt.ylabel('Activity (Bq/cc)')
# plt.legend(loc = 7, fontsize = 'x-small')
# plt.savefig(f'Rb82/Blood_Curves/Rb82_Blood_Curves_All_Tau.png')
# plt.close()
# #plt.show()



# Define parameters for both tracers

tissue_type = 'Blockage'        # Either 1TCM or Blockage depending on what model you want to use to generate a tissue curve
vb_coeff = 0.9

rb82_C0 = np.array(rb82_data_int.Blood)
rb82_C0_disp = np.array(rb82_data_int.Blood_Disp)

init = [0.0]
init2 = [0.0, 0.0]

p1 = [0.47, 0.12, 0.48]         # K1, k2, vb
p2 = [1.08, 0.21, 0.50]
k1_ratio = p1[0]/p2[0]
k2_ratio = p1[1]/p2[1]

p1_blockage = [p1[0], p1[1], k1_ratio * p1[0], k2_ratio * p1[1], 1/0.75, p1[2]]
p2_blockage = [p2[0], p2[1], k1_ratio * p2[0], k2_ratio * p2[1], 1/0.75, p2[2]]

if tissue_type == '1TCM':
    x1, t1 = RK4(comp_ode_model1, rb82_C0, init, rb82_dt, rb82_T_f, rb82_T0, p1)
    x2, t2 = RK4(comp_ode_model1, rb82_C0, init, rb82_dt, rb82_T_f, rb82_T0, p2)

    rb82_data_int.Heart_Rest = (1-p1[-1]) * x1[:, 0] + p1[-1] * rb82_C0             # How does vb calculation work for this system?
    rb82_data_int.Heart_Stress = (1-p2[-1]) * x2[:, 0] + p2[-1] * rb82_C0

elif tissue_type == 'Blockage':
    x1, t1 = RK4(comp_ode_model_blockage2, rb82_C0, init2, rb82_dt, rb82_T_f, rb82_T0, p1_blockage)
    x2, t2 = RK4(comp_ode_model_blockage2, rb82_C0, init2, rb82_dt, rb82_T_f, rb82_T0, p2_blockage)

    rb82_data_int.Heart_Rest = (1 - p1[-1]) * x1[:, 1] + p1[-1] * (vb_coeff * rb82_C0 + (1 - vb_coeff) * x1[:, 0])
    rb82_data_int.Heart_Stress = (1 - p2[-1]) * x2[:, 1] + p2[-1] * (vb_coeff * rb82_C0 + (1 - vb_coeff) * x2[:, 0])


rb82_data_int.Heart_Rest_no_vb = x1[:, 0]           
rb82_data_int.Heart_Stress_no_vb = x2[:, 0]

hr, hr_func = interpolate_frames(rb82_data_int.Time, rb82_data_int.Heart_Rest, 'cubic', rb82_frametimes_mid_m)
hs, hs_func = interpolate_frames(rb82_data_int.Time, rb82_data_int.Heart_Stress, 'cubic', rb82_frametimes_mid_m)

rb82_data['Heart_Rest'] = hr
rb82_data['Heart_Stress'] = hs

#print(rb82_data)


##########################
# SUBPLOT FOR IEEE ABSTRACT 
IEEE_figure('IEEE_Abstract')

# Plot the tissue curve graphs

# Rb82 standard tissue curves
# plt.scatter(rb82_data.Time, rb82_data.Blood, s = 7, label = 'Blood', color = 'r')
# #plt.scatter(rb82_data.Time, rb82_data.Blood_Disp, s = 7, label = 'Blood (Disp)', color = 'm')
# plt.plot(t1, ((1-p1[-1]) * x1[:, 0] + p1[-1] * rb82_C0), label = 'Tissue', color = 'b')
# plt.title('Tissue Curve (Rb82 Rest)')
# plt.xlabel('Time (minutes)')
# plt.ylabel('Activity (kBq/cc)')
# plt.legend(loc = 7, fontsize = 'x-small')
# plt.savefig('Rb82/Standard_Tissue_Curves/Rb82_tissue_curve_rest')
# plt.close()

# plt.scatter(rb82_data.Time, rb82_data.Blood, s = 7, label = 'Blood', color = 'r')
# #plt.scatter(rb82_data.Time, rb82_data.Blood_Disp, s = 7, label = 'Blood (Disp)', color = 'm')
# plt.plot(t2, ((1-p2[-1]) * x2[:, 0] + p2[-1] * rb82_C0), label = 'Tissue', color = 'b')
# plt.title('Tissue Curve (Rb82 Stress)')
# plt.xlabel('Time (minutes)')
# plt.ylabel('Activity (kBq/cc)')
# plt.legend(loc = 7, fontsize = 'x-small')
# plt.savefig('Rb82/Standard_Tissue_Curves/Rb82_tissue_curve_stress')
# plt.close()

# plt.scatter(rb82_data.Time, rb82_data.Blood, s = 7, label = 'Blood', color = 'r')
# #plt.scatter(rb82_data.Time, rb82_data.Blood_Disp, s = 7, label = 'Blood (Disp)', color = 'm')
# plt.plot(t1, ((1-p1[-1]) * x1[:, 0] + p1[-1] * rb82_C0), label = 'Tissue (Rest)', color = 'b')
# plt.plot(t2, ((1-p2[-1]) * x2[:, 0] + p2[-1] * rb82_C0), label = 'Tissue (Stress)', color = 'g')
# plt.title('Tissue Curves (Rb82)')
# plt.xlabel('Time (minutes)')
# plt.ylabel('Activity (kBq/cc)')
# plt.legend(loc = 7, fontsize = 'x-small')
# plt.savefig('Rb82/Standard_Tissue_Curves/Rb82_tissue_curve_both')
# plt.close()



# # Rb82 calculated tissue curves
# plt.scatter(rb82_data.Time, rb82_data.Blood, s = 7, label = 'Blood', color = 'r')
# plt.scatter(rb82_data.Time, rb82_data.Blood_Disp, s = 7, label = 'Blood (Disp)', color = 'm')
# plt.plot(rb82_data_int.Time, rb82_data_int.Heart_Rest, label = 'Tissue', color = 'b')
# plt.title(f'Tissue Curve (Rb82 Rest) with vb_coeff = {vb_coeff}')
# plt.xlabel('Time (minutes)')
# plt.ylabel('Activity (kBq/cc)')
# plt.legend(loc = 7, fontsize = 'x-small')
# plt.savefig(f'Rb82/Calculated_Tissue_Curves/Rb82_tissue_curve_{vb_coeff}_rest.png')
# plt.close()

# plt.scatter(rb82_data.Time, rb82_data.Blood, s = 7, label = 'Blood', color = 'r')
# plt.scatter(rb82_data.Time, rb82_data.Blood_Disp, s = 7, label = 'Blood (Disp)', color = 'm')
# plt.plot(rb82_data_int.Time, rb82_data_int.Heart_Stress, label = 'Tissue', color = 'b')
# plt.title(f'Tissue Curve (Rb82 Stress) with vb_coeff = {vb_coeff}')
# plt.xlabel('Time (minutes)')
# plt.ylabel('Activity (kBq/cc)')
# plt.legend(loc = 7, fontsize = 'x-small')
# plt.savefig(f'Rb82/Calculated_Tissue_Curves/Rb82_tissue_curve_{vb_coeff}_stress.png')
# plt.close()

# plt.scatter(rb82_data.Time, rb82_data.Blood, s = 7, label = 'Blood', color = 'r')
# plt.scatter(rb82_data.Time, rb82_data.Blood_Disp, s = 7, label = 'Blood (Disp)', color = 'm')
# plt.plot(rb82_data_int.Time, rb82_data_int.Heart_Rest, label = 'Tissue (rest)', color = 'b')
# plt.plot(rb82_data_int.Time, rb82_data_int.Heart_Stress, label = 'Tissue (stress)', color = 'g')
# plt.title(f'Tissue Curves (Rb82) with vb_coeff = {vb_coeff}')
# plt.xlabel('Time (minutes)')
# plt.ylabel('Activity (kBq/cc)')
# plt.legend(loc = 7, fontsize = 'x-small')
# plt.savefig(f'Rb82/Calculated_Tissue_Curves/Rb82_tissue_curve_{vb_coeff}_both.png')
# plt.close()


####

tracer = 'Rb82'
state = 'Stress'
mode = 'None' #'1TCM_fit_noise_realisations'

folder = 'Rb82'
sub_folder = '1TCM_fitting_control_tissue_noise_realisations'

gens = 100

rb82_tissue_values = pd.DataFrame(columns = ['Time', 'Heart_Tau_0.05', 'Heart_Tau_0.10', 'Heart_Tau_0.15', 'Heart_Tau_0.20', 'Heart_Tau_0.25', 'Heart_Tau_0.30', 
                                                'Heart_Tau_0.35', 'Heart_Tau_0.40', 'Heart_Tau_0.45', 'Heart_Tau_0.50'])
rb82_tissue_values = rb82_tissue_values.reset_index()
rb82_tissue_values = rb82_tissue_values.drop('index', axis=1)
rb82_tissue_values.Time = rb82_time

tau_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
title_values = ['Heart_Tau_0.05', 'Heart_Tau_0.10', 'Heart_Tau_0.15', 'Heart_Tau_0.20', 'Heart_Tau_0.25', 'Heart_Tau_0.30', 'Heart_Tau_0.35', 'Heart_Tau_0.40', 'Heart_Tau_0.45', 'Heart_Tau_0.50']


if mode == 'Blockage1':
    for i in range(len(tau_values)):
        tau = tau_values[i]
        y_time = rb82_data.Time
        y_0 = rb82_data.Blood
        C0 = rb82_C0
        C0_disp = generate_C0_disp(rb82_data_int['Blood'], rb82_data_int, tau)

        y_dat = rb82_data.Heart_Rest
        C1_data = rb82_data_int.Heart_Rest

        if state == 'Rest':
            p = [0.47, 0.12, 0.48]
            tag = 'Heart_Rest'
        elif state == 'Stress':
            p = [1.08, 0.21, 0.50]
            tag = 'Heart_Stress'


        params = lmfit.Parameters()
        params.add('K1', p[0], min=0.0, max=5.0)
        params.add('k2', p[1], min=0.0, max=5.0)
        params.add('K1_d', 0.5 * p[0], min=0.0, max=5.0)
        params.add('k2_d', 0.5 * p[1], min=0.0, max=5.0)
        params.add('R', 1/tau)
        params.add('vb', p[2], vary = True, min=0.0, max=1.0)

        x1, t1 = RK4_disp(comp_ode_model_blockage1, rb82_C0, rb82_C0_disp, init2, rb82_dt, rb82_T_f, rb82_T0, [params['K1'].value, params['k2'].value, params['K1_d'].value, params['k2_d'].value, params['vb'].value])
        val1, val2, val3, val4, valvb = [params['K1'].value, params['k2'].value, params['K1_d'].value, params['k2_d'].value, params['vb'].value]

        #with_vb = (1 - params['vb'].value) * x1[:, 0] + params['vb'].value * (0.66666 * C0 + 0.33333 * C0_disp)          # Unsure about this?
        with_vb = (1 - params['vb'].value) * x1[:, 0] + params['vb'].value * 0.5 *( C0 + C0_disp)
        rb82_tissue_values[title_values[i]] = with_vb
        
        plt.scatter(rb82_data.Time, rb82_data.Blood, s = 7, label = 'C0', color = 'g')
        plt.scatter(rb82_data.Time, rb82_data.Blood_Disp, s = 7, label = 'C0\'', color = 'r')
        plt.plot(t1, with_vb, label = 'C1', color = 'b')
        plt.title(f'Tissue Curve ({tracer} {state}) K1 = {val1:.3f}, k2 = {val2:.3f}, \n K1_d = {val3:.3f}  k2_d = {val4:.3f}, vb = {valvb:.3f}')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Activity (Bq/cc)')
        plt.legend(loc = 7, fontsize = 'x-small')
        plt.savefig(f'{folder}\{sub_folder}\Blockage1\Rb82_{state}_Tau{tau}.png')
        plt.close()
        #plt.show()

    rb82_tissue_values.to_excel(f'{folder}\{sub_folder}\Blockage1\Rb82_tissue_values_{state}_All_Tau_{date}.xlsx')

    plt.scatter(rb82_data.Time, rb82_data.Blood, s = 7, label = 'C0', color = 'g')
    plt.scatter(rb82_data.Time, rb82_data.Blood_Disp, s = 7, label = 'C0\'', color = 'r')
    plt.plot(rb82_data_int.Time, rb82_data_int[tag], label = 'C1 (Original)', color = 'm')
    plt.plot(rb82_tissue_values.Time, rb82_tissue_values['Heart_Tau_0.05'], label = 'C1 (Tau = 0.05)')
    plt.plot(rb82_tissue_values.Time, rb82_tissue_values['Heart_Tau_0.10'], label = 'C1 (Tau = 0.10)')
    plt.plot(rb82_tissue_values.Time, rb82_tissue_values['Heart_Tau_0.15'], label = 'C1 (Tau = 0.15)')
    plt.plot(rb82_tissue_values.Time, rb82_tissue_values['Heart_Tau_0.20'], label = 'C1 (Tau = 0.20)')
    plt.plot(rb82_tissue_values.Time, rb82_tissue_values['Heart_Tau_0.25'], label = 'C1 (Tau = 0.25)')
    plt.plot(rb82_tissue_values.Time, rb82_tissue_values['Heart_Tau_0.30'], label = 'C1 (Tau = 0.30)')
    plt.plot(rb82_tissue_values.Time, rb82_tissue_values['Heart_Tau_0.35'], label = 'C1 (Tau = 0.35)')
    plt.plot(rb82_tissue_values.Time, rb82_tissue_values['Heart_Tau_0.40'], label = 'C1 (Tau = 0.40)')
    plt.plot(rb82_tissue_values.Time, rb82_tissue_values['Heart_Tau_0.45'], label = 'C1 (Tau = 0.45)')
    plt.plot(rb82_tissue_values.Time, rb82_tissue_values['Heart_Tau_0.50'], label = 'C1 (Tau = 0.50)')
    plt.title(f'Tissue Curve ({tracer} {state}) K1 = {val1:.3f}, k2 = {val2:.3f}, \n K1_d = {val3:.3f}  k2_d = {val4:.3f}, vb = {valvb:.3f}')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Activity (Bq/cc)')
    plt.legend(loc = 7, fontsize = 'x-small')
    plt.savefig(f'{folder}\{sub_folder}\Blockage1\Rb82_{state}_All_Tau.png')
    plt.close()

elif mode == 'Blockage2':
    for i in range(len(tau_values)):
        tau = tau_values[i]
        y_time = rb82_data.Time
        y_0 = rb82_data.Blood
        C0 = rb82_C0
        #C0_disp = generate_C0_disp(rb82_data_int['Blood'], rb82_data_int, tau)

        y_dat = rb82_data.Heart_Rest
        C1_data = rb82_data_int.Heart_Rest

        if state == 'Rest':
            p = [0.47, 0.12, 0.48]
            tag = 'Heart_Rest'
        elif state == 'Stress':
            p = [1.08, 0.21, 0.50]
            tag = 'Heart_Stress'

        init2 = [0.0, 0.0]

        params = lmfit.Parameters()
        params.add('K1', p[0], min=0.0, max=5.0)
        params.add('k2', p[1], min=0.0, max=5.0)
        params.add('K1_d', 0.5 * p[0], min=0.0, max=5.0)
        params.add('k2_d', 0.5 * p[1], min=0.0, max=5.0)
        params.add('R', 1/tau)
        params.add('vb', p[2], vary = True, min=0.0, max=1.0)

        x1, t1 = RK4(comp_ode_model_blockage2, rb82_C0, init2, rb82_dt, rb82_T_f, rb82_T0, [params['K1'].value, params['k2'].value, params['K1_d'].value, params['k2_d'].value, params['R'].value, params['vb'].value])
        val1, val2, val3, val4, valR, valvb = [params['K1'].value, params['k2'].value, params['K1_d'].value, params['k2_d'].value, params['R'].value, params['vb'].value]

        #with_vb = (1 - params['vb'].value) * x1[:, 0] + params['vb'].value * (0.66666 * C0 + 0.33333 * C0_disp)          # Unsure about this?
        with_vb = (1 - params['vb'].value) * x1[:, 1] + params['vb'].value * 0.5 *( C0 + x1[:, 0])
        rb82_tissue_values[title_values[i]] = with_vb
        
        plt.scatter(rb82_data.Time, rb82_data.Blood, s = 7, label = 'C0', color = 'g')
        plt.plot(t1, x1[:, 0], label = 'C0\'', color = 'r')
        plt.plot(t1, x1[:, 1], label = 'C1', color = 'm')
        plt.plot(t1, with_vb, label = 'Tissue', color = 'b')
        plt.title(f'Tissue Curve ({tracer} {state}) K1 = {val1:.3f}, k2 = {val2:.3f}, \n K1_d = {val3:.3f}  k2_d = {val4:.3f}, R = {valR:.3f}, vb = {valvb:.3f}')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Activity (Bq/cc)')
        plt.legend(loc = 7, fontsize = 'x-small')
        plt.savefig(f'{folder}\{sub_folder}\Blockage2\Rb82_{state}_Tau{tau}.png')
        plt.close()
        #plt.show()

    rb82_tissue_values.to_excel(f'{folder}\{sub_folder}\Blockage2\Rb82_tissue_values_{state}_All_Tau_{date}.xlsx')

    plt.scatter(rb82_data.Time, rb82_data.Blood, s = 7, label = 'C0', color = 'g')
    plt.scatter(rb82_data.Time, rb82_data.Blood_Disp, s = 7, label = 'C0\'', color = 'r')
    plt.plot(rb82_data_int.Time, rb82_data_int[tag], label = 'C1 (Original)', color = 'm')
    plt.plot(rb82_tissue_values.Time, rb82_tissue_values['Heart_Tau_0.05'], label = 'C1 (Tau = 0.05)')
    plt.plot(rb82_tissue_values.Time, rb82_tissue_values['Heart_Tau_0.10'], label = 'C1 (Tau = 0.10)')
    plt.plot(rb82_tissue_values.Time, rb82_tissue_values['Heart_Tau_0.15'], label = 'C1 (Tau = 0.15)')
    plt.plot(rb82_tissue_values.Time, rb82_tissue_values['Heart_Tau_0.20'], label = 'C1 (Tau = 0.20)')
    plt.plot(rb82_tissue_values.Time, rb82_tissue_values['Heart_Tau_0.25'], label = 'C1 (Tau = 0.25)')
    plt.plot(rb82_tissue_values.Time, rb82_tissue_values['Heart_Tau_0.30'], label = 'C1 (Tau = 0.30)')
    plt.plot(rb82_tissue_values.Time, rb82_tissue_values['Heart_Tau_0.35'], label = 'C1 (Tau = 0.35)')
    plt.plot(rb82_tissue_values.Time, rb82_tissue_values['Heart_Tau_0.40'], label = 'C1 (Tau = 0.40)')
    plt.plot(rb82_tissue_values.Time, rb82_tissue_values['Heart_Tau_0.45'], label = 'C1 (Tau = 0.45)')
    plt.plot(rb82_tissue_values.Time, rb82_tissue_values['Heart_Tau_0.50'], label = 'C1 (Tau = 0.50)')
    plt.title(f'Tissue Curve ({tracer} {state}) K1 = {val1:.3f}, k2 = {val2:.3f}, \n K1_d = {val3:.3f}  k2_d = {val4:.3f}, vb = {valvb:.3f}')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Activity (Bq/cc)')
    plt.legend(loc = 7, fontsize = 'x-small')
    plt.savefig(f'{folder}\{sub_folder}\Blockage2\Rb82_{state}_All_Tau.png')
    plt.close()

elif mode == 'Blockage3':
    tau = 0.75
    y_time = rb82_data.Time
    y_0 = rb82_data.Blood
    C0 = rb82_C0
    #C0_disp = generate_C0_disp(rb82_data_int['Blood'], rb82_data_int, tau)

    

    if state == 'Rest':
        p = [0.47, 0.12, 0.48]
        tag = 'Heart_Rest'
        y_dat = rb82_data.Heart_Rest
        C1_data = rb82_data_int.Heart_Rest
        test_C1 = rb82_data_int.Heart_Rest_no_vb
    elif state == 'Stress':
        p = [1.08, 0.21, 0.50]
        tag = 'Heart_Stress'
        y_dat = rb82_data.Heart_Stress
        C1_data = rb82_data_int.Heart_Stress
        test_C1 = rb82_data_int.Heart_Stress_no_vb

    init2 = [0.0, 0.0]

    params = lmfit.Parameters()
    params.add('K1', p[0], min=0.0, max=5.0)
    params.add('k2', p[1], min=0.0, max=5.0)
    params.add('K1_d', k1_ratio * p[0], min=0.0, max=5.0)
    params.add('k2_d', k2_ratio * p[1], min=0.0, max=5.0)
    params.add('R', 1/tau)
    params.add('vb', p[2], min=0.0, max=1.0)
    
    fit = lmfit.minimize(resid1_weighted, params, args = (C0, rb82_data, y_time, y_dat, rb82_frame_lengths_m, rb82_frametimes_mid_m, tracer), method = 'leastsq', max_nfev = 2500)
    lmfit.report_fit(fit)

    val1, val2, val3, val4, valR, valvb = [params['K1'].value, params['k2'].value, params['K1_d'].value, params['k2_d'].value, params['R'].value, params['vb'].value]
    
    x1, t1 = RK4(comp_ode_model1, rb82_C0, init, rb82_dt, rb82_T_f, rb82_T0, [params['K1'].value, params['k2'].value, params['vb'].value])
    x2, t2 = RK4(comp_ode_model_blockage_midstep, rb82_C0, init2, rb82_dt, rb82_T_f, rb82_T0, [params['K1'].value, params['k2'].value, params['R'].value, params['vb'].value])
    x3, t3 = RK4(comp_ode_model_blockage2, rb82_C0, init2, rb82_dt, rb82_T_f, rb82_T0, [params['K1'].value, params['k2'].value, params['K1_d'].value, params['k2_d'].value, params['R'].value, params['vb'].value])

    C0_coeffs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    C0_coeff_titles = ['vb_1', 'vb_2', 'vb_3', 'vb_4', 'vb_5', 'vb_6', 'vb_7', 'vb_8', 'vb_9', 'vb_10']

    vb_test = pd.DataFrame(columns = ['Time', 'vb_1', 'vb_2', 'vb_3', 'vb_4', 'vb_5', 'vb_6', 
                                                'vb_7', 'vb_8', 'vb_9', 'vb_10'])
    vb_test = vb_test.reset_index()
    vb_test = vb_test.drop('index', axis=1)
    vb_test.Time = rb82_time
    
    for i in range(len(C0_coeffs)):
        x4 = (1 - params['vb'].value) * x1[:, 0] + params['vb'].value * C0
        x5 = (1 - params['vb'].value) * x2[:, 1] + params['vb'].value * (C0_coeffs[i] * C0 + (1 - C0_coeffs[i]) * x2[:, 0])
        x6 = (1 - params['vb'].value) * x3[:, 1] + params['vb'].value * (C0_coeffs[i] * C0 + (1 - C0_coeffs[i]) * x3[:, 0])

        vb_test[C0_coeff_titles[i]] = x6
        
        plt.scatter(rb82_data.Time, rb82_data.Blood, s = 7, label = 'C0', color = 'r')
        plt.plot(t1, x1[:, 0], label = 'C1', color = 'b')
        plt.plot(t1, x4, label = 'PET Activity', color = 'g')
        plt.plot(rb82_data_int.Time, C1_data, label = 'Tissue', color = 'm')
        plt.title(f'Tissue Curve ({tracer} {state}) K1 = {val1:.3f}, k2 = {val2:.3f}, vb = {valvb:.3f}')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Activity (Bq/cc)')
        plt.legend(loc = 7, fontsize = 'x-small')
        plt.savefig(f'{folder}\{sub_folder}\Step_by_Step\Rb82_{state}_C0_coeff_{C0_coeffs[i]}_Step1.png')
        plt.close()
        #plt.show()

        plt.scatter(rb82_data.Time, rb82_data.Blood, s = 7, label = 'C0', color = 'r')
        plt.plot(t2, x2[:, 0], label = 'C0\'', color = 'g')
        plt.plot(t2, x2[:, 1], label = 'C1', color = 'b')
        plt.plot(t2, x5, label = 'PET Activity', color = 'm')
        plt.plot(rb82_data_int.Time, C1_data, label = 'Tissue', color = 'm')
        plt.title(f'Tissue Curve ({tracer} {state}) K1 = {val1:.3f}, k2 = {val2:.3f}, \n R = {valR:.3f}, vb = {valvb:.3f}')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Activity (Bq/cc)')
        plt.legend(loc = 7, fontsize = 'x-small')
        plt.savefig(f'{folder}\{sub_folder}\Step_by_Step\Rb82_{state}_C0_coeff_{C0_coeffs[i]}_Step2.png')
        plt.close()
        #plt.show()

        # print('C0\':')
        # print(x2[:25, 0])
        # print('C1:')
        # print(x2[:25, 1])

        plt.scatter(rb82_data.Time, rb82_data.Blood, s = 7, label = 'C0', color = 'r')
        plt.plot(t3, x3[:, 0], label = 'C0\'', color = 'g')
        plt.plot(t3, x3[:, 1], label = 'C1', color = 'b')
        plt.plot(t3, x6, label = 'PET Activity', color = 'm')
        plt.plot(rb82_data_int.Time, C1_data, label = 'Tissue', color = 'm')
        plt.title(f'Tissue Curve ({tracer} {state}) K1 = {val1:.3f}, k2 = {val2:.3f}, \n K1_d = {val3:.3f}  k2_d = {val4:.3f}, R = {valR:.3f}, vb = {valvb:.3f}')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Activity (Bq/cc)')
        plt.legend(loc = 7, fontsize = 'x-small')
        plt.savefig(f'{folder}\{sub_folder}\Step_by_Step\Rb82_{state}_C0_coeff_{C0_coeffs[i]}_Step3.png')
        plt.close()
        #plt.show()

    plt.plot(rb82_data_int['Time'], rb82_data_int['Blood'], label = 'Blood', color = 'r')
    plt.plot(rb82_data_int['Time'], rb82_data_int['Blood_Disp'], label = 'Blood (Dispersed)', color = 'b')
    plt.plot(rb82_data_int['Time'], 0.25 * rb82_data_int['Blood'], label = 'Blood (25%% Lumen Occupancy)', color = 'm')
    plt.title(f'Blood Curves (Rb82), Tau = {tau}, R = {1/tau}')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Activity (Bq/cc)')
    plt.legend(loc = 7, fontsize = 'x-small')
    plt.savefig(f'{folder}\{sub_folder}\Step_by_Step\Rb82_Blood_Curves_Tau_{tau}.png')
    plt.close()
    #plt.show()
    
    plt.scatter(rb82_data.Time, rb82_data.Blood, s = 7, label = 'C0', color = 'r')
    plt.plot(t3, x3[:, 0], label = 'C0\'', color = 'g')
    plt.plot(vb_test.Time, vb_test['vb_1'], label = 'vb = 0.1')
    plt.plot(vb_test.Time, vb_test['vb_2'], label = 'vb = 0.2')
    plt.plot(vb_test.Time, vb_test['vb_3'], label = 'vb = 0.3')
    plt.plot(vb_test.Time, vb_test['vb_4'], label = 'vb = 0.4')
    plt.plot(vb_test.Time, vb_test['vb_5'], label = 'vb = 0.5')
    plt.plot(vb_test.Time, vb_test['vb_6'], label = 'vb = 0.6')
    plt.plot(vb_test.Time, vb_test['vb_7'], label = 'vb = 0.7')
    plt.plot(vb_test.Time, vb_test['vb_8'], label = 'vb = 0.8')
    plt.plot(vb_test.Time, vb_test['vb_9'], label = 'vb = 0.9')
    plt.plot(vb_test.Time, vb_test['vb_10'], label = 'vb = 1.0')
    plt.title(f'PET Activity Curve ({tracer} {state}) K1 = {val1:.3f}, k2 = {val2:.3f}, \n K1_d = {val3:.3f}  k2_d = {val4:.3f}, vb = {valvb:.3f}')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Activity (Bq/cc)')
    plt.legend(loc = 7, fontsize = 'x-small')
    plt.savefig(f'{folder}\{sub_folder}\Step_by_Step\Rb82_{state}_all_vb.png')
    plt.close()

elif mode == '1TCM_fit':
    tau = 0.75
    y_time = rb82_data.Time
    y_0 = rb82_data.Blood
    C0 = rb82_C0
    #C0_disp = generate_C0_disp(rb82_data_int['Blood'], rb82_data_int, tau)

    rb82_param_values = pd.DataFrame(columns = ['Num_Evals', 'Num_Points', 'ChiSq', 'Red_ChiSq', 'AIC', 'BIC', 'K1', 'K1_err', 'k2', 'k2_err', 'vb', 'vb_err', 
                                            'K1_k2_corr', 'K1_vb_corr', 'k2_vb_corr'])
    
    if state == 'Rest':
        p = [0.47, 0.12, 0.48]
        tag = 'Heart_Rest'
        y_dat = rb82_data.Heart_Rest
        C1_data = rb82_data_int.Heart_Rest
        test_C1 = rb82_data_int.Heart_Rest_no_vb
    elif state == 'Stress':
        p = [1.08, 0.21, 0.50]
        tag = 'Heart_Stress'
        y_dat = rb82_data.Heart_Stress
        C1_data = rb82_data_int.Heart_Stress
        test_C1 = rb82_data_int.Heart_Stress_no_vb

    init = [0.0]

    params = lmfit.Parameters()
    params.add('K1', p[0], min=0.0, max=5.0)
    params.add('k2', p[1], min=0.0, max=5.0)
    params.add('vb', p[2], min=0.0, max=1.0)
    
    fit = lmfit.minimize(resid1_weighted, params, args = (C0, rb82_data, y_time, y_dat, rb82_frame_lengths_m, rb82_frametimes_mid_m, tracer), method = 'leastsq', max_nfev = 2500)
    lmfit.report_fit(fit)

    val1, val2, valvb = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value]
    
    x1, t1 = RK4(comp_ode_model1, rb82_C0, init, rb82_dt, rb82_T_f, rb82_T0, [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value])

    if fit.params['K1'].correl != None:
            correls = [fit.params['K1'].correl['k2'], fit.params['K1'].correl['vb'], fit.params['k2'].correl['vb']]
    else:
        correls = [0.0, 0.0, 0.0]
    
    stats = pd.DataFrame(columns = ['Num_Evals', 'Num_Points', 'ChiSq', 'Red_ChiSq', 'AIC', 'BIC', 'K1', 'K1_err', 'k2', 'k2_err', 'vb', 'vb_err', 
                        'K1_k2_corr', 'K1_vb_corr', 'k2_vb_corr'], data = [[fit.nfev, fit.ndata, fit.chisqr, fit.redchi, fit.aic, fit.bic, fit.params['K1'].value, 
                        fit.params['K1'].stderr, fit.params['k2'].value, fit.params['k2'].stderr, fit.params['vb'].value, fit.params['vb'].stderr, correls[0], 
                        correls[1], correls[2]]])
    
    rb82_param_values = pd.concat([rb82_param_values, stats])

    plt.scatter(rb82_data.Time, rb82_data.Blood, s = 7, label = 'Blood', color = 'g')
    plt.scatter(rb82_data.Time, y_dat, s = 7, label = 'Tissue', color = 'r')
    plt.plot(t1, ((1 - fit.params['vb'].value) * x1[:, 0] + fit.params['vb'].value * C0), label = 'Model Fit', color = 'b')
    plt.title(f'1TCM fit of \'blocked\' tissue curve ({tracer} {state}) \n K1 = {val1:.3f}, k2 = {val2:.3f}, vb = {valvb:.3f}')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Activity (Bq/cc)')
    plt.legend(loc = 7, fontsize = 'x-small')
    plt.savefig(f'{folder}/{sub_folder}/Rb82_{state}_vb_{vb_coeff}.png')
    plt.close()
    #plt.show()
    
    print(rb82_param_values)
    rb82_param_values.to_excel(f'{folder}/{sub_folder}/Rb82_{state}_vb_{vb_coeff}.xlsx')

elif mode == '1TCM_fit_noise_realisations':
    tau = 0.75
    y_time = rb82_data.Time
    y_0 = rb82_data.Blood
    C0 = rb82_C0
    #C0_disp = generate_C0_disp(rb82_data_int['Blood'], rb82_data_int, tau)

    rb82_param_values = pd.DataFrame(columns = ['Num_Evals', 'Num_Points', 'ChiSq', 'Red_ChiSq', 'AIC', 'BIC', 'K1', 'K1_err', 'k2', 'k2_err', 'vb', 'vb_err', 
                                            'K1_k2_corr', 'K1_vb_corr', 'k2_vb_corr'])
    
    if state == 'Rest':
        p = [0.47, 0.12, 0.48]
        tag = 'Heart_Rest'
    elif state == 'Stress':
        p = [1.08, 0.21, 0.50]
        tag = 'Heart_Stress'

    for i in range(gens):
        y_dat, a, tissue_noise = noise_gen(rb82_data, rb82_data_int, 'Rb82', state)    
        C1_data, C1_data_inter_func = interpolate(rb82_data.Time, y_dat, 'cubic', rb82_T0, rb82_T_f, rb82_dt)

        init = [0.0]

        params = lmfit.Parameters()
        params.add('K1', p[0], min=0.0, max=5.0)
        params.add('k2', p[1], min=0.0, max=5.0)
        params.add('vb', p[2], min=0.0, max=1.0)
        
        fit = lmfit.minimize(resid1_weighted, params, args = (C0, rb82_data, y_time, y_dat, rb82_frame_lengths_m, rb82_frametimes_mid_m, tracer), method = 'leastsq', max_nfev = 2500)
        lmfit.report_fit(fit)

        val1, val2, valvb = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value]
        
        x1, t1 = RK4(comp_ode_model1, rb82_C0, init, rb82_dt, rb82_T_f, rb82_T0, [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value])

        if fit.params['K1'].correl != None:
                correls = [fit.params['K1'].correl['k2'], fit.params['K1'].correl['vb'], fit.params['k2'].correl['vb']]
        else:
            correls = [0.0, 0.0, 0.0]
        
        stats = pd.DataFrame(columns = ['Num_Evals', 'Num_Points', 'ChiSq', 'Red_ChiSq', 'AIC', 'BIC', 'K1', 'K1_err', 'k2', 'k2_err', 'vb', 'vb_err', 
                            'K1_k2_corr', 'K1_vb_corr', 'k2_vb_corr'], data = [[fit.nfev, fit.ndata, fit.chisqr, fit.redchi, fit.aic, fit.bic, fit.params['K1'].value, 
                            fit.params['K1'].stderr, fit.params['k2'].value, fit.params['k2'].stderr, fit.params['vb'].value, fit.params['vb'].stderr, correls[0], 
                            correls[1], correls[2]]])
        
        rb82_param_values = pd.concat([rb82_param_values, stats])

        plt.scatter(rb82_data.Time, rb82_data.Blood, s = 7, label = 'Blood', color = 'g')
        plt.scatter(y_time, y_dat, s = 7, label = 'Tissue (Noisy)', color = 'r')
        plt.plot(t1, ((1 - fit.params['vb'].value) * x1[:, 0] + fit.params['vb'].value * C0), label = 'Model Fit', color = 'b')
        plt.title(f'1TCM fit of noisy \'blocked\' tissue curve ({tracer}, {state}) \n K1 = {val1:.3f}, k2 = {val2:.3f}, vb = {valvb:.3f}')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Activity (Bq/cc)')
        plt.legend(loc = 7, fontsize = 'x-small')
        plt.savefig(f'{folder}/{sub_folder}/Rb82_{state}_control_{i}.png')
        plt.close() 
        #plt.show()
    
    print(rb82_param_values)
    rb82_param_values.to_excel(f'{folder}/{sub_folder}/Rb82_{state}_control_{gens}_realisations.xlsx')

elif mode == 'Boxplots':
    boxplots_vb_coeff('Rb82/1TCM_fitting_blocked_tissue_noise_realisations/Rb82_Summary_100_realisations', 'Rb82/1TCM_fitting_blocked_tissue_noise_realisations/Boxplots', 'Rb82', '0.1')
    boxplots_vb_coeff('Rb82/1TCM_fitting_blocked_tissue_noise_realisations/Rb82_Summary_100_realisations', 'Rb82/1TCM_fitting_blocked_tissue_noise_realisations/Boxplots', 'Rb82', '0.5')
    boxplots_vb_coeff('Rb82/1TCM_fitting_blocked_tissue_noise_realisations/Rb82_Summary_100_realisations', 'Rb82/1TCM_fitting_blocked_tissue_noise_realisations/Boxplots', 'Rb82', '0.9')

elif mode == 'ANOVA':
    # f_stat, pvalue = f_oneway([243, 251, 275, 291, 347, 354, 380, 392], [206, 210, 226, 249, 255, 273, 285, 295, 309], [241, 258, 270, 293, 328])
    # print(f'F statistic of {f_stat}')
    # print(f'P-value of {pvalue}')

    ANOVA('Rb82/1TCM_fitting_blocked_tissue_noise_realisations/Rb82_Summary_100_realisations', 'Rb82/1TCM_fitting_blocked_tissue_noise_realisations/ANOVA', 'Rb82', 'Rest')
    ANOVA('Rb82/1TCM_fitting_blocked_tissue_noise_realisations/Rb82_Summary_100_realisations', 'Rb82/1TCM_fitting_blocked_tissue_noise_realisations/ANOVA', 'Rb82', 'Stress')



#### Calculate the flow values from the exported parameter tables

# Rb82
add_flow('Rb82/1TCM_fitting_blocked_tissue_noise_realisations/Rb82_Rest_vb_0.1_100_realisations', 'Rb82/1TCM_fitting_blocked_tissue_noise_realisations/With_Flow', 'Rb82', 'Rest', 0.1)
add_flow('Rb82/1TCM_fitting_blocked_tissue_noise_realisations/Rb82_Stress_vb_0.1_100_realisations', 'Rb82/1TCM_fitting_blocked_tissue_noise_realisations/With_Flow', 'Rb82', 'Stress', 0.1)
add_flow('Rb82/1TCM_fitting_blocked_tissue_noise_realisations/Rb82_Rest_vb_0.5_100_realisations', 'Rb82/1TCM_fitting_blocked_tissue_noise_realisations/With_Flow', 'Rb82', 'Rest', 0.5)
add_flow('Rb82/1TCM_fitting_blocked_tissue_noise_realisations/Rb82_Stress_vb_0.5_100_realisations', 'Rb82/1TCM_fitting_blocked_tissue_noise_realisations/With_Flow', 'Rb82', 'Stress', 0.5)
add_flow('Rb82/1TCM_fitting_blocked_tissue_noise_realisations/Rb82_Rest_vb_0.9_100_realisations', 'Rb82/1TCM_fitting_blocked_tissue_noise_realisations/With_Flow', 'Rb82', 'Rest', 0.9)
add_flow('Rb82/1TCM_fitting_blocked_tissue_noise_realisations/Rb82_Stress_vb_0.9_100_realisations', 'Rb82/1TCM_fitting_blocked_tissue_noise_realisations/With_Flow', 'Rb82', 'Stress', 0.9)

add_flow('Rb82/1TCM_fitting_control_tissue_noise_realisations/Rb82_Rest_control_100_realisations', 'Rb82/1TCM_fitting_control_tissue_noise_realisations/With_Flow', 'Rb82', 'Rest', 'control')
add_flow('Rb82/1TCM_fitting_control_tissue_noise_realisations/Rb82_Stress_control_100_realisations', 'Rb82/1TCM_fitting_control_tissue_noise_realisations/With_Flow', 'Rb82', 'Stress', 'control')




####

# tracer = 'Rb82'
# state = 'Rest'

# folder = 'Testing'
# sub_folder = 'Model_Fits'

# rb82_param_values = pd.DataFrame(columns = ['Num_Evals', 'Num_Points', 'ChiSq', 'Red_ChiSq', 'AIC', 'BIC', 'K1', 'K1_err', 'k2', 'k2_err', 'K1_d', 'K1_err', 'k2_d', 'k2_err', 'R', 'R_err', 
#                         'vb', 'vb_err', 'K1_k2_corr', 'K1_K1_d_corr', 'K1_k2_d_corr', 'K1_R_corr', 'K1_vb_corr', 
#                         'k2_K1_d_corr', 'k2_k2_d_corr', 'k2_R_corr', 'k2_vb_corr', 
#                         'K1_d_k2_d_corr', 'K1_d_R_corr', 'K1_d_vb_corr', 
#                         'k2_d_R_corr', 'k2_d_vb_corr', 
#                         'R_vb_corr'])

# tau_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
# tau_values = [0.50]

# for tau in tau_values:
#     y_time = rb82_data.Time
#     y_0 = rb82_data.Blood
#     C0 = rb82_C0
#     C0_disp = generate_C0_disp(rb82_data_int['Blood'], rb82_data_int, tau)

#     y_dat = rb82_data.Heart_Rest
#     C1_data = rb82_data_int.Heart_Rest

#     init = [0.0]

#     params = lmfit.Parameters()
#     params.add('K1', 0.5, min=0.0, max=5.0)
#     params.add('k2', 0.1, min=0.0, max=5.0)
#     params.add('K1_d', 0.5, min=0.0, max=5.0)
#     params.add('k2_d', 0.1, min=0.0, max=5.0)
#     params.add('R', 7, min = 1, max = 20)
#     params.add('vb', 0.3, vary = True, min=0.0, max=1.0)

#     fit = lmfit.minimize(resid_blockage, params, args = (C0, C0_disp, rb82_data, y_time, y_dat, rb82_frame_lengths_m, rb82_frametimes_mid_m, tracer), method = 'leastsq', max_nfev = 2500)
#     #lmfit.report_fit(fit)

#     if fit.params['K1'].correl != None:
#         correls = [fit.params['K1'].correl['k2'], fit.params['K1'].correl['K1_d'], fit.params['K1'].correl['k2_d'], fit.params['K1'].correl['R'], fit.params['K1'].correl['vb'], 
#                    fit.params['k2'].correl['K1_d'], fit.params['k2'].correl['k2_d'], fit.params['k2'].correl['R'], fit.params['k2'].correl['vb'], 
#                    fit.params['K1_d'].correl['k2_d'], fit.params['K1_d'].correl['R'], fit.params['K1_d'].correl['vb'],
#                    fit.params['k2_d'].correl['R'], fit.params['k2_d'].correl['vb'],
#                    fit.params['R'].correl['vb']]
#     else:
#         correls = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

#     stats = pd.DataFrame(columns = ['Num_Evals', 'Num_Points', 'ChiSq', 'Red_ChiSq', 'AIC', 'BIC', 'K1', 'K1_err', 'k2', 'k2_err', 'K1_d', 'K1_err', 'k2_d', 'k2_err', 'R', 'R_err', 
#                         'vb', 'vb_err', 'K1_k2_corr', 'K1_K1_d_corr', 'K1_k2_d_corr', 'K1_R_corr', 'K1_vb_corr', 
#                         'k2_K1_d_corr', 'k2_k2_d_corr', 'k2_R_corr', 'k2_vb_corr', 
#                         'K1_d_k2_d_corr', 'K1_d_R_corr', 'K1_d_vb_corr', 
#                         'k2_d_R_corr', 'k2_d_vb_corr', 
#                         'R_vb_corr'], 
#                         data = [[fit.nfev, fit.ndata, fit.chisqr, fit.redchi, fit.aic, fit.bic, fit.params['K1'].value, fit.params['K1'].stderr, fit.params['k2'].value, fit.params['k2'].stderr, 
#                                  fit.params['K1_d'].value, fit.params['K1_d'].stderr, fit.params['k2_d'].value, fit.params['k2_d'].stderr, fit.params['R'].value, fit.params['R'].stderr, 
#                                  fit.params['vb'].value, fit.params['vb'].stderr, correls[0], correls[1], correls[2], correls[3], correls[4], correls[5], correls[6], correls[7], correls[8], 
#                                  correls[9], correls[10], correls[11], correls[12], correls[13], correls[14]]])

#     rb82_param_values = pd.concat([rb82_param_values, stats])

#     x1, t1 = RK4_disp(comp_ode_model_blockage, rb82_C0, rb82_C0_disp, init2, rb82_dt, rb82_T_f, rb82_T0, [fit.params['K1'].value, fit.params['k2'].value, fit.params['K1_d'].value, fit.params['k2_d'].value, fit.params['R'].value, fit.params['vb'].value])
#     val1, val2, val3, val4, valR, valvb = [fit.params['K1'].value, fit.params['k2'].value, fit.params['K1_d'].value, fit.params['k2_d'].value, fit.params['R'].value, fit.params['vb'].value]

#     plt.scatter(rb82_data.Time, rb82_data.Blood, s = 7, label = 'Blood', color = 'g')
#     plt.scatter(y_time, y_dat, s = 7, label = 'Tissue', color = 'r')
#     plt.plot(t1, ((1 - fit.params['vb'].value) * x1[:-1, 0] + fit.params['vb'].value * C0), label = 'Model Fit', color = 'b')
#     plt.title(f'Tissue Curve with Noise ({tracer} {state}) \n K1 = {val1:.3f}, k2 = {val2:.3f}, K1_d = {val3:.3f} \n k2_d = {val4:.3f}, R = {valR:.3f}, vb = {valvb:.3f}')
#     plt.xlabel('Time (minutes)')
#     plt.ylabel('Activity (Bq/cc)')
#     plt.legend(loc = 7, fontsize = 'x-small')
#     plt.savefig(f'{folder}\{sub_folder}\Rb82_{state}_Tau{tau}')
#     plt.close()
#     #plt.show()

# rb82_param_values.to_excel(f'{folder}\{sub_folder}\Rb82_param_values_{state}_Tau{tau}_{date}.xlsx')