import numpy as np
import math
import lmfit
import pandas as pd
#import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
import random
import csv
from datetime import datetime
from scipy.optimize import fsolve
from scipy.stats import f_oneway

import matplotlib
matplotlib.use('TkAgg',force=True)
from matplotlib import pyplot as plt
print("Switched to:",matplotlib.get_backend())



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

    ind = int(round((t - T0)/dt))# -1)

    # dC_1 / dt 
    du = K_1 * C0[ind] - k_2 * u[0] 

    # u[0] = C_1

    return du

def comp_ode_model1_deg(u, C0_deg, t, T0_deg, p):
    K_1, k_2, vb = p

    du = np.zeros(1)

    # test whether any concentrations are negative
    # if len(u[u < -1E-12]) > 0:
    #     print("negative u value!")

    ind = int(round((t - T0_deg)/dt_deg))

    # if ind == int(round((T_f - T0)/dt)):
    #     return None

    # dC_1 / dt 
    du = K_1 * C0_deg[ind] - k_2 * u[0] 

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
        scale_factor = 0.0006
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
    weights = np.sqrt(weights)
    weights[np.isnan(weights)] = 0.01

    return (weights * resids)
    #return resids

def resid1_deg_weighted(params, C0_deg, data, y_time_deg, y_dat_deg, y_dat, frame_lengths_m, framemidtimes_m, tracer):
    K_1 = params['K1'].value
    k_2 = params['k2'].value
    vb = params['vb'].value
    
    p = [K_1, k_2, vb]

    dt = 1/600
    T_f_deg, T0_deg, dt_deg, time = time_vars(data, dt)
    #C0_new = shift(C0, delay)
    ''' Delay stuff would go here in the order of things'''

    u_out, t = RK4(comp_ode_model1_deg, C0_deg, init, dt_deg, T_f_deg, T0_deg, p)
    
    annoying = np.array(C0_deg, dtype=float)
    model = (1 - vb) * u_out[:, 0] + vb * annoying[:-1]


    func = interp1d(np.array(t, dtype=float), np.array(model, dtype=float), kind='cubic', fill_value = 'extrapolate')
    model = func(np.array(y_time_deg, dtype=float))     # This is the model fit refitted into the original 33 time points

    result = integrateTrapezium(frame_lengths_m, y_dat)     # This is the approximate decay corrected PET data (TAC) to be used in the weighting calculating
    
    resids = model - np.array(y_dat_deg, dtype=float)       # This is the plain residuals to be returned from the function after being multiplied by the weights, final five values are to replace any zero values in result

    #scale_factor = 0.71
    if tracer == 'Rb82':
        decay_const = 0.55121048155860461981489631925104
        scale_factor = 0.002
    elif tracer == 'NH3':
        decay_const = 0.0695232879197537923186792498955
        scale_factor = 0.0006
    elif tracer == 'Water':
        decay_const = 0.34015041658021623807954727632776
        scale_factor = 0.5
                   # minutes
    #dec_const = math.log(2) / 109.771           # minutes 
    frame_dur = np.zeros(len(frame_lengths_m))
    exp_decay = np.zeros(len(frame_lengths_m))

    # weights = 1 / np.sqrt(result)
    # weights_mean = np.mean(weights)

    # weights[np.isnan(weights)] = 1.0
    # weights[np.isinf(weights)] = 1.0
    # print(weights)

    for i in range(len(frame_lengths_m)):
        frame_dur[i] = frame_lengths_m[i]
        exp_decay[i] = math.exp(- decay_const * framemidtimes_m[i])

        if result[i] == 0:
            result[i] = np.mean(resids[-3:])        # Maybe replace this value with an average of the last 5 residuals, 3 for degrado

        # if weights[i] == np.NaN or weights[i] == np.inf:
        #     weights[i] == weights_mean

    sigma_sq = scale_factor * (result / (frame_dur * exp_decay))
    #sigma_sq = 0.05 * (result / (frame_dur * exp_decay))            # Changed scale factor to 0.05 for comparison with results from PMOD using 0.05 as the scale factor
    weights = 1 / sigma_sq
    weights = np.sqrt(weights)
    weights[np.isnan(weights)] = 0.01
    #print(weights)

    return (weights[:21] * resids)
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
        scale_factor = 0.0006
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
    weights = np.sqrt(weights)
    weights[np.isnan(weights)] = 0.01

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

def expo(x, A, lambd, line, t_peak):
    p1, p3, p5 = A        
    p2, p4, p6 = lambd
    a, b = line

    result = np.array([])
    j = 0
    for i in x:
        if i < -(b/a):
            result = np.append(result, [0])
        elif i >= -(b/a) and i < t_peak:
            result = np.append(result, [a * i + b]) 
        elif i >= t_peak:
            result = np.append(result, [p1*np.exp(-p2*(i - t_peak)) + p3*np.exp(-p4*(i - t_peak)) + p5*np.exp(-p6*(i - t_peak))]) 
        print(j)
        j+=1
    return result

def expo_resid(params_exp, t_peak):
    coeff = [params_exp['p1'].value, params_exp['p3'].value, params_exp['p5'].value]
    d_const = [params_exp['p2'].value, params_exp['p4'].value, params_exp['p6'].value]
    line = [params_exp['a'].value, params_exp['b'].value]

    model = expo(mouse.Time, coeff, d_const, line, t_peak)
    #print(model)
    print(np.isnan(model - y_0).any())
    print(np.isinf(model - y_0).any())

    return model - y_0

def expo3(x, A, lambd, line, t_peak):
    p1, p3, p5 = A        
    p2, p4, p6 = lambd
    a, b = line

    result = np.array([])
   
    for i in x:
        result = np.append(result, [p1*np.exp(-p2*(i - t_peak)) + p3*np.exp(-p4*(i - t_peak)) + p5*np.exp(-p6*(i - t_peak))])
    
    return result

def expo3_resid(params_exp, t_peak):
    coeff = [params_exp['p1'].value, params_exp['p3'].value, params_exp['p5'].value]
    d_const = [params_exp['p2'].value, params_exp['p4'].value, params_exp['p6'].value]
    line = [params_exp['a'].value, params_exp['b'].value]

    model = expo3(mouse.Time, coeff, d_const, line, t_peak)
    #print(model)
    print(np.isnan(model - y_0).any())
    print(np.isinf(model - y_0).any())

    return model - y_0

def blood_curve(sheet):
    data = pd.read_excel('NH3_Blood_Fit_Params.xlsx', sheet_name=sheet, engine = 'openpyxl', dtype = {'AA01_Rest' : float, 'AA01_Stress' : float, 'AA02_Rest' : float, 
                        'AA02_Stress' : float, 'AA03_Rest' : float, 'AA03_Stress' : float, 'AA04_Rest' : float, 'AA05_Rest' : float, 'AA05_Stress' : float})
    data.columns = ['Parameter', 'AA01_Rest', 'AA01_Stress', 'AA02_Rest', 'AA02_Stress', 'AA03_Rest', 'AA03_Stress', 'AA04_Rest', 'AA05_Rest', 'AA05_Stress']
    #data = data.drop('index', axis=1)
    data = data.set_index('Parameter')

    return data

def blood_curve_fitting(mouse, m, p):
    params_exp = lmfit.Parameters()
    params_exp.add('p1', p[0])               # p1 = 6.677
    params_exp.add('p2', p[1])               # p2 = 9.066
    params_exp.add('p3', p[2], vary = True)               # p3 = 0.727
    params_exp.add('p4', p[3], vary = True)               # p4 = 0.043
    params_exp.add('p5', p[4], vary = True)               # p5 = 1.880
    params_exp.add('p6', p[5], vary = True)               # p6 = 0.530
    params_exp.add('a', p[6], min = 0)       # a = 8.761
    params_exp.add('b', p[7])              # b = -0.00349
    # p = [10, 10, 1, 1, 1, 1, 10, 1]

    #t_peak = mouse_int.loc[mouse_int['Vena_Cava'].idxmax()]['Time']
    t_peak = mouse.loc[mouse['Vena_Cava'].idxmax()]['Time']
    # print(t_peak)
    # print(mouse)

    #bolus_fit = lmfit.minimize(expo3_resid, params_exp, args = ([t_peak]), method = 'leastsq', max_nfev = 1000)
    #lmfit.report_fit(bolus_fit)

    bolus_fitted = expo3(mouse.Time, [params_exp['p1'].value, params_exp['p3'].value, params_exp['p5'].value],
                                            [params_exp['p2'].value, params_exp['p4'].value, params_exp['p6'].value],
                                            [params_exp['a'].value, params_exp['b'].value], t_peak)
    
    # with open(f'{folder}\Blood_Curves\Blood_Curve_Fits\{m}_blood_curve_fit.csv', 'w', newline = '') as f:
    #     writer = csv.writer(f)
    #     writer.writerows([[m]])
    #     writer.writerow([' '])
    #     writer.writerows([[bolus_fit.params['p1'].value, bolus_fit.params['p3'].value, bolus_fit.params['p5'].value],
    #                                     [bolus_fit.params['p2'].value, bolus_fit.params['p4'].value, bolus_fit.params['p6'].value],
    #                                     [bolus_fit.params['a'].value, bolus_fit.params['b'].value]])
    #     writer.writerow([' '])
    #     writer.writerows([bolus_fitted])
            
    plt.scatter(mouse.Time, mouse.Vena_Cava, s = 7, color='r', label='Original')
    plt.plot(mouse.Time, bolus_fitted, color='b', label='Fitted')
    plt.title(f'Comparison of original blood curve to \n fitted blood curve for Mouse {m}')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Activity Concentration (kBq/cc)')
    plt.legend(loc = 7, fontsize = 'x-small')
    plt.savefig(f'{folder}\Blood_Curves\Blood_Curve_Fits\{m}_blood_curve_fit')
    plt.close()

def sensitivity_analysis(fit, param, delta_param, model):
    h_scale = delta_param

    if model == 'Degrado':
        if param == 'K1':
            K_1 = fit.params['K1'].value
            k_2 = fit.params['k2'].value
            vb = fit.params['vb'].value

            h = h_scale * K_1
            
            x1, t1 = RK4(comp_ode_model1_deg, C0_deg, init, dt_deg, T_f_deg, T0_deg, [K_1 + h, k_2, vb])
            x2, t2 = RK4(comp_ode_model1_deg, C0_deg, init, dt_deg, T_f_deg, T0_deg, [K_1 - h, k_2, vb])

            x1 = (1 - vb) * x1[:, 0] + vb * C0_deg[:-1]
            x2 = (1 - vb) * x2[:, 0] + vb * C0_deg[:-1]

            deriv = (x1 - x2)/2*h

        elif param == 'k2':
            K_1 = fit.params['K1'].value
            k_2 = fit.params['k2'].value
            vb = fit.params['vb'].value

            h = h_scale * k_2

            x1, t1 = RK4(comp_ode_model1_deg, C0_deg, init, dt_deg, T_f_deg, T0_deg, [K_1, k_2 + h, vb])
            x2, t2 = RK4(comp_ode_model1_deg, C0_deg, init, dt_deg, T_f_deg, T0_deg, [K_1, k_2 - h, vb])

            x1 = (1 - vb) * x1[:, 0] + vb * C0_deg[:-1]
            x2 = (1 - vb) * x2[:, 0] + vb * C0_deg[:-1]

            deriv = (x1 - x2)/2*h

        elif param == 'vb':
            K_1 = fit.params['K1'].value
            k_2 = fit.params['k2'].value
            vb = fit.params['vb'].value

            h = h_scale * vb

            x1, t1 = RK4(comp_ode_model1_deg, C0_deg, init, dt_deg, T_f_deg, T0_deg, [K_1, k_2, vb])

            x2 = (1 - (vb + h)) * x1[:, 0] + (vb + h) * C0_deg[:-1]
            x3 = (1 - (vb - h)) * x1[:, 0] + (vb - h) * C0_deg[:-1]

            deriv = (x2 - x3)/2*h

        elif param == 'F':
            K_1 = fit.params['K1'].value
            k_2 = fit.params['k2'].value
            vb = fit.params['vb'].value

            a = ps_df.loc[organ, 'a']
            b = ps_df.loc[organ, 'b']
            f_values = np.array([])

            for i in np.linspace(-9, 9, 19):
                f_root = fsolve(flow_func, [K_1 + (i/10)*K_1], args = (K_1, a, b))
                f = f_root[0]
                f_values = np.append(f_values, f)
            f = find_nearest(f_values, K_1)

            h = h_scale * f

            x1, t1 = RK4(comp_ode_model1_deg, C0_deg, init, dt_deg, T_f_deg, T0_deg, [k1_flow(f + h, a, b), k_2, vb])
            x2, t2 = RK4(comp_ode_model1_deg, C0_deg, init, dt_deg, T_f_deg, T0_deg, [k1_flow(f - h, a, b), k_2, vb])

            x1 = (1 - vb) * x1[:, 0] + vb * C0_deg[:-1]
            x2 = (1 - vb) * x2[:, 0] + vb * C0_deg[:-1]

            deriv = (x1 - x2)/2*h

            return deriv, t1, f, f_root

    elif model == '1TCM':
        if param == 'K1':
            K_1 = fit.params['K1'].value
            k_2 = fit.params['k2'].value
            vb = fit.params['vb'].value

            h = h_scale * K_1
            
            x1, t1 = RK4(comp_ode_model1, C0_orig, init, dt, T_f, T0, [K_1 + h, k_2, vb])
            x2, t2 = RK4(comp_ode_model1, C0_orig, init, dt, T_f, T0, [K_1 - h, k_2, vb])

            x1 = (1 - vb) * x1[:, 0] + vb * C0_orig
            x2 = (1 - vb) * x2[:, 0] + vb * C0_orig

            deriv = (x1 - x2)/2*h

        elif param == 'k2':
            K_1 = fit.params['K1'].value
            k_2 = fit.params['k2'].value
            vb = fit.params['vb'].value

            h = h_scale * k_2

            x1, t1 = RK4(comp_ode_model1, C0_orig, init, dt, T_f, T0, [K_1, k_2 + h, vb])
            x2, t2 = RK4(comp_ode_model1, C0_orig, init, dt, T_f, T0, [K_1, k_2 - h, vb])

            x1 = (1 - vb) * x1[:, 0] + vb * C0_orig
            x2 = (1 - vb) * x2[:, 0] + vb * C0_orig

            deriv = (x1 - x2)/2*h

        elif param == 'vb':
            K_1 = fit.params['K1'].value
            k_2 = fit.params['k2'].value
            vb = fit.params['vb'].value

            h = h_scale * vb

            x1, t1 = RK4(comp_ode_model1, C0_orig, init, dt, T_f, T0, [K_1, k_2, vb])

            x2 = (1 - (vb + h)) * x1[:, 0] + (vb + h) * C0_orig
            x3 = (1 - (vb - h)) * x1[:, 0] + (vb - h) * C0_orig

            deriv = (x2 - x3)/2*h

        elif param == 'F':
            K_1 = fit.params['K1'].value
            k_2 = fit.params['k2'].value
            vb = fit.params['vb'].value

            a = ps_df.loc[organ, 'a']
            b = ps_df.loc[organ, 'b']
            f_values = np.array([])

            for i in np.linspace(-9, 9, 19):
                f_root = fsolve(flow_func, [K_1 + (i/10)*K_1], args = (K_1, a, b))
                f = f_root[0]
                f_values = np.append(f_values, f)
            f = find_nearest(f_values, K_1)

            h = h_scale * f

            x1, t1 = RK4(comp_ode_model1, C0_orig, init, dt, T_f, T0, [k1_flow(f + h, a, b), k_2, vb])
            x2, t2 = RK4(comp_ode_model1, C0_orig, init, dt, T_f, T0, [k1_flow(f - h, a, b), k_2, vb])

            x1 = (1 - vb) * x1[:, 0] + vb * C0_orig
            x2 = (1 - vb) * x2[:, 0] + vb * C0_orig

            deriv = (x1 - x2)/2*h

            return deriv, t1, f, f_root

    elif model == '2TCM':
        if param == 'K1':
            K_1 = fit.params['K1'].value
            k_2 = fit.params['k2'].value
            k_3 = fit.params['k3'].value
            k_4 = fit.params['k4'].value
            vb = fit.params['vb'].value

            h = h_scale * K_1

            x1, t1 = RK4(comp_ode_model2, C0_orig, init2, dt, T_f, T0, [K_1 + h, k_2, k_3, k_4, vb])
            x2, t2 = RK4(comp_ode_model2, C0_orig, init2, dt, T_f, T0, [K_1 - h, k_2, k_3, k_4, vb])

            x1 = (1 - vb) * (x1[:, 0] + x1[:, 1]) + vb * C0_orig
            x2 = (1 - vb) * (x2[:, 0] + x2[:, 1]) + vb * C0_orig

            deriv = (x1 - x2)/2*h

        elif param == 'k2':
            K_1 = fit.params['K1'].value
            k_2 = fit.params['k2'].value
            k_3 = fit.params['k3'].value
            k_4 = fit.params['k4'].value
            vb = fit.params['vb'].value

            h = h_scale * k_2

            x1, t1 = RK4(comp_ode_model2, C0_orig, init2, dt, T_f, T0, [K_1, k_2 + h, k_3, k_4, vb])
            x2, t2 = RK4(comp_ode_model2, C0_orig, init2, dt, T_f, T0, [K_1, k_2 - h, k_3, k_4, vb])

            x1 = (1 - vb) * (x1[:, 0] + x1[:, 1]) + vb * C0_orig
            x2 = (1 - vb) * (x2[:, 0] + x2[:, 1]) + vb * C0_orig

            deriv = (x1 - x2)/2*h

        elif param == 'k3':
            K_1 = fit.params['K1'].value
            k_2 = fit.params['k2'].value
            k_3 = fit.params['k3'].value
            k_4 = fit.params['k4'].value
            vb = fit.params['vb'].value

            h = h_scale * k_3

            x1, t1 = RK4(comp_ode_model2, C0_orig, init2, dt, T_f, T0, [K_1, k_2, k_3 + h, k_4, vb])
            x2, t2 = RK4(comp_ode_model2, C0_orig, init2, dt, T_f, T0, [K_1, k_2, k_3 - h, k_4, vb])

            x1 = (1 - vb) * (x1[:, 0] + x1[:, 1]) + vb * C0_orig
            x2 = (1 - vb) * (x2[:, 0] + x2[:, 1]) + vb * C0_orig

            deriv = (x1 - x2)/2*h

        elif param == 'k4':
            K_1 = fit.params['K1'].value
            k_2 = fit.params['k2'].value
            k_3 = fit.params['k3'].value
            k_4 = fit.params['k4'].value
            vb = fit.params['vb'].value

            h = h_scale * k_4

            x1, t1 = RK4(comp_ode_model2, C0_orig, init2, dt, T_f, T0, [K_1, k_2, k_3, k_4 + h, vb])
            x2, t2 = RK4(comp_ode_model2, C0_orig, init2, dt, T_f, T0, [K_1, k_2, k_3, k_4 - h, vb])

            x1 = (1 - vb) * (x1[:, 0] + x1[:, 1]) + vb * C0_orig
            x2 = (1 - vb) * (x2[:, 0] + x2[:, 1]) + vb * C0_orig

            deriv = (x1 - x2)/2*h

        elif param == 'vb':
            K_1 = fit.params['K1'].value
            k_2 = fit.params['k2'].value
            k_3 = fit.params['k3'].value
            k_4 = fit.params['k4'].value
            vb = fit.params['vb'].value

            h = h_scale * vb

            x1, t1 = RK4(comp_ode_model2, C0_orig, init2, dt, T_f, T0, [K_1, k_2, k_3, k_4, vb])

            x2 = (1 - (vb + h)) * (x1[:, 0] + x1[:, 1]) + (vb + h) * C0_orig
            x3 = (1 - (vb - h)) * (x1[:, 0] + x1[:, 1]) + (vb - h) * C0_orig

            deriv = (x2 - x3)/2*h

        elif param == 'F':
            K_1 = fit.params['K1'].value
            k_2 = fit.params['k2'].value
            k_3 = fit.params['k3'].value
            k_4 = fit.params['k4'].value
            vb = fit.params['vb'].value

            a = ps_df.loc[organ, 'a']
            b = ps_df.loc[organ, 'b']
            f_values = np.array([])


            for i in np.linspace(-9, 9, 19):
                f_root = fsolve(flow_func, [K_1 + (i/10)*K_1], args = (K_1, a, b))
                f = f_root[0]
                f_values = np.append(f_values, f)
            f = find_nearest(f_values, K_1)

            h = h_scale * f

            x1, t1 = RK4(comp_ode_model2, C0_orig, init2, dt, T_f, T0, [k1_flow(f + h, a, b), k_2, k_3, k_4, vb])
            x2, t2 = RK4(comp_ode_model2, C0_orig, init2, dt, T_f, T0, [k1_flow(f - h, a, b), k_2, k_3, k_4, vb])

            x1 = (1 - vb) * x1[:, 0] + vb * C0_orig
            x2 = (1 - vb) * x2[:, 0] + vb * C0_orig

            deriv = (x1 - x2)/2*h

            return deriv, t1, f, f_root
    
    return deriv, t1

def flow_func(F, K1, a, b):
    return  F * (1 - a * np.exp(-b/F)) - K1

def k1_flow(F, a, b):
    k1 = F * (1 - a * np.exp(-b/F))
    
    return k1

def ext_frac(F, a, b):
    return (1 - a * np.exp(-b/F))
            
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    
    return array[idx]

def sensitivity_analysis_display_ch4(folder, sub_folder, data, params, error, data_AA05, params_AA05, error_AA05, organ, model, state):
    if state == 'Rest':
        if model == '2TCM':
            fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15), (ax16, ax17, ax18, ax19, ax20), (ax21, ax22, ax23, ax24, ax25)) = plt.subplots(5, 5, figsize = (15,15), constrained_layout = False, tight_layout = True)
            fig.suptitle(f'Sensitivity analysis of all parameters for the {organ} at {state}')

            # First row
            ax1.plot(data[0][0], data[0][1], label = 'K1', color = 'b')
            ax1.set_title(f'Mouse AA01 \n K1 = {params[0, 0]:.3f}')
            ax1.set_ylabel('dPET/dK1')

            ax2.plot(data[1][0], data[1][1], label = 'K1', color = 'b')
            ax2.set_title(f'Mouse AA02 \n K1 = {params[0, 1]:.3f}')

            ax3.plot(data[2][0], data[2][1], label = 'K1', color = 'b')
            ax3.set_title(f'Mouse AA03 \n K1 = {params[0, 2]:.3f}')

            ax4.plot(data[3][0], data[3][1], label = 'K1', color = 'b')
            ax4.set_title(f'Mouse AA04 \n K1 = {params[0, 3]:.3f}')

            ax5.plot(data_AA05[0][0], data_AA05[0][1], label = 'K1', color = 'b')
            ax5.set_title(f'Mouse AA05 \n K1 = {params_AA05[0, 0]:.3f}')

            # Second row
            ax6.plot(data[0][0], data[0][2], label = 'k2', color = 'g')
            ax6.set_title(f'k2 = {params[1, 0]:.3f}')
            ax6.set_ylabel('dPET/dk2')

            ax7.plot(data[1][0], data[1][2], label = 'k2', color = 'g')
            ax7.set_title(f'k2 = {params[1, 1]:.3f}')

            ax8.plot(data[2][0], data[2][2], label = 'k2', color = 'g')
            ax8.set_title(f'k2 = {params[1, 2]:.3f}')

            ax9.plot(data[3][0], data[3][2], label = 'k2', color = 'g')
            ax9.set_title(f'k2 = {params[1, 3]:.3f}')

            ax10.plot(data_AA05[0][0], data_AA05[0][2], label = 'k2', color = 'g')
            ax10.set_title(f'k2 = {params_AA05[1, 0]:.3f}')

            # Third row
            ax11.plot(data[0][0], data[0][5], label = 'k3', color = 'y')
            ax11.set_title(f'k3 = {params[2, 0]:.3f}')
            ax11.set_ylabel('dPET/dk3')

            ax12.plot(data[1][0], data[1][5], label = 'k3', color = 'y')
            ax12.set_title(f'k3 = {params[2, 1]:.3f}')

            ax13.plot(data[2][0], data[2][5], label = 'k3', color = 'y')
            ax13.set_title(f'k3 = {params[2, 2]:.3f}')

            ax14.plot(data[3][0], data[3][5], label = 'k3', color = 'y')
            ax14.set_title(f'k3 = {params[2, 3]:.3f}')

            ax15.plot(data_AA05[0][0], data_AA05[0][5], label = 'k3', color = 'y')
            ax15.set_title(f'k3 = {params_AA05[2, 0]:.3f}')

            # Fourth row
            ax16.plot(data[0][0], data[0][3], label = 'vb', color = 'r')
            ax16.set_title(f'vb = {params[3, 0]:.3f}')
            ax16.set_ylabel('dPET/dvb')

            ax17.plot(data[1][0], data[1][3], label = 'vb', color = 'r')
            ax17.set_title(f'vb = {params[3, 1]:.3f}')

            ax18.plot(data[2][0], data[2][3], label = 'vb', color = 'r')
            ax18.set_title(f'vb = {params[3, 2]:.3f}')

            ax19.plot(data[3][0], data[3][3], label = 'vb', color = 'r')
            ax19.set_title(f'vb = {params[3, 3]:.3f}')

            ax20.plot(data_AA05[0][0], data_AA05[0][3], label = 'vb', color = 'r')
            ax20.set_title(f'vb = {params_AA05[3, 0]:.3f}')

            # Fifth row
            ax21.plot(data[0][0], data[0][4], label = 'F', color = 'm')
            ax21.set_title(f'F = {params[4, 0]:.3f}')
            ax21.set_ylabel('dPET/dF')

            ax22.plot(data[1][0], data[1][4], label = 'F', color = 'm')
            ax22.set_title(f'F = {params[4, 1]:.3f}')

            ax23.plot(data[2][0], data[2][4], label = 'F', color = 'm')
            ax23.set_title(f'F = {params[4, 2]:.3f}')

            ax24.plot(data[3][0], data[3][4], label = 'F', color = 'm')
            ax24.set_title(f'F = {params[4, 3]:.3f}')

            ax25.plot(data_AA05[0][0], data_AA05[0][4], label = 'F', color = 'm')
            ax25.set_title(f'F = {params_AA05[4, 0]:.3f}')

        else:
            fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15), (ax16, ax17, ax18, ax19, ax20)) = plt.subplots(4, 5, figsize = (15,12), constrained_layout = False, tight_layout = True)
            fig.suptitle(f'Sensitivity analysis of all parameters for {organ} at {state}')

            # First row
            ax1.plot(data[0][0], data[0][1], label = 'K1', color = 'b')
            ax1.set_title(f'Mouse AA01 \n K1 = {params[0, 0]:.3f}')
            ax1.set_ylabel('dPET/dK1')

            ax2.plot(data[1][0], data[1][1], label = 'K1', color = 'b')
            ax2.set_title(f'Mouse AA02 \n K1 = {params[0, 1]:.3f}')

            ax3.plot(data[2][0], data[2][1], label = 'K1', color = 'b')
            ax3.set_title(f'Mouse AA03 \n K1 = {params[0, 2]:.3f}')

            ax4.plot(data[3][0], data[3][1], label = 'K1', color = 'b')
            ax4.set_title(f'Mouse AA04 \n K1 = {params[0, 3]:.3f}')

            ax5.plot(data_AA05[0][0], data_AA05[0][1], label = 'K1', color = 'b')
            ax5.set_title(f'Mouse AA05 \n K1 = {params_AA05[0, 0]:.3f}')

            # Second row
            ax6.plot(data[0][0], data[0][2], label = 'k2', color = 'g')
            ax6.set_title(f'k2 = {params[1, 0]:.3f}')
            ax6.set_ylabel('dPET/dk2')

            ax7.plot(data[1][0], data[1][2], label = 'k2', color = 'g')
            ax7.set_title(f'k2 = {params[1, 1]:.3f}')

            ax8.plot(data[2][0], data[2][2], label = 'k2', color = 'g')
            ax8.set_title(f'k2 = {params[1, 2]:.3f}')

            ax9.plot(data[3][0], data[3][2], label = 'k2', color = 'g')
            ax9.set_title(f'k2 = {params[1, 3]:.3f}')

            ax10.plot(data_AA05[0][0], data_AA05[0][2], label = 'k2', color = 'g')
            ax10.set_title(f'k2 = {params_AA05[1, 0]:.3f}')

            # Third row
            ax11.plot(data[0][0], data[0][3], label = 'vb', color = 'r')
            ax11.set_title(f'vb = {params[2, 0]:.3f}')
            ax11.set_ylabel('dPET/dvb')

            ax12.plot(data[1][0], data[1][3], label = 'vb', color = 'r')
            ax12.set_title(f'vb = {params[2, 1]:.3f}')

            ax13.plot(data[2][0], data[2][3], label = 'vb', color = 'r')
            ax13.set_title(f'vb = {params[2, 2]:.3f}')

            ax14.plot(data[3][0], data[3][3], label = 'vb', color = 'r')
            ax14.set_title(f'vb = {params[2, 3]:.3f}')

            ax15.plot(data_AA05[0][0], data_AA05[0][3], label = 'vb', color = 'r')
            ax15.set_title(f'vb = {params_AA05[2, 0]:.3f}')

            # Fourth row
            ax16.plot(data[0][0], data[0][4], label = 'F', color = 'm')
            ax16.set_title(f'F = {params[3, 0]:.3f}')
            ax16.set_ylabel('dPET/dF')

            ax17.plot(data[1][0], data[1][4], label = 'F', color = 'm')
            ax17.set_title(f'F = {params[3, 1]:.3f}')

            ax18.plot(data[2][0], data[2][4], label = 'F', color = 'm')
            ax18.set_title(f'F = {params[3, 2]:.3f}')

            ax19.plot(data[3][0], data[3][4], label = 'F', color = 'm')
            ax19.set_title(f'F = {params[3, 3]:.3f}')

            ax20.plot(data_AA05[0][0], data_AA05[0][4], label = 'F', color = 'm')
            ax20.set_title(f'F = {params_AA05[3, 0]:.3f}')
            
        plt.savefig(f'{folder}\{sub_folder}/SA/{organ}_{state}_SA')
        plt.close()

        with open(f'{folder}\{sub_folder}/SA/{organ}_{state}params.csv', 'w', newline = '') as f:
            writer = csv.writer(f)
            writer.writerows([['Mouse AA01', 'Mouse AA02', 'Mouse AA03', 'Mouse AA04']])
            writer.writerows(params)
            writer.writerows([' '])
            writer.writerows(error)
            writer.writerows([' '])
            writer.writerows([['Mouse AA05']])
            writer.writerows(params_AA05)
            writer.writerows([' '])
            writer.writerows(error_AA05)

    elif state == 'Stress':
        if model == '2TCM':
            fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12), (ax13, ax14, ax15, ax16), (ax17, ax18, ax19, ax20)) = plt.subplots(5, 4, figsize = (12,15), constrained_layout = False, tight_layout = True)
            fig.suptitle(f'Sensitivity analysis of all parameters for the {organ} at {state}')

            # First row
            ax1.plot(data[0][0], data[0][1], label = 'K1', color = 'b')
            ax1.set_title(f'Mouse AA01 \n K1 = {params[0, 0]:.3f}')
            ax1.set_ylabel('dPET/dK1')

            ax2.plot(data[1][0], data[1][1], label = 'K1', color = 'b')
            ax2.set_title(f'Mouse AA02 \n K1 = {params[0, 1]:.3f}')

            ax3.plot(data[2][0], data[2][1], label = 'K1', color = 'b')
            ax3.set_title(f'Mouse AA03 \n K1 = {params[0, 2]:.3f}')

            ax4.plot(data_AA05[0][0], data_AA05[0][1], label = 'K1', color = 'b')
            ax4.set_title(f'Mouse AA05 \n K1 = {params_AA05[0, 0]:.3f}')

            # Second row
            ax5.plot(data[0][0], data[0][2], label = 'k2', color = 'g')
            ax5.set_title(f'k2 = {params[1, 0]:.3f}')
            ax5.set_ylabel('dPET/dk2')

            ax6.plot(data[1][0], data[1][2], label = 'k2', color = 'g')
            ax6.set_title(f'k2 = {params[1, 1]:.3f}')

            ax7.plot(data[2][0], data[2][2], label = 'k2', color = 'g')
            ax7.set_title(f'k2 = {params[1, 2]:.3f}')

            ax8.plot(data_AA05[0][0], data_AA05[0][2], label = 'k2', color = 'g')
            ax8.set_title(f'k2 = {params_AA05[1, 0]:.3f}')

            # Third row
            ax9.plot(data[0][0], data[0][5], label = 'k3', color = 'y')
            ax9.set_title(f'k3 = {params[2, 0]:.3f}')
            ax9.set_ylabel('dPET/dk3')

            ax10.plot(data[1][0], data[1][5], label = 'k3', color = 'y')
            ax10.set_title(f'k3 = {params[2, 1]:.3f}')

            ax11.plot(data[2][0], data[2][5], label = 'k3', color = 'y')
            ax11.set_title(f'k3 = {params[2, 2]:.3f}')

            ax12.plot(data_AA05[0][0], data_AA05[0][5], label = 'k3', color = 'y')
            ax12.set_title(f'k3 = {params_AA05[2, 0]:.3f}')

            # Fourth row
            ax13.plot(data[0][0], data[0][3], label = 'vb', color = 'r')
            ax13.set_title(f'vb = {params[3, 0]:.3f}')
            ax13.set_ylabel('dPET/dvb')

            ax14.plot(data[1][0], data[1][3], label = 'vb', color = 'r')
            ax14.set_title(f'vb = {params[3, 1]:.3f}')

            ax15.plot(data[2][0], data[2][3], label = 'vb', color = 'r')
            ax15.set_title(f'vb = {params[3, 2]:.3f}')

            ax16.plot(data_AA05[0][0], data_AA05[0][3], label = 'vb', color = 'r')
            ax16.set_title(f'vb = {params_AA05[3, 0]:.3f}')

            # Fifth row
            ax17.plot(data[0][0], data[0][4], label = 'F', color = 'm')
            ax17.set_title(f'F = {params[4, 0]:.3f}')
            ax17.set_ylabel('dPET/dF')

            ax18.plot(data[1][0], data[1][4], label = 'F', color = 'm')
            ax18.set_title(f'F = {params[4, 1]:.3f}')

            ax19.plot(data[2][0], data[2][4], label = 'F', color = 'm')
            ax19.set_title(f'F = {params[4, 2]:.3f}')

            ax20.plot(data_AA05[0][0], data_AA05[0][4], label = 'F', color = 'm')
            ax20.set_title(f'F = {params_AA05[4, 0]:.3f}')

        else:
            fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12), (ax13, ax14, ax15, ax16)) = plt.subplots(4, 4, figsize = (12,12), constrained_layout = False, tight_layout = True)
            fig.suptitle(f'Sensitivity analysis of all parameters for {organ} at {state}')

            # First row
            ax1.plot(data[0][0], data[0][1], label = 'K1', color = 'b')
            ax1.set_title(f'Mouse AA01 \n K1 = {params[0, 0]:.3f}')
            ax1.set_ylabel('dPET/dK1')

            ax2.plot(data[1][0], data[1][1], label = 'K1', color = 'b')
            ax2.set_title(f'Mouse AA02 \n K1 = {params[0, 1]:.3f}')

            ax3.plot(data[2][0], data[2][1], label = 'K1', color = 'b')
            ax3.set_title(f'Mouse AA03 \n K1 = {params[0, 2]:.3f}')

            ax4.plot(data_AA05[0][0], data_AA05[0][1], label = 'K1', color = 'b')
            ax4.set_title(f'Mouse AA05 \n K1 = {params_AA05[0, 0]:.3f}')

            # Second row
            ax5.plot(data[0][0], data[0][2], label = 'k2', color = 'g')
            ax5.set_title(f'k2 = {params[1, 0]:.3f}')
            ax5.set_ylabel('dPET/dk2')

            ax6.plot(data[1][0], data[1][2], label = 'k2', color = 'g')
            ax6.set_title(f'k2 = {params[1, 1]:.3f}')

            ax7.plot(data[2][0], data[2][2], label = 'k2', color = 'g')
            ax7.set_title(f'k2 = {params[1, 2]:.3f}')

            ax8.plot(data_AA05[0][0], data_AA05[0][2], label = 'k2', color = 'g')
            ax8.set_title(f'k2 = {params_AA05[1, 0]:.3f}')

            # Third row
            ax9.plot(data[0][0], data[0][3], label = 'vb', color = 'r')
            ax9.set_title(f'vb = {params[2, 0]:.3f}')
            ax9.set_ylabel('dPET/dvb')

            ax10.plot(data[1][0], data[1][3], label = 'vb', color = 'r')
            ax10.set_title(f'vb = {params[2, 1]:.3f}')

            ax11.plot(data[2][0], data[2][3], label = 'vb', color = 'r')
            ax11.set_title(f'vb = {params[2, 2]:.3f}')

            ax12.plot(data_AA05[0][0], data_AA05[0][3], label = 'vb', color = 'r')
            ax12.set_title(f'vb = {params_AA05[2, 0]:.3f}')

            # Fourth row
            ax13.plot(data[0][0], data[0][4], label = 'F', color = 'm')
            ax13.set_title(f'F = {params[3, 0]:.3f}')
            ax13.set_ylabel('dPET/dF')

            ax14.plot(data[1][0], data[1][4], label = 'F', color = 'm')
            ax14.set_title(f'F = {params[3, 1]:.3f}')

            ax15.plot(data[2][0], data[2][4], label = 'F', color = 'm')
            ax15.set_title(f'F = {params[3, 2]:.3f}')

            ax16.plot(data_AA05[0][0], data_AA05[0][4], label = 'F', color = 'm')
            ax16.set_title(f'F = {params_AA05[3, 0]:.3f}')
            
        plt.savefig(f'{folder}\{sub_folder}/SA/{organ}_{state}_SA')
        plt.close()

        with open(f'{folder}\{sub_folder}/SA/{organ}_{state}params.csv', 'w', newline = '') as f:
            writer = csv.writer(f)
            writer.writerows([['Mouse AA01', 'Mouse AA02', 'Mouse AA03']])
            writer.writerows(params)
            writer.writerows([' '])
            writer.writerows(error)
            writer.writerows([' '])
            writer.writerows([['Mouse AA05']])
            writer.writerows(params_AA05)
            writer.writerows([' '])
            writer.writerows(error_AA05)

def sensitivity_analysis_display_ch4_combined(folder, sub_folder, data_heart, params_heart, data_AA05_heart, params_AA05_heart, data_lungs, params_lungs, data_AA05_lungs, params_AA05_lungs, data_liver, params_liver, data_AA05_liver, params_AA05_liver, data_kidneys, params_kidneys, data_AA05_kidneys, params_AA05_kidneys, data_femur, params_femur, data_AA05_femur, params_AA05_femur, state):
    if state == 'Rest':
        fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15), (ax16, ax17, ax18, ax19, ax20), (ax21, ax22, ax23, ax24, ax25)) = plt.subplots(5, 5, figsize = (15,15), constrained_layout = False, tight_layout = True)
        fig.suptitle(f'Sensitivity analysis of K1, F and vb for all mice and organs at {state}', fontsize = 'x-large')

        # First row (Heart)
        ax1.plot(data_heart[0][0], data_heart[0][1], label = 'K1', color = 'b')
        ax1.plot(data_heart[0][0], data_heart[0][4], label = 'F', color = 'm')
        ax1.plot(data_heart[0][0], data_heart[0][3], label = 'vb', color = 'r')
        ax1.set_title(f'Mouse AA01: K1 = {params_heart[0, 0]:.3f}, \n F = {params_heart[4, 0]:.3f}, vb = {params_heart[3, 0]:.3f}')
        ax1.set_ylabel('Heart')

        ax2.plot(data_heart[1][0], data_heart[1][1], label = 'K1', color = 'b')
        ax2.plot(data_heart[1][0], data_heart[1][4], label = 'F', color = 'm')
        ax2.plot(data_heart[1][0], data_heart[1][3], label = 'vb', color = 'r')
        ax2.set_title(f'Mouse AA02: K1 = {params_heart[0, 1]:.3f}, \n F = {params_heart[4, 1]:.3f}, vb = {params_heart[3, 1]:.3f}')

        ax3.plot(data_heart[2][0], data_heart[2][1], label = 'K1', color = 'b')
        ax3.plot(data_heart[2][0], data_heart[2][4], label = 'F', color = 'm')
        ax3.plot(data_heart[2][0], data_heart[2][3], label = 'vb', color = 'r')
        ax3.set_title(f'Mouse AA03: K1 = {params_heart[0, 2]:.3f}, \n F = {params_heart[4, 2]:.3f}, vb = {params_heart[3, 2]:.3f}')

        ax4.plot(data_heart[3][0], data_heart[3][1], label = 'K1', color = 'b')
        ax4.plot(data_heart[3][0], data_heart[3][4], label = 'F', color = 'm')
        ax4.plot(data_heart[3][0], data_heart[3][3], label = 'vb', color = 'r')
        ax4.set_title(f'Mouse AA04: K1 = {params_heart[0, 3]:.3f}, \n F = {params_heart[4, 3]:.3f}, vb = {params_heart[3, 3]:.3f}')

        ax5.plot(data_AA05_heart[0][0], data_AA05_heart[0][1], label = 'K1', color = 'b')
        ax5.plot(data_AA05_heart[0][0], data_AA05_heart[0][4], label = 'F', color = 'm')
        ax5.plot(data_AA05_heart[0][0], data_AA05_heart[0][3], label = 'vb', color = 'r')
        ax5.set_title(f'Mouse AA05: K1 = {params_AA05_heart[0, 0]:.3f}, \n F = {params_AA05_heart[4, 0]:.3f}, vb = {params_AA05_heart[3, 0]:.3f}')


        # Second row
        ax6.plot(data_lungs[0][0], data_lungs[0][1], label = 'K1', color = 'b')
        ax6.plot(data_lungs[0][0], data_lungs[0][4], label = 'F', color = 'm')
        ax6.plot(data_lungs[0][0], data_lungs[0][3], label = 'vb', color = 'r')
        ax6.set_title(f'K1 = {params_lungs[0, 0]:.3f}, \n F = {params_lungs[3, 0]:.3f}, vb = {params_lungs[2, 0]:.3f}')
        ax6.set_ylabel('Lungs')

        ax7.plot(data_lungs[1][0], data_lungs[1][1], label = 'K1', color = 'b')
        ax7.plot(data_lungs[1][0], data_lungs[1][4], label = 'F', color = 'm')
        ax7.plot(data_lungs[1][0], data_lungs[1][3], label = 'vb', color = 'r')
        ax7.set_title(f'K1 = {params_lungs[0, 1]:.3f}, \n F = {params_lungs[3, 1]:.3f}, vb = {params_lungs[2, 1]:.3f}')

        ax8.plot(data_lungs[2][0], data_lungs[2][1], label = 'K1', color = 'b')
        ax8.plot(data_lungs[2][0], data_lungs[2][4], label = 'F', color = 'm')
        ax8.plot(data_lungs[2][0], data_lungs[2][3], label = 'vb', color = 'r')
        ax8.set_title(f'K1 = {params_lungs[0, 2]:.3f}, \n F = {params_lungs[3, 2]:.3f}, vb = {params_lungs[2, 2]:.3f}')

        ax9.plot(data_lungs[3][0], data_lungs[3][1], label = 'K1', color = 'b')
        ax9.plot(data_lungs[3][0], data_lungs[3][4], label = 'F', color = 'm')
        ax9.plot(data_lungs[3][0], data_lungs[3][3], label = 'vb', color = 'r')
        ax9.set_title(f'K1 = {params_lungs[0, 3]:.3f}, \n F = {params_lungs[3, 3]:.3f}, vb = {params_lungs[2, 3]:.3f}')

        ax10.plot(data_AA05_lungs[0][0], data_AA05_lungs[0][1], label = 'K1', color = 'b')
        ax10.plot(data_AA05_lungs[0][0], data_AA05_lungs[0][4], label = 'F', color = 'm')
        ax10.plot(data_AA05_lungs[0][0], data_AA05_lungs[0][3], label = 'vb', color = 'r')
        ax10.set_title(f'K1 = {params_AA05_lungs[0, 0]:.3f}, \n F = {params_AA05_lungs[3, 0]:.3f}, vb = {params_AA05_lungs[2, 0]:.3f}')

        # Third row (Liver)
        ax11.plot(data_liver[0][0], data_liver[0][1], label = 'K1', color = 'b')
        ax11.plot(data_liver[0][0], data_liver[0][4], label = 'F', color = 'm')
        ax11.plot(data_liver[0][0], data_liver[0][3], label = 'vb', color = 'r')
        ax11.set_title(f'K1 = {params_liver[0, 0]:.3f}, \n F = {params_liver[3, 0]:.3f}, vb = {params_liver[2, 0]:.3f}')
        ax11.set_ylabel('Liver')

        ax12.plot(data_liver[1][0], data_liver[1][1], label = 'K1', color = 'b')
        ax12.plot(data_liver[1][0], data_liver[1][4], label = 'F', color = 'm')
        ax12.plot(data_liver[1][0], data_liver[1][3], label = 'vb', color = 'r')
        ax12.set_title(f'K1 = {params_liver[0, 1]:.3f}, \n F = {params_liver[3, 1]:.3f}, vb = {params_liver[2, 1]:.3f}')

        ax13.plot(data_liver[2][0], data_liver[2][1], label = 'K1', color = 'b')
        ax13.plot(data_liver[2][0], data_liver[2][4], label = 'F', color = 'm')
        ax13.plot(data_liver[2][0], data_liver[2][3], label = 'vb', color = 'r')
        ax13.set_title(f'K1 = {params_liver[0, 2]:.3f}, \n F = {params_liver[3, 2]:.3f}, vb = {params_liver[2, 2]:.3f}')

        ax14.plot(data_liver[3][0], data_liver[3][1], label = 'K1', color = 'b')
        ax14.plot(data_liver[3][0], data_liver[3][4], label = 'F', color = 'm')
        ax14.plot(data_liver[3][0], data_liver[3][3], label = 'vb', color = 'r')
        ax14.set_title(f'K1 = {params_liver[0, 3]:.3f}, \n F = {params_liver[3, 3]:.3f}, vb = {params_liver[2, 3]:.3f}')

        ax15.plot(data_AA05_liver[0][0], data_AA05_liver[0][1], label = 'K1', color = 'b')
        ax15.plot(data_AA05_liver[0][0], data_AA05_liver[0][4], label = 'F', color = 'm')
        ax15.plot(data_AA05_liver[0][0], data_AA05_liver[0][3], label = 'vb', color = 'r')
        ax15.set_title(f'K1 = {params_AA05_liver[0, 0]:.3f}, \n F = {params_AA05_liver[3, 0]:.3f}, vb = {params_AA05_liver[2, 0]:.3f}')

        # Fourth row (Kidneys)
        ax16.plot(data_kidneys[0][0], data_kidneys[0][1], label = 'K1', color = 'b')
        ax16.plot(data_kidneys[0][0], data_kidneys[0][4], label = 'F', color = 'm')
        ax16.plot(data_kidneys[0][0], data_kidneys[0][3], label = 'vb', color = 'r')
        ax16.set_title(f'K1 = {params_kidneys[0, 0]:.3f}, \n F = {params_kidneys[3, 0]:.3f}, vb = {params_kidneys[2, 0]:.3f}')
        ax16.set_ylabel('Kidneys')

        ax17.plot(data_kidneys[1][0], data_kidneys[1][1], label = 'K1', color = 'b')
        ax17.plot(data_kidneys[1][0], data_kidneys[1][4], label = 'F', color = 'm')
        ax17.plot(data_kidneys[1][0], data_kidneys[1][3], label = 'vb', color = 'r')
        ax17.set_title(f'K1 = {params_kidneys[0, 1]:.3f}, \n F = {params_kidneys[3, 1]:.3f}, vb = {params_kidneys[2, 1]:.3f}')

        ax18.plot(data_kidneys[2][0], data_kidneys[2][1], label = 'K1', color = 'b')
        ax18.plot(data_kidneys[2][0], data_kidneys[2][4], label = 'F', color = 'm')
        ax18.plot(data_kidneys[2][0], data_kidneys[2][3], label = 'vb', color = 'r')
        ax18.set_title(f'K1 = {params_kidneys[0, 2]:.3f}, \n F = {params_kidneys[3, 2]:.3f}, vb = {params_kidneys[2, 2]:.3f}')

        ax19.plot(data_kidneys[3][0], data_kidneys[3][1], label = 'K1', color = 'b')
        ax19.plot(data_kidneys[3][0], data_kidneys[3][4], label = 'F', color = 'm')
        ax19.plot(data_kidneys[3][0], data_kidneys[3][3], label = 'vb', color = 'r')
        ax19.set_title(f'K1 = {params_kidneys[0, 3]:.3f}, \n F = {params_kidneys[3, 3]:.3f}, vb = {params_kidneys[2, 3]:.3f}')

        ax20.plot(data_AA05_kidneys[0][0], data_AA05_kidneys[0][1], label = 'K1', color = 'b')
        ax20.plot(data_AA05_kidneys[0][0], data_AA05_kidneys[0][4], label = 'F', color = 'm')
        ax20.plot(data_AA05_kidneys[0][0], data_AA05_kidneys[0][3], label = 'vb', color = 'r')
        ax20.set_title(f'K1 = {params_AA05_kidneys[0, 0]:.3f}, \n F = {params_AA05_kidneys[3, 0]:.3f}, vb = {params_AA05_kidneys[2, 0]:.3f}')

        # Fifth row (Femur)

        ax21.plot(data_femur[0][0], data_femur[0][1], label = 'K1', color = 'b')
        ax21.plot(data_femur[0][0], data_femur[0][4], label = 'F', color = 'm')
        ax21.plot(data_femur[0][0], data_femur[0][3], label = 'vb', color = 'r')
        ax21.set_title(f'K1 = {params_femur[0, 0]:.3f}, \n F = {params_femur[3, 0]:.3f}, vb = {params_femur[2, 0]:.3f}')
        ax21.set_ylabel('Femur')

        ax22.plot(data_femur[1][0], data_femur[1][1], label = 'K1', color = 'b')
        ax22.plot(data_femur[1][0], data_femur[1][4], label = 'F', color = 'm')
        ax22.plot(data_femur[1][0], data_femur[1][3], label = 'vb', color = 'r')
        ax22.set_title(f'K1 = {params_femur[0, 1]:.3f}, \n F = {params_femur[3, 1]:.3f}, vb = {params_femur[2, 1]:.3f}')

        ax23.plot(data_femur[2][0], data_femur[2][1], label = 'K1', color = 'b')
        ax23.plot(data_femur[2][0], data_femur[2][4], label = 'F', color = 'm')
        ax23.plot(data_femur[2][0], data_femur[2][3], label = 'vb', color = 'r')
        ax23.set_title(f'K1 = {params_femur[0, 2]:.3f}, \n F = {params_femur[3, 2]:.3f}, vb = {params_femur[2, 2]:.3f}')

        ax24.plot(data_femur[3][0], data_femur[3][1], label = 'K1', color = 'b')
        ax24.plot(data_femur[3][0], data_femur[3][4], label = 'F', color = 'm')
        ax24.plot(data_femur[3][0], data_femur[3][3], label = 'vb', color = 'r')
        ax24.set_title(f'K1 = {params_femur[0, 3]:.3f}, \n F = {params_femur[3, 3]:.3f}, vb = {params_femur[2, 3]:.3f}')

        ax25.plot(data_AA05_femur[0][0], data_AA05_femur[0][1], label = 'K1', color = 'b')
        ax25.plot(data_AA05_femur[0][0], data_AA05_femur[0][4], label = 'F', color = 'm')
        ax25.plot(data_AA05_femur[0][0], data_AA05_femur[0][3], label = 'vb', color = 'r')
        ax25.set_title(f'K1 = {params_AA05_femur[0, 0]:.3f}, \n F = {params_AA05_femur[3, 0]:.3f}, vb = {params_AA05_femur[2, 0]:.3f}')
        
        plt.savefig(f'{folder}\{sub_folder}/SA/{state}_combined_SA')
        plt.close()

        with open(f'{folder}\{sub_folder}/SA/{state}_combined_params.csv', 'w', newline = '') as f:
            writer = csv.writer(f)
            writer.writerows([['Heart', 'Lungs', 'Liver', 'Kidneys', 'Femur']])
            writer.writerows([[params_heart, params_lungs, params_liver, params_kidneys, params_femur]])
            writer.writerows([' '])
            writer.writerows([['Heart', 'Lungs', 'Liver', 'Kidneys', 'Femur']])
            writer.writerows([[params_AA05_heart, params_AA05_lungs, params_AA05_liver, params_AA05_kidneys, params_AA05_femur]])
            writer.writerows([' '])


    elif state == 'Stress':
        fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12), (ax13, ax14, ax15, ax16), (ax17, ax18, ax19, ax20)) = plt.subplots(5, 4, figsize = (12,15), constrained_layout = False, tight_layout = True)
        fig.suptitle(f'Sensitivity analysis of K1, F and vb for all mice and organs at {state}', fontsize = 'x-large')

        # First row (Heart)
        ax1.plot(data_heart[0][0], data_heart[0][1], label = 'K1', color = 'b')
        ax1.plot(data_heart[0][0], data_heart[0][4], label = 'F', color = 'm')
        ax1.plot(data_heart[0][0], data_heart[0][3], label = 'vb', color = 'r')
        ax1.set_title(f'Mouse AA01: K1 = {params_heart[0, 0]:.3f}, \n F = {params_heart[3, 0]:.3f}, vb = {params_heart[2, 0]:.3f}')
        ax1.set_ylabel('Heart')

        ax2.plot(data_heart[1][0], data_heart[1][1], label = 'K1', color = 'b')
        ax2.plot(data_heart[1][0], data_heart[1][4], label = 'F', color = 'm')
        ax2.plot(data_heart[1][0], data_heart[1][3], label = 'vb', color = 'r')
        ax2.set_title(f'Mouse AA02: K1 = {params_heart[0, 1]:.3f}, \n F = {params_heart[3, 1]:.3f}, vb = {params_heart[2, 1]:.3f}')

        ax3.plot(data_heart[2][0], data_heart[2][1], label = 'K1', color = 'b')
        ax3.plot(data_heart[2][0], data_heart[2][4], label = 'F', color = 'm')
        ax3.plot(data_heart[2][0], data_heart[2][3], label = 'vb', color = 'r')
        ax3.set_title(f'Mouse AA03: K1 = {params_heart[0, 2]:.3f}, \n F = {params_heart[3, 2]:.3f}, vb = {params_heart[2, 2]:.3f}')

        ax4.plot(data_AA05_heart[0][0], data_AA05_heart[0][1], label = 'K1', color = 'b')
        ax4.plot(data_AA05_heart[0][0], data_AA05_heart[0][4], label = 'F', color = 'm')
        ax4.plot(data_AA05_heart[0][0], data_AA05_heart[0][3], label = 'vb', color = 'r')
        ax4.set_title(f'Mouse AA05: K1 = {params_AA05_heart[0, 0]:.3f}, \n F = {params_AA05_heart[3, 0]:.3f}, vb = {params_AA05_heart[2, 0]:.3f}')


        # Second row
        ax5.plot(data_lungs[0][0], data_lungs[0][1], label = 'K1', color = 'b')
        ax5.plot(data_lungs[0][0], data_lungs[0][4], label = 'F', color = 'm')
        ax5.plot(data_lungs[0][0], data_lungs[0][3], label = 'vb', color = 'r')
        ax5.set_title(f'K1 = {params_lungs[0, 0]:.3f}, \n F = {params_lungs[3, 0]:.3f}, vb = {params_lungs[2, 0]:.3f}')
        ax5.set_ylabel('Lungs')

        ax6.plot(data_lungs[1][0], data_lungs[1][1], label = 'K1', color = 'b')
        ax6.plot(data_lungs[1][0], data_lungs[1][4], label = 'F', color = 'm')
        ax6.plot(data_lungs[1][0], data_lungs[1][3], label = 'vb', color = 'r')
        ax6.set_title(f'K1 = {params_lungs[0, 1]:.3f}, \n F = {params_lungs[3, 1]:.3f}, vb = {params_lungs[2, 1]:.3f}')

        ax7.plot(data_lungs[2][0], data_lungs[2][1], label = 'K1', color = 'b')
        ax7.plot(data_lungs[2][0], data_lungs[2][4], label = 'F', color = 'm')
        ax7.plot(data_lungs[2][0], data_lungs[2][3], label = 'vb', color = 'r')
        ax7.set_title(f'K1 = {params_lungs[0, 2]:.3f}, \n F = {params_lungs[3, 2]:.3f}, vb = {params_lungs[2, 2]:.3f}')

        ax8.plot(data_AA05_lungs[0][0], data_AA05_lungs[0][1], label = 'K1', color = 'b')
        ax8.plot(data_AA05_lungs[0][0], data_AA05_lungs[0][4], label = 'F', color = 'm')
        ax8.plot(data_AA05_lungs[0][0], data_AA05_lungs[0][3], label = 'vb', color = 'r')
        ax8.set_title(f'K1 = {params_AA05_lungs[0, 0]:.3f}, \n F = {params_AA05_lungs[3, 0]:.3f}, vb = {params_AA05_lungs[2, 0]:.3f}')

        # Third row (Liver)
        ax9.plot(data_liver[0][0], data_liver[0][1], label = 'K1', color = 'b')
        ax9.plot(data_liver[0][0], data_liver[0][4], label = 'F', color = 'm')
        ax9.plot(data_liver[0][0], data_liver[0][3], label = 'vb', color = 'r')
        ax9.set_title(f'K1 = {params_liver[0, 0]:.3f}, \n F = {params_liver[4, 0]:.3f}, vb = {params_liver[3, 0]:.3f}')
        ax9.set_ylabel('Liver')

        ax10.plot(data_liver[1][0], data_liver[1][1], label = 'K1', color = 'b')
        ax10.plot(data_liver[1][0], data_liver[1][4], label = 'F', color = 'm')
        ax10.plot(data_liver[1][0], data_liver[1][3], label = 'vb', color = 'r')
        ax10.set_title(f'K1 = {params_liver[0, 1]:.3f}, \n F = {params_liver[4, 1]:.3f}, vb = {params_liver[3, 1]:.3f}')

        ax11.plot(data_liver[2][0], data_liver[2][1], label = 'K1', color = 'b')
        ax11.plot(data_liver[2][0], data_liver[2][4], label = 'F', color = 'm')
        ax11.plot(data_liver[2][0], data_liver[2][3], label = 'vb', color = 'r')
        ax11.set_title(f'K1 = {params_liver[0, 2]:.3f}, \n F = {params_liver[4, 2]:.3f}, vb = {params_liver[3, 2]:.3f}')

        ax12.plot(data_AA05_liver[0][0], data_AA05_liver[0][1], label = 'K1', color = 'b')
        ax12.plot(data_AA05_liver[0][0], data_AA05_liver[0][4], label = 'F', color = 'm')
        ax12.plot(data_AA05_liver[0][0], data_AA05_liver[0][3], label = 'vb', color = 'r')
        ax12.set_title(f'K1 = {params_AA05_liver[0, 0]:.3f}, \n F = {params_AA05_liver[4, 0]:.3f}, vb = {params_AA05_liver[3, 0]:.3f}')

        # Fourth row (Kidneys)
        ax13.plot(data_kidneys[0][0], data_kidneys[0][1], label = 'K1', color = 'b')
        ax13.plot(data_kidneys[0][0], data_kidneys[0][4], label = 'F', color = 'm')
        ax13.plot(data_kidneys[0][0], data_kidneys[0][3], label = 'vb', color = 'r')
        ax13.set_title(f'K1 = {params_kidneys[0, 0]:.3f}, \n F = {params_kidneys[4, 0]:.3f}, vb = {params_kidneys[3, 0]:.3f}')
        ax13.set_ylabel('Kidneys')

        ax14.plot(data_kidneys[1][0], data_kidneys[1][1], label = 'K1', color = 'b')
        ax14.plot(data_kidneys[1][0], data_kidneys[1][4], label = 'F', color = 'm')
        ax14.plot(data_kidneys[1][0], data_kidneys[1][3], label = 'vb', color = 'r')
        ax14.set_title(f'K1 = {params_kidneys[0, 1]:.3f}, \n F = {params_kidneys[4, 1]:.3f}, vb = {params_kidneys[3, 1]:.3f}')

        ax15.plot(data_kidneys[2][0], data_kidneys[2][1], label = 'K1', color = 'b')
        ax15.plot(data_kidneys[2][0], data_kidneys[2][4], label = 'F', color = 'm')
        ax15.plot(data_kidneys[2][0], data_kidneys[2][3], label = 'vb', color = 'r')
        ax15.set_title(f'K1 = {params_kidneys[0, 2]:.3f}, \n F = {params_kidneys[4, 2]:.3f}, vb = {params_kidneys[3, 2]:.3f}')

        ax16.plot(data_AA05_kidneys[0][0], data_AA05_kidneys[0][1], label = 'K1', color = 'b')
        ax16.plot(data_AA05_kidneys[0][0], data_AA05_kidneys[0][4], label = 'F', color = 'm')
        ax16.plot(data_AA05_kidneys[0][0], data_AA05_kidneys[0][3], label = 'vb', color = 'r')
        ax16.set_title(f'K1 = {params_AA05_kidneys[0, 0]:.3f}, \n F = {params_AA05_kidneys[4, 0]:.3f}, vb = {params_AA05_kidneys[3, 0]:.3f}')

        # Fifth row (Femur)

        ax17.plot(data_femur[0][0], data_femur[0][1], label = 'K1', color = 'b')
        ax17.plot(data_femur[0][0], data_femur[0][4], label = 'F', color = 'm')
        ax17.plot(data_femur[0][0], data_femur[0][3], label = 'vb', color = 'r')
        ax17.set_title(f'K1 = {params_femur[0, 0]:.3f}, \n F = {params_femur[4, 0]:.3f}, vb = {params_femur[3, 0]:.3f}')
        ax17.set_ylabel('Femur')

        ax18.plot(data_femur[1][0], data_femur[1][1], label = 'K1', color = 'b')
        ax18.plot(data_femur[1][0], data_femur[1][4], label = 'F', color = 'm')
        ax18.plot(data_femur[1][0], data_femur[1][3], label = 'vb', color = 'r')
        ax18.set_title(f'K1 = {params_femur[0, 1]:.3f}, \n F = {params_femur[4, 1]:.3f}, vb = {params_femur[3, 1]:.3f}')

        ax19.plot(data_femur[2][0], data_femur[2][1], label = 'K1', color = 'b')
        ax19.plot(data_femur[2][0], data_femur[2][4], label = 'F', color = 'm')
        ax19.plot(data_femur[2][0], data_femur[2][3], label = 'vb', color = 'r')
        ax19.set_title(f'K1 = {params_femur[0, 2]:.3f}, \n F = {params_femur[4, 2]:.3f}, vb = {params_femur[3, 2]:.3f}')

        ax20.plot(data_AA05_femur[0][0], data_AA05_femur[0][1], label = 'K1', color = 'b')
        ax20.plot(data_AA05_femur[0][0], data_AA05_femur[0][4], label = 'F', color = 'm')
        ax20.plot(data_AA05_femur[0][0], data_AA05_femur[0][3], label = 'vb', color = 'r')
        ax20.set_title(f'K1 = {params_AA05_femur[0, 0]:.3f}, \n F = {params_AA05_femur[4, 0]:.3f}, vb = {params_AA05_femur[3, 0]:.3f}')
        
        plt.savefig(f'{folder}\{sub_folder}/SA/{state}_combined_SA')
        plt.close()

        with open(f'{folder}\{sub_folder}/SA/{state}_combined_params.csv', 'w', newline = '') as f:
            writer = csv.writer(f)
            writer.writerows([['Heart', 'Lungs', 'Liver', 'Kidneys', 'Femur']])
            writer.writerows([[params_heart, params_lungs, params_liver, params_kidneys, params_femur]])
            writer.writerows([' '])
            writer.writerows([['Heart', 'Lungs', 'Liver', 'Kidneys', 'Femur']])
            writer.writerows([[params_AA05_heart, params_AA05_lungs, params_AA05_liver, params_AA05_kidneys, params_AA05_femur]])
            writer.writerows([' '])
        

def Ext_Frac_graph(ps_df, folder, sub_folder):
    x = np.linspace(0, 10, 101)
    organs = ['Heart', 'Lungs', 'Kidneys', 'Liver', 'Femur']
    efs = np.zeros((5, 101))

    i = 0
    for organ in organs:
        a = ps_df.loc[organ, 'a']
        b = ps_df.loc[organ, 'b']
        ef = ext_frac(x, a, b)
        efs[i] = ef

        plt.plot(x, ef, label = 'Extraction Fraction', color = 'b')
        #plt.plot(x, x, label = 'Identity Line', color = 'r', linestyle = '-')
        plt.title(f'Extraction Fraction vs Flow for \n {organ} in mice using NH3')
        plt.xlabel('Flow (mL/mL*min)')
        plt.ylabel('Extraction Fraction')
        plt.savefig(f'{folder}\{sub_folder}/Extraction_Fraction/{organ}_NH3_mice')
        plt.close()

        i+=1

    plt.plot(x, efs[0], label = 'Heart', color = 'b')
    #plt.plot(x, efs[1], label = 'Lungs', color = 'g')
    plt.plot(x, efs[2], label = 'Kidneys', color = 'm')
    plt.plot(x, efs[3], label = 'Liver', color = 'g')
    #plt.plot(x, efs[4], label = 'Femur', color = 'y')
    #plt.plot(x, x, label = 'Identity Line', color = 'r', linestyle = '-')
    plt.title(f'Extraction Fraction vs Flow for \n all organs in mice using NH3')
    plt.xlabel('Flow (mL/mL*min)')
    plt.ylabel('Extraction Fraction')
    plt.legend(loc = 7, fontsize = 'x-small')
    plt.savefig(f'{folder}\{sub_folder}/Extraction_Fraction/All_organs_NH3_mice')
    plt.close()



def input_NH3_data(sheet, deg):

    ### Read in the mouse data from the Excel spreadsheet

    data = pd.read_excel('NH3_Mice_TACs_SUVs.xlsx', sheet_name=sheet, engine = 'openpyxl', usecols = 'B:H', dtype = {'Time (min)' : float, 'Heart' : float, 'Liver' : float, 
    'Lungs' : float, 'Kidneys' : float, 'Femur' : float, 'Vena_Cava' : float})       # 'Des Aorta Amide', 'TACs_new_VOIs_fitted_input_functions.xlsx'
    data.columns = ['Time', 'Heart', 'Liver', 'Lungs', 'Kidneys', 'Femur', 'Vena_Cava']
    
    data = data.drop(0, axis=0)
    data = data.dropna()
    
    #data = data[~(data == 0).any(axis=1)]       # Used to remove any zeros
    data = data.reset_index()
    data = data.drop('index', axis=1)
    #data.Time = data.Time - data.Time[0]
    #print(data)

    #
    if deg == True:
        data = data.iloc[:21]   # This slices the data to the first 4 minutes
    #

    # print(data)
   
    T_f = data.iloc[-1, 0]
    T_0 = data.iloc[0, 0]
    dt = 1/600
   
    time = np.arange(T_0, T_f + dt, dt)

    ### Resample the data so that it's evenly-spaced and place in new dataframe

    data_int = pd.DataFrame(columns = ['Time', 'Heart', 'Liver', 'Lungs', 'Kidneys', 'Femur', 'Vena_Cava'])
    data_int.Time = time - T_0

    #weights = pd.DataFrame(columns = ['Heart', 'Lungs', 'Kidneys', 'Bladder', 'Femur', 'Liver', 'Vena_Cava'])
    
    
    for i in data.columns:
        if i != 'Time':
            data_int[i], inter_func = interpolate(data.Time, data[i], 'cubic', T_0, T_f, dt)

            #weights[i], zero_count = weights_dc(inter_func, data, i, 0.0063128158521)       # currently in minutes, in seconds: 0.0001052135975

    data = data.apply(pd.to_numeric, errors = 'coerce', axis=0)

    t_peak = data.loc[data['Vena_Cava'].idxmax()]['Time']


    return data, data_int, [T_f, T_0, dt, time]



### MAIN

frametimes_s = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 190, 220, 265, 355, 475, 685, 985, 1165, 1225])
frametimes_m = frametimes_s / 60

frametimes_mid_s = np.array([5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155, 165, 180, 205, 242.5, 310, 415, 580, 835, 1075, 1195])
frametimes_mid_m = frametimes_mid_s / 60

frame_lengths_s = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 20, 30, 45, 90, 120, 210, 300, 180, 60])
frame_lengths_m = frame_lengths_s / 60


# Dataframe with a and b values used for determining flow according to the generalized Renkin-Crone model

ps_df = pd.DataFrame(data = {'Organ' : ['Heart', 'Lungs', 'Liver', 'Femur', 'Kidneys'], 'a' : [0.607, 0.607, 1, 1, 0.83],
                             'b' : [1.25, 1.25, 1.18, 1.18, 1.353]}, index = ['Heart', 'Lungs', 'Liver', 'Femur', 'Kidneys'])   # Lungs and Femur values unknown, set to same as heart and liver, respectively


model_selection_df = pd.DataFrame(data = { 'Heart' : ['2TCM', '1TCM', '2TCM', '1TCM', '2TCM', '1TCM', '2TCM', '2TCM', '1TCM'], 'Lungs' : ['1TCM', '1TCM', '1TCM', '1TCM', '1TCM', '1TCM', '1TCM', '1TCM', '1TCM'],
                                          'Liver' : ['1TCM', '2TCM', '1TCM', '2TCM', '1TCM', '2TCM', '1TCM', '1TCM', '2TCM'], 'Femur' : ['1TCM', '2TCM', '1TCM', '2TCM', '1TCM', '2TCM', '1TCM', '1TCM', '2TCM'], 
                                          'Kidneys' : ['Degrado', '2TCM', 'Degrado', '2TCM', 'Degrado', '2TCM', 'Degrado', 'Degrado', '2TCM']}, 
                                          index = ['AA01_Rest', 'AA01_Stress', 'AA02_Rest', 'AA02_Stress', 'AA03_Rest', 'AA03_Stress', 'AA04_Rest', 'AA05_Rest', 'AA05_Stress'])


# Sensitivity Analysis
# Rest
sens_heart_R = np.zeros((4, 6, 11901))      # 2TCM
sens_lungs_R = np.zeros((4, 5, 11901))      # 1TCM
sens_liver_R = np.zeros((4, 5, 11901))      # 1TCM
sens_kidneys_R = np.zeros((4, 5, 3051))     # Degrado  
sens_femur_R = np.zeros((4, 5, 11901))      # 1TCM

sens_heart_params_R = np.zeros((5, 4))
sens_lungs_params_R = np.zeros((4, 4))
sens_liver_params_R = np.zeros((4, 4))
sens_kidneys_params_R = np.zeros((4, 4))
sens_femur_params_R = np.zeros((4, 4))

sens_heart_err_R = np.zeros((5, 4))
sens_lungs_err_R = np.zeros((4, 4))
sens_liver_err_R = np.zeros((4, 4))
sens_kidneys_err_R = np.zeros((4, 4))
sens_femur_err_R = np.zeros((4, 4))

# AA05 shenanigans

sens_heart_AA05_R = np.zeros((1, 6, 5901))      # 2TCM
sens_lungs_AA05_R = np.zeros((1, 5, 5901))      # 1TCM
sens_liver_AA05_R = np.zeros((1, 5, 5901))      # 1TCM
sens_kidneys_AA05_R = np.zeros((1, 5, 3051))     # Degrado  
sens_femur_AA05_R = np.zeros((1, 5, 5901))      # 1TCM

sens_heart_params_AA05_R = np.zeros((5, 1))
sens_lungs_params_AA05_R = np.zeros((4, 1))
sens_liver_params_AA05_R = np.zeros((4, 1))
sens_kidneys_params_AA05_R = np.zeros((4, 1))
sens_femur_params_AA05_R = np.zeros((4, 1))

sens_heart_err_AA05_R = np.zeros((5, 1))
sens_lungs_err_AA05_R = np.zeros((4, 1))
sens_liver_err_AA05_R = np.zeros((4, 1))
sens_kidneys_err_AA05_R = np.zeros((4, 1))
sens_femur_err_AA05_R = np.zeros((4, 1))

#Stress
sens_heart_S = np.zeros((3, 5, 11901))      # 1TCM
sens_lungs_S = np.zeros((3, 5, 11901))      # 1TCM
sens_liver_S = np.zeros((3, 6, 11901))      # 2TCM
sens_kidneys_S = np.zeros((3, 6, 11901))     # 2TCM       
sens_femur_S = np.zeros((3, 6, 11901))      # 2TCM

sens_heart_params_S = np.zeros((4, 3))
sens_lungs_params_S = np.zeros((4, 3))
sens_liver_params_S = np.zeros((5, 3))
sens_kidneys_params_S = np.zeros((5, 3))
sens_femur_params_S = np.zeros((5, 3))

sens_heart_err_S = np.zeros((4, 3))
sens_lungs_err_S = np.zeros((4, 3))
sens_liver_err_S = np.zeros((5, 3))
sens_kidneys_err_S = np.zeros((5, 3))
sens_femur_err_S = np.zeros((5, 3))

# AA05 shenanigans

sens_heart_AA05_S = np.zeros((1, 5, 5901))      # 1TCM
sens_lungs_AA05_S = np.zeros((1, 5, 5901))      # 1TCM
sens_liver_AA05_S = np.zeros((1, 6, 5901))      # 2TCM
sens_kidneys_AA05_S = np.zeros((1, 6, 5901))     # 2TCM  
sens_femur_AA05_S = np.zeros((1, 6, 5901))      # 2TCM

sens_heart_params_AA05_S = np.zeros((4, 1))
sens_lungs_params_AA05_S = np.zeros((4, 1))
sens_liver_params_AA05_S = np.zeros((5, 1))
sens_kidneys_params_AA05_S = np.zeros((5, 1))
sens_femur_params_AA05_S = np.zeros((5, 1))

sens_heart_err_AA05_S = np.zeros((4, 1))
sens_lungs_err_AA05_S = np.zeros((4, 1))
sens_liver_err_AA05_S = np.zeros((5, 1))
sens_kidneys_err_AA05_S = np.zeros((5, 1))
sens_femur_err_AA05_S = np.zeros((5, 1))

#############################################################

folder = 'Mouse_NH3_Fits'
sub_folder = 'Model_Fits/Final_FINAL_SF_fixed'

state = 'Rest'
#model = '2TCM'

Rest_mice = ['AA01_Rest', 'AA02_Rest', 'AA03_Rest', 'AA04_Rest', 'AA05_Rest']
Stress_mice = ['AA01_Stress', 'AA02_Stress', 'AA03_Stress', 'AA05_Stress']
all_mice = ['AA01_Rest', 'AA01_Stress', 'AA02_Rest', 'AA02_Stress', 'AA03_Rest', 'AA03_Stress', 'AA04_Rest', 'AA05_Rest', 'AA05_Stress']

if state == 'Rest':
    mice = Rest_mice
elif state == 'Stress':
    mice = Stress_mice
elif state == 'Both':
    mice = all_mice


k = 0
for m in mice:
    mouse, mouse_int, mouse_time = input_NH3_data(m, False)

    T_f, T0, dt, t_array = mouse_time

    C0_orig = mouse_int.Vena_Cava
    C0_orig_time = mouse_int.Time
    y_time = mouse.Time
    y_0 = mouse.Vena_Cava
    
    mouse_deg, mouse_int_deg, mouse_time_deg = input_NH3_data(m, True)
    
    T_f_deg, T0_deg, dt_deg, t_array_deg = mouse_time_deg
    C0_deg_time = mouse_int_deg.Time
    C0_deg = mouse_int_deg.Vena_Cava
    y_time_deg = mouse_deg.Time

    tag = f'{m}'
    #organ_plot_nb(mouse_deg, True, tag)        # For visualising relations between tissue curves and blood curve

    # AA05 has weird frametimes
    if m == 'AA05_Rest' or m == 'AA05_Stress':
        frametimes_s = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 190, 220, 265, 355, 475, 565, 625])
        frametimes_m = frametimes_s / 60

        frametimes_mid_s = np.array([5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155, 165, 180, 205, 242.5, 310, 415, 520, 595])
        frametimes_mid_m = frametimes_mid_s / 60

        frame_lengths_s = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 20, 30, 45, 90, 120, 90, 60])
        frame_lengths_m = frame_lengths_s / 60
    
    for organ in ['Heart', 'Lungs', 'Kidneys', 'Liver', 'Femur']:   #['Heart', 'Lungs', 'Kidneys', 'Liver', 'Femur']

        C1_data = mouse_int[organ]
        y_dat = mouse[organ]
        
        C1_data_deg = mouse_int_deg[organ]
        y_dat_deg = mouse_deg[organ] 

        name = f'{m}_{organ}'

        model = model_selection_df.loc[m, organ]

        # scale_factor = scale_factor_df.loc[organ, m]
        # delay = delays_df.loc[organ, m] / 60

        # shift values down by X time steps (0.1 seconds for the interpolated data), add zero to the front
        # for opposite direction pad with average of last 5 values
        # Checking times of each delay shift, it is 1 second not 0.1 second

        if model == '2TCM':
        
            init2 = [0.0, 0.0]

            params = lmfit.Parameters()
            params.add('K1', 1.0, min=0.0, max=10.0)
            params.add('k2', 0.1, min=0.0, max=5.0)
            params.add('k3', 0.1, min=0.0, max=5.0)
            params.add('k4', 0.0, vary = False)                     # This is standard for NH3 
            params.add('vb', 0.1, vary = True, min=0.0, max=1.0)
            
            fit = lmfit.minimize(resid2_weighted, params, args = (C0_orig, mouse, y_time, y_dat, frame_lengths_m, frametimes_mid_m, 'NH3'),  method = 'leastsq', max_nfev = 1000)
            #lmfit.report_fit(fit)

            residuals = resid2_weighted(fit.params, C0_orig, mouse, y_time, y_dat, frame_lengths_m, frametimes_mid_m, 'NH3')
            
            ststics = [['Number of evaluations:', fit.nfev], ['Number of data points:', fit.ndata], ['Chi-Squared:', fit.chisqr], ['Reduced Chi-Squared:', fit.redchi], 
            ['Akaike Information Criterion:', fit.aic], ['Bayesian Information Criterion:', fit.bic]]

            vrbles = [[fit.var_names[0], fit.params['K1'].value, fit.params['K1'].stderr], 
            [fit.var_names[1], fit.params['k2'].value, fit.params['k2'].stderr],
            [fit.var_names[2], fit.params['k3'].value, fit.params['k3'].stderr],
            ['k4', fit.params['k4'].value, fit.params['k4'].stderr],
            [fit.var_names[3], fit.params['vb'].value, fit.params['vb'].stderr]]
            
            if fit.params['K1'].correl != None:
                correls = [[fit.params['K1'].correl['k2'], fit.params['K1'].correl['k3'], 0, fit.params['K1'].correl['vb']],
                        [fit.params['k2'].correl['k3'], 0, fit.params['k2'].correl['vb']], 
                        [0, fit.params['k3'].correl['vb']],
                        [0]]
            else:
                correls = []
                
            with open(f'{folder}\{sub_folder}/{name}_2TCM_Stats.csv', 'w', newline = '') as f:
                writer = csv.writer(f)
                writer.writerows(ststics)
                writer.writerow([' '])
                writer.writerows(vrbles)
                writer.writerow([' '])
                writer.writerows(correls)

            deriv_k1, deriv_time_k1 = sensitivity_analysis(fit, 'K1', 0.1, '2TCM')
            deriv_k2, deriv_time_k2 = sensitivity_analysis(fit, 'k2', 0.1, '2TCM')
            deriv_k3, deriv_time_k3 = sensitivity_analysis(fit, 'k3', 0.1, '2TCM')
            deriv_vb, deriv_time_vb = sensitivity_analysis(fit, 'vb', 0.1, '2TCM')
            deriv_F, deriv_time_F, f_value, f_roots = sensitivity_analysis(fit, 'F', 0.1, '2TCM')

            if m in Rest_mice:
                if m == 'AA05_Rest':
                    if organ == 'Heart':
                        sens_heart_AA05_R[0][0] = deriv_time_k1
                        sens_heart_AA05_R[0][1] = deriv_k1
                        sens_heart_AA05_R[0][2] = deriv_k2
                        sens_heart_AA05_R[0][3] = deriv_vb
                        sens_heart_AA05_R[0][4] = deriv_F
                        sens_heart_AA05_R[0][5] = deriv_k3

                        sens_heart_params_AA05_R[:, 0] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['k3'].value,fit.params['vb'].value, f_value]
                        sens_heart_err_AA05_R[:, 0] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['k3'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Lungs':
                        sens_lungs_AA05_R[0][0] = deriv_time_k1
                        sens_lungs_AA05_R[0][1] = deriv_k1
                        sens_lungs_AA05_R[0][2] = deriv_k2
                        sens_lungs_AA05_R[0][3] = deriv_vb
                        sens_lungs_AA05_R[0][4] = deriv_F
                        sens_lungs_AA05_R[0][5] = deriv_k3

                        sens_lungs_params_AA05_R[:, 0] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['k3'].value,fit.params['vb'].value, f_value]
                        sens_lungs_err_AA05_R[:, 0] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['k3'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Liver':
                        sens_liver_AA05_R[0][0] = deriv_time_k1
                        sens_liver_AA05_R[0][1] = deriv_k1
                        sens_liver_AA05_R[0][2] = deriv_k2
                        sens_liver_AA05_R[0][3] = deriv_vb
                        sens_liver_AA05_R[0][4] = deriv_F
                        sens_liver_AA05_R[0][5] = deriv_k3

                        sens_liver_params_AA05_R[:, 0] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['k3'].value,fit.params['vb'].value, f_value]
                        sens_liver_err_AA05_R[:, 0] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['k3'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Kidneys':
                        sens_kidneys_AA05_R[0][0] = deriv_time_k1
                        sens_kidneys_AA05_R[0][1] = deriv_k1
                        sens_kidneys_AA05_R[0][2] = deriv_k2
                        sens_kidneys_AA05_R[0][3] = deriv_vb
                        sens_kidneys_AA05_R[0][4] = deriv_F
                        sens_kidneys_AA05_R[0][5] = deriv_k3

                        sens_kidneys_params_AA05_R[:, 0] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['k3'].value,fit.params['vb'].value, f_value]
                        sens_kidneys_err_AA05_R[:, 0] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['k3'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Femur':
                        sens_femur_AA05_R[0][0] = deriv_time_k1
                        sens_femur_AA05_R[0][1] = deriv_k1
                        sens_femur_AA05_R[0][2] = deriv_k2
                        sens_femur_AA05_R[0][3] = deriv_vb
                        sens_femur_AA05_R[0][4] = deriv_F
                        sens_femur_AA05_R[0][5] = deriv_k3

                        sens_femur_params_AA05_R[:, 0] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['k3'].value,fit.params['vb'].value, f_value]
                        sens_femur_err_AA05_R[:, 0] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['k3'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]
                
                else:
                    if organ == 'Heart':
                        sens_heart_R[k][0] = deriv_time_k1
                        sens_heart_R[k][1] = deriv_k1
                        sens_heart_R[k][2] = deriv_k2
                        sens_heart_R[k][3] = deriv_vb
                        sens_heart_R[k][4] = deriv_F
                        sens_heart_R[k][5] = deriv_k3

                        sens_heart_params_R[:, k] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['k3'].value, fit.params['vb'].value, f_value]
                        sens_heart_err_R[:, k] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['k3'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Lungs':
                        sens_lungs_R[k][0] = deriv_time_k1
                        sens_lungs_R[k][1] = deriv_k1
                        sens_lungs_R[k][2] = deriv_k2
                        sens_lungs_R[k][3] = deriv_vb
                        sens_lungs_R[k][4] = deriv_F
                        sens_lungs_R[k][5] = deriv_k3

                        sens_lungs_params_R[:, k] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['k3'].value, fit.params['vb'].value, f_value]
                        sens_lungs_err_R[:, k] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['k3'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Liver':
                        sens_liver_R[k][0] = deriv_time_k1
                        sens_liver_R[k][1] = deriv_k1
                        sens_liver_R[k][2] = deriv_k2
                        sens_liver_R[k][3] = deriv_vb
                        sens_liver_R[k][4] = deriv_F
                        sens_liver_R[k][5] = deriv_k3

                        sens_liver_params_R[:, k] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['k3'].value,fit.params['vb'].value, f_value]
                        sens_liver_err_R[:, k] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['k3'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Kidneys':
                        sens_kidneys_R[k][0] = deriv_time_k1
                        sens_kidneys_R[k][1] = deriv_k1
                        sens_kidneys_R[k][2] = deriv_k2
                        sens_kidneys_R[k][3] = deriv_vb
                        sens_kidneys_R[k][4] = deriv_F
                        sens_kidneys_R[k][5] = deriv_k3

                        sens_kidneys_params_R[:, k] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['k3'].value,fit.params['vb'].value, f_value]
                        sens_kidneys_err_R[:, k] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['k3'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Femur':
                        sens_femur_R[k][0] = deriv_time_k1
                        sens_femur_R[k][1] = deriv_k1
                        sens_femur_R[k][2] = deriv_k2
                        sens_femur_R[k][3] = deriv_vb
                        sens_femur_R[k][4] = deriv_F
                        sens_femur_R[k][5] = deriv_k3

                        sens_femur_params_R[:, k] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['k3'].value,fit.params['vb'].value, f_value]
                        sens_femur_err_R[:, k] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['k3'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]
                
            elif m in Stress_mice:
                if m == 'AA05_Stress':
                    if organ == 'Heart':
                        sens_heart_AA05_S[0][0] = deriv_time_k1
                        sens_heart_AA05_S[0][1] = deriv_k1
                        sens_heart_AA05_S[0][2] = deriv_k2
                        sens_heart_AA05_S[0][3] = deriv_vb
                        sens_heart_AA05_S[0][4] = deriv_F
                        sens_heart_AA05_S[0][5] = deriv_k3

                        sens_heart_params_AA05_S[:, 0] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['k3'].value,fit.params['vb'].value, f_value]
                        sens_heart_err_AA05_S[:, 0] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['k3'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Lungs':
                        sens_lungs_AA05_S[0][0] = deriv_time_k1
                        sens_lungs_AA05_S[0][1] = deriv_k1
                        sens_lungs_AA05_S[0][2] = deriv_k2
                        sens_lungs_AA05_S[0][3] = deriv_vb
                        sens_lungs_AA05_S[0][4] = deriv_F
                        sens_lungs_AA05_S[0][5] = deriv_k3

                        sens_lungs_params_AA05_S[:, 0] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['k3'].value,fit.params['vb'].value, f_value]
                        sens_lungs_err_AA05_S[:, 0] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['k3'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Liver':
                        sens_liver_AA05_S[0][0] = deriv_time_k1
                        sens_liver_AA05_S[0][1] = deriv_k1
                        sens_liver_AA05_S[0][2] = deriv_k2
                        sens_liver_AA05_S[0][3] = deriv_vb
                        sens_liver_AA05_S[0][4] = deriv_F
                        sens_liver_AA05_S[0][5] = deriv_k3

                        sens_liver_params_AA05_S[:, 0] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['k3'].value,fit.params['vb'].value, f_value]
                        sens_liver_err_AA05_S[:, 0] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['k3'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Kidneys':
                        sens_kidneys_AA05_S[0][0] = deriv_time_k1
                        sens_kidneys_AA05_S[0][1] = deriv_k1
                        sens_kidneys_AA05_S[0][2] = deriv_k2
                        sens_kidneys_AA05_S[0][3] = deriv_vb
                        sens_kidneys_AA05_S[0][4] = deriv_F
                        sens_kidneys_AA05_S[0][5] = deriv_k3

                        sens_kidneys_params_AA05_S[:, 0] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['k3'].value,fit.params['vb'].value, f_value]
                        sens_kidneys_err_AA05_S[:, 0] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['k3'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Femur':
                        sens_femur_AA05_S[0][0] = deriv_time_k1
                        sens_femur_AA05_S[0][1] = deriv_k1
                        sens_femur_AA05_S[0][2] = deriv_k2
                        sens_femur_AA05_S[0][3] = deriv_vb
                        sens_femur_AA05_S[0][4] = deriv_F
                        sens_femur_AA05_S[0][5] = deriv_k3

                        sens_femur_params_AA05_S[:, 0] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['k3'].value,fit.params['vb'].value, f_value]
                        sens_femur_err_AA05_S[:, 0] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['k3'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                else:
                    if organ == 'Heart':
                        sens_heart_S[k][0] = deriv_time_k1
                        sens_heart_S[k][1] = deriv_k1
                        sens_heart_S[k][2] = deriv_k2
                        sens_heart_S[k][3] = deriv_vb
                        sens_heart_S[k][4] = deriv_F
                        sens_heart_S[k][5] = deriv_k3

                        sens_heart_params_S[:, k] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['k3'].value,fit.params['vb'].value, f_value]
                        sens_heart_err_S[:, k] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['k3'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Lungs':
                        sens_lungs_S[k][0] = deriv_time_k1
                        sens_lungs_S[k][1] = deriv_k1
                        sens_lungs_S[k][2] = deriv_k2
                        sens_lungs_S[k][3] = deriv_vb
                        sens_lungs_S[k][4] = deriv_F
                        sens_lungs_S[k][5] = deriv_k3

                        sens_lungs_params_S[:, k] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['k3'].value,fit.params['vb'].value, f_value]
                        sens_lungs_err_S[:, k] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['k3'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Liver':
                        sens_liver_S[k][0] = deriv_time_k1
                        sens_liver_S[k][1] = deriv_k1
                        sens_liver_S[k][2] = deriv_k2
                        sens_liver_S[k][3] = deriv_vb
                        sens_liver_S[k][4] = deriv_F
                        sens_liver_S[k][5] = deriv_k3

                        sens_liver_params_S[:, k] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['k3'].value,fit.params['vb'].value, f_value]
                        sens_liver_err_S[:, k] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['k3'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Kidneys':
                        sens_kidneys_S[k][0] = deriv_time_k1
                        sens_kidneys_S[k][1] = deriv_k1
                        sens_kidneys_S[k][2] = deriv_k2
                        sens_kidneys_S[k][3] = deriv_vb
                        sens_kidneys_S[k][4] = deriv_F
                        sens_kidneys_S[k][5] = deriv_k3

                        sens_kidneys_params_S[:, k] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['k3'].value,fit.params['vb'].value, f_value]
                        sens_kidneys_err_S[:, k] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['k3'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Femur':
                        sens_femur_S[k][0] = deriv_time_k1
                        sens_femur_S[k][1] = deriv_k1
                        sens_femur_S[k][2] = deriv_k2
                        sens_femur_S[k][3] = deriv_vb
                        sens_femur_S[k][4] = deriv_F
                        sens_femur_S[k][5] = deriv_k3

                        sens_femur_params_S[:, k] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['k3'].value,fit.params['vb'].value, f_value]
                        sens_femur_err_S[:, k] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['k3'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

            a = ps_df.loc[organ, 'a']
            b = ps_df.loc[organ, 'b']
            print(f'{organ}:')
            print(f'a = {a}, b = {b}, f_roots = {f_roots}')
            print(f'Flow value of {organ} for mouse {m} = {f_value:.3f}')

            val1, val2, val3, val4, valvb = [fit.params['K1'].value, fit.params['k2'].value,  fit.params['k3'].value, fit.params['k4'].value, fit.params['vb'].value]

            x1, t1 = RK4(comp_ode_model2, C0_orig, init2, dt, T_f, T0, [fit.params['K1'].value, fit.params['k2'].value, fit.params['k3'].value, fit.params['k4'].value, fit.params['vb'].value])


            plt.scatter(y_time, mouse.Vena_Cava, s = 7, label = 'Blood', color = 'g')
            plt.scatter(y_time, y_dat, s = 7, label = 'Tissue', color = 'r')
            plt.plot(t1, ((1 - fit.params['vb'].value) * (x1[:, 0] + x1[:, 1]) + fit.params['vb'].value * C0_orig), label = 'Model Fit', color = 'b')
            plt.title(f'{m}, {organ} \n K1 = {val1:.3f}, k2 = {val2:.3f},  k3 = {val3:.3f}, vb = {valvb:.3f}')
            plt.xlabel('Time (minutes)')
            plt.ylabel('SUV (g/mL)')
            plt.legend(loc = 7, fontsize = 'x-small')
            plt.savefig(f'{folder}\{sub_folder}/{name}_2TCM')
            plt.close()

            plt.scatter(y_time, residuals, s = 7, label = 'Residuals', color = 'b')
            plt.title(f'Residuals of 2TCM fit for {m}, {organ} \n K1 = {val1:.3f}, k2 = {val2:.3f}, k3 = {val3:.3f}, vb = {valvb:.3f}')
            plt.axhline(y=0.0, color='r', linestyle='-')
            plt.xlabel('Time (minutes)')
            plt.ylabel('Residuals')
            plt.legend(loc = 7, fontsize = 'x-small')
            plt.savefig(f'{folder}\{sub_folder}/Residuals/{name}_2TCM_Residuals')
            plt.close()

            plt.plot(y_time, mouse.Vena_Cava, label = 'Blood', color = 'g')
            plt.plot(y_time, y_dat, label = 'Tissue', color = 'r')
            plt.title(f'Comparison of blood curve and tissue curve \n for {m}, {organ}')
            plt.xlabel('Time (minutes)')
            plt.ylabel('SUV (g/mL)')
            plt.legend(loc = 7, fontsize = 'x-small')
            plt.savefig(f'{folder}\{sub_folder}/Blood_Tissue_Comparison/{name}_2TCM')
            plt.close()

        if model == '1TCM':
            init = [0.0]

            params = lmfit.Parameters()
            params.add('K1', 1.0, min=0.0, max=10.0)
            params.add('k2', 0.1, min=0.0, max=5.0)
            params.add('vb', 0.1, vary = True, min=0.0, max=1.0)
            
            fit = lmfit.minimize(resid1_weighted, params, args = (C0_orig, mouse, y_time, y_dat, frame_lengths_m, frametimes_mid_m, 'NH3'), method = 'leastsq', max_nfev = 1000)
            #lmfit.report_fit(fit)

            residuals = resid1_weighted(fit.params, C0_orig, mouse, y_time, y_dat, frame_lengths_m, frametimes_mid_m, 'NH3')
            
            ststics = [['Number of evaluations:', fit.nfev], ['Number of data points:', fit.ndata], ['Chi-Squared:', fit.chisqr], ['Reduced Chi-Squared:', fit.redchi], 
            ['Akaike Information Criterion:', fit.aic], ['Bayesian Information Criterion:', fit.bic]]

            vrbles = [[fit.var_names[0], fit.params['K1'].value, fit.params['K1'].stderr], 
            [fit.var_names[1], fit.params['k2'].value, fit.params['k2'].stderr],
            [fit.var_names[2], fit.params['vb'].value, fit.params['vb'].stderr]]
            
            if fit.params['K1'].correl != None:
                correls = [[fit.params['K1'].correl['k2'], fit.params['K1'].correl['vb']], [fit.params['k2'].correl['vb']]]
            else:
                correls = []
                
            with open(f'{folder}/{sub_folder}/{name}_1TCM_Stats.csv', 'w', newline = '') as f:
                writer = csv.writer(f)
                writer.writerows(ststics)
                writer.writerow([' '])
                writer.writerows(vrbles)
                writer.writerow([' '])
                writer.writerows(correls)

            deriv_k1, deriv_time_k1 = sensitivity_analysis(fit, 'K1', 0.1, '1TCM')
            deriv_k2, deriv_time_k2 = sensitivity_analysis(fit, 'k2', 0.1, '1TCM')
            deriv_vb, deriv_time_vb = sensitivity_analysis(fit, 'vb', 0.1, '1TCM')
            deriv_F, deriv_time_F, f_value, f_roots = sensitivity_analysis(fit, 'F', 0.1, '1TCM')

            if m in Rest_mice:
                if m == 'AA05_Rest':
                    if organ == 'Heart':
                        sens_heart_AA05_R[0][0] = deriv_time_k1
                        sens_heart_AA05_R[0][1] = deriv_k1
                        sens_heart_AA05_R[0][2] = deriv_k2
                        sens_heart_AA05_R[0][3] = deriv_vb
                        sens_heart_AA05_R[0][4] = deriv_F

                        sens_heart_params_AA05_R[:, 0] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_heart_err_AA05_R[:, 0] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Lungs':
                        sens_lungs_AA05_R[0][0] = deriv_time_k1
                        sens_lungs_AA05_R[0][1] = deriv_k1
                        sens_lungs_AA05_R[0][2] = deriv_k2
                        sens_lungs_AA05_R[0][3] = deriv_vb
                        sens_lungs_AA05_R[0][4] = deriv_F

                        sens_lungs_params_AA05_R[:, 0] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_lungs_err_AA05_R[:, 0] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Liver':
                        sens_liver_AA05_R[0][0] = deriv_time_k1
                        sens_liver_AA05_R[0][1] = deriv_k1
                        sens_liver_AA05_R[0][2] = deriv_k2
                        sens_liver_AA05_R[0][3] = deriv_vb
                        sens_liver_AA05_R[0][4] = deriv_F

                        sens_liver_params_AA05_R[:, 0] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_liver_err_AA05_R[:, 0] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Kidneys':
                        sens_kidneys_AA05_R[0][0] = deriv_time_k1
                        sens_kidneys_AA05_R[0][1] = deriv_k1
                        sens_kidneys_AA05_R[0][2] = deriv_k2
                        sens_kidneys_AA05_R[0][3] = deriv_vb
                        sens_kidneys_AA05_R[0][4] = deriv_F

                        sens_kidneys_params_AA05_R[:, 0] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_kidneys_err_AA05_R[:, 0] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Femur':
                        sens_femur_AA05_R[0][0] = deriv_time_k1
                        sens_femur_AA05_R[0][1] = deriv_k1
                        sens_femur_AA05_R[0][2] = deriv_k2
                        sens_femur_AA05_R[0][3] = deriv_vb
                        sens_femur_AA05_R[0][4] = deriv_F

                        sens_femur_params_AA05_R[:, 0] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_femur_err_AA05_R[:, 0] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                else:
                    if organ == 'Heart':
                        sens_heart_R[k][0] = deriv_time_k1
                        sens_heart_R[k][1] = deriv_k1
                        sens_heart_R[k][2] = deriv_k2
                        sens_heart_R[k][3] = deriv_vb
                        sens_heart_R[k][4] = deriv_F

                        sens_heart_params_R[:, k] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_heart_err_R[:, k] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Lungs':
                        sens_lungs_R[k][0] = deriv_time_k1
                        sens_lungs_R[k][1] = deriv_k1
                        sens_lungs_R[k][2] = deriv_k2
                        sens_lungs_R[k][3] = deriv_vb
                        sens_lungs_R[k][4] = deriv_F

                        sens_lungs_params_R[:, k] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_lungs_err_R[:, k] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Liver':
                        sens_liver_R[k][0] = deriv_time_k1
                        sens_liver_R[k][1] = deriv_k1
                        sens_liver_R[k][2] = deriv_k2
                        sens_liver_R[k][3] = deriv_vb
                        sens_liver_R[k][4] = deriv_F

                        sens_liver_params_R[:, k] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_liver_err_R[:, k] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Kidneys':
                        sens_kidneys_R[k][0] = deriv_time_k1
                        sens_kidneys_R[k][1] = deriv_k1
                        sens_kidneys_R[k][2] = deriv_k2
                        sens_kidneys_R[k][3] = deriv_vb
                        sens_kidneys_R[k][4] = deriv_F

                        sens_kidneys_params_R[:, k] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_kidneys_err_R[:, k] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Femur':
                        sens_femur_R[k][0] = deriv_time_k1
                        sens_femur_R[k][1] = deriv_k1
                        sens_femur_R[k][2] = deriv_k2
                        sens_femur_R[k][3] = deriv_vb
                        sens_femur_R[k][4] = deriv_F

                        sens_femur_params_R[:, k] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_femur_err_R[:, k] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

            elif m in Stress_mice:
                if m == 'AA05_Stress':
                    if organ == 'Heart':
                        sens_heart_AA05_S[0][0] = deriv_time_k1
                        sens_heart_AA05_S[0][1] = deriv_k1
                        sens_heart_AA05_S[0][2] = deriv_k2
                        sens_heart_AA05_S[0][3] = deriv_vb
                        sens_heart_AA05_S[0][4] = deriv_F

                        sens_heart_params_AA05_S[:, 0] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_heart_err_AA05_S[:, 0] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Lungs':
                        sens_lungs_AA05_S[0][0] = deriv_time_k1
                        sens_lungs_AA05_S[0][1] = deriv_k1
                        sens_lungs_AA05_S[0][2] = deriv_k2
                        sens_lungs_AA05_S[0][3] = deriv_vb
                        sens_lungs_AA05_S[0][4] = deriv_F

                        sens_lungs_params_AA05_S[:, 0] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_lungs_err_AA05_S[:, 0] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Liver':
                        sens_liver_AA05_S[0][0] = deriv_time_k1
                        sens_liver_AA05_S[0][1] = deriv_k1
                        sens_liver_AA05_S[0][2] = deriv_k2
                        sens_liver_AA05_S[0][3] = deriv_vb
                        sens_liver_AA05_S[0][4] = deriv_F

                        sens_liver_params_AA05_S[:, 0] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_liver_err_AA05_S[:, 0] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Kidneys':
                        sens_kidneys_AA05_S[0][0] = deriv_time_k1
                        sens_kidneys_AA05_S[0][1] = deriv_k1
                        sens_kidneys_AA05_S[0][2] = deriv_k2
                        sens_kidneys_AA05_S[0][3] = deriv_vb
                        sens_kidneys_AA05_S[0][4] = deriv_F

                        sens_kidneys_params_AA05_S[:, 0] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_kidneys_err_AA05_S[:, 0] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Femur':
                        sens_femur_AA05_S[0][0] = deriv_time_k1
                        sens_femur_AA05_S[0][1] = deriv_k1
                        sens_femur_AA05_S[0][2] = deriv_k2
                        sens_femur_AA05_S[0][3] = deriv_vb
                        sens_femur_AA05_S[0][4] = deriv_F

                        sens_femur_params_AA05_S[:, 0] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_femur_err_AA05_S[:, 0] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                else:
                    if organ == 'Heart':
                        sens_heart_S[k][0] = deriv_time_k1
                        sens_heart_S[k][1] = deriv_k1
                        sens_heart_S[k][2] = deriv_k2
                        sens_heart_S[k][3] = deriv_vb
                        sens_heart_S[k][4] = deriv_F

                        sens_heart_params_S[:, k] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_heart_err_S[:, k] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Lungs':
                        sens_lungs_S[k][0] = deriv_time_k1
                        sens_lungs_S[k][1] = deriv_k1
                        sens_lungs_S[k][2] = deriv_k2
                        sens_lungs_S[k][3] = deriv_vb
                        sens_lungs_S[k][4] = deriv_F

                        sens_lungs_params_S[:, k] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_lungs_err_S[:, k] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Liver':
                        sens_liver_S[k][0] = deriv_time_k1
                        sens_liver_S[k][1] = deriv_k1
                        sens_liver_S[k][2] = deriv_k2
                        sens_liver_S[k][3] = deriv_vb
                        sens_liver_S[k][4] = deriv_F

                        sens_liver_params_S[:, k] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_liver_err_S[:, k] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Kidneys':
                        sens_kidneys_S[k][0] = deriv_time_k1
                        sens_kidneys_S[k][1] = deriv_k1
                        sens_kidneys_S[k][2] = deriv_k2
                        sens_kidneys_S[k][3] = deriv_vb
                        sens_kidneys_S[k][4] = deriv_F

                        sens_kidneys_params_S[:, k] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_kidneys_err_S[:, k] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Femur':
                        sens_femur_S[k][0] = deriv_time_k1
                        sens_femur_S[k][1] = deriv_k1
                        sens_femur_S[k][2] = deriv_k2
                        sens_femur_S[k][3] = deriv_vb
                        sens_femur_S[k][4] = deriv_F

                        sens_femur_params_S[:, k] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_femur_err_S[:, k] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]
            
            a = ps_df.loc[organ, 'a']
            b = ps_df.loc[organ, 'b']
            print(f'{organ}:')
            print(f'a = {a}, b = {b}, f_roots = {f_roots}')
            print(f'Flow value of {organ} for mouse {m} = {f_value:.3f}')

            x1, t1 = RK4(comp_ode_model1, C0_orig, init, dt, T_f, T0, [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value])

            val1, val2, valvb = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value]

            plt.scatter(y_time, mouse.Vena_Cava, s = 7, label = 'Blood', color = 'g')
            plt.scatter(y_time, y_dat, s = 7, label = 'Tissue', color = 'r')
            plt.plot(t1, ((1 - fit.params['vb'].value) * x1[:, 0] + fit.params['vb'].value * C0_orig), label = 'Model Fit', color = 'b')
            plt.title(f'{m}, {organ} \n K1 = {val1:.3f}, k2 = {val2:.3f}, vb = {valvb:.3f}')
            plt.xlabel('Time (minutes)')
            plt.ylabel('SUV (g/mL)')
            plt.legend(loc = 7, fontsize = 'x-small')
            plt.savefig(f'{folder}\{sub_folder}/{name}_1TCM')
            plt.close()

            plt.scatter(y_time, residuals, s = 7, label = 'Residuals', color = 'b')
            plt.title(f'Residuals of 1TCM fit for {m}, {organ} \n K1 = {val1:.3f}, k2 = {val2:.3f}, vb = {valvb:.3f}')
            plt.axhline(y=0.0, color='r', linestyle='-')
            plt.xlabel('Time (minutes)')
            plt.ylabel('Residuals')
            plt.legend(loc = 7, fontsize = 'x-small')
            plt.savefig(f'{folder}\{sub_folder}/Residuals/{name}_1TCM_Residuals')
            plt.close()

            plt.plot(y_time, mouse.Vena_Cava, label = 'Blood', color = 'g')
            plt.plot(y_time, y_dat, label = 'Tissue', color = 'r')
            plt.title(f'Comparison of blood curve and tissue curve \n for {m}, {organ}')
            plt.xlabel('Time (minutes)')
            plt.ylabel('SUV (g/mL)')
            plt.legend(loc = 7, fontsize = 'x-small')
            plt.savefig(f'{folder}\{sub_folder}/Blood_Tissue_Comparison/{name}_2TCM')
            plt.close()

        if model == 'Degrado':
            init = [0.0]

            params = lmfit.Parameters()
            params.add('K1', 1.0, min=0.0, max=10.0)
            params.add('k2', 0.1, min=0.0, max=5.0)
            params.add('vb', 0.1, vary = True, min=0.0, max=1.0)
            
            fit = lmfit.minimize(resid1_deg_weighted, params, args = (C0_deg, mouse_deg, y_time_deg, y_dat_deg, y_dat, frame_lengths_m, frametimes_mid_m, 'NH3'), method = 'leastsq', max_nfev = 1000)
            #lmfit.report_fit(fit)

            residuals = resid1_deg_weighted(fit.params, C0_deg, mouse_deg, y_time_deg, y_dat_deg, y_dat, frame_lengths_m, frametimes_mid_m, 'NH3')

            
            ststics = [['Number of evaluations:', fit.nfev], ['Number of data points:', fit.ndata], ['Chi-Squared:', fit.chisqr], ['Reduced Chi-Squared:', fit.redchi], 
            ['Akaike Information Criterion:', fit.aic], ['Bayesian Information Criterion:', fit.bic]]

            vrbles = [[fit.var_names[0], fit.params['K1'].value, fit.params['K1'].stderr], 
            [fit.var_names[1], fit.params['k2'].value, fit.params['k2'].stderr],
            [fit.var_names[2], fit.params['vb'].value, fit.params['vb'].stderr]]
            
            if fit.params['K1'].correl != None:
                correls = [[fit.params['K1'].correl['k2'], fit.params['K1'].correl['vb']], [fit.params['k2'].correl['vb']]]
            else:
                correls = []
                
            with open(f'{folder}\{sub_folder}/{name}_Degrado_Stats.csv', 'w', newline = '') as f:
                writer = csv.writer(f)
                writer.writerows(ststics)
                writer.writerow([' '])
                writer.writerows(vrbles)
                writer.writerow([' '])
                writer.writerows(correls)

            deriv_k1, deriv_time_k1 = sensitivity_analysis(fit, 'K1', 0.1, 'Degrado')
            deriv_k2, deriv_time_k2 = sensitivity_analysis(fit, 'k2', 0.1, 'Degrado')
            deriv_vb, deriv_time_vb = sensitivity_analysis(fit, 'vb', 0.1, 'Degrado')
            deriv_F, deriv_time_F, f_value, f_roots = sensitivity_analysis(fit, 'F', 0.1, 'Degrado')

            if m in Rest_mice:
                if m == 'AA05_Rest':
                    if organ == 'Heart':
                        sens_heart_AA05_R[0][0] = deriv_time_k1
                        sens_heart_AA05_R[0][1] = deriv_k1
                        sens_heart_AA05_R[0][2] = deriv_k2
                        sens_heart_AA05_R[0][3] = deriv_vb
                        sens_heart_AA05_R[0][4] = deriv_F

                        sens_heart_params_AA05_R[:, 0] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_heart_err_AA05_R[:, 0] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Lungs':
                        sens_lungs_AA05_R[0][0] = deriv_time_k1
                        sens_lungs_AA05_R[0][1] = deriv_k1
                        sens_lungs_AA05_R[0][2] = deriv_k2
                        sens_lungs_AA05_R[0][3] = deriv_vb
                        sens_lungs_AA05_R[0][4] = deriv_F

                        sens_lungs_params_AA05_R[:, 0] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_lungs_err_AA05_R[:, 0] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Liver':
                        sens_liver_AA05_R[0][0] = deriv_time_k1
                        sens_liver_AA05_R[0][1] = deriv_k1
                        sens_liver_AA05_R[0][2] = deriv_k2
                        sens_liver_AA05_R[0][3] = deriv_vb
                        sens_liver_AA05_R[0][4] = deriv_F

                        sens_liver_params_AA05_R[:, 0] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_liver_err_AA05_R[:, 0] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Kidneys':
                        sens_kidneys_AA05_R[0][0] = deriv_time_k1
                        sens_kidneys_AA05_R[0][1] = deriv_k1
                        sens_kidneys_AA05_R[0][2] = deriv_k2
                        sens_kidneys_AA05_R[0][3] = deriv_vb
                        sens_kidneys_AA05_R[0][4] = deriv_F

                        sens_kidneys_params_AA05_R[:, 0] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_kidneys_err_AA05_R[:, 0] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Femur':
                        sens_femur_AA05_R[0][0] = deriv_time_k1
                        sens_femur_AA05_R[0][1] = deriv_k1
                        sens_femur_AA05_R[0][2] = deriv_k2
                        sens_femur_AA05_R[0][3] = deriv_vb
                        sens_femur_AA05_R[0][4] = deriv_F

                        sens_femur_params_AA05_R[:, 0] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_femur_err_AA05_R[:, 0] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                else:
                    if organ == 'Heart':
                        sens_heart_R[k][0] = deriv_time_k1
                        sens_heart_R[k][1] = deriv_k1
                        sens_heart_R[k][2] = deriv_k2
                        sens_heart_R[k][3] = deriv_vb
                        sens_heart_R[k][4] = deriv_F

                        sens_heart_params_R[:, k] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_heart_err_R[:, k] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Lungs':
                        sens_lungs_R[k][0] = deriv_time_k1
                        sens_lungs_R[k][1] = deriv_k1
                        sens_lungs_R[k][2] = deriv_k2
                        sens_lungs_R[k][3] = deriv_vb
                        sens_lungs_R[k][4] = deriv_F

                        sens_lungs_params_R[:, k] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_lungs_err_R[:, k] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Liver':
                        sens_liver_R[k][0] = deriv_time_k1
                        sens_liver_R[k][1] = deriv_k1
                        sens_liver_R[k][2] = deriv_k2
                        sens_liver_R[k][3] = deriv_vb
                        sens_liver_R[k][4] = deriv_F

                        sens_liver_params_R[:, k] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_liver_err_R[:, k] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Kidneys':
                        sens_kidneys_R[k][0] = deriv_time_k1
                        sens_kidneys_R[k][1] = deriv_k1
                        sens_kidneys_R[k][2] = deriv_k2
                        sens_kidneys_R[k][3] = deriv_vb
                        sens_kidneys_R[k][4] = deriv_F

                        sens_kidneys_params_R[:, k] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_kidneys_err_R[:, k] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Femur':
                        sens_femur_R[k][0] = deriv_time_k1
                        sens_femur_R[k][1] = deriv_k1
                        sens_femur_R[k][2] = deriv_k2
                        sens_femur_R[k][3] = deriv_vb
                        sens_femur_R[k][4] = deriv_F

                        sens_femur_params_R[:, k] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_femur_err_R[:, k] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

            elif m in Stress_mice:
                if m == 'AA05_Stress':
                    if organ == 'Heart':
                        sens_heart_AA05_S[0][0] = deriv_time_k1
                        sens_heart_AA05_S[0][1] = deriv_k1
                        sens_heart_AA05_S[0][2] = deriv_k2
                        sens_heart_AA05_S[0][3] = deriv_vb
                        sens_heart_AA05_S[0][4] = deriv_F

                        sens_heart_params_AA05_S[:, 0] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_heart_err_AA05_S[:, 0] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Lungs':
                        sens_lungs_AA05_S[0][0] = deriv_time_k1
                        sens_lungs_AA05_S[0][1] = deriv_k1
                        sens_lungs_AA05_S[0][2] = deriv_k2
                        sens_lungs_AA05_S[0][3] = deriv_vb
                        sens_lungs_AA05_S[0][4] = deriv_F

                        sens_lungs_params_AA05_S[:, 0] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_lungs_err_AA05_S[:, 0] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Liver':
                        sens_liver_AA05_S[0][0] = deriv_time_k1
                        sens_liver_AA05_S[0][1] = deriv_k1
                        sens_liver_AA05_S[0][2] = deriv_k2
                        sens_liver_AA05_S[0][3] = deriv_vb
                        sens_liver_AA05_S[0][4] = deriv_F

                        sens_liver_params_AA05_S[:, 0] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_liver_err_AA05_S[:, 0] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Kidneys':
                        sens_kidneys_AA05_S[0][0] = deriv_time_k1
                        sens_kidneys_AA05_S[0][1] = deriv_k1
                        sens_kidneys_AA05_S[0][2] = deriv_k2
                        sens_kidneys_AA05_S[0][3] = deriv_vb
                        sens_kidneys_AA05_S[0][4] = deriv_F

                        sens_kidneys_params_AA05_S[:, 0] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_kidneys_err_AA05_S[:, 0] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Femur':
                        sens_femur_AA05_S[0][0] = deriv_time_k1
                        sens_femur_AA05_S[0][1] = deriv_k1
                        sens_femur_AA05_S[0][2] = deriv_k2
                        sens_femur_AA05_S[0][3] = deriv_vb
                        sens_femur_AA05_S[0][4] = deriv_F

                        sens_femur_params_AA05_S[:, 0] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_femur_err_AA05_S[:, 0] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                else:
                    if organ == 'Heart':
                        sens_heart_S[k][0] = deriv_time_k1
                        sens_heart_S[k][1] = deriv_k1
                        sens_heart_S[k][2] = deriv_k2
                        sens_heart_S[k][3] = deriv_vb
                        sens_heart_S[k][4] = deriv_F

                        sens_heart_params_S[:, k] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_heart_err_S[:, k] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Lungs':
                        sens_lungs_S[k][0] = deriv_time_k1
                        sens_lungs_S[k][1] = deriv_k1
                        sens_lungs_S[k][2] = deriv_k2
                        sens_lungs_S[k][3] = deriv_vb
                        sens_lungs_S[k][4] = deriv_F

                        sens_lungs_params_S[:, k] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_lungs_err_S[:, k] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Liver':
                        sens_liver_S[k][0] = deriv_time_k1
                        sens_liver_S[k][1] = deriv_k1
                        sens_liver_S[k][2] = deriv_k2
                        sens_liver_S[k][3] = deriv_vb
                        sens_liver_S[k][4] = deriv_F

                        sens_liver_params_S[:, k] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_liver_err_S[:, k] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Kidneys':
                        sens_kidneys_S[k][0] = deriv_time_k1
                        sens_kidneys_S[k][1] = deriv_k1
                        sens_kidneys_S[k][2] = deriv_k2
                        sens_kidneys_S[k][3] = deriv_vb
                        sens_kidneys_S[k][4] = deriv_F

                        sens_kidneys_params_S[:, k] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_kidneys_err_S[:, k] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

                    elif organ == 'Femur':
                        sens_femur_S[k][0] = deriv_time_k1
                        sens_femur_S[k][1] = deriv_k1
                        sens_femur_S[k][2] = deriv_k2
                        sens_femur_S[k][3] = deriv_vb
                        sens_femur_S[k][4] = deriv_F

                        sens_femur_params_S[:, k] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                        sens_femur_err_S[:, k] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

            a = ps_df.loc[organ, 'a']
            b = ps_df.loc[organ, 'b']
            print(f'{organ}:')
            print(f'a = {a}, b = {b}, f_roots = {f_roots}')
            print(f'Flow value of {organ} for mouse {m} = {f_value:.3f}')

            x1, t1 = RK4(comp_ode_model1_deg, C0_deg, init, dt_deg, T_f_deg, T0_deg, [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value])

            val1, val2, valvb = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value]

            plt.scatter(y_time_deg, mouse_deg.Vena_Cava, s = 7, label = 'Blood', color = 'g')
            plt.scatter(y_time_deg, y_dat_deg, s = 7, label = 'Tissue', color = 'r')
            plt.plot(t1, ((1 - fit.params['vb'].value) * x1[:, 0] + fit.params['vb'].value * C0_deg[:-1]), label = 'Model Fit', color = 'b')
            plt.title(f'{m}, {organ} \n K1 = {val1:.3f}, k2 = {val2:.3f}, vb = {valvb:.3f}')
            plt.xlabel('Time (minutes)')
            plt.ylabel('SUV (g/mL)')
            plt.legend(loc = 7, fontsize = 'x-small')
            plt.savefig(f'{folder}\{sub_folder}/{name}_Degrado')
            plt.close()
            
            plt.scatter(y_time_deg, residuals, s = 7, label = 'Residuals', color = 'b')
            plt.title(f'Residuals of Degrado fit for {m}, {organ} \n K1 = {val1:.3f}, k2 = {val2:.3f}, vb = {valvb:.3f}')
            plt.axhline(y=0.0, color='r', linestyle='-')
            plt.xlabel('Time (minutes)')
            plt.ylabel('Residuals')
            plt.legend(loc = 7, fontsize = 'x-small')
            plt.savefig(f'{folder}\{sub_folder}/Residuals/{name}_Degrado_Residuals')
            plt.close()

            plt.plot(y_time, mouse.Vena_Cava, label = 'Blood', color = 'g')
            plt.plot(y_time, y_dat, label = 'Tissue', color = 'r')
            plt.title(f'Comparison of blood curve and tissue curve \n for {m}, {organ}')
            plt.xlabel('Time (minutes)')
            plt.ylabel('SUV (g/mL)')
            plt.legend(loc = 7, fontsize = 'x-small')
            plt.savefig(f'{folder}\{sub_folder}/Blood_Tissue_Comparison/{name}_2TCM')
            plt.close()

    k += 1


### REST ORGANS ###
if state == 'Rest':
    sensitivity_analysis_display_ch4(folder, sub_folder, sens_heart_R, sens_heart_params_R, sens_heart_err_R, sens_heart_AA05_R, sens_heart_params_AA05_R, sens_heart_err_AA05_R, 'Heart', '2TCM', 'Rest')
    sensitivity_analysis_display_ch4(folder, sub_folder, sens_lungs_R, sens_lungs_params_R, sens_lungs_err_R, sens_lungs_AA05_R, sens_lungs_params_AA05_R, sens_lungs_err_AA05_R, 'Lungs', '1TCM', 'Rest')
    sensitivity_analysis_display_ch4(folder, sub_folder, sens_liver_R, sens_liver_params_R, sens_liver_err_R, sens_liver_AA05_R, sens_liver_params_AA05_R, sens_liver_err_AA05_R, 'Liver', '1TCM', 'Rest')
    sensitivity_analysis_display_ch4(folder, sub_folder, sens_kidneys_R, sens_kidneys_params_R, sens_kidneys_err_R, sens_kidneys_AA05_R, sens_kidneys_params_AA05_R, sens_kidneys_err_AA05_R, 'Kidneys', 'Degrado', 'Rest')
    sensitivity_analysis_display_ch4(folder, sub_folder, sens_femur_R, sens_femur_params_R, sens_femur_err_R, sens_femur_AA05_R, sens_femur_params_AA05_R, sens_femur_err_AA05_R, 'Femur', '1TCM', 'Rest')

    sensitivity_analysis_display_ch4_combined(folder, sub_folder, sens_heart_R, sens_heart_params_R, sens_heart_AA05_R, sens_heart_params_AA05_R, 
                                              sens_lungs_R, sens_lungs_params_R, sens_lungs_AA05_R, sens_lungs_params_AA05_R,
                                              sens_liver_R, sens_liver_params_R, sens_liver_AA05_R, sens_liver_params_AA05_R,
                                              sens_kidneys_R, sens_kidneys_params_R, sens_kidneys_AA05_R, sens_kidneys_params_AA05_R,
                                              sens_femur_R, sens_femur_params_R, sens_femur_AA05_R, sens_femur_params_AA05_R, 'Rest')

### STRESS ORGANS ###
elif state == 'Stress':
    sensitivity_analysis_display_ch4(folder, sub_folder, sens_heart_S, sens_heart_params_S, sens_heart_err_S, sens_heart_AA05_S, sens_heart_params_AA05_S, sens_heart_err_AA05_S, 'Heart', '1TCM', 'Stress')
    sensitivity_analysis_display_ch4(folder, sub_folder, sens_lungs_S, sens_lungs_params_S, sens_lungs_err_S, sens_lungs_AA05_S, sens_lungs_params_AA05_S, sens_lungs_err_AA05_S, 'Lungs', '1TCM', 'Stress')
    sensitivity_analysis_display_ch4(folder, sub_folder, sens_liver_S, sens_liver_params_S, sens_liver_err_S, sens_liver_AA05_S, sens_liver_params_AA05_S, sens_liver_err_AA05_S, 'Liver', '2TCM', 'Stress')
    sensitivity_analysis_display_ch4(folder, sub_folder, sens_kidneys_S, sens_kidneys_params_S, sens_kidneys_err_S, sens_kidneys_AA05_S, sens_kidneys_params_AA05_S, sens_kidneys_err_AA05_S, 'Kidneys', '2TCM', 'Stress')
    sensitivity_analysis_display_ch4(folder, sub_folder, sens_femur_S, sens_femur_params_S, sens_femur_err_S, sens_femur_AA05_S, sens_femur_params_AA05_S, sens_femur_err_AA05_S, 'Femur', '2TCM', 'Stress')

    sensitivity_analysis_display_ch4_combined(folder, sub_folder, sens_heart_S, sens_heart_params_S, sens_heart_AA05_S, sens_heart_params_AA05_S, 
                                              sens_lungs_S, sens_lungs_params_S, sens_lungs_AA05_S, sens_lungs_params_AA05_S,
                                              sens_liver_S, sens_liver_params_S, sens_liver_AA05_S, sens_liver_params_AA05_S,
                                              sens_kidneys_S, sens_kidneys_params_S, sens_kidneys_AA05_S, sens_kidneys_params_AA05_S,
                                              sens_femur_S, sens_femur_params_S, sens_femur_AA05_S, sens_femur_params_AA05_S, 'Stress')