import pandas as pd
import math
import numpy as np
import scipy as sp
import scipy.io as scio
import lmfit
import random
import csv
from time import sleep
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import fsolve


def print_full(x):
    pd.set_option('display.max_rows', 500)
    print(x)
    pd.reset_option('display.max_rows')

def interpolate(x, y, method, T0, Tf, dt):
    x_new = np.arange(T0, Tf + dt, dt)
    f = interp1d(x.astype(float), y.astype(float), kind=method, fill_value = 'extrapolate')
    y_new = f(x_new)

    for i in range(len(y_new)):
        if y_new[i] < 0:
            y_new[i] = 0


    return y_new, f

def organ_plot(data):
    plt.plot(data.Time, data.Heart, label = 'Heart') 
    plt.plot(data.Time, data.Lungs, label = 'Lungs') 
    plt.plot(data.Time, data.Kidneys, label = 'Kidneys') 
    plt.plot(data.Time, data.Bladder, label = 'Bladder')
    plt.plot(data.Time, data.Femur, label = 'Femur') 
    plt.plot(data.Time, data.Liver, label = 'Liver') 
    plt.plot(data.Time, data.Vena_Cava, label = 'Vena Cava')
    plt.title('Comparison of uptake for all organs')
    plt.xlabel('Time (min)')
    plt.ylabel('Tracer concentration')
    plt.legend(loc = 7, fontsize = 'x-small')
    plt.show()

def organ_plot_nb(data):
    plt.plot(data.Time, data.Heart, label = 'Heart') 
    plt.plot(data.Time, data.Lungs, label = 'Lungs') 
    plt.plot(data.Time, data.Kidneys, label = 'Kidneys') 
    plt.plot(data.Time, data.Femur, label = 'Femur') 
    plt.plot(data.Time, data.Liver, label = 'Liver') 
    plt.plot(data.Time, data.Vena_Cava, label = 'Vena Cava')
    plt.title('Comparison of uptake for all organs (except the bladder)')
    plt.xlabel('Time (min)')
    plt.ylabel('Tracer concentration')
    plt.legend(loc = 7, fontsize = 'x-small')
    plt.show()

def aif_plot(data):
    plt.plot(data.Time, data.Vena_Cava, label = 'Vena Cava')
    plt.title('Comparison of uptake for all organs')
    plt.xlabel('Time (min)')
    plt.ylabel('Tracer concentration')
    plt.legend(loc = 7, fontsize = 'x-small')
    plt.show()


def comp_ode_model1(u, C0, t, T0, p):
    K_1, k_2, vb = p
    du = np.zeros(1)
    dt =1/60

    ind = int(round((t - T0)/dt))# -1)

    # dC_1 / dt 
    du = K_1 * C0[ind] - k_2 * u[0] 

    # u[0] = C_1

    return du

def comp_ode_model1_deg(u, C0_deg, t, T0_deg, p):
    K_1, k_2, vb = p

    du = np.zeros(1)
    dt_deg = 1/60
    # test whether any concentrations are negative
    # if len(u[u < -1E-12]) > 0:
    #     print("negative u value!")
    
    ind = int(round((t - T0_deg)/dt_deg))
    
    # if ind == int(round((T_f - T0)/dt)):
    #     return None

    # dC_1 / dt 
    # if ind < 265:
    #     du = K_1 * C0_deg[ind] - k_2 * u[0] 

    du = K_1 * C0_deg[ind] - k_2 * u[0]

    # u[0] = C_1

    return du

def comp_ode_model2(u, C0, t, T0, p):
    K_1, k_2, k_3, k_4, vb = p

    du = np.zeros(2)
    dt = 1/60
    ind = int(round((t - T0)/dt))

    # dC_1 / dt 
    du[0] = K_1 * C0[ind] - (k_2 + k_3) * u[0] + k_4 * u[1]

    # dC_2 / dt
    du[1] = k_3 * u[0] - k_4 * u[1]

    # u[0] = C_1, u[1] = C_2

    return du

def comp_ode_model2_kidney(u, C0, t, T0, p):
    K_1, k_2, k_3, k_4, vb= p

    du = np.zeros(2)
    dt = 1/60

    ind = int(round((t - T0)/dt))

    # dC_1 / dt 
    du[0] = K_1 * C0[ind] - (k_2 + k_3) * u[0] 

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



def resid1_weighted(params, C0, mouse_time, y_time, y_dat, frame_lengths_m, framemidtimes_m, scale_factor):
    K_1 = params['K1'].value
    k_2 = params['k2'].value
    vb = params['vb'].value
    
    p = [K_1, k_2, vb]
    init = [0.0]
    dt = 1/60
    T_f, T0, dt, time = mouse_time
    
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

    decay_const = math.log(2) / 109.771           # minutes 
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

def resid1_deg_weighted(params, C0_deg, mouse_deg, y_time_deg, y_dat_deg, y_dat, frame_lengths_m, framemidtimes_m, scale_factor):
    K_1 = params['K1'].value
    k_2 = params['k2'].value
    vb = params['vb'].value
    
    p = [K_1, k_2, vb]
    init = [0.0]
    dt = 1/60
    T_f_deg, T0_deg, dt_deg, time = time_vars(mouse_deg, dt)

    u_out, t = RK4(comp_ode_model1_deg, C0_deg, init, dt_deg, T_f_deg, T0_deg, p)
    
    annoying = np.array(C0_deg, dtype=float)
    model = (1 - vb) * u_out[:, 0] + vb * annoying


    func = interp1d(np.array(t, dtype=float), np.array(model, dtype=float), kind='cubic', fill_value = 'extrapolate')
    model = func(np.array(y_time_deg, dtype=float))     # This is the model fit refitted into the original 33 time points

    result = integrateTrapezium(frame_lengths_m, y_dat)     # This is the approximate decay corrected PET data (TAC) to be used in the weighting calculating
    
    resids = model - np.array(y_dat_deg, dtype=float)       # This is the plain residuals to be returned from the function after being multiplied by the weights, final five values are to replace any zero values in result

    #scale_factor = 0.71
    decay_const = math.log(2) / 109.771           # minutes 
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

def resid2_weighted(params, C0, mouse, y_time, y_dat, frame_lengths_m, framemidtimes_m, scale_factor):
    K_1 = params['K1'].value
    k_2 = params['k2'].value
    k_3 = params['k3'].value
    k_4 = params['k4'].value
    vb = params['vb'].value

    p = [K_1, k_2, k_3, k_4, vb]
    init2 = [0.0, 0.0]
    dt = 1/60
    T_f, T0, dt, time = time_vars(mouse, dt)

    u_out, t = RK4(comp_ode_model2, C0, init2, dt, T_f, T0, p)

    model = (1 - vb) * (u_out[:, 0] + u_out[:, 1]) + vb * C0

    func = interp1d(np.array(t, dtype=float), np.array(model, dtype=float), kind='cubic', fill_value = 'extrapolate')
    model = func(np.array(y_time, dtype=float))     # This is the model fit refitted into the original 33 time points

    #result = integrateTrapezium(frame_lengths_m, y_dat)     # This is the approximate decay corrected PET data (TAC) to be used in the weighting calculating
    result = np.array(y_dat)

    resids = model - np.array(y_dat, dtype=float)       # This is the plain residuals to be returned from the function after being multiplied by the weights, final five values are to replace any zero values in result

    decay_const = math.log(2) / 109.771           # minutes 
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


def resid2_kidneys_weighted(params, C0, mouse, y_time, y_dat, frame_lengths_m, framemidtimes_m, scale_factor):
    K_1 = params['K1'].value
    k_2 = params['k2'].value
    k_3 = params['k3'].value
    k_4 = params['k4'].value
    vb = params['vb'].value

    p = [K_1, k_2, k_3, k_4, vb]
    init2 = [0.0, 0.0]
    dt = 1/60
    T_f, T0, dt, time = time_vars(mouse, dt)

    u_out, t = RK4(comp_ode_model2_kidney, C0, init2, dt, T_f, T0, p)

    model = (1 - vb) * (u_out[:, 0] + u_out[:,1]) + vb * C0

    func = interp1d(np.array(t, dtype=float), np.array(model, dtype=float), kind='cubic', fill_value = 'extrapolate')
    model = func(np.array(y_time, dtype=float))     # This is the model fit refitted into the original 33 time points

    #result = integrateTrapezium(frame_lengths_m, y_dat)     # This is the approximate decay corrected PET data (TAC) to be used in the weighting calculating
    result = np.array(y_dat)
    #print(result)

    #result2 = np.trapz(y_dat, framemidtimes_m, axis = 1)
    #print(result2)
    
    resids = model - np.array(y_dat, dtype=float)       # This is the plain residuals to be returned from the function after being multiplied by the weights, final five values are to replace any zero values in result

    #scale_factor = 0.71
    dec_const = math.log(2) / 109.771           # minutes 
    frame_dur = np.zeros(len(frame_lengths_m))
    exp_decay = np.zeros(len(frame_lengths_m))

    for i in range(len(frame_lengths_m)):
        frame_dur[i] = frame_lengths_m[i]
        exp_decay[i] = math.exp(- dec_const * framemidtimes_m[i])

        if result[i] == 0:
            result[i] = np.mean(resids[-5:])        # Maybe replace this value with an average of the last 5 residuals, 3 for degrado

    sigma_sq = scale_factor * (result / (frame_dur * exp_decay))
    #sigma_sq = 0.05 * (result / (frame_dur * exp_decay))            # Changed scale factor to 0.05 for comparison with results from PMOD using 0.05 as the scale factor
    weights = 1 / sigma_sq
    weights = np.sqrt(weights)
    weights[np.isnan(weights)] = 0.01
    #print(weights)

    return (weights * resids)


def time_vars(data, dt):
    T_f = data.iloc[-1, 0]
    T0 = data.iloc[0, 0]
    time = np.arange(T0, T_f + dt, dt)

    return T_f, T0, dt, time

def expo(x, A, lambd, line, t_peak):
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

def expo_resid(params_exp):
    coeff = [params_exp['p1'].value, params_exp['p3'].value, params_exp['p5'].value]
    d_const = [params_exp['p2'].value, params_exp['p4'].value, params_exp['p6'].value]
    line = [params_exp['a'].value, params_exp['b'].value]

    model = expo(mouse3_int.Time, coeff, d_const, line, t_peak)

    #print(np.isnan(model - C0).any())

    return model - C0


def step(x):
    # if (x >= data_int.Time.min()) and (x <= data_int.Time.max()):
    #     return step_const*1
    # else:
    #     return 0

    func = interp1d(x.Time, x.Vena_Cava, kind='cubic', fill_value = 'extrapolate')

    counts, error = quad(func, frametimes_m[0], frametimes_m[-1])

    step_const = counts/ (frametimes_m[-1] - frametimes_m[0])

    return step_const * np.ones(len(x.Time))

def const_infusion(data):
    result = np.ones(len(data.Time))
    
    for i in range(len(data.Time)):
        integral = np.trapz(np.array(data.loc[:i, 'Vena_Cava']), x = np.array(data.loc[:i, 'Time']))
        integral2 = integral / (frametimes_m[-1] - frametimes_m[0])
        result[i] = integral2

    return result

def bolus_infusion(data, kbol):
    result = np.ones(len(data.Time))
    
    for i in range(len(data.Time)):
        integral = np.trapz(np.array(data.loc[:i, 'Vena_Cava']), x = np.array(data.loc[:i, 'Time']))
        num = kbol * data.loc[i, 'Vena_Cava'] + integral
        denom = kbol + (frametimes_m[-1] - frametimes_m[0])
        result[i] = num / denom

    return result

def weights_dc(func, data, col, decay):
    frametimes = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 210, 240, 300, 420, 540, 840, 1140, 1440, 1740, 2040, 2340, 2640, 2940, 3240, 3540])
    frametimes = frametimes / 60
    weights = np.zeros(len(frametimes)-1)
    zero_count = 0

    # This won't be the case for mice that start with zero values

    for i in range(len(frametimes)-1):
        if data.iloc[i, 1] == 0:
            weights[i] = 0
            zero_count += 1
        else:
            frame_length = frametimes[i+1] - frametimes[i]
            counts, error = quad(func, frametimes[i], frametimes[i+1], limit = 100)
            corr_fac = (np.exp(decay * frametimes[i]) * decay * frame_length) / (1 - np.exp(-decay * frame_length))

            weights[i] = (frame_length ** 2) / (counts * (corr_fac ** 2))           # Weighting 1
            #weights[i] = frame_length * np.exp(-decay * frametimes[i+1])          # Weighting 2

    return weights, zero_count


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



def noise_gen(aif, bolus_df, sim_aif, ff):
    
    func = interp1d(np.array(bolus_df.iloc[:, 0], dtype=float), np.array(bolus_df.iloc[:, 1], dtype=float), kind='cubic', fill_value = 'extrapolate')
    a = func(np.array(frametimes_mid_m, dtype = float))
    
    bolus_noise = np.array([])

    for i in range(len(frametimes_m) - 1):
        delta_t = frame_lengths_m[i]
        decay = np.exp(-0.0063128158521 * ((frametimes_m[i+1] + frametimes_m[i])/2))
        counts = a[i]
        #counts = aif.iloc[i, 7]

        cv = np.sqrt(ff/(counts * delta_t * decay))
        rn = random.gauss(0, 1)
        noise = cv * rn
        bolus_noise = np.append(bolus_noise, [noise])

    for i in range(len(bolus_noise)):
        if bolus_noise[i] == np.Inf or bolus_noise[i] == np.NINF:
            bolus_noise[i] = 0

    noisy_signal = a * (1 + bolus_noise)

    return noisy_signal, a, bolus_noise #, bolus_integrate


def std_mice(mouse1, mouse2, mouse3, organ):
    std = np.array([])

    df = pd.DataFrame(data = {'Mouse1' : mouse1[organ], 'Mouse2' : mouse2[organ], 'Mouse3' : mouse3[organ]})
    df['Average'] = df.mean(axis = 1)
    mean = np.array(df['Average'])


    for i in range(len(mouse1[organ])):
        row = np.array(df.iloc[i])
        s = np.std(row)
        std = np.append(std, s)

    var1 = np.var(mouse1[organ])
    var2 = np.var(mouse2[organ])
    var3 = np.var(mouse3[organ])
    mean2 = np.mean([var1, var2, var3])
    std2 = np.sqrt(mean2)

    return std, std2, mean

def sensitivity_analysis(data, param_values, param, delta_param, input_function, model, organ):
    h_scale = delta_param
    deriv = None
    t1 = None
    if model == 'Degrado':
        T_f_deg, T0_deg, dt_deg, time = time_vars(data, 1/60)
        init = [0.0]
        if param == 'K1':
            K_1, k_2, vb = param_values

            h = h_scale * K_1
            
            x1, t1 = RK4(comp_ode_model1_deg, input_function, init, dt_deg, T_f_deg, T0_deg, [K_1 + h, k_2, vb])
            x2, t2 = RK4(comp_ode_model1_deg, input_function, init, dt_deg, T_f_deg, T0_deg, [K_1 - h, k_2, vb])

            x1 = (1 - vb) * x1[:, 0] + vb * input_function
            x2 = (1 - vb) * x2[:, 0] + vb * input_function

            deriv = (x1 - x2)/2*h

        elif param == 'k2':
            K_1, k_2, vb = param_values

            h = h_scale * k_2

            x1, t1 = RK4(comp_ode_model1_deg, input_function, init, dt_deg, T_f_deg, T0_deg, [K_1, k_2 + h, vb])
            x2, t2 = RK4(comp_ode_model1_deg, input_function, init, dt_deg, T_f_deg, T0_deg, [K_1, k_2 - h, vb])

            x1 = (1 - vb) * x1[:, 0] + vb * input_function
            x2 = (1 - vb) * x2[:, 0] + vb * input_function

            deriv = (x1 - x2)/2*h

        elif param == 'vb':
            K_1, k_2, vb = param_values

            h = h_scale * vb

            x1, t1 = RK4(comp_ode_model1_deg, input_function, init, dt_deg, T_f_deg, T0_deg, [K_1, k_2, vb])

            x2 = (1 - (vb + h)) * x1[:, 0] + (vb + h) * input_function
            x3 = (1 - (vb - h)) * x1[:, 0] + (vb - h) * input_function

            deriv = (x2 - x3)/2*h

        elif param == 'F':
            K_1, k_2, vb = param_values

            ps = ps_df.loc[organ, 'PS']
            f_values = np.array([])

            for i in np.linspace(-9, 9, 19):
                f_root = fsolve(flow_func, [K_1 + (i/10)*K_1], args = (K_1, ps))
                f = f_root[0]
                f_values = np.append(f_values, f)
            f = find_nearest(f_values, K_1)

            h = h_scale * f

            x1, t1 = RK4(comp_ode_model1_deg, input_function, init, dt_deg, T_f_deg, T0_deg, [k1_flow(f + h, ps), k_2, vb])
            x2, t2 = RK4(comp_ode_model1_deg, input_function, init, dt_deg, T_f_deg, T0_deg, [k1_flow(f - h, ps), k_2, vb])

            x1 = (1 - vb) * x1[:, 0] + vb * input_function
            x2 = (1 - vb) * x2[:, 0] + vb * input_function

            deriv = (x1 - x2)/2*h

            return deriv, t1, f, f_root

    elif model == '1TCM':
        T_f, T0, dt, time = time_vars(data, 1/60)
        init = [0.0]
        if param == 'K1':
            K_1, k_2, vb = param_values

            h = h_scale * K_1
            
            x1, t1 = RK4(comp_ode_model1, input_function, init, dt, T_f, T0, [K_1 + h, k_2, vb])
            x2, t2 = RK4(comp_ode_model1, input_function, init, dt, T_f, T0, [K_1 - h, k_2, vb])

            x1 = (1 - vb) * x1[:, 0] + vb * input_function
            x2 = (1 - vb) * x2[:, 0] + vb * input_function

            deriv = (x1 - x2)/2*h

        elif param == 'k2':
            K_1, k_2, vb = param_values

            h = h_scale * k_2

            x1, t1 = RK4(comp_ode_model1, input_function, init, dt, T_f, T0, [K_1, k_2 + h, vb])
            x2, t2 = RK4(comp_ode_model1, input_function, init, dt, T_f, T0, [K_1, k_2 - h, vb])

            x1 = (1 - vb) * x1[:, 0] + vb * input_function
            x2 = (1 - vb) * x2[:, 0] + vb * input_function

            deriv = (x1 - x2)/2*h

        elif param == 'vb':
            K_1, k_2, vb = param_values

            h = h_scale * vb

            x1, t1 = RK4(comp_ode_model1, input_function, init, dt, T_f, T0, [K_1, k_2, vb])

            x2 = (1 - (vb + h)) * x1[:, 0] + (vb + h) * input_function
            x3 = (1 - (vb - h)) * x1[:, 0] + (vb - h) * input_function

            deriv = (x2 - x3)/2*h

        elif param == 'F':
            K_1, k_2, vb = param_values

            ps = ps_df.loc[organ, 'PS']
            f_values = np.array([])

            for i in np.linspace(-9, 9, 19):
                f_root = fsolve(flow_func, [K_1 + (i/10)*K_1], args = (K_1, ps))
                f = f_root[0]
                f_values = np.append(f_values, f)
            f = find_nearest(f_values, K_1)

            h = h_scale * f

            x1, t1 = RK4(comp_ode_model1, input_function, init, dt, T_f, T0, [k1_flow(f + h, ps), k_2, vb])
            x2, t2 = RK4(comp_ode_model1, input_function, init, dt, T_f, T0, [k1_flow(f - h, ps), k_2, vb])

            x1 = (1 - vb) * x1[:, 0] + vb * input_function
            x2 = (1 - vb) * x2[:, 0] + vb * input_function

            deriv = (x1 - x2)/2*h

            return deriv, t1, f, f_root

    elif model == '2TCM':
        T_f, T0, dt, time = time_vars(data, 1/60)
        init2 = [0.0, 0.0]
        if param == 'K1':
            K_1, k_2, k_3, k_4, vb = param_values

            h = h_scale * K_1

            x1, t1 = RK4(comp_ode_model2, input_function, init2, dt, T_f, T0, [K_1 + h, k_2, k_3, k_4, vb])
            x2, t2 = RK4(comp_ode_model2, input_function, init2, dt, T_f, T0, [K_1 - h, k_2, k_3, k_4, vb])

            x1 = (1 - vb) * (x1[:, 0] + x1[:, 1]) + vb * input_function
            x2 = (1 - vb) * (x2[:, 0] + x2[:, 1]) + vb * input_function

            deriv = (x1 - x2)/2*h

        elif param == 'k2':
            K_1, k_2, k_3, k_4, vb = param_values

            h = h_scale * k_2

            x1, t1 = RK4(comp_ode_model2, input_function, init2, dt, T_f, T0, [K_1, k_2 + h, k_3, k_4, vb])
            x2, t2 = RK4(comp_ode_model2, input_function, init2, dt, T_f, T0, [K_1, k_2 - h, k_3, k_4, vb])

            x1 = (1 - vb) * (x1[:, 0] + x1[:, 1]) + vb * input_function
            x2 = (1 - vb) * (x2[:, 0] + x2[:, 1]) + vb * input_function

            deriv = (x1 - x2)/2*h

        elif param == 'k3':
            K_1, k_2, k_3, k_4, vb = param_values

            h = h_scale * k_3

            x1, t1 = RK4(comp_ode_model2, input_function, init2, dt, T_f, T0, [K_1, k_2, k_3 + h, k_4, vb])
            x2, t2 = RK4(comp_ode_model2, input_function, init2, dt, T_f, T0, [K_1, k_2, k_3 - h, k_4, vb])

            x1 = (1 - vb) * (x1[:, 0] + x1[:, 1]) + vb * input_function
            x2 = (1 - vb) * (x2[:, 0] + x2[:, 1]) + vb * input_function

            deriv = (x1 - x2)/2*h

        elif param == 'k4':
            K_1, k_2, k_3, k_4, vb = param_values

            h = h_scale * k_4

            x1, t1 = RK4(comp_ode_model2, input_function, init2, dt, T_f, T0, [K_1, k_2, k_3, k_4 + h, vb])
            x2, t2 = RK4(comp_ode_model2, input_function, init2, dt, T_f, T0, [K_1, k_2, k_3, k_4 - h, vb])

            x1 = (1 - vb) * (x1[:, 0] + x1[:, 1]) + vb * input_function
            x2 = (1 - vb) * (x2[:, 0] + x2[:, 1]) + vb * input_function

            deriv = (x1 - x2)/2*h

        elif param == 'vb':
            K_1, k_2, k_3, k_4, vb = param_values

            h = h_scale * vb

            x1, t1 = RK4(comp_ode_model2, input_function, init2, dt, T_f, T0, [K_1, k_2, k_3, k_4, vb])

            x2 = (1 - (vb + h)) * (x1[:, 0] + x1[:, 1]) + (vb + h) * input_function
            x3 = (1 - (vb - h)) * (x1[:, 0] + x1[:, 1]) + (vb - h) * input_function

            deriv = (x2 - x3)/2*h

        elif param == 'F':
            K_1, k_2, k_3, k_4, vb = param_values

            ps = ps_df.loc[organ, 'PS']
            f_values = np.array([])


            for i in np.linspace(-9, 9, 19):
                f_root = fsolve(flow_func, [K_1 + (i/10)*K_1], args = (K_1, ps))
                f = f_root[0]
                f_values = np.append(f_values, f)
            f = find_nearest(f_values, K_1)

            h = h_scale * f

            x1, t1 = RK4(comp_ode_model2, input_function, init2, dt, T_f, T0, [k1_flow(f + h, ps), k_2, k_3, k_4, vb])
            x2, t2 = RK4(comp_ode_model2, input_function, init2, dt, T_f, T0, [k1_flow(f - h, ps), k_2, k_3, k_4, vb])

            x1 = (1 - vb) * (x1[:, 0] + x1[:, 1]) + vb * input_function
            x2 = (1 - vb) * (x2[:, 0] + x2[:, 1]) + vb * input_function

            deriv = (x1 - x2)/2*h

            return deriv, t1, f, f_root
    
    return deriv, t1

def flow_func(F, K1, ps):
    return  F * (1 - np.exp(-ps/F)) - K1

def k1_flow(F, ps):
    k1 = F * (1 - np.exp(-ps/F))
    
    return k1

def ext_frac(F, a, b):
    return (1 - a * np.exp(-b/F))
            
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    
    return array[idx]


def blood_curve(sheet):
    data = pd.read_excel('Blood_fits_parameters.xlsx', sheet_name=sheet, engine = 'openpyxl', dtype = {'Params' : float})

    return data

def delay_shift(time, out, delay):
    if delay != 0:
        extra_time_points = np.array([])
        for i in range(len(time)):
            new_val = time[i] - delay
            if (new_val > time[0] and delay > 0):
                extra_time_points = np.append(extra_time_points, new_val)
            elif (new_val < time[-1] and delay < 0):
                extra_time_points = np.append(extra_time_points, new_val)
        #print(extra_time_points)
        #print(len(time))
        #print(len(extra_time_points))
        f = interp1d(time, out, kind="cubic")
        interp_out = f(extra_time_points)
    else:
        conv_bolus = out
    #print(interp_out)
    # shift the interpolated time points and put them in a new array
    if delay > 0:
        conv_bolus = np.full(len(time), out[0])
        #print(conv_bolus)
        for i in range(-1, -len(interp_out), -1):
            conv_bolus[i] = abs(interp_out[i])
    elif delay < 0:
        conv_bolus = np.full(len(time), out[-1])
        for i in range(len(interp_out)):
            conv_bolus[i] = abs(interp_out[i])
        #delay *= -1.0 # if I have changed the external value


def input_data(sheet, deg):

    ### Read in the mouse data from the Excel spreadsheet

    data = pd.read_excel('TACs_new_VOIs_fitted_input_functions.xlsx', sheet_name=sheet, engine = 'openpyxl', usecols = 'Y:AF', dtype = {'Time (min)' : float, 'Heart' : float, 'Lungs merge' : float, 
    'Kidneys merge' : float, 'Bladder' : float, 'Femur' : float, 'Liver' : float, 'Des Aorta fitted' : float})       # 'Des Aorta Amide', 'TACs_new_VOIs_fitted_input_functions.xlsx'
    data.columns = ['Time', 'Heart', 'Lungs', 'Kidneys', 'Bladder', 'Femur', 'Liver', 'Vena_Cava']

    data = data.drop(0, axis=0)
    #data = data[~(data == 0).any(axis=1)]       # Used to remove any zeros
    data = data.reset_index()
    data = data.drop('index', axis=1)
    #data.Time = data.Time - data.Time[0]

    #
    if deg == True:
        data = data.iloc[:21]   # This slices the data to the first 4 minutes
    #

    T_f = data.iloc[-1, 0]

    '''
    i = 0
    for row in data.Heart:
        if row != 0:
            break
        else:
            i += 1

    T_0 = data.iloc[i, 0]
    '''
    T_0 = data.iloc[0, 0]
    #dt = data.iloc[i+1, 0] - data.iloc[i, 0]
    dt = 1/60
    #time = np.linspace(T_0, T_f, int((T_f - T_0)/dt) + 1)
    time = np.arange(T_0, T_f + dt, dt)

    ### Resample the data so that it's evenly-spaced and place in new dataframe

    data_int = pd.DataFrame(columns = ['Time', 'Heart', 'Lungs', 'Kidneys', 'Bladder', 'Femur', 'Liver', 'Vena_Cava'])
    data_int.Time = time - T_0

    weights = pd.DataFrame(columns = ['Heart', 'Lungs', 'Kidneys', 'Bladder', 'Femur', 'Liver', 'Vena_Cava'])
    
    
    for i in data.columns:
        if i != 'Time':
            data_int[i], inter_func = interpolate(data.Time, data[i], 'cubic', T_0, T_f, dt)

            #weights[i], zero_count = weights_dc(inter_func, data, i, 0.0063128158521)       # currently in minutes, in seconds: 0.0001052135975
            zero_count = 0

    data = data.apply(pd.to_numeric, errors = 'coerce', axis=0)

    t_peak = data.loc[data['Vena_Cava'].idxmax()]['Time']

    if sheet not in ['Mouse_6_17A1101B', 'Mouse_9_17A1010A', 'Mouse_16_17A1101C', 'Mouse_17_17A1101D']:
        blood_params_df = blood_curve(sheet)
        blood = expo(data.Time, [blood_params_df.iloc[0, 0], blood_params_df.iloc[2, 0], blood_params_df.iloc[4, 0]], [blood_params_df.iloc[1, 0], blood_params_df.iloc[3, 0], blood_params_df.iloc[5, 0]], [blood_params_df.iloc[6, 0], blood_params_df.iloc[7, 0]], t_peak)  # , [blood_params_df.iloc[6, 0], blood_params_df.iloc[7, 0]]
        blood_int = expo(data_int.Time, [blood_params_df.iloc[0, 0], blood_params_df.iloc[2, 0], blood_params_df.iloc[4, 0]], [blood_params_df.iloc[1, 0], blood_params_df.iloc[3, 0], blood_params_df.iloc[5, 0]], [blood_params_df.iloc[6, 0], blood_params_df.iloc[7, 0]], t_peak)

        # plt.plot(data.Time, data.Vena_Cava, 'r', label = 'Blood Data')
        # plt.plot(data.Time, blood, 'b', label = 'Fitted Blood Data')
        # plt.xlabel('Time (minutes)')
        # plt.ylabel('SUV (g/ml)')
        # plt.legend(loc = 7, fontsize = 'x-small')
        # plt.show()
        #print(sheet)

        data.Vena_Cava = blood
        data_int.Vena_Cava = blood_int

    return data, data_int, [T_f, T_0, dt, time], weights, zero_count


def create_sim_data(m, organ, model):
    k_dict = {'Heart' : 0, 'Lungs' : 1, 'Liver' : 2, 'Kidneys' : 0, 'Femur' : 1}
    k = k_dict[organ]

    if model == 'Degrado':
        mouse_deg, mouse_int_deg, mouse_time_deg, mouse_weights_deg, mouse_zc_deg = input_data(m, True)
    
        T_f_deg, T0_deg, dt_deg, t_array_deg = mouse_time_deg

        C0_orig_deg_time = mouse_int_deg.Time
        C0_orig_deg = mouse_int_deg.Vena_Cava
        y_time_deg = mouse_deg.Time

        C1_data_deg = mouse_int_deg[organ]
        y_dat_deg = mouse_deg[organ] 
        w_deg = mouse_weights_deg[organ]

        name = f'{m}_{organ}'

        scale_factor = scale_factor_df.loc[organ, m]

        if organ == 'Lungs':
            scale_factor = 0.005

        k1, k2, k3, vb, delay = [organ_params_df.loc['K1', organ], organ_params_df.loc['k2', organ], organ_params_df.loc['k3', organ], organ_params_df.loc['vb', organ], organ_params_df.loc['Delay', organ]]
        
        print(k1)
        print(k2)
        print(k3)
        print(vb)

        sim = mouse_int_deg.copy()
        sim['Bolus'] = bolus_fitted_deg
        sim['Constant'] = ci_deg
        sim['Bol_Inf'] = bolus_inf_deg

        #print(sim)

        init = [0.0]

        # BOLUS
        C0_bolus_deg = sim.Bolus
        
        bolus_x, bolus_t = RK4(comp_ode_model1_deg, C0_bolus_deg, init, dt_deg, T_f_deg, T0_deg, [k1, k2, vb])
        bolus_bv = ((1 - vb) * bolus_x[:, 0]  + vb * C0_bolus_deg)

        deriv_k1, deriv_time_k1 = sensitivity_analysis(mouse_deg, [k1, k2, vb], 'K1', 0.1, C0_bolus_deg,'Degrado', organ)
        deriv_k2, deriv_time_k2 = sensitivity_analysis(mouse_deg, [k1, k2, vb], 'k2', 0.1, C0_bolus_deg, 'Degrado', organ)
        deriv_vb, deriv_time_vb = sensitivity_analysis(mouse_deg, [k1, k2, vb], 'vb', 0.1, C0_bolus_deg, 'Degrado', organ)
        deriv_F, deriv_time_F, f_value, f_roots = sensitivity_analysis(mouse_deg, [k1, k2, vb], 'F', 0.1, C0_bolus_deg, 'Degrado', organ)

        #print(len(deriv_k1))
        sens_bolus_deg[k][0] = deriv_time_k1
        sens_bolus_deg[k][1] = deriv_k1
        sens_bolus_deg[k][2] = deriv_k2
        sens_bolus_deg[k][3] = deriv_vb
        sens_bolus_deg[k][4] = deriv_F

        sens_bolus_params_deg[:, k] = [k1, k2, vb, f_value]
        
        # CONSTANT INFUSION
        C0_con_deg = sim.Constant
        
        con_x, con_t = RK4(comp_ode_model1_deg, C0_con_deg, init, dt_deg, T_f_deg, T0_deg, [k1, k2, vb])
        con_bv = ((1 - vb) * con_x[:, 0] + vb * C0_con_deg)

        deriv_k1, deriv_time_k1 = sensitivity_analysis(mouse_deg, [k1, k2, vb], 'K1', 0.1, C0_con_deg,'Degrado', organ)
        deriv_k2, deriv_time_k2 = sensitivity_analysis(mouse_deg, [k1, k2, vb], 'k2', 0.1, C0_con_deg, 'Degrado', organ)
        deriv_vb, deriv_time_vb = sensitivity_analysis(mouse_deg, [k1, k2, vb], 'vb', 0.1, C0_con_deg, 'Degrado', organ)
        deriv_F, deriv_time_F, f_value, f_roots = sensitivity_analysis(mouse_deg, [k1, k2, vb], 'F', 0.1, C0_con_deg, 'Degrado', organ)

        sens_con_deg[k][0] = deriv_time_k1
        sens_con_deg[k][1] = deriv_k1
        sens_con_deg[k][2] = deriv_k2
        sens_con_deg[k][3] = deriv_vb
        sens_con_deg[k][4] = deriv_F

        sens_con_params_deg[:, k] = [k1, k2, vb, f_value]

        # BOLUS INFUSION
        C0_bolinf_deg = sim.Bol_Inf
        
        bol_inf_x, bol_inf_t = RK4(comp_ode_model1_deg, C0_bolinf_deg, init, dt_deg, T_f_deg, T0_deg, [k1, k2, vb])
        bol_inf_bv = ((1 - vb) * bol_inf_x[:, 0] + vb * C0_bolinf_deg)

        deriv_k1, deriv_time_k1 = sensitivity_analysis(mouse_deg, [k1, k2, vb], 'K1', 0.1, C0_bolinf_deg,'Degrado', organ)
        deriv_k2, deriv_time_k2 = sensitivity_analysis(mouse_deg, [k1, k2, vb], 'k2', 0.1, C0_bolinf_deg, 'Degrado', organ)
        deriv_vb, deriv_time_vb = sensitivity_analysis(mouse_deg, [k1, k2, vb], 'vb', 0.1, C0_bolinf_deg, 'Degrado', organ)
        deriv_F, deriv_time_F, f_value, f_roots = sensitivity_analysis(mouse_deg, [k1, k2, vb], 'F', 0.1, C0_bolinf_deg, 'Degrado', organ)

        sens_bolinf_deg[k][0] = deriv_time_k1
        sens_bolinf_deg[k][1] = deriv_k1
        sens_bolinf_deg[k][2] = deriv_k2
        sens_bolinf_deg[k][3] = deriv_vb
        sens_bolinf_deg[k][4] = deriv_F

        sens_bolinf_params_deg[:, k] = [k1, k2, vb, f_value]

        plt.plot(bolus_t, bolus_bv, color = 'b', label = 'Bolus')
        plt.plot(con_t, con_bv, color = 'g', label = 'Constant Infusion')
        plt.plot(bol_inf_t, bol_inf_bv, color = 'm', label = 'Bolus Infusion')
        plt.title(f'{organ} data derived from a simulated input function')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Activity concentration (MBq/ml)')
        plt.legend(loc = 7, fontsize = 'x-small')
        plt.savefig(f'Tissue_Data/{organ}_tissue_data_comparison_Degrado')
        plt.close()

    else:
        mouse, mouse_int, mouse_time, mouse_weights, mouse_zc = input_data(m, False)
        
        T_f, T0, dt, t_array = mouse_time

        C0_orig = mouse_int.Vena_Cava
        C0_orig_time = mouse_int.Time
        y_time = mouse.Time

        tag = f'{m}'

        C1_data = mouse_int[organ]
        y_dat = mouse[organ]
        w = mouse_weights[organ]

        name = f'{m}_{organ}'

        scale_factor = scale_factor_df.loc[organ, m]

        if organ == 'Lungs':
            scale_factor = 0.005

        k1, k2, k3, k4, vb, delay = [organ_params_df.loc['K1', organ], organ_params_df.loc['k2', organ], organ_params_df.loc['k3', organ], organ_params_df.loc['k4', organ], organ_params_df.loc['vb', organ], organ_params_df.loc['Delay', organ]]
        
        print(k1)
        print(k2)
        print(k3)
        print(k4)
        print(vb)

        sim = mouse_int.copy()
        sim['Bolus'] = bolus_fitted
        sim['Constant'] = ci
        sim['Bol_Inf'] = bolus_inf

        if model == '1TCM':
            init = [0.0]

            # BOLUS
            C0_bolus = sim.Bolus
            bolus_x, bolus_t = RK4(comp_ode_model1, C0_bolus, init, dt, T_f, T0, [k1, k2, vb])
            bolus_bv = ((1 - vb) * bolus_x[:, 0]  + vb * C0_bolus)

            deriv_k1, deriv_time_k1 = sensitivity_analysis(mouse, [k1, k2, vb], 'K1', 0.1, C0_bolus,'1TCM', organ)
            deriv_k2, deriv_time_k2 = sensitivity_analysis(mouse, [k1, k2, vb], 'k2', 0.1, C0_bolus, '1TCM', organ)
            deriv_k3, deriv_time_k3 = sensitivity_analysis(mouse, [k1, k2, vb], 'k3', 0.1, C0_bolus, '1TCM', organ)
            deriv_vb, deriv_time_vb = sensitivity_analysis(mouse, [k1, k2, vb], 'vb', 0.1, C0_bolus, '1TCM', organ)
            deriv_F, deriv_time_F, f_value, f_roots = sensitivity_analysis(mouse, [k1, k2, vb], 'F', 0.1, C0_bolus, '1TCM', organ)

            print(len(deriv_k1))
            sens_bolus[k][0] = deriv_time_k1
            sens_bolus[k][1] = deriv_k1
            sens_bolus[k][2] = deriv_k2
            sens_bolus[k][3] = deriv_vb
            sens_bolus[k][4] = deriv_F
            sens_bolus[k][5] = deriv_k3

            sens_bolus_params[:, k] = [k1, k2, vb, f_value, k3]

            # CONSTANT INFUSION
            C0_con = sim.Constant
            con_x, con_t = RK4(comp_ode_model1, C0_con, init, dt, T_f, T0, [k1, k2, vb])
            con_bv = ((1 - vb) * con_x[:, 0] + vb * C0_con)

            deriv_k1, deriv_time_k1 = sensitivity_analysis(mouse, [k1, k2, vb], 'K1', 0.1, C0_con,'1TCM', organ)
            deriv_k2, deriv_time_k2 = sensitivity_analysis(mouse, [k1, k2, vb], 'k2', 0.1, C0_con, '1TCM', organ)
            deriv_k3, deriv_time_k3 = sensitivity_analysis(mouse, [k1, k2, vb], 'k3', 0.1, C0_con, '1TCM', organ)
            deriv_vb, deriv_time_vb = sensitivity_analysis(mouse, [k1, k2, vb], 'vb', 0.1, C0_con, '1TCM', organ)
            deriv_F, deriv_time_F, f_value, f_roots = sensitivity_analysis(mouse, [k1, k2, vb], 'F', 0.1, C0_con, '1TCM', organ)

            sens_con[k][0] = deriv_time_k1
            sens_con[k][1] = deriv_k1
            sens_con[k][2] = deriv_k2
            sens_con[k][3] = deriv_vb
            sens_con[k][4] = deriv_F
            sens_con[k][5] = deriv_k3

            sens_con_params[:, k] = [k1, k2, vb, f_value, k3]

            #BOLUS INFUSION
            C0_bolinf = sim.Bol_Inf
            bol_inf_x, bol_inf_t = RK4(comp_ode_model1_deg, C0_bolinf, init, dt, T_f, T0, [k1, k2, vb])
            bol_inf_bv = ((1 - vb) * bol_inf_x[:, 0] + vb * C0_bolinf)

            deriv_k1, deriv_time_k1 = sensitivity_analysis(mouse, [k1, k2, vb], 'K1', 0.1, C0_bolinf, '1TCM', organ)
            deriv_k2, deriv_time_k2 = sensitivity_analysis(mouse, [k1, k2, vb], 'k2', 0.1, C0_bolinf, '1TCM', organ)
            deriv_k3, deriv_time_k3 = sensitivity_analysis(mouse, [k1, k2, vb], 'k3', 0.1, C0_bolinf, '1TCM', organ)
            deriv_vb, deriv_time_vb = sensitivity_analysis(mouse, [k1, k2, vb], 'vb', 0.1, C0_bolinf, '1TCM', organ)
            deriv_F, deriv_time_F, f_value, f_roots = sensitivity_analysis(mouse, [k1, k2, vb], 'F', 0.1, C0_bolinf, '1TCM', organ)

            sens_bolinf[k][0] = deriv_time_k1
            sens_bolinf[k][1] = deriv_k1
            sens_bolinf[k][2] = deriv_k2
            sens_bolinf[k][3] = deriv_vb
            sens_bolinf[k][4] = deriv_F
            sens_bolinf[k][5] = deriv_k3

            sens_bolinf_params[:, k] = [k1, k2, vb, f_value, k3]

            
            plt.plot(bolus_t, bolus_bv, color = 'b', label = 'Bolus')
            plt.plot(con_t, con_bv, color = 'g', label = 'Constant Infusion')
            plt.plot(bol_inf_t, bol_inf_bv, color = 'm', label = 'Bolus Infusion')
            plt.title(f'{organ} data derived from a simulated input function')
            plt.xlabel('Time (minutes)')
            plt.ylabel('Activity concentration (MBq/ml)')
            plt.legend(loc = 7, fontsize = 'x-small')
            plt.savefig(f'Tissue_Data/{organ}_tissue_data_comparison_1TCM')
            plt.close()


        elif model == '2TCM':
            init = [0.0, 0.0]

            # BOLUS
            C0_bolus = sim.Bolus
            bolus_x, bolus_t = RK4(comp_ode_model2, C0_bolus, init, dt, T_f, T0, [k1, k2, k3, k4, vb])
            bolus_bv = ((1 - vb) * (bolus_x[:, 0] + bolus_x[:, 1]) + vb * C0_bolus)

            deriv_k1, deriv_time_k1 = sensitivity_analysis(mouse, [k1, k2, k3, k4, vb], 'K1', 0.1, C0_bolus,'2TCM', organ)
            deriv_k2, deriv_time_k2 = sensitivity_analysis(mouse, [k1, k2, k3, k4, vb], 'k2', 0.1, C0_bolus, '2TCM', organ)
            deriv_k3, deriv_time_k3 = sensitivity_analysis(mouse, [k1, k2, k3, k4, vb], 'k3', 0.1, C0_bolus, '2TCM', organ)
            deriv_k4, deriv_time_k4 = sensitivity_analysis(mouse, [k1, k2, k3, k4, vb], 'k4', 0.1, C0_bolus, '2TCM', organ)
            deriv_vb, deriv_time_vb = sensitivity_analysis(mouse, [k1, k2, k3, k4, vb], 'vb', 0.1, C0_bolus, '2TCM', organ)
            deriv_F, deriv_time_F, f_value, f_roots = sensitivity_analysis(mouse, [k1, k2, k3, k4, vb], 'F', 0.1, C0_bolus, '2TCM', organ)

            sens_bolus[k][0] = deriv_time_k1
            sens_bolus[k][1] = deriv_k1
            sens_bolus[k][2] = deriv_k2
            sens_bolus[k][3] = deriv_vb
            sens_bolus[k][4] = deriv_F
            sens_bolus[k][5] = deriv_k3
            sens_bolus[k][6] = deriv_k4

            sens_bolus_params[:, k] = [k1, k2, vb, f_value, k3, k4]

            # CONSTANT INFUSION
            C0_con = sim.Constant
            con_x, con_t = RK4(comp_ode_model2, C0_con, init, dt, T_f, T0, [k1, k2, k3, k4, vb])
            con_bv = ((1 - vb) * (con_x[:, 0] + con_x[:, 1]) + vb * C0_con)

            deriv_k1, deriv_time_k1 = sensitivity_analysis(mouse, [k1, k2, k3, k4, vb], 'K1', 0.1, C0_con,'2TCM', organ)
            deriv_k2, deriv_time_k2 = sensitivity_analysis(mouse, [k1, k2, k3, k4, vb], 'k2', 0.1, C0_con, '2TCM', organ)
            deriv_k3, deriv_time_k3 = sensitivity_analysis(mouse, [k1, k2, k3, k4, vb], 'k3', 0.1, C0_con, '2TCM', organ)
            deriv_k4, deriv_time_k4 = sensitivity_analysis(mouse, [k1, k2, k3, k4, vb], 'k4', 0.1, C0_con, '2TCM', organ)
            deriv_vb, deriv_time_vb = sensitivity_analysis(mouse, [k1, k2, k3, k4, vb], 'vb', 0.1, C0_con, '2TCM', organ)
            deriv_F, deriv_time_F, f_value, f_roots = sensitivity_analysis(mouse, [k1, k2, k3, k4, vb], 'F', 0.1, C0_con, '2TCM', organ)

            sens_con[k][0] = deriv_time_k1
            sens_con[k][1] = deriv_k1
            sens_con[k][2] = deriv_k2
            sens_con[k][3] = deriv_vb
            sens_con[k][4] = deriv_F
            sens_con[k][5] = deriv_k3
            sens_con[k][6] = deriv_k4

            sens_con_params[:, k] = [k1, k2, vb, f_value, k3, k4]

            # BOLUS INFUSION
            C0_bolinf = sim.Bol_Inf
            bol_inf_x, bol_inf_t = RK4(comp_ode_model2, C0_bolinf, init, dt, T_f, T0, [k1, k2, k3, k4, vb])
            bol_inf_bv = ((1 - vb) * (bol_inf_x[:, 0] + bol_inf_x[:, 1]) + vb * C0_bolinf)

            deriv_k1, deriv_time_k1 = sensitivity_analysis(mouse, [k1, k2, k3, k4, vb], 'K1', 0.1, C0_bolinf, '2TCM', organ)
            deriv_k2, deriv_time_k2 = sensitivity_analysis(mouse, [k1, k2, k3, k4, vb], 'k2', 0.1, C0_bolinf, '2TCM', organ)
            deriv_k3, deriv_time_k3 = sensitivity_analysis(mouse, [k1, k2, k3, k4, vb], 'k3', 0.1, C0_bolinf, '2TCM', organ)
            deriv_k4, deriv_time_k4 = sensitivity_analysis(mouse, [k1, k2, k3, k4, vb], 'k4', 0.1, C0_bolinf, '2TCM', organ)
            deriv_vb, deriv_time_vb = sensitivity_analysis(mouse, [k1, k2, k3, k4, vb], 'vb', 0.1, C0_bolinf, '2TCM', organ)
            deriv_F, deriv_time_F, f_value, f_roots = sensitivity_analysis(mouse, [k1, k2, k3, k4, vb], 'F', 0.1, C0_bolinf, '2TCM', organ)

            sens_bolinf[k][0] = deriv_time_k1
            sens_bolinf[k][1] = deriv_k1
            sens_bolinf[k][2] = deriv_k2
            sens_bolinf[k][3] = deriv_vb
            sens_bolinf[k][4] = deriv_F
            sens_bolinf[k][5] = deriv_k3
            sens_bolinf[k][6] = deriv_k4

            sens_bolinf_params[:, k] = [k1, k2, vb, f_value, k3, k4]

            plt.plot(bolus_t, bolus_bv, color = 'b', label = 'Bolus')
            plt.plot(con_t, con_bv, color = 'g', label = 'Constant Infusion')
            plt.plot(bol_inf_t, bol_inf_bv, color = 'm', label = 'Bolus Infusion')
            plt.title(f'{organ} data derived from a simulated input function')
            plt.xlabel('Time (minutes)')
            plt.ylabel('Activity concentration (MBq/ml)')
            plt.legend(loc = 7, fontsize = 'x-small')
            plt.savefig(f'Tissue_Data/{organ}_tissue_data_comparison_2TCM')
            plt.close()

    num = 50

    bolus_50 = np.zeros((num, 3, (len(frametimes_m) - 1)))
    con_50 = np.zeros((num, 3, (len(frametimes_m) - 1)))
    bolinf_50 = np.zeros((num, 3, (len(frametimes_m) - 1)))

    organ_df = pd.DataFrame(data = {'Bol_Time' : bolus_t, 'Bolus' : bolus_bv, 'Con_Time' : con_t, 'Con' : con_bv, 'BolInf_Time' : bol_inf_t, 'BolInf' : bol_inf_bv})
    #print(lungs_df[['Bol_Time', 'Bolus']])

    for i in range(num):
        noisy_bolus, bolus_new, bolus_noise = noise_gen(mouse3, organ_df[['Bol_Time', 'Bolus']], sim.Bolus, scale_factor)
        noisy_con, con_new, con_noise = noise_gen(mouse3, organ_df[['Con_Time', 'Con']], sim.Constant, scale_factor)
        noisy_bolinf, bolinf_new, bolinf_noise = noise_gen(mouse3, organ_df[['BolInf_Time', 'BolInf']], sim.Bol_Inf, scale_factor)  # This should be good for the lungs?

        bolus_50[i, :] = [noisy_bolus, bolus_new, bolus_noise]         
        con_50[i, :] = [noisy_con, con_new, con_noise]
        bolinf_50[i, :] = [noisy_bolinf, bolinf_new, bolinf_noise]

    file_data = np.array([bolus_50, con_50, bolinf_50])

    matfile = f'{organ}.mat'
    scio.savemat(matfile, mdict={'out': file_data}, oned_as='row')

    return file_data


def sensitivity_analysis_display(folder, data, params, data_deg, params_deg, ip, percent):
    if percent:    
        fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15), (ax16, ax17, ax18, ax19, ax20), (ax21, ax22, ax23, ax24, ax25)) = plt.subplots(5, 5, figsize = (15,15), sharey = 'row', sharex = 'col', constrained_layout = False, tight_layout = True)
        fig.suptitle(f'Sensitivity analysis of all parameters for all organs using {ip}')

        # First row
        ax1.plot(data_deg[0][0], (data_deg[0][1] / params_deg[0, 0]) * 100, label = 'Heart K1', color = 'b')
        ax1.set_title(f'Heart \n K1 = {params_deg[0, 0]:.3f}')
        ax1.set_ylabel('dPET/dK1')
    

        ax2.plot(data_deg[1][0], (data_deg[1][1] / params_deg[0, 1]) * 100, label = 'Lungs K1', color = 'b')
        ax2.set_title(f'Lungs \n K1 = {params_deg[0, 1]:.3f}')
        ax2.tick_params(axis = 'both')

        ax3.plot(data_deg[2][0], (data_deg[2][1] / params_deg[0, 2]) * 100, label = 'Liver K1', color = 'b')
        ax3.set_title(f'Liver \n K1 = {params_deg[0, 2]:.3f}')
        ax3.tick_params()

        ax4.plot(data[0][0], (data[0][1] / params[0, 0]) * 100, label = 'Kidneys K1', color = 'b')
        ax4.set_title(f'Kidneys \n K1 = {params[0, 0]:.3f}')
        ax4.tick_params()

        ax5.plot(data[1][0], (data[1][1] / params[0, 1]) * 100, label = 'Femur K1', color = 'b')
        ax5.set_title(f'Femur \n K1 = {params[0, 1]:.3f}')
        ax5.tick_params()

        # Second row
        ax6.plot(data_deg[0][0], (data_deg[0][2] / params_deg[1, 0]) * 100, label = 'Heart k2', color = 'g')
        ax6.set_title(f'k2 = {params_deg[1, 0]:.3f}')
        ax6.set_ylabel('dPET/dk2')

        ax7.plot(data_deg[1][0], (data_deg[1][2] / params_deg[1, 1]) * 100, label = 'Lungs k2', color = 'g')
        ax7.set_title(f'k2 = {params_deg[1, 1]:.3f}')

        ax8.plot(data_deg[2][0], (data_deg[2][2] / params_deg[1, 2]) * 100, label = 'Liver k2', color = 'g')
        ax8.set_title(f'k2 = {params_deg[1, 2]:.3f}')

        ax9.plot(data[0][0], (data[0][2] / params[1, 0]) * 100, label = 'Kidneys k2', color = 'g')
        ax9.set_title(f'k2 = {params[1, 0]:.3f}')

        ax10.plot(data[1][0], (data[1][2] / params[1, 1]) * 100, label = 'Femur k2', color = 'g')
        ax10.set_title(f'k2 = {params[1, 1]:.3f}')


        # Third row
        ax11.plot(data_deg[0][0], (data_deg[0][3] / params_deg[2, 0]) * 100, label = 'Heart vb', color = 'r')
        ax11.set_title(f'vb = {params_deg[2, 0]:.3f}')
        ax11.set_ylabel('dPET/dvb')

        ax12.plot(data_deg[1][0], (data_deg[1][3] / params_deg[2, 1]) * 100, label = 'Lungs vb', color = 'r')
        ax12.set_title(f'vb = {params_deg[2, 1]:.3f}')

        ax13.plot(data_deg[2][0], (data_deg[2][3] / params_deg[2, 2]) * 100, label = 'Liver vb', color = 'r')
        ax13.set_title(f'vb = {params_deg[2, 2]:.3f}')

        ax14.plot(data[0][0], (data[0][3] / params[2, 0]) * 100, label = 'Kidneys vb', color = 'r')
        ax14.set_title(f'vb = {params[2, 0]:.3f}')

        ax15.plot(data[1][0], (data[1][3] / params[2, 1]) * 100, label = 'Femur vb', color = 'r')
        ax15.set_title(f'vb = {params[2, 1]:.3f}')

        # Fourth row
        ax16.plot(data_deg[0][0], (data_deg[0][4] / params_deg[3, 0]) * 100, label = 'F', color = 'm')
        ax16.set_title(f'F = {params_deg[3, 0]:.3f}')
        ax16.set_ylabel('dPET/dF')

        ax17.plot(data_deg[1][0], (data_deg[1][4] / params_deg[3, 1]) * 100, label = 'F', color = 'm')
        ax17.set_title(f'F = {params_deg[3, 1]:.3f}')

        ax18.plot(data_deg[2][0], (data_deg[2][4] / params_deg[3, 2]) * 100, label = 'F', color = 'm')
        ax18.set_title(f'F = {params_deg[3, 2]:.3f}')

        ax19.plot(data[0][0], (data[0][4] / params[3, 0]) * 100, label = 'F', color = 'm')
        ax19.set_title(f'F = {params[3, 0]:.3f}')

        ax20.plot(data[1][0], (data[1][4] / params[3, 1]) * 100, label = 'F', color = 'm')
        ax20.set_title(f'F = {params[3, 1]:.3f}')
        
        # Fifth row
        # ax21.plot(data_deg[0][0], data_deg[0][5], label = 'k3', color = 'y')
        # ax21.set_title(f'k3 = {params_deg[2, 0]:.3f}')
        ax21.set_ylabel('dPET/dk3')

        # ax22.plot(data_deg[1][0], data_deg[1][5], label = 'k3', color = 'y')
        # ax22.set_title(f'k3 = {params_deg[2, 1]:.3f}')

        # ax23.plot(data_deg[2][0], data_deg[2][5], label = 'k3', color = 'y')
        # ax23.set_title(f'k3 = {params_deg[2, 2]:.3f}')

        # ax24.plot(data[0][0], data[0][4], label = 'k3', color = 'y')
        # ax24.set_title(f'k3 = {params[4, 0]:.3f}')

        ax25.plot(data[1][0], (data[1][5] / params[4, 1]) * 100, label = 'k3', color = 'y')
        ax25.set_title(f'k3 = {params[4, 1]:.3f}')

    elif not percent:
        fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15), (ax16, ax17, ax18, ax19, ax20), (ax21, ax22, ax23, ax24, ax25), (ax26, ax27, ax28, ax29, ax30)) = plt.subplots(6, 5, figsize = (15,18), constrained_layout = False, tight_layout = True)
        fig.suptitle(f'Sensitivity analysis of all parameters for all organs using {ip}')

        # First row

        ax1.plot(data[0][0], data[0][1], label = 'Kidneys K1', color = 'b')
        ax1.set_title(f'Kidneys \n K1 = {params[0, 0]:.3f}')
        ax1.set_ylabel('dPET/dK1')

        ax2.plot(data[1][0], data[1][1], label = 'Femur K1', color = 'b')
        ax2.set_title(f'Femur \n K1 = {params[0, 1]:.3f}')

        ax3.plot(data_deg[0][0], data_deg[0][1], label = 'Heart K1', color = 'b')
        ax3.set_title(f'Heart \n K1 = {params_deg[0, 0]:.3f}')

        ax4.plot(data_deg[1][0], data_deg[1][1], label = 'Lungs K1', color = 'b')
        ax4.set_title(f'Lungs \n K1 = {params_deg[0, 1]:.3f}')

        ax5.plot(data_deg[2][0], data_deg[2][1], label = 'Liver K1', color = 'b')
        ax5.set_title(f'Liver \n K1 = {params_deg[0, 2]:.3f}')

        

        # Second row
        ax6.plot(data[0][0], data[0][2], label = 'Kidneys k2', color = 'g')
        ax6.set_title(f'k2 = {params[1, 0]:.3f}')
        ax6.set_ylabel('dPET/dk2')

        ax7.plot(data[1][0], data[1][2], label = 'Femur k2', color = 'g')
        ax7.set_title(f'k2 = {params[1, 1]:.3f}')

        ax8.plot(data_deg[0][0], data_deg[0][2], label = 'Heart k2', color = 'g')
        ax8.set_title(f'k2 = {params_deg[1, 0]:.3f}')

        ax9.plot(data_deg[1][0], data_deg[1][2], label = 'Lungs k2', color = 'g')
        ax9.set_title(f'k2 = {params_deg[1, 1]:.3f}')

        ax10.plot(data_deg[2][0], data_deg[2][2], label = 'Liver k2', color = 'g')
        ax10.set_title(f'k2 = {params_deg[1, 2]:.3f}')

        


        # Third row
        ax11.plot(data[0][0], data[0][3], label = 'Kidneys vb', color = 'r')
        ax11.set_title(f'vb = {params[2, 0]:.3f}')
        ax11.set_ylabel('dPET/dvb')

        ax12.plot(data[1][0], data[1][3], label = 'Femur vb', color = 'r')
        ax12.set_title(f'vb = {params[2, 1]:.3f}')
        
        ax13.plot(data_deg[0][0], data_deg[0][3], label = 'Heart vb', color = 'r')
        ax13.set_title(f'vb = {params_deg[2, 0]:.3f}')

        ax14.plot(data_deg[1][0], data_deg[1][3], label = 'Lungs vb', color = 'r')
        ax14.set_title(f'vb = {params_deg[2, 1]:.3f}')

        ax15.plot(data_deg[2][0], data_deg[2][3], label = 'Liver vb', color = 'r')
        ax15.set_title(f'vb = {params_deg[2, 2]:.3f}')

        
        # Fourth row
        ax16.plot(data[0][0], data[0][4], label = 'F', color = 'm')
        ax16.set_title(f'F = {params[3, 0]:.3f}')
        ax16.set_ylabel('dPET/dF')

        ax17.plot(data[1][0], data[1][4], label = 'F', color = 'm')
        ax17.set_title(f'F = {params[3, 1]:.3f}')

        ax18.plot(data_deg[0][0], data_deg[0][4], label = 'F', color = 'm')
        ax18.set_title(f'F = {params_deg[3, 0]:.3f}')

        ax19.plot(data_deg[1][0], data_deg[1][4], label = 'F', color = 'm')
        ax19.set_title(f'F = {params_deg[3, 1]:.3f}')

        ax20.plot(data_deg[2][0], data_deg[2][4], label = 'F', color = 'm')
        ax20.set_title(f'F = {params_deg[3, 2]:.3f}')

        
        
        # Fifth row
        ax21.plot(data[0][0], data[0][4], label = 'k3', color = 'y')
        ax21.set_title(f'k3 = {params[4, 0]:.3f}')
        ax21.set_ylabel('dPET/dk3')

        ax22.plot(data[1][0], data[1][5], label = 'k3', color = 'y')
        ax22.set_title(f'k3 = {params[4, 1]:.3f}')

        # ax21.plot(data_deg[0][0], data_deg[0][5], label = 'k3', color = 'y')
        # ax21.set_title(f'k3 = {params_deg[2, 0]:.3f}')

        # ax22.plot(data_deg[1][0], data_deg[1][5], label = 'k3', color = 'y')
        # ax22.set_title(f'k3 = {params_deg[2, 1]:.3f}')

        # ax23.plot(data_deg[2][0], data_deg[2][5], label = 'k3', color = 'y')
        # ax23.set_title(f'k3 = {params_deg[2, 2]:.3f}')

        # Sixth row

        ax26.plot(data[0][0], data[0][5], label = 'k4', color = 'c')
        ax26.set_title(f'k3 = {params[5, 0]:.3f}')
        ax26.set_ylabel('dPET/dk4')

        

    fig.delaxes(ax23)
    fig.delaxes(ax24)
    fig.delaxes(ax25)

    fig.delaxes(ax27)
    fig.delaxes(ax28)
    fig.delaxes(ax29)
    fig.delaxes(ax30)

    plt.savefig(f'{folder}/{ip}_SA')
    plt.close()

    with open(f'{folder}\{ip}_params.csv', 'w', newline = '') as f:
        writer = csv.writer(f)
        writer.writerows([['Heart', 'Lungs', 'Liver']])
        writer.writerows(params_deg)
        writer.writerows([' '])
        writer.writerows([['Kidneys', 'Femur']])
        writer.writerows(params)
      

def model_data(path, file_data, mouse, mouse_deg, organ, C0s, C0_degs, framemidtimes_m, scale_factor):
    inputs = ['bolus', 'constant', 'bolinf']
    #C0s = [C0_bolus, C0_constant, C0_bolinf]
    #C0_degs = [C0_deg_bolus, C0_deg_constant, C0_deg_bolinf]
    n = len(file_data[0])
    #print(file_data)
    #print(len(file_data[0]))
    # print(file_data[0, 0])
    # print(file_data[0, 0, 0])
    # print(file_data[0, 0, 0, 0])

    if organ == 'Heart' or organ == 'Lungs' or organ == 'Liver':
        organ_params = np.zeros((3, n, 3)) 

        for j in range(len(inputs)):
            input = inputs[j]
            C0 = C0s[j]
            C0_deg = C0_degs[j]

            for i in range(n):
                y_dat = file_data[j, i, 0, :]
                y_dat_clean = file_data[j, i, 1, :]
                y_dat_deg = y_dat[:21]
                y_dat_deg_clean = y_dat_clean[:21]
                #print(len(y_dat))
                #print(y_dat)
                y_time = mouse.Time
                #y_time = y_time[:-1]
                y_time_deg = mouse_deg.Time
                #y_time_deg = y_time_deg[:-1]
                

                params = lmfit.Parameters()
                params.add('K1', 1, min=0.00001, max=5.0)
                params.add('k2', 0.1, min = 0.00001, max = 5.0)
                params.add('vb', 0.1, vary = True, min=0.0, max=1.0)

                fit2 = lmfit.minimize(resid1_deg_weighted, params, args = (C0_deg, mouse_deg, y_time_deg, y_dat_deg, y_dat, frame_lengths_m, framemidtimes_m, scale_factor), method = 'leastsq', max_nfev = 1000, nan_policy = 'propagate')
                
                # val1, val2, valvb = [fit2.params['K1'].value, fit2.params['k2'].value, fit2.params['vb'].value]
                # T_f_deg, T0_deg, dt_deg, time = time_vars(mouse_deg, 1/60)
                # x1, t1 = RK4(comp_ode_model1_deg, C0_deg, [0.0, 0.0], dt_deg, T_f_deg, T0_deg, [fit2.params['K1'].value, fit2.params['k2'].value, fit2.params['vb'].value])

                # plt.scatter(y_time_deg, y_dat_deg_clean, s = 7, label = 'Blood', color = 'r')
                # plt.scatter(y_time_deg, y_dat_deg, s = 7, label = 'Noisy Blood', color = 'g')
                # plt.plot(t1, ((1 - fit2.params['vb'].value) * (x1[:, 0]) + fit2.params['vb'].value * C0_deg), label = 'Model Fit', color = 'b')
                # plt.title(f'{organ} \n K1 = {val1:.3f}, k2 = {val2:.3f}, vb = {valvb:.3f}')
                # plt.xlabel('Time (minutes)')
                # plt.ylabel('Activity Concentration (MBq/cc)')
                # plt.legend(loc = 7, fontsize = 'x-small')
                # plt.savefig(f'Testing/{organ}_{input}_{i}')
                # plt.close()

                with open(f"{path}/{organ}_{input}_params.txt", "a") as lungs_param_out:
                    for name, param in fit2.params.items():
                        #lungs_param_out.write('{:7s} {:11.6f} {:11.6f}'.format(name, param.value, param.stderr))
                        lungs_param_out.write(f'{param.value:.3f}' + ' ')
                    lungs_param_out.write('\n')
                
                organ_params[j, i] = [fit2.params['K1'].value, fit2.params['k2'].value, fit2.params['vb'].value]

    elif organ == 'Femur':
        organ_params = np.zeros((3, n, 4))

        for j in range(len(inputs)):
            input = inputs[j]
            C0 = C0s[j]
            C0_deg = C0_degs[j]

            for i in range(n):
                y_dat = file_data[j, i, 0, :]
                y_dat_clean = file_data[j, i, 1, :]
                #y_dat_deg = y_dat[:21]
                y_time = mouse.Time
                #y_time_deg = mouse_deg.Time

                params = lmfit.Parameters()
                params.add('K1', 1, min=0.00001, max=5.0)
                params.add('k2', 0.1, min = 0.00001, max = 5.0)
                params.add('vb', 0.1, vary = True, min=0.0, max=1.0)
                params.add('k3', 0.1, min=0.0, max=5.0)
                params.add('k4', 0.0, vary = False, min = 0.0, max = 5.0)
                
                fit2 = lmfit.minimize(resid2_weighted, params, args = (C0, mouse, y_time, y_dat, frame_lengths_m, framemidtimes_m, scale_factor), method = 'leastsq', max_nfev = 1000, nan_policy = 'propagate')
                
                # val1, val2, valvb, val3 = [fit2.params['K1'].value, fit2.params['k2'].value, fit2.params['vb'].value, fit2.params['k3'].value]
                # T_f, T0, dt, time = time_vars(mouse, 1/60)
                # x1, t1 = RK4(comp_ode_model2, C0, [0.0, 0.0], dt, T_f, T0, [fit2.params['K1'].value, fit2.params['k2'].value, fit2.params['k3'].value, fit2.params['k4'].value, fit2.params['vb'].value])

                # plt.scatter(y_time, y_dat_clean, s = 7, label = 'Blood', color = 'r')
                # plt.scatter(y_time, y_dat, s = 7, label = 'Noisy Blood', color = 'g')
                # plt.plot(t1, ((1 - fit2.params['vb'].value) * (x1[:, 0] + x1[:, 1]) + fit2.params['vb'].value * C0), label = 'Model Fit', color = 'b')
                # plt.title(f'{organ} \n K1 = {val1:.3f}, k2 = {val2:.3f},  k3 = {val3:.3f}, vb = {valvb:.3f}')
                # plt.xlabel('Time (minutes)')
                # plt.ylabel('Activity Concentration (MBq/cc)')
                # plt.legend(loc = 7, fontsize = 'x-small')
                # plt.savefig(f'Testing/{organ}_{input}_{i}')
                # plt.close()

                with open(f"{path}/{organ}_{input}_params.txt", "a") as femur_param_out:
                    for name, param in fit2.params.items():
                        #lungs_param_out.write('{:7s} {:11.6f} {:11.6f}'.format(name, param.value, param.stderr))
                        femur_param_out.write(f'{param.value:.3f}' + ' ')
                    femur_param_out.write('\n')


                organ_params[j, i] = [fit2.params['K1'].value, fit2.params['k2'].value, fit2.params['vb'].value, fit2.params['k3'].value]
                

    elif organ == 'Kidneys':
        organ_params = np.zeros((3, n, 5))

        for j in range(len(inputs)):
            input = inputs[j]
            C0 = C0s[j]
            C0_deg = C0_degs[j]

            for i in range(n):
                y_dat = file_data[j, i, 0, :]
                y_dat_clean = file_data[j, i, 1, :]
                #y_dat_deg = y_dat[:21]
                y_time = mouse.Time
                #y_time_deg = mouse_deg.Time
                

                params = lmfit.Parameters()
                params.add('K1', 1, min=0.00001, max=5.0)
                params.add('k2', 0.1, min = 0.00001, max = 5.0)
                params.add('vb', 0.1, vary = True, min=0.0, max=1.0)
                params.add('k3', 0.1, min=0.0, max=5.0)
                params.add('k4', 0.1, vary = True, min = 0.0, max = 5.0)
                
                fit2 = lmfit.minimize(resid2_kidneys_weighted, params, args = (C0, mouse, y_time, y_dat, frame_lengths_m, framemidtimes_m, scale_factor), method = 'leastsq', max_nfev = 1000, nan_policy = 'propagate')

                # val1, val2, valvb, val3, val4 = [fit2.params['K1'].value, fit2.params['k2'].value, fit2.params['vb'].value, fit2.params['k3'].value, fit2.params['k4'].value]
                # T_f, T0, dt, time = time_vars(mouse, 1/60)
                # x1, t1 = RK4(comp_ode_model2_kidney, C0, [0.0, 0.0], dt, T_f, T0, [fit2.params['K1'].value, fit2.params['k2'].value, fit2.params['k3'].value, fit2.params['k4'].value, fit2.params['vb'].value])

                # plt.scatter(y_time, y_dat_clean, s = 7, label = 'Blood', color = 'r')
                # plt.scatter(y_time, y_dat, s = 7, label = 'Noisy Blood', color = 'g')
                # plt.plot(t1, ((1 - fit2.params['vb'].value) * (x1[:, 0] + x1[:, 1]) + fit2.params['vb'].value * C0), label = 'Model Fit', color = 'b')
                # plt.title(f'{organ} \n K1 = {val1:.3f}, k2 = {val2:.3f},  k3 = {val3:.3f}, vb = {valvb:.3f}')
                # plt.xlabel('Time (minutes)')
                # plt.ylabel('Activity Concentration (MBq/cc)')
                # plt.legend(loc = 7, fontsize = 'x-small')
                # plt.savefig(f'Testing/{organ}_{input}_{i}')
                # plt.close()

                with open(f"{path}/{organ}_{input}_params.txt", "a") as femur_param_out:
                    for name, param in fit2.params.items():
                        #lungs_param_out.write('{:7s} {:11.6f} {:11.6f}'.format(name, param.value, param.stderr))
                        femur_param_out.write(f'{param.value:.3f}' + ' ')
                    femur_param_out.write('\n')
                
                organ_params[j, i] = [fit2.params['K1'].value, fit2.params['k2'].value, fit2.params['vb'].value, fit2.params['k3'].value, fit2.params['k4'].value]

    return organ_params

                

def table_generator(x, optimal):
    sum = np.array([])
    
    for i in range(len(x)):
        sum = np.append(sum, (x[i] - optimal))

    bias = np.sum(sum) / len(x)

    bias_per = (bias/optimal) * 100

    std_dev = np.std(x) / np.sqrt(len(x))

    mse = np.square(std_dev) + np.square(bias)

    # print(bias)
    # print(std_dev)
    # print(mse)
    # print(bias_per)

    return round(bias, 3), round(std_dev, 3), round(mse, 3), round(bias_per, 1)


def calculate_data(path1, path2, organ, organ_params_df):
    if organ == 'Heart' or organ == 'Lungs' or organ == 'Liver':
        organ_bolus = np.loadtxt(f'{path1}/{organ}_bolus_params.txt')
        organ_constant = np.loadtxt(f'{path1}/{organ}_constant_params.txt')
        organ_bolinf = np.loadtxt(f'{path1}/{organ}_bolinf_params.txt')

        organ_df = pd.DataFrame(data = {'K1_bolus' : organ_bolus[:, 0], 'k2_bolus' : organ_bolus[:, 1], 'vb_bolus' : organ_bolus[:, 2], 
                                        'K1_constant' : organ_constant[:, 0], 'k2_constant' : organ_constant[:, 1],  'vb_constant' : organ_constant[:, 2],
                                        'K1_bolinf' : organ_bolinf[:, 0], 'k2_bolinf' : organ_bolinf[:, 1], 'vb_bolinf' : organ_bolinf[:, 2]})

        organ_bolus_k1_bias, organ_bolus_k1_std, organ_bolus_k1_mse, organ_bolus_k1_bias_per = table_generator(organ_df['K1_bolus'], organ_params_df[organ][0])
        organ_bolus_k2_bias, organ_bolus_k2_std, organ_bolus_k2_mse, organ_bolus_k2_bias_per = table_generator(organ_df['k2_bolus'], organ_params_df[organ][1])
        organ_bolus_vb_bias, organ_bolus_vb_std, organ_bolus_vb_mse, organ_bolus_vb_bias_per = table_generator(organ_df['vb_bolus'], organ_params_df[organ][4])

        organ_constant_k1_bias, organ_constant_k1_std, organ_constant_k1_mse, organ_constant_k1_bias_per = table_generator(organ_df['K1_constant'], organ_params_df[organ][0])
        organ_constant_k2_bias, organ_constant_k2_std, organ_constant_k2_mse, organ_constant_k2_bias_per = table_generator(organ_df['k2_constant'], organ_params_df[organ][1])
        organ_constant_vb_bias, organ_constant_vb_std, organ_constant_vb_mse, organ_constant_vb_bias_per = table_generator(organ_df['vb_constant'], organ_params_df[organ][4])

        organ_bolinf_k1_bias, organ_bolinf_k1_std, organ_bolinf_k1_mse, organ_bolinf_k1_bias_per = table_generator(organ_df['K1_bolinf'], organ_params_df[organ][0])
        organ_bolinf_k2_bias, organ_bolinf_k2_std, organ_bolinf_k2_mse, organ_bolinf_k2_bias_per = table_generator(organ_df['k2_bolinf'], organ_params_df[organ][1])
        organ_bolinf_vb_bias, organ_bolinf_vb_std, organ_bolinf_vb_mse, organ_bolinf_vb_bias_per = table_generator(organ_df['vb_bolinf'], organ_params_df[organ][4])

        with open(f'{path2}/{organ}_bias.csv', 'w', newline = '') as f:
            writer = csv.writer(f)
            writer.writerows([['Bolus Injection'],
                              ['Param', 'Bias', 'STDEV', 'MSE', 'Bias (%)'],
                              ['K1', organ_bolus_k1_bias, organ_bolus_k1_std, organ_bolus_k1_mse, organ_bolus_k1_bias_per], 
                              ['k2', organ_bolus_k2_bias, organ_bolus_k2_std, organ_bolus_k2_mse, organ_bolus_k2_bias_per],
                              ['vb', organ_bolus_vb_bias, organ_bolus_vb_std, organ_bolus_vb_mse, organ_bolus_vb_bias_per]])
            writer.writerow([' '])
            writer.writerows([['Constant Infusion'],
                              ['Param', 'Bias', 'STDEV', 'MSE', 'Bias (%)'],
                              ['K1', organ_constant_k1_bias, organ_constant_k1_std, organ_constant_k1_mse, organ_constant_k1_bias_per], 
                              ['k2', organ_constant_k2_bias, organ_constant_k2_std, organ_constant_k2_mse, organ_constant_k2_bias_per],
                              ['vb', organ_constant_vb_bias, organ_constant_vb_std, organ_constant_vb_mse, organ_constant_vb_bias_per]])
            writer.writerow([' '])
            writer.writerows([['Bolus Infusion'],
                              ['Param', 'Bias', 'STDEV', 'MSE', 'Bias (%)'],
                              ['K1', organ_bolinf_k1_bias, organ_bolinf_k1_std, organ_bolinf_k1_mse, organ_bolinf_k1_bias_per], 
                              ['k2', organ_bolinf_k2_bias, organ_bolinf_k2_std, organ_bolinf_k2_mse, organ_bolinf_k2_bias_per],
                              ['vb', organ_bolinf_vb_bias, organ_bolinf_vb_std, organ_bolinf_vb_mse, organ_bolinf_vb_bias_per]])
            
            
    elif organ == 'Kidneys':
        organ_bolus = np.loadtxt(f'{path1}/{organ}_bolus_params.txt')
        organ_constant = np.loadtxt(f'{path1}/{organ}_constant_params.txt')
        organ_bolinf = np.loadtxt(f'{path1}/{organ}_bolinf_params.txt')

        organ_df = pd.DataFrame(data = {'K1_bolus' : organ_bolus[:, 0], 'k2_bolus' : organ_bolus[:, 1], 'k3_bolus' : organ_bolus[:, 3], 'k4_bolus' : organ_bolus[:, 4],'vb_bolus' : organ_bolus[:, 2], 
                                        'K1_constant' : organ_constant[:, 0], 'k2_constant' : organ_constant[:, 1], 'k3_constant' : organ_constant[:, 3], 'k4_constant' : organ_constant[:, 4], 'vb_constant' : organ_constant[:, 2],
                                        'K1_bolinf' : organ_bolinf[:, 0], 'k2_bolinf' : organ_bolinf[:, 1], 'k3_bolinf' : organ_bolinf[:, 3], 'k4_bolinf' : organ_bolinf[:, 4], 'vb_bolinf' : organ_bolinf[:, 2]})

        organ_bolus_k1_bias, organ_bolus_k1_std, organ_bolus_k1_mse, organ_bolus_k1_bias_per = table_generator(organ_df['K1_bolus'], organ_params_df[organ][0])
        organ_bolus_k2_bias, organ_bolus_k2_std, organ_bolus_k2_mse, organ_bolus_k2_bias_per = table_generator(organ_df['k2_bolus'], organ_params_df[organ][1])
        organ_bolus_k3_bias, organ_bolus_k3_std, organ_bolus_k3_mse, organ_bolus_k3_bias_per = table_generator(organ_df['k3_bolus'], organ_params_df[organ][2])
        organ_bolus_k4_bias, organ_bolus_k4_std, organ_bolus_k4_mse, organ_bolus_k4_bias_per = table_generator(organ_df['k4_bolus'], organ_params_df[organ][3])
        organ_bolus_vb_bias, organ_bolus_vb_std, organ_bolus_vb_mse, organ_bolus_vb_bias_per = table_generator(organ_df['vb_bolus'], organ_params_df[organ][4])

        organ_constant_k1_bias, organ_constant_k1_std, organ_constant_k1_mse, organ_constant_k1_bias_per = table_generator(organ_df['K1_constant'], organ_params_df[organ][0])
        organ_constant_k2_bias, organ_constant_k2_std, organ_constant_k2_mse, organ_constant_k2_bias_per = table_generator(organ_df['k2_constant'], organ_params_df[organ][1])
        organ_constant_k3_bias, organ_constant_k3_std, organ_constant_k3_mse, organ_constant_k3_bias_per = table_generator(organ_df['k3_constant'], organ_params_df[organ][2])
        organ_constant_k4_bias, organ_constant_k4_std, organ_constant_k4_mse, organ_constant_k4_bias_per = table_generator(organ_df['k4_constant'], organ_params_df[organ][3])
        organ_constant_vb_bias, organ_constant_vb_std, organ_constant_vb_mse, organ_constant_vb_bias_per = table_generator(organ_df['vb_constant'], organ_params_df[organ][4])

        organ_bolinf_k1_bias, organ_bolinf_k1_std, organ_bolinf_k1_mse, organ_bolinf_k1_bias_per = table_generator(organ_df['K1_bolinf'], organ_params_df[organ][0])
        organ_bolinf_k2_bias, organ_bolinf_k2_std, organ_bolinf_k2_mse, organ_bolinf_k2_bias_per = table_generator(organ_df['k2_bolinf'], organ_params_df[organ][1])
        organ_bolinf_k3_bias, organ_bolinf_k3_std, organ_bolinf_k3_mse, organ_bolinf_k3_bias_per = table_generator(organ_df['k3_bolinf'], organ_params_df[organ][2])
        organ_bolinf_k4_bias, organ_bolinf_k4_std, organ_bolinf_k4_mse, organ_bolinf_k4_bias_per = table_generator(organ_df['k4_bolinf'], organ_params_df[organ][3])
        organ_bolinf_vb_bias, organ_bolinf_vb_std, organ_bolinf_vb_mse, organ_bolinf_vb_bias_per = table_generator(organ_df['vb_bolinf'], organ_params_df[organ][4])

        with open(f'{path2}/{organ}_bias.csv', 'w', newline = '') as f:
            writer = csv.writer(f)
            writer.writerows([['Bolus Injection'],
                              ['Param', 'Bias', 'STDEV', 'MSE', 'Bias (%)'],
                              ['K1', organ_bolus_k1_bias, organ_bolus_k1_std, organ_bolus_k1_mse, organ_bolus_k1_bias_per], 
                              ['k2', organ_bolus_k2_bias, organ_bolus_k2_std, organ_bolus_k2_mse, organ_bolus_k2_bias_per],
                              ['k3', organ_bolus_k3_bias, organ_bolus_k3_std, organ_bolus_k3_mse, organ_bolus_k3_bias_per],
                              ['k4', organ_bolus_k4_bias, organ_bolus_k4_std, organ_bolus_k4_mse, organ_bolus_k4_bias_per],
                              ['vb', organ_bolus_vb_bias, organ_bolus_vb_std, organ_bolus_vb_mse, organ_bolus_vb_bias_per]
                              ])
            writer.writerow([' '])
            writer.writerows([['Constant Infusion'],
                              ['Param', 'Bias', 'STDEV', 'MSE', 'Bias (%)'],
                              ['K1', organ_constant_k1_bias, organ_constant_k1_std, organ_constant_k1_mse, organ_constant_k1_bias_per], 
                              ['k2', organ_constant_k2_bias, organ_constant_k2_std, organ_constant_k2_mse, organ_constant_k2_bias_per],
                              ['k3', organ_constant_k3_bias, organ_constant_k3_std, organ_constant_k3_mse, organ_constant_k3_bias_per],
                              ['k4', organ_constant_k4_bias, organ_constant_k4_std, organ_constant_k4_mse, organ_constant_k4_bias_per],
                              ['vb', organ_constant_vb_bias, organ_constant_vb_std, organ_constant_vb_mse, organ_constant_vb_bias_per]])
            writer.writerow([' '])
            writer.writerows([['Bolus Infusion'],
                              ['Param', 'Bias', 'STDEV', 'MSE', 'Bias (%)'],
                              ['K1', organ_bolinf_k1_bias, organ_bolinf_k1_std, organ_bolinf_k1_mse, organ_bolinf_k1_bias_per], 
                              ['k2', organ_bolinf_k2_bias, organ_bolinf_k2_std, organ_bolinf_k2_mse, organ_bolinf_k2_bias_per],
                              ['k3', organ_bolinf_k3_bias, organ_bolinf_k3_std, organ_bolinf_k3_mse, organ_bolinf_k3_bias_per],
                              ['k4', organ_bolinf_k4_bias, organ_bolinf_k4_std, organ_bolinf_k4_mse, organ_bolinf_k4_bias_per],
                              ['vb', organ_bolinf_vb_bias, organ_bolinf_vb_std, organ_bolinf_vb_mse, organ_bolinf_vb_bias_per]])
            
        
    elif organ == 'Femur':
        organ_bolus = np.loadtxt(f'{path1}/{organ}_bolus_params.txt')
        organ_constant = np.loadtxt(f'{path1}/{organ}_constant_params.txt')
        organ_bolinf = np.loadtxt(f'{path1}/{organ}_bolinf_params.txt')

        organ_df = pd.DataFrame(data = {'K1_bolus' : organ_bolus[:, 0], 'k2_bolus' : organ_bolus[:, 1], 'k3_bolus' : organ_bolus[:, 3], 'vb_bolus' : organ_bolus[:, 2], 
                                        'K1_constant' : organ_constant[:, 0], 'k2_constant' : organ_constant[:, 1], 'k3_constant' : organ_constant[:, 3], 'vb_constant' : organ_constant[:, 2],
                                        'K1_bolinf' : organ_bolinf[:, 0], 'k2_bolinf' : organ_bolinf[:, 1], 'k3_bolinf' : organ_bolinf[:, 3], 'vb_bolinf' : organ_bolinf[:, 2]})

        organ_bolus_k1_bias, organ_bolus_k1_std, organ_bolus_k1_mse, organ_bolus_k1_bias_per = table_generator(organ_df['K1_bolus'], organ_params_df[organ][0])
        organ_bolus_k2_bias, organ_bolus_k2_std, organ_bolus_k2_mse, organ_bolus_k2_bias_per = table_generator(organ_df['k2_bolus'], organ_params_df[organ][1])
        organ_bolus_k3_bias, organ_bolus_k3_std, organ_bolus_k3_mse, organ_bolus_k3_bias_per = table_generator(organ_df['k3_bolus'], organ_params_df[organ][2])
        organ_bolus_vb_bias, organ_bolus_vb_std, organ_bolus_vb_mse, organ_bolus_vb_bias_per = table_generator(organ_df['vb_bolus'], organ_params_df[organ][4])

        organ_constant_k1_bias, organ_constant_k1_std, organ_constant_k1_mse, organ_constant_k1_bias_per = table_generator(organ_df['K1_constant'], organ_params_df[organ][0])
        organ_constant_k2_bias, organ_constant_k2_std, organ_constant_k2_mse, organ_constant_k2_bias_per = table_generator(organ_df['k2_constant'], organ_params_df[organ][1])
        organ_constant_k3_bias, organ_constant_k3_std, organ_constant_k3_mse, organ_constant_k3_bias_per = table_generator(organ_df['k3_constant'], organ_params_df[organ][2])
        organ_constant_vb_bias, organ_constant_vb_std, organ_constant_vb_mse, organ_constant_vb_bias_per = table_generator(organ_df['vb_constant'], organ_params_df[organ][4])

        organ_bolinf_k1_bias, organ_bolinf_k1_std, organ_bolinf_k1_mse, organ_bolinf_k1_bias_per = table_generator(organ_df['K1_bolinf'], organ_params_df[organ][0])
        organ_bolinf_k2_bias, organ_bolinf_k2_std, organ_bolinf_k2_mse, organ_bolinf_k2_bias_per = table_generator(organ_df['k2_bolinf'], organ_params_df[organ][1])
        organ_bolinf_k3_bias, organ_bolinf_k3_std, organ_bolinf_k3_mse, organ_bolinf_k3_bias_per = table_generator(organ_df['k3_bolinf'], organ_params_df[organ][2])
        organ_bolinf_vb_bias, organ_bolinf_vb_std, organ_bolinf_vb_mse, organ_bolinf_vb_bias_per = table_generator(organ_df['vb_bolinf'], organ_params_df[organ][4])

        with open(f'{path2}/{organ}_bias.csv', 'w', newline = '') as f:
            writer = csv.writer(f)
            writer.writerows([['Bolus Injection'],
                              ['Param', 'Bias', 'STDEV', 'MSE', 'Bias (%)'],
                              ['K1', organ_bolus_k1_bias, organ_bolus_k1_std, organ_bolus_k1_mse, organ_bolus_k1_bias_per], 
                              ['k2', organ_bolus_k2_bias, organ_bolus_k2_std, organ_bolus_k2_mse, organ_bolus_k2_bias_per],
                              ['k3', organ_bolus_k3_bias, organ_bolus_k3_std, organ_bolus_k3_mse, organ_bolus_k3_bias_per],
                              ['vb', organ_bolus_vb_bias, organ_bolus_vb_std, organ_bolus_vb_mse, organ_bolus_vb_bias_per]
                              ])
            writer.writerow([' '])
            writer.writerows([['Constant Infusion'],
                              ['Param', 'Bias', 'STDEV', 'MSE', 'Bias (%)'],
                              ['K1', organ_constant_k1_bias, organ_constant_k1_std, organ_constant_k1_mse, organ_constant_k1_bias_per], 
                              ['k2', organ_constant_k2_bias, organ_constant_k2_std, organ_constant_k2_mse, organ_constant_k2_bias_per],
                              ['k3', organ_constant_k3_bias, organ_constant_k3_std, organ_constant_k3_mse, organ_constant_k3_bias_per],
                              ['vb', organ_constant_vb_bias, organ_constant_vb_std, organ_constant_vb_mse, organ_constant_vb_bias_per]])
            writer.writerow([' '])
            writer.writerows([['Bolus Infusion'],
                              ['Param', 'Bias', 'STDEV', 'MSE', 'Bias (%)'],
                              ['K1', organ_bolinf_k1_bias, organ_bolinf_k1_std, organ_bolinf_k1_mse, organ_bolinf_k1_bias_per], 
                              ['k2', organ_bolinf_k2_bias, organ_bolinf_k2_std, organ_bolinf_k2_mse, organ_bolinf_k2_bias_per],
                              ['k3', organ_bolinf_k3_bias, organ_bolinf_k3_std, organ_bolinf_k3_mse, organ_bolinf_k3_bias_per],
                              ['vb', organ_bolinf_vb_bias, organ_bolinf_vb_std, organ_bolinf_vb_mse, organ_bolinf_vb_bias_per]])
            




        

### Testing

# bolus = np.zeros((2, 3, 3))
# constant = np.zeros((2, 3, 3))
# a = [1, 2, 3]
# b = [4, 5, 6]
# c = [7, 8, 9]

# d = [10, 20, 30]
# e = [40, 50, 60]
# f = [70, 80, 90]

# g = [100, 200, 300]
# h = [400, 500, 600]
# k = [700, 800, 900]

# l = [1000, 2000, 3000]
# m = [4000, 5000, 6000]
# n = [7000, 8000, 9000]

# bolus[0, :] = [a, b, c]
# bolus[1, :] = [d, e, f]

# constant[0, :] = [g, h, k]
# constant[1, :] = [l, m, n]

# data = np.array([bolus, constant])

# # print(bolus)
# # print(bolus[0, :])
# # print(bolus[0, 0, 1])
# # print(bolus[0, 0])

# print(data[0, 0, 0, 0])
# print(data[0, 0, 0, :])
# print(data[0, 1, 0, :])
# print(data[0, :, 0, :])


### MAIN

mouse3, mouse3_int, mouse3_time, mouse3_weights, mouse3_zc = input_data('Mouse_3_16A0823', False)
mouse3_deg, mouse3_int_deg, mouse3_time_deg, mouse3_weights_deg, mouse3_zc_deg = input_data('Mouse_3_16A0823', True)

C0 = mouse3_int.Vena_Cava
t_peak = mouse3_int.loc[mouse3_int['Vena_Cava'].idxmax()]['Time']

params_exp = lmfit.Parameters()
params_exp.add('p1', 6.677)               # p1 = 6.677
params_exp.add('p2', 9.066)               # p2 = 9.066
params_exp.add('p3', 0.727)               # p3 = 0.727
params_exp.add('p4', 0.043)               # p4 = 0.043
params_exp.add('p5', 1.880)               # p5 = 1.880
params_exp.add('p6', 0.530)               # p6 = 0.530
params_exp.add('a', 8.761, min = 0)       # a = 8.761
params_exp.add('b', -0.00349)              # b = -0.00349



frametimes_s = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 210, 240, 300, 420, 540, 840, 1140, 1440, 1740, 2040, 2340, 2640, 2940, 3240, 3540])
frametimes_m = frametimes_s / 60

frametimes_mid_m = (frametimes_m[1:] + frametimes_m[:-1]) / 2


frame_lengths_s = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 30, 30, 60, 120, 120, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300])
frame_lengths_m = frame_lengths_s / 60


bolus_fitted = expo(mouse3_int.Time, [params_exp['p1'].value, params_exp['p3'].value, params_exp['p5'].value],
                                         [params_exp['p2'].value, params_exp['p4'].value, params_exp['p6'].value],
                                         [params_exp['a'].value, params_exp['b'].value], t_peak)

bolus_fitted_deg = expo(mouse3_int_deg.Time, [params_exp['p1'].value, params_exp['p3'].value, params_exp['p5'].value],
                                         [params_exp['p2'].value, params_exp['p4'].value, params_exp['p6'].value],
                                         [params_exp['a'].value, params_exp['b'].value], t_peak)



# T_f_deg, T0_deg, dt_deg, time = mouse3_time_deg
# time = time[:-1]
# list = []
# C0_list = []

# init = [0.0]
# N_t = int(round((T_f_deg - T0_deg)/dt_deg))# - 1
# u = np.zeros((N_t + 1, len(init)))
# t = np.linspace(T0_deg, T0_deg + N_t*dt_deg, len(u))

# print(N_t)
# print(len(u))
# print(t)
# print(len(t))

# #print(len(np.arange(T0_deg, T_f_deg + dt_deg, dt_deg)))
# # print(T_f_deg)
# print(len(time))
# # print((time - T0_deg)/dt_deg)
# # print((T_f_deg - T0_deg)/dt_deg)
# print((t - T0_deg)/dt_deg)
# for t in time:
#     ind = int(round((t - T0_deg)/dt_deg))
#     list.append(ind)
#     C0_list.append(bolus_fitted_deg[ind])



# #print(len(list))
# #print(list[-10:])

# #print(C0_list)
# #print(len(C0_list))
# print(len(mouse3_int.Vena_Cava))
# print(len(mouse3_int_deg.Vena_Cava))
# print(len(bolus_fitted_deg))

# print(mouse3_time_deg)
# print(mouse3_time_deg[3])

# print(len(mouse3_time_deg))
# print(len(mouse3_time_deg[3]))



ci = const_infusion(mouse3_int)
ci_deg = const_infusion(mouse3_int_deg)

bolus_inf = bolus_infusion(mouse3_int, 30)      # Kbol of 30 was chosen as appropriate here but could be changed if needed
bolus_inf_deg = bolus_infusion(mouse3_int_deg, 30)

# # Comparsion of input functions graph
# plt.plot(mouse3_int.Time, bolus_fitted, color = 'b', label = 'Bolus')
# plt.plot(mouse3_int.Time, ci, color = 'g', label = 'Constant Infusion')
# plt.plot(mouse3_int.Time, bolus_inf, color = 'm', label = 'Bolus Infusion')
# plt.title('Comparison of Input Functions')
# plt.xlabel('Time (Minutes)')
# plt.ylabel('Activity Concentration (kBq/cc)')
# plt.legend(loc = 7, fontsize = 'x-small')
# plt.show()


# # Comparison of input functions subplot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (8,8), constrained_layout = False, tight_layout = True)
fig.supxlabel('Time (minutes)') #, y = 0.0)
fig.supylabel('Activity concentration (MBq/ml)') #, x = 0.06)
fig.suptitle('Comparison of arterial input functions for different injection protocols')

ax1.plot(mouse3_int.Time, bolus_fitted, color = 'b', label = 'Bolus')
ax1.plot(mouse3_int.Time, ci, color = 'g', label = 'Constant Infusion')
ax1.plot(mouse3_int.Time, bolus_inf, color = 'm', label = 'Bolus Infusion')
ax1.set_title('All Injection Protocols')
ax1.legend(loc = 7, fontsize = 'x-small')
#ax1.set_ylim(0, 10)

ax2.plot(mouse3_int.Time, bolus_fitted, color = 'b', label = 'Bolus')
ax2.set_title('Bolus Injection')
ax2.legend(loc = 7, fontsize = 'x-small')
ax2.set_ylim(0, 10)

ax3.plot(mouse3_int.Time, ci, color = 'g', label = 'Constant Infusion')
ax3.set_title('Constant Infusion')
ax3.legend(loc = 7, fontsize = 'x-small')
#ax3.set_ylim(0, 10)

ax4.plot(mouse3_int.Time, bolus_inf, color = 'm', label = 'Bolus Infusion')
ax4.set_title('Bolus Infusion')
ax4.legend(loc = 7, fontsize = 'x-small')
#ax4.set_ylim(0, 10)

plt.savefig('Input_Function/Comparison_input_functions')
plt.close()

# Comparison of input functions subplot (Degrado)
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (8,8), constrained_layout = False, tight_layout = True)
# fig.supxlabel('Time (minutes)') #, y = 0.0)
# fig.supylabel('Activity concentration (kBq/ml)') #, x = 0.06)
# fig.suptitle('Comparison of arterial input functions for different injection protocols (Degrado)')

# ax1.plot(mouse3_int_deg.Time, bolus_fitted_deg, color = 'b', label = 'Bolus')
# ax1.plot(mouse3_int_deg.Time, ci_deg, color = 'g', label = 'Constant Infusion')
# ax1.plot(mouse3_int_deg.Time, bolus_inf_deg, color = 'm', label = 'Bolus Infusion')
# ax1.set_title('All Injection Protocols')
# ax1.legend(loc = 7, fontsize = 'x-small')
# #ax1.set_ylim(0, 10)

# ax2.plot(mouse3_int_deg.Time, bolus_fitted_deg, color = 'b', label = 'Bolus')
# ax2.set_title('Bolus Injection')
# ax2.legend(loc = 7, fontsize = 'x-small')
# ax2.set_ylim(0, 10)

# ax3.plot(mouse3_int_deg.Time, ci_deg, color = 'g', label = 'Constant Infusion')
# ax3.set_title('Constant Infusion')
# ax3.legend(loc = 7, fontsize = 'x-small')
# #ax3.set_ylim(0, 10)

# ax4.plot(mouse3_int_deg.Time, bolus_inf_deg, color = 'm', label = 'Bolus Infusion')
# ax4.set_title('Bolus Infusion')
# ax4.legend(loc = 7, fontsize = 'x-small')
# #ax4.set_ylim(0, 10)

# plt.show()


####### SIMULATIONS ##########

#mice = ['Mouse_1_16A0818A', 'Mouse_2_16A0818B', 'Mouse_3_16A0823', 'Mouse_4_17A1012C', 'Mouse_5_17A1101A', 'Mouse_6_17A1101B']
#mice = ['Mouse_2_16A0818B']
#mice = ['Mouse_6_17A1101B']

mice = ['Mouse_2_16A0818B', 'Mouse_3_16A0823', 'Mouse_4_17A1012C', 'Mouse_5_17A1101A']  # Mouse 1 and Mouse 6 are troublemakers

# Putting these here just in case they're needed

scale_factor_df = pd.DataFrame(data = {'Mouse_2_16A0818B' : [0.05, 0.05, 0.05, 0.05, 0.05], 'Mouse_3_16A0823' : [0.05, 0.05, 0.05, 0.05, 0.05], 
                                       'Mouse_4_17A1012C' : [0.05, 0.05, 0.05, 0.005, 0.05], 'Mouse_5_17A1101A' : [0.005, 0.005, 0.05, 0.005, 0.005], 
                                        'Mouse_9_17A1010A' : [0.0005, 0.005, 0.05, 0.0005, 0.05], 'Mouse_16_17A1101C' : [0.05, 0.05, 0.05, 0.05, 0.005], 
                                        'Mouse_17_17A1101D' : [0.05, 0.05, 0.05, 0.05, 0.05]}, index = ['Heart', 'Lungs', 'Kidneys', 'Liver', 'Femur'])


ps_df = pd.DataFrame(data = {'Organ' : ['Heart', 'Lungs', 'Liver', 'Femur', 'Kidneys'], 'PS' : [0.874, 0.874, 0.874, 0.650, 3.532]}, index = ['Heart', 'Lungs', 'Liver', 'Femur', 'Kidneys'])


# Dataframes containing 'optimal' parameters obtained from fitting in Aim 1

heart_params_df = pd.DataFrame(data = {'Mouse_2' : [0.202, 0.000, 0.857, 0.204, 0.0],  'Mouse_3' : [0.373, 0.475, 0.756, 0.429, 0.0], 
                                       'Mouse_4' : [0.682, 0.695, 0.090, 1.689, 1.0], 'Mouse_5' : [0.950, 1.180, 0.588, np.NaN, 0.0]}, index = ['K1', 'k2', 'vb', 'F', 'Delay'])
lungs_params_df = pd.DataFrame(data = {'Mouse_2' : [0.046, 0.0, 0.480, 0.046, 0.0],  'Mouse_3' : [0.185, 0.746, 0.372, 0.187, 0.0], 
                                       'Mouse_4' : [0.338, 0.480, 0.042, 0.375, 1.0], 'Mouse_5' : [0.273, 0.482, 0.305, 0.286, 0.0]}, index = ['K1', 'k2', 'vb', 'F', 'Delay'])
liver_params_df = pd.DataFrame(data = {'Mouse_2' : [0.109, 0.050, 0.294, 0.109, 1.0],  'Mouse_3' : [0.201, 0.353, 0.337, 0.204, 0.0], 
                                       'Mouse_4' : [0.354, 0.403, 0.034, 0.399, 9.0], 'Mouse_5' : [0.407, 0.652, 0.077, 0.488, 2.0]}, index = ['K1', 'k2', 'vb', 'F', 'Delay'])
kidneys_params_df = pd.DataFrame(data = {'Mouse_2' : [1.326, 0.445, 0.256, 1.453, 0.0],  'Mouse_3' : [2.043, 0.967, 0.182, 2.902, 0.0], 
                                       'Mouse_4' : [1.296, 0.295, 0.039, 1.411, 1.0], 'Mouse_5' : [0.967, 0.412, 0.140, 0.996, 2.0]}, index = ['K1', 'k2', 'vb', 'F', 'Delay'])
femur_params_df = pd.DataFrame(data = {'Mouse_2' : [0.178, 0.000, 0.219, 0.000, 0.183, 0.0],  'Mouse_3' : [0.268, 0.057, 0.071, 0.025, 0.304, 4.0], 
                                       'Mouse_4' : [0.268, 0.000, 1.182, 0.000, 0.304, 8.0], 'Mouse_5' : [0.153, 0.036, 0.045, 0.000, 0.156, 0.0]}, 
                                       index = ['K1', 'k2', 'k3', 'vb', 'F', 'Delay'])

# Heart = Mouse 2, Lungs = Mouse 2, Liver = Mouse 2, Kidneys = Mouse 4, Femur = Mouse 3
organ_params_df = pd.DataFrame(data = {'Heart' : [0.705, 1.181, 0.000, 0.000, 0.146, 1.275, 0.0], 'Lungs' : [0.450, 1.539, 0.000, 0.000, 0.304, 0.524, 0.0], 
                                       'Liver' : [0.314, 0.527, 0.000, 0.000, 0.060, 0.436, 0.0], 'Kidneys' : [1.513, 0.031, 1.269, 1.260, 0.100, 0.000, 0.0], 
                                       'Femur' : [0.187, 0.019, 0.029, 0.000, 0.000, 0.332, 2.0]}, index = ['K1', 'k2', 'k3', 'k4', 'vb', 'F', 'Delay'])


# Sensitivity Analysis 2651 and 33851 or 266 and 3386
sens_bolus = np.zeros((2, 7, 3386))      # 2TCM
sens_con = np.zeros((2, 7, 3386))      
sens_bolinf = np.zeros((2, 7, 3386))      

sens_bolus_params = np.zeros((6, 2))
sens_con_params = np.zeros((6, 2))
sens_bolinf_params = np.zeros((6, 2))

sens_bolus_deg = np.zeros((3, 5, 266))      # 2TCM
sens_con_deg = np.zeros((3, 5, 266))      
sens_bolinf_deg = np.zeros((3, 5, 266))      

sens_bolus_params_deg = np.zeros((4, 3))
sens_con_params_deg = np.zeros((4, 3))
sens_bolinf_params_deg = np.zeros((4, 3))

# sens_bolus_err = np.zeros((5, 4))
# sens_con_err = np.zeros((5, 4))
# sens_bolinf_err = np.zeros((5, 4))


# 'Mouse_2_16A0818B', 'Mouse_3_16A0823', 'Mouse_4_17A1012C', 'Mouse_5_17A1101A'

# Heart_data = create_sim_data('Mouse_9_17A1010A', 'Heart', 'Degrado')
# Lungs_data = create_sim_data('Mouse_3_16A0823', 'Lungs', 'Degrado')
# Liver_data = create_sim_data('Mouse_9_17A1010A', 'Liver', 'Degrado')
# Kidneys_data = create_sim_data('Mouse_5_17A1101A', 'Kidneys', '2TCM')
# Femur_data = create_sim_data('Mouse_16_17A1101C', 'Femur', '2TCM')
  

# sensitivity_analysis_display('SA_Graphs', sens_bolus, sens_bolus_params, sens_bolus_deg, sens_bolus_params_deg, 'Bolus-Injection', percent = False)    
# sensitivity_analysis_display('SA_Graphs', sens_con, sens_con_params, sens_con_deg, sens_con_params_deg, 'Constant-Infusion', percent = False)
# sensitivity_analysis_display('SA_Graphs', sens_bolinf, sens_bolinf_params, sens_bolinf_deg, sens_bolinf_params_deg, 'Bolus-Infusion', percent = False)

# heart_params = model_data('Params', Heart_data, mouse3, mouse3_deg, 'Heart', [bolus_fitted, ci, bolus_inf], [bolus_fitted_deg, ci_deg, bolus_inf_deg], frametimes_mid_m, scale_factor_df.loc['Heart', 'Mouse_9_17A1010A'])
# lungs_params = model_data('Params', Lungs_data, mouse3, mouse3_deg, 'Lungs', [bolus_fitted, ci, bolus_inf], [bolus_fitted_deg, ci_deg, bolus_inf_deg], frametimes_mid_m, scale_factor_df.loc['Lungs', 'Mouse_3_16A0823'])
# liver_params = model_data('Params', Liver_data, mouse3, mouse3_deg, 'Liver', [bolus_fitted, ci, bolus_inf], [bolus_fitted_deg, ci_deg, bolus_inf_deg], frametimes_mid_m, scale_factor_df.loc['Liver', 'Mouse_9_17A1010A'])
# kidneys_params = model_data('Params', Kidneys_data, mouse3, mouse3_deg, 'Kidneys', [bolus_fitted, ci, bolus_inf], [bolus_fitted_deg, ci_deg, bolus_inf_deg], frametimes_mid_m, scale_factor_df.loc['Kidneys', 'Mouse_5_17A1101A'])
# femur_params = model_data('Params', Femur_data, mouse3, mouse3_deg, 'Femur', [bolus_fitted, ci, bolus_inf], [bolus_fitted_deg, ci_deg, bolus_inf_deg], frametimes_mid_m, scale_factor_df.loc['Femur', 'Mouse_16_17A1101C'])

calculate_data('Params', 'Params/Bias', 'Heart', organ_params_df)
calculate_data('Params', 'Params/Bias', 'Lungs', organ_params_df)
calculate_data('Params', 'Params/Bias', 'Liver', organ_params_df)
calculate_data('Params', 'Params/Bias', 'Kidneys', organ_params_df)
calculate_data('Params', 'Params/Bias', 'Femur', organ_params_df)

# plt.plot(sens_bolus_deg[0][0], sens_bolus_deg[0][1], color = 'r', label = 'K1')
# plt.plot(sens_bolus_deg[0][0], sens_bolus_deg[0][2], color = 'b', label = 'k2')
# plt.plot(sens_bolus_deg[0][0], sens_bolus_deg[0][3], color = 'g', label = 'vb')
# plt.plot(sens_bolus_deg[0][0], sens_bolus_deg[0][4], color = 'm', label = 'Flow')
# plt.legend(loc = 7, fontsize = 'x-small')
# plt.show()


# plt.plot(sens_bolus_deg[0][0], sens_bolus_deg[0][1], color = 'r', label = 'Heart')
# plt.plot(sens_bolus_deg[1][0], sens_bolus_deg[1][1], color = 'b', label = 'Lungs')
# plt.plot(sens_bolus_deg[2][0], sens_bolus_deg[2][1], color = 'g', label = 'Liver')
# plt.legend(loc = 7, fontsize = 'x-small')
# plt.show()

# plt.plot(sens_bolus[0][0], sens_bolus[0][1], color = 'm', label = 'Kidneys')
# plt.plot(sens_bolus[1][0], sens_bolus[1][1], color = 'c', label = 'Femur')
# plt.legend(loc = 7, fontsize = 'x-small')
# plt.show()

# # Now load in the data from the .mat that was just saved
        # matdata = scio.loadmat("lungs.mat")
        # # And just to check if the data is the same:
        # assert np.all(file_data == matdata['out']), 'Computer go brrrrr'
        # print(matdata['out'].shape)


