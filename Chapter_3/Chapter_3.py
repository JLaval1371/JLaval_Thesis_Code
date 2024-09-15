import pandas as pd
import numpy as np
import math
import scipy as sp
import lmfit
import matplotlib.pyplot as plt
import csv
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
    plt.ylabel('SUV (g/ml)')
    plt.legend(loc = 7, fontsize = 'x-small')
    plt.show()

def organ_plot_nb(data, plot, tag):
    plt.plot(data.Time, data.Heart, label = 'Heart') 
    plt.plot(data.Time, data.Lungs, label = 'Lungs') 
    plt.plot(data.Time, data.Kidneys, label = 'Kidneys') 
    plt.plot(data.Time, data.Femur, label = 'Femur') 
    plt.plot(data.Time, data.Liver, label = 'Liver') 
    plt.plot(data.Time, data.Vena_Cava, label = 'Vena Cava')
    plt.title('Comparison of uptake for all organs (except the bladder)')
    plt.xlabel('Time (min)')
    plt.ylabel('SUV (g/ml)')
    plt.legend(loc = 7, fontsize = 'x-small')

    if plot == True:
        plt.show()
    else:
        plt.savefig(f'{folder}\{tag}')

def aif_plot(data):
    plt.plot(data.Time, data.Vena_Cava, label = 'Vena Cava')
    plt.title('Comparison of uptake for all organs')
    plt.xlabel('Time (min)')
    plt.ylabel('Tracer concentration')
    plt.legend(loc = 7, fontsize = 'x-small')
    plt.show()

def comp_ode_model1(u, t, p):
    K_1 = p[0]
    k_2 = p[1]
    du = np.zeros(1)

    # test whether any concentrations are negative
    # if len(u[u < -1E-12]) > 0:
    #     print("negative u value!")

    ind = int(round((t - T0)/dt))

    # if ind == int(round((T_f - T0)/dt)):
    #     return None

    # dC_1 / dt 
    du = K_1 * C0[ind] - k_2 * u[0] 

    # u[0] = C_1

    return du

def comp_ode_model1_deg(u, t, p):
    K_1 = p[0]
    k_2 = p[1]
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

def comp_ode_model2(u, t, p):
    K_1, k_2, k_3, k_4 = p

    du = np.zeros(2)

    # test whether any concentrations are negative
    # if len(u[u < -1E-12]) > 0:
    #     print("negative u value!")

    ind = int(round((t - T0)/dt))

    # if ind == int(round((T_f - T0)/dt)):
    #     return None



    # dC_1 / dt 
    du[0] = K_1 * C0[ind] - (k_2 + k_3) * u[0] + k_4 * u[1]

    # dC_2 / dt
    du[1] = k_3 * u[0] - k_4 * u[1]

    # u[0] = C_1, u[1] = C_2

    return du

def comp_ode_model2_kidney(u, t, p):
    K_1, k_2, k_3, k_4 = p

    du = np.zeros(2)

    # test whether any concentrations are negative
    # if len(u[u < -1E-12]) > 0:
    #     print("negative u value!")

    ind = int(round((t - T0)/dt))

    # if ind == int(round((T_f - T0)/dt)):
    #     return None



    # dC_1 / dt 
    du[0] = K_1 * C0[ind] - (k_2 + k_3) * u[0] 

    # dC_2 / dt
    du[1] = k_3 * u[0] - k_4 * u[1]

    # u[0] = C_1, u[1] = C_2

    return du

def RK4(func, init, dt, T_f, T0, p):
    N_t = int(round((T_f - T0)/dt))# - 1
    f_ = lambda u, t, p: np.asarray(func(u, t, p))
    u = np.zeros((N_t + 1, len(init)))
    k1 = np.zeros((N_t + 1, len(init)))
    k2 = np.zeros((N_t + 1, len(init)))
    k3 = np.zeros((N_t + 1, len(init)))
    k4 = np.zeros((N_t + 1, len(init)))
    t = np.linspace(T0, T0 + N_t*dt, len(u))
    u[0] = init
    
    for n in range(N_t):
        k1[n] = dt * f_(u[n], t[n], p)
        k2[n] = dt * f_(u[n] + k1[n]/2.0, t[n] + dt/2.0, p)
        k3[n] = dt * f_(u[n] + k2[n]/2.0, t[n] + dt/2.0, p)
        k4[n] = dt * f_(u[n] + k3[n], t[n] + dt, p)
        u[n+1] = u[n] + (k1[n] + 2.0 * (k2[n] + k3[n]) + k4[n])/6.0
    
    return u, t

def resid1(params):
    K_1 = params['K1'].value
    k_2 = params['k2'].value
    vb = params['vb'].value
    
    p = [K_1, k_2]
    
    u_out, t = RK4(comp_ode_model1, init, dt, T_f, T0, p)
    
    # plt.scatter(t, u_out, label = 'Model')
    # plt.scatter(t, C1_data, label = 'Data')
    # plt.show()

    #print(u_out)

    model = (1 - vb) * u_out[:, 0] + vb * C0

    #print(model)
    #print(C1_data)
    #print(model - C1_data)

    return (model - C1_data)

def resid1_weighted(params):
    K_1 = params['K1'].value
    k_2 = params['k2'].value
    vb = params['vb'].value
    
    p = [K_1, k_2]
    
    u_out, t = RK4(comp_ode_model1, init, dt, T_f, T0, p)

    model = (1 - vb) * u_out[:, 0] + vb * C0

    func = interp1d(np.array(t, dtype=float), np.array(model, dtype=float), kind='cubic', fill_value = 'extrapolate')
    model = func(np.array(y_time, dtype=float))     # This is the model fit refitted into the original 33 time points

    #result = integrateTrapezium(frame_lengths_m, y_dat)     # This is the approximate decay corrected PET data (TAC) to be used in the weighting calculating
    result = np.array(y_dat)
    # print(result)
    # print(y_dat)
    # print(y_dat - result)
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
    #return resids

def resid1_deg(params):
    K_1 = params['K1'].value
    k_2 = params['k2'].value
    #delay 
    vb = params['vb'].value
    
    p = [K_1, k_2]

    #C0_new = shift(C0, delay)
    ''' Delay stuff would go here in the order of things'''

    u_out, t = RK4(comp_ode_model1_deg, init, dt_deg, T_f_deg, T0_deg, p)

    model = (1 - vb) * u_out[:, 0] + vb * C0_deg

    return (model - C1_data_deg)

def resid1_deg_weighted(params):
    K_1 = params['K1'].value
    k_2 = params['k2'].value
    vb = params['vb'].value
    
    p = [K_1, k_2]

    #C0_new = shift(C0, delay)
    ''' Delay stuff would go here in the order of things'''
    
    u_out, t = RK4(comp_ode_model1_deg, init, dt_deg, T_f_deg, T0_deg, p)
    
    model = (1 - vb) * u_out[:, 0] + vb * np.array(C0_deg, dtype=float)


    func = interp1d(np.array(t, dtype=float), np.array(model, dtype=float), kind='cubic', fill_value = 'extrapolate')
    model = func(np.array(y_time_deg, dtype=float))     # This is the model fit refitted into the original 33 time points

    result = integrateTrapezium(frame_lengths_m, y_dat)     # This is the approximate decay corrected PET data (TAC) to be used in the weighting calculating
   
    resids = model - np.array(y_dat_deg, dtype=float)       # This is the plain residuals to be returned from the function after being multiplied by the weights, final five values are to replace any zero values in result

    #scale_factor = 0.71
    dec_const = math.log(2) / 109.771           # minutes 
    frame_dur = np.zeros(len(frame_lengths_m))
    exp_decay = np.zeros(len(frame_lengths_m))

    for i in range(len(frame_lengths_m)):
        frame_dur[i] = frame_lengths_m[i]
        exp_decay[i] = math.exp(- dec_const * framemidtimes_m[i])

        if result[i] == 0:
            result[i] = np.mean(resids[-3:])        # Maybe replace this value with an average of the last 5 residuals, 3 for degrado

    sigma_sq = scale_factor * (result / (frame_dur * exp_decay))
    #sigma_sq = 0.05 * (result / (frame_dur * exp_decay))            # Changed scale factor to 0.05 for comparison with results from PMOD using 0.05 as the scale factor
    weights = 1 / sigma_sq
    weights = np.sqrt(weights)
    weights[np.isnan(weights)] = 0.01
    #print(weights)


    return (weights[:21] * resids)
    #return resids

def resid2(params):
    K_1 = params['K1'].value
    k_2 = params['k2'].value
    k_3 = params['k3'].value
    k_4 = params['k4'].value
    vb = params['vb'].value

    p = [K_1, k_2, k_3, k_4]

    u_out, t = RK4(comp_ode_model2, init2, dt, T_f, T0, p)

    model = (1 - vb) * (u_out[:, 0] + u_out[:,1]) + vb * C0

    # df = pd.DataFrame({'Time' : t, 'C1' : model})
    # func = interp1d(np.array(t, dtype=float), np.array(model, dtype=float), kind='cubic', fill_value = 'extrapolate')
    # result = pd.Series([], dtype = 'float64')

    # plt.plot(t, func(t), label = 'Interpolation', color = 'b')
    # plt.plot(y_time, y_dat, label = 'Original Data', color = 'r')
    # for i in range(len(frametimes_m) - 1):
    #     plt.axvline(x = frametimes_m[i+1], color = 'g', lw = 0.75)
    # plt.legend()
    # plt.show()

    # sqrt_weights = np.zeros(len(frametimes_m) - 1, dtype = 'float64')

    #result = pd.Series(integrateTrapezium(frame_lengths_m, y_dat), dtype = 'float64')


    # result = result / frame_lengths_m
    # sqrt_weights = np.sqrt(result)

    # for i in range(len(frametimes_m) - 1):
    #     if i < zc:
    #         result[i] = 0
    #         sqrt_weights[i] = 1
        # else:
        #     # frame = df[df['Time'].between(i, i+1, inclusive=True)]
        #     # counts, error = quad(func, frametimes_m[i], frametimes_m[i+1])

        #     sqrt_weights[i] = np.sqrt(counts)
        #     result[i] = counts

        #     #result[i] = frame.C1.sum() / ((i+1) - i)

    # print(sqrt_weights)
    # print(result)
    # print(y_dat)
    # print(y_time)

    # plt.plot(t, C1_data, color = 'r')
    # plt.plot(y_time, result, 'bo')
    # for i in range(len(frametimes_m) - 1):
    #     plt.axvline(x = frametimes_m[i+1], color = 'g', lw = 0.75)
    # plt.show()

    # print(w)
    # print((result - y_dat) * w)
    # print(result - y_dat)
    # print(model - C1_data)

    #print(np.sum((result - y_dat) * w))
    #print(np.sum(model - C1_data))
    
    # print(params['K1'].value)
    # print(params['k2'].value)
    # print(params['k3'].value)
    # print(params['k4'].value)
    # print(params['vb'].value)
    
    return np.array((model - C1_data) * 1, dtype = 'float64')   
    #return np.array((result - y_dat) * 1, dtype = 'float64')

def resid2_weighted(params):
    K_1 = params['K1'].value
    k_2 = params['k2'].value
    k_3 = params['k3'].value
    k_4 = params['k4'].value
    vb = params['vb'].value

    p = [K_1, k_2, k_3, k_4]

    u_out, t = RK4(comp_ode_model2, init2, dt, T_f, T0, p)

    model = (1 - vb) * (u_out[:, 0] + u_out[:,1]) + vb * C0

    func = interp1d(np.array(t, dtype=float), np.array(model, dtype=float), kind='cubic', fill_value = 'extrapolate')
    model = func(np.array(y_time, dtype=float))     # This is the model fit refitted into the original 33 time points

    result = integrateTrapezium(frame_lengths_m, y_dat)     # This is the approximate decay corrected PET data (TAC) to be used in the weighting calculating

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
    #return resids

def resid2_kidneys(params):
    K_1 = params['K1'].value
    k_2 = params['k2'].value
    k_3 = params['k3'].value
    k_4 = params['k4'].value
    vb = params['vb'].value

    p = [K_1, k_2, k_3, k_4]

    u_out, t = RK4(comp_ode_model2_kidney, init2, dt, T_f, T0, p)

    model = (1 - vb) * (u_out[:, 0] + u_out[:,1]) + vb * C0

    return np.array((model - C1_data) * 1, dtype = 'float64')

def resid2_kidneys_weighted(params):
    K_1 = params['K1'].value
    k_2 = params['k2'].value
    k_3 = params['k3'].value
    k_4 = params['k4'].value
    vb = params['vb'].value

    p = [K_1, k_2, k_3, k_4]

    u_out, t = RK4(comp_ode_model2_kidney, init2, dt, T_f, T0, p)

    model = (1 - vb) * (u_out[:, 0] + u_out[:,1]) + vb * C0

    func = interp1d(np.array(t, dtype=float), np.array(model, dtype=float), kind='cubic', fill_value = 'extrapolate')
    model = func(np.array(y_time, dtype=float))     # This is the model fit refitted into the original 33 time points

    result = integrateTrapezium(frame_lengths_m, y_dat)     # This is the approximate decay corrected PET data (TAC) to be used in the weighting calculating
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
    #return resids

# def expo(x, A, lambd):
#     p1, p3, p5 = A        # p1 = 3.7, p2 = 3
#     p2, p4, p6 = lambd

#     return p1*np.exp(-p2*x) + p3*np.exp(-p4*x) + p5*np.exp(-p6*x)

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

def expo2(x, A, lambd, t_peak):
    p1, p3, p5 = A        
    p2, p4, p6 = lambd
    
    result = np.array([])

    for i in x:
        result = np.append(result, [p1*np.exp(-p2*(i - t_peak)) + p3*np.exp(-p4*(i - t_peak)) + p5*np.exp(-p6*(i - t_peak))]) 
         
    return result

def expo_resid(params_exp):
    coeff = [params_exp['p1'].value, params_exp['p3'].value, params_exp['p5'].value]
    d_const = [params_exp['p2'].value, params_exp['p4'].value, params_exp['p6'].value]
    line = [params_exp['a'].value, params_exp['b'].value]

    model = expo(mouse_int.Time, coeff, d_const, line, t_peak)

    #print(np.isnan(model - C0).any())

    return model - C0_orig

def step(x):
    # if (x >= data_int.Time.min()) and (x <= data_int.Time.max()):
    #     return step_const*1
    # else:
    #     return 0

    func = interp1d(x.Time, x.Vena_Cava, kind='cubic', fill_value = 'extrapolate')

    counts, error = quad(func, frametimes_m[0], frametimes_m[-1])

    step_const = counts/ (frametimes_m[-1] - frametimes_m[0])

    return step_const * np.ones(len(x.Time))

def weights_dc(func, data, col, decay):
    frametimes = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 210, 240, 300]) #, 420, 540, 840, 1140, 1440, 1740, 2040, 2340, 2640, 2940, 3240, 3540])
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

def shift(arr, num):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = 0
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = np.mean(arr[-5:])
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    
    return result

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

    return conv_bolus

def blood_curve(sheet):
    data = pd.read_excel('Blood_fits_parameters.xlsx', sheet_name=sheet, engine = 'openpyxl', dtype = {'Params' : float})

    return data

def sensitivity_analysis(fit, param, delta_param, model, organ):
    h_scale = delta_param

    if model == 'Degrado':
        if param == 'K1':
            K_1 = fit.params['K1'].value
            k_2 = fit.params['k2'].value
            vb = fit.params['vb'].value

            h = h_scale * K_1
            
            x1, t1 = RK4(comp_ode_model1_deg, init, dt_deg, T_f_deg, T0_deg, [K_1 + h, k_2])
            x2, t2 = RK4(comp_ode_model1_deg, init, dt_deg, T_f_deg, T0_deg, [K_1 - h, k_2])

            x1 = (1 - vb) * x1[:, 0] + vb * C0_deg
            x2 = (1 - vb) * x2[:, 0] + vb * C0_deg

            deriv = (x1 - x2)/2*h

        elif param == 'k2':
            K_1 = fit.params['K1'].value
            k_2 = fit.params['k2'].value
            vb = fit.params['vb'].value

            h = h_scale * k_2

            x1, t1 = RK4(comp_ode_model1_deg, init, dt_deg, T_f_deg, T0_deg, [K_1, k_2 + h])
            x2, t2 = RK4(comp_ode_model1_deg, init, dt_deg, T_f_deg, T0_deg, [K_1, k_2 - h])

            x1 = (1 - vb) * x1[:, 0] + vb * C0_deg
            x2 = (1 - vb) * x2[:, 0] + vb * C0_deg

            deriv = (x1 - x2)/2*h

        elif param == 'vb':
            K_1 = fit.params['K1'].value
            k_2 = fit.params['k2'].value
            vb = fit.params['vb'].value

            h = h_scale * vb

            x1, t1 = RK4(comp_ode_model1_deg, init, dt_deg, T_f_deg, T0_deg, [K_1, k_2])

            x2 = (1 - (vb + h)) * x1[:, 0] + (vb + h) * C0_deg
            x3 = (1 - (vb - h)) * x1[:, 0] + (vb - h) * C0_deg

            deriv = (x2 - x3)/2*h

        elif param == 'F':
            K_1 = fit.params['K1'].value
            k_2 = fit.params['k2'].value
            vb = fit.params['vb'].value

            ps = ps_df.loc[organ, 'PS']
            f_values = np.array([])

            for i in np.linspace(-9, 9, 19):
                f_root = fsolve(flow_func, [K_1 + (i/10)*K_1], args = (K_1, ps))
                f = f_root[0]
                f_values = np.append(f_values, f)
            
            f = find_nearest(f_values, K_1)

            h = h_scale * f

            x1, t1 = RK4(comp_ode_model1_deg, init, dt_deg, T_f_deg, T0_deg, [k1_flow(f + h, ps), k_2])
            x2, t2 = RK4(comp_ode_model1_deg, init, dt_deg, T_f_deg, T0_deg, [k1_flow(f - h, ps), k_2])

            x1 = (1 - vb) * x1[:, 0] + vb * C0_deg
            x2 = (1 - vb) * x2[:, 0] + vb * C0_deg

            deriv = (x1 - x2)/2*h

            return deriv, t1, f

    elif model == '1TCM':
        if param == 'K1':
            K_1 = fit.params['K1'].value
            k_2 = fit.params['k2'].value
            vb = fit.params['vb'].value

            h = h_scale * K_1
            
            x1, t1 = RK4(comp_ode_model1, init, dt, T_f, T0, [K_1 + h, k_2])
            x2, t2 = RK4(comp_ode_model1, init, dt, T_f, T0, [K_1 - h, k_2])

            x1 = (1 - vb) * x1[:, 0] + vb * C0
            x2 = (1 - vb) * x2[:, 0] + vb * C0

            deriv = (x1 - x2)/2*h

        elif param == 'k2':
            K_1 = fit.params['K1'].value
            k_2 = fit.params['k2'].value
            vb = fit.params['vb'].value

            h = h_scale * k_2

            x1, t1 = RK4(comp_ode_model1, init, dt, T_f, T0, [K_1, k_2 + h])
            x2, t2 = RK4(comp_ode_model1, init, dt, T_f, T0, [K_1, k_2 - h])

            x1 = (1 - vb) * x1[:, 0] + vb * C0
            x2 = (1 - vb) * x2[:, 0] + vb * C0

            deriv = (x1 - x2)/2*h

        elif param == 'vb':
            K_1 = fit.params['K1'].value
            k_2 = fit.params['k2'].value
            vb = fit.params['vb'].value

            h = h_scale * vb

            x1, t1 = RK4(comp_ode_model1, init, dt, T_f, T0, [K_1, k_2])

            x2 = (1 - (vb + h)) * x1[:, 0] + (vb + h) * C0
            x3 = (1 - (vb - h)) * x1[:, 0] + (vb - h) * C0

            deriv = (x2 - x3)/2*h

        elif param == 'F':
            K_1 = fit.params['K1'].value
            k_2 = fit.params['k2'].value
            vb = fit.params['vb'].value

            ps = ps_df.loc[organ, 'PS']
            f_values = np.array([])

            for i in np.linspace(-9, 9, 19):
                f_root = fsolve(flow_func, [K_1 + (i/10)*K_1], args = (K_1, ps))
                f = f_root[0]
                f_values = np.append(f_values, f)
            f = find_nearest(f_values, K_1)

            h = h_scale * f

            x1, t1 = RK4(comp_ode_model1, init, dt, T_f, T0, [k1_flow(f + h, ps), k_2])
            x2, t2 = RK4(comp_ode_model1, init, dt, T_f, T0, [k1_flow(f - h, ps), k_2])

            x1 = (1 - vb) * x1[:, 0] + vb * C0
            x2 = (1 - vb) * x2[:, 0] + vb * C0

            deriv = (x1 - x2)/2*h

            return deriv, t1, f

    elif model == '2TCM':
        if organ == 'Kidneys':
            if param == 'K1':
                K_1 = fit.params['K1'].value
                k_2 = fit.params['k2'].value
                k_3 = fit.params['k3'].value
                k_4 = fit.params['k4'].value
                vb = fit.params['vb'].value

                h = h_scale * K_1

                x1, t1 = RK4(comp_ode_model2, init2, dt, T_f, T0, [K_1 + h, k_2, k_3, k_4])
                x2, t2 = RK4(comp_ode_model2, init2, dt, T_f, T0, [K_1 - h, k_2, k_3, k_4])

                x1 = (1 - vb) * (x1[:, 0] + x1[:, 1]) + vb * C0
                x2 = (1 - vb) * (x2[:, 0] + x2[:, 1]) + vb * C0

                deriv = (x1 - x2)/2*h

            elif param == 'k2':
                K_1 = fit.params['K1'].value
                k_2 = fit.params['k2'].value
                k_3 = fit.params['k3'].value
                k_4 = fit.params['k4'].value
                vb = fit.params['vb'].value

                h = h_scale * k_2

                x1, t1 = RK4(comp_ode_model2, init2, dt, T_f, T0, [K_1, k_2 + h, k_3, k_4])
                x2, t2 = RK4(comp_ode_model2, init2, dt, T_f, T0, [K_1, k_2 - h, k_3, k_4])

                x1 = (1 - vb) * (x1[:, 0] + x1[:, 1]) + vb * C0
                x2 = (1 - vb) * (x2[:, 0] + x2[:, 1]) + vb * C0

                deriv = (x1 - x2)/2*h

            elif param == 'k3':
                K_1 = fit.params['K1'].value
                k_2 = fit.params['k2'].value
                k_3 = fit.params['k3'].value
                k_4 = fit.params['k4'].value
                vb = fit.params['vb'].value

                h = h_scale * k_3

                x1, t1 = RK4(comp_ode_model2, init2, dt, T_f, T0, [K_1, k_2, k_3 + h, k_4])
                x2, t2 = RK4(comp_ode_model2, init2, dt, T_f, T0, [K_1, k_2, k_3 - h, k_4])

                x1 = (1 - vb) * (x1[:, 0] + x1[:, 1]) + vb * C0
                x2 = (1 - vb) * (x2[:, 0] + x2[:, 1]) + vb * C0

                deriv = (x1 - x2)/2*h

            elif param == 'k4':
                K_1 = fit.params['K1'].value
                k_2 = fit.params['k2'].value
                k_3 = fit.params['k3'].value
                k_4 = fit.params['k4'].value
                vb = fit.params['vb'].value

                h = h_scale * k_4

                x1, t1 = RK4(comp_ode_model2, init2, dt, T_f, T0, [K_1, k_2, k_3, k_4 + h])
                x2, t2 = RK4(comp_ode_model2, init2, dt, T_f, T0, [K_1, k_2, k_3, k_4 - h])

                x1 = (1 - vb) * (x1[:, 0] + x1[:, 1]) + vb * C0
                x2 = (1 - vb) * (x2[:, 0] + x2[:, 1]) + vb * C0

                deriv = (x1 - x2)/2*h

            elif param == 'vb':
                K_1 = fit.params['K1'].value
                k_2 = fit.params['k2'].value
                k_3 = fit.params['k3'].value
                k_4 = fit.params['k4'].value
                vb = fit.params['vb'].value

                h = h_scale * vb

                x1, t1 = RK4(comp_ode_model2, init2, dt, T_f, T0, [K_1, k_2, k_3, k_4])

                x2 = (1 - (vb + h)) * (x1[:, 0] + x1[:, 1]) + (vb + h) * C0
                x3 = (1 - (vb - h)) * (x1[:, 0] + x1[:, 1]) + (vb - h) * C0

                deriv = (x2 - x3)/2*h

            elif param == 'F':
                K_1 = fit.params['K1'].value
                k_2 = fit.params['k2'].value
                k_3 = fit.params['k3'].value
                k_4 = fit.params['k4'].value
                vb = fit.params['vb'].value

                ps = ps_df.loc[organ, 'PS']
                f_values = np.array([])

                for i in np.linspace(-9, 9, 19):
                    f_root = fsolve(flow_func, [K_1 + (i/10)*K_1], args = (K_1, ps))
                    f = f_root[0]
                    f_values = np.append(f_values, f)
                
                f = find_nearest(f_values, K_1)

                h = h_scale * f

                x1, t1 = RK4(comp_ode_model2, init2, dt, T_f, T0, [k1_flow(f + h, ps), k_2, k_3, k_4])
                x2, t2 = RK4(comp_ode_model2, init2, dt, T_f, T0, [k1_flow(f - h, ps), k_2, k_3, k_4])

                x1 = (1 - vb) * x1[:, 0] + vb * C0
                x2 = (1 - vb) * x2[:, 0] + vb * C0

                deriv = (x1 - x2)/2*h

                return deriv, t1, f
        
        elif organ == 'Femur':
            if param == 'K1':
                K_1 = fit.params['K1'].value
                k_2 = fit.params['k2'].value
                k_3 = fit.params['k3'].value
                k_4 = fit.params['k4'].value
                vb = fit.params['vb'].value

                h = h_scale * K_1

                x1, t1 = RK4(comp_ode_model2, init2, dt, T_f, T0, [K_1 + h, k_2, k_3, k_4])
                x2, t2 = RK4(comp_ode_model2, init2, dt, T_f, T0, [K_1 - h, k_2, k_3, k_4])

                x1 = (1 - vb) * (x1[:, 0] + x1[:, 1]) + vb * C0
                x2 = (1 - vb) * (x2[:, 0] + x2[:, 1]) + vb * C0

                deriv = (x1 - x2)/2*h

            elif param == 'k2':
                K_1 = fit.params['K1'].value
                k_2 = fit.params['k2'].value
                k_3 = fit.params['k3'].value
                k_4 = fit.params['k4'].value
                vb = fit.params['vb'].value

                h = h_scale * k_2

                x1, t1 = RK4(comp_ode_model2, init2, dt, T_f, T0, [K_1, k_2 + h, k_3, k_4])
                x2, t2 = RK4(comp_ode_model2, init2, dt, T_f, T0, [K_1, k_2 - h, k_3, k_4])

                x1 = (1 - vb) * (x1[:, 0] + x1[:, 1]) + vb * C0
                x2 = (1 - vb) * (x2[:, 0] + x2[:, 1]) + vb * C0

                deriv = (x1 - x2)/2*h

            elif param == 'k3':
                K_1 = fit.params['K1'].value
                k_2 = fit.params['k2'].value
                k_3 = fit.params['k3'].value
                k_4 = fit.params['k4'].value
                vb = fit.params['vb'].value

                h = h_scale * k_3

                x1, t1 = RK4(comp_ode_model2, init2, dt, T_f, T0, [K_1, k_2, k_3 + h, k_4])
                x2, t2 = RK4(comp_ode_model2, init2, dt, T_f, T0, [K_1, k_2, k_3 - h, k_4])

                x1 = (1 - vb) * (x1[:, 0] + x1[:, 1]) + vb * C0
                x2 = (1 - vb) * (x2[:, 0] + x2[:, 1]) + vb * C0

                deriv = (x1 - x2)/2*h

            elif param == 'vb':
                K_1 = fit.params['K1'].value
                k_2 = fit.params['k2'].value
                k_3 = fit.params['k3'].value
                k_4 = fit.params['k4'].value
                vb = fit.params['vb'].value

                h = h_scale * vb

                x1, t1 = RK4(comp_ode_model2, init2, dt, T_f, T0, [K_1, k_2, k_3, k_4])

                x2 = (1 - (vb + h)) * (x1[:, 0] + x1[:, 1]) + (vb + h) * C0
                x3 = (1 - (vb - h)) * (x1[:, 0] + x1[:, 1]) + (vb - h) * C0

                deriv = (x2 - x3)/2*h

            elif param == 'F':
                K_1 = fit.params['K1'].value
                k_2 = fit.params['k2'].value
                k_3 = fit.params['k3'].value
                k_4 = fit.params['k4'].value
                vb = fit.params['vb'].value

                ps = ps_df.loc[organ, 'PS']
                f_values = np.array([])

                for i in np.linspace(-9, 9, 19):
                    f_root = fsolve(flow_func, [K_1 + (i/10)*K_1], args = (K_1, ps))
                    f = f_root[0]
                    f_values = np.append(f_values, f)
                f = find_nearest(f_values, K_1)

                h = h_scale * f

                x1, t1 = RK4(comp_ode_model2, init2, dt, T_f, T0, [k1_flow(f + h, ps), k_2, k_3, k_4])
                x2, t2 = RK4(comp_ode_model2, init2, dt, T_f, T0, [k1_flow(f - h, ps), k_2, k_3, k_4])

                x1 = (1 - vb) * x1[:, 0] + vb * C0
                x2 = (1 - vb) * x2[:, 0] + vb * C0

                deriv = (x1 - x2)/2*h

                return deriv, t1, f

        else:
            if param == 'K1':
                K_1 = fit.params['K1'].value
                k_2 = fit.params['k2'].value
                k_3 = fit.params['k3'].value
                k_4 = fit.params['k4'].value
                vb = fit.params['vb'].value

                h = h_scale * K_1

                x1, t1 = RK4(comp_ode_model2, init2, dt, T_f, T0, [K_1 + h, k_2, k_3, k_4])
                x2, t2 = RK4(comp_ode_model2, init2, dt, T_f, T0, [K_1 - h, k_2, k_3, k_4])

                x1 = (1 - vb) * (x1[:, 0] + x1[:, 1]) + vb * C0
                x2 = (1 - vb) * (x2[:, 0] + x2[:, 1]) + vb * C0

                deriv = (x1 - x2)/2*h

            elif param == 'k2':
                K_1 = fit.params['K1'].value
                k_2 = fit.params['k2'].value
                k_3 = fit.params['k3'].value
                k_4 = fit.params['k4'].value
                vb = fit.params['vb'].value

                h = h_scale * k_2

                x1, t1 = RK4(comp_ode_model2, init2, dt, T_f, T0, [K_1, k_2 + h, k_3, k_4])
                x2, t2 = RK4(comp_ode_model2, init2, dt, T_f, T0, [K_1, k_2 - h, k_3, k_4])

                x1 = (1 - vb) * (x1[:, 0] + x1[:, 1]) + vb * C0
                x2 = (1 - vb) * (x2[:, 0] + x2[:, 1]) + vb * C0

                deriv = (x1 - x2)/2*h

            elif param == 'k3':
                K_1 = fit.params['K1'].value
                k_2 = fit.params['k2'].value
                k_3 = fit.params['k3'].value
                k_4 = fit.params['k4'].value
                vb = fit.params['vb'].value

                h = h_scale * k_3

                x1, t1 = RK4(comp_ode_model2, init2, dt, T_f, T0, [K_1, k_2, k_3 + h, k_4])
                x2, t2 = RK4(comp_ode_model2, init2, dt, T_f, T0, [K_1, k_2, k_3 - h, k_4])

                x1 = (1 - vb) * (x1[:, 0] + x1[:, 1]) + vb * C0
                x2 = (1 - vb) * (x2[:, 0] + x2[:, 1]) + vb * C0

                deriv = (x1 - x2)/2*h

            elif param == 'k4':
                K_1 = fit.params['K1'].value
                k_2 = fit.params['k2'].value
                k_3 = fit.params['k3'].value
                k_4 = fit.params['k4'].value
                vb = fit.params['vb'].value

                h = h_scale * k_4

                x1, t1 = RK4(comp_ode_model2, init2, dt, T_f, T0, [K_1, k_2, k_3, k_4 + h])
                x2, t2 = RK4(comp_ode_model2, init2, dt, T_f, T0, [K_1, k_2, k_3, k_4 - h])

                x1 = (1 - vb) * (x1[:, 0] + x1[:, 1]) + vb * C0
                x2 = (1 - vb) * (x2[:, 0] + x2[:, 1]) + vb * C0

                deriv = (x1 - x2)/2*h

            elif param == 'vb':
                K_1 = fit.params['K1'].value
                k_2 = fit.params['k2'].value
                k_3 = fit.params['k3'].value
                k_4 = fit.params['k4'].value
                vb = fit.params['vb'].value

                h = h_scale * vb

                x1, t1 = RK4(comp_ode_model2, init2, dt, T_f, T0, [K_1, k_2, k_3, k_4])

                x2 = (1 - (vb + h)) * (x1[:, 0] + x1[:, 1]) + (vb + h) * C0
                x3 = (1 - (vb - h)) * (x1[:, 0] + x1[:, 1]) + (vb - h) * C0

                deriv = (x2 - x3)/2*h

            elif param == 'F':
                K_1 = fit.params['K1'].value
                k_2 = fit.params['k2'].value
                k_3 = fit.params['k3'].value
                k_4 = fit.params['k4'].value
                vb = fit.params['vb'].value

                ps = ps_df.loc[organ, 'PS']
                f_values = np.array([])


                for i in np.linspace(-9, 9, 19):
                    f_root = fsolve(flow_func, [K_1 + (i/10)*K_1], args = (K_1, ps))
                    f = f_root[0]
                    f_values = np.append(f_values, f)
                f = find_nearest(f_values, K_1)

                h = h_scale * f

                x1, t1 = RK4(comp_ode_model2, init2, dt, T_f, T0, [k1_flow(f + h, ps), k_2, k_3, k_4])
                x2, t2 = RK4(comp_ode_model2, init2, dt, T_f, T0, [k1_flow(f - h, ps), k_2, k_3, k_4])

                x1 = (1 - vb) * x1[:, 0] + vb * C0
                x2 = (1 - vb) * x2[:, 0] + vb * C0

                deriv = (x1 - x2)/2*h

                return deriv, t1, f
    
    return deriv, t1

def flow_func(F, K1, PS):
    return  F * (1 - np.exp(-PS/F)) - K1

def get_flow(K_1, organ):
    ps = ps_df.loc[organ, 'PS']
    f_values = np.array([])
    
    f_root = fsolve(flow_func, K_1, args = (K_1, ps))
    possible_f = f_root[0]
    f_values = np.append(f_values, possible_f)
            
    f = find_nearest(f_values, K_1)

    return f

def k1_flow(F, PS):
    k1 = F * (1 - np.exp(-PS/F))
    
    return k1

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    
    return array[idx]

def sensitivity_analysis_display_ch3(folder, sub_folder, data, params, error, organ):
    if organ == 'Heart' or organ == 'Lungs' or organ == 'Liver':
        fig, ((ax1, ax2, ax3, ax4, ax5, ax6, ax7), (ax8, ax9, ax10, ax11, ax12, ax13, ax14), (ax15, ax16, ax17, ax18, ax19, ax20, ax21), (ax22, ax23, ax24, ax25, ax26, ax27, ax28)) = plt.subplots(4, 7, figsize = (21,12), constrained_layout = False, tight_layout = True)
        fig.suptitle(f'Sensitivity analysis of all parameters for the {organ}', fontsize = 'x-large')

        # First row (K1)
        ax1.plot(data[0][0], data[0][1], label = 'K1', color = 'b')
        ax1.set_title(f'Mouse 2 \n K1 = {params[0, 0]:.3f}')
        ax1.set_ylabel('dPET/dK1')

        ax2.plot(data[1][0], data[1][1], label = 'K1', color = 'b')
        ax2.set_title(f'Mouse 3 \n K1 = {params[0, 1]:.3f}')

        ax3.plot(data[2][0], data[2][1], label = 'K1', color = 'b')
        ax3.set_title(f'Mouse 4 \n K1 = {params[0, 2]:.3f}')

        ax4.plot(data[3][0], data[3][1], label = 'K1', color = 'b')
        ax4.set_title(f'Mouse 5 \n K1 = {params[0, 3]:.3f}')

        ax5.plot(data[4][0], data[4][1], label = 'K1', color = 'b')
        ax5.set_title(f'Mouse 9 \n K1 = {params[0, 4]:.3f}')

        ax6.plot(data[5][0], data[5][1], label = 'K1', color = 'b')
        ax6.set_title(f'Mouse 16 \n K1 = {params[0, 5]:.3f}')

        ax7.plot(data[6][0], data[6][1], label = 'K1', color = 'b')
        ax7.set_title(f'Mouse 17 \n K1 = {params[0, 6]:.3f}')

        # Second row (k2)
        ax8.plot(data[0][0], data[0][2], label = 'k2', color = 'g')
        ax8.set_title(f'k2 = {params[1, 0]:.3f}')
        ax8.set_ylabel('dPET/dk2')

        ax9.plot(data[1][0], data[1][2], label = 'k2', color = 'g')
        ax9.set_title(f'k2 = {params[1, 1]:.3f}')

        ax10.plot(data[2][0], data[2][2], label = 'k2', color = 'g')
        ax10.set_title(f'k2 = {params[1, 2]:.3f}')

        ax11.plot(data[3][0], data[3][2], label = 'k2', color = 'g')
        ax11.set_title(f'k2 = {params[1, 3]:.3f}')

        ax12.plot(data[4][0], data[4][2], label = 'k2', color = 'g')
        ax12.set_title(f'k2 = {params[1, 4]:.3f}')

        ax13.plot(data[5][0], data[5][2], label = 'k2', color = 'g')
        ax13.set_title(f'k2 = {params[1, 5]:.3f}')

        ax14.plot(data[6][0], data[6][2], label = 'k2', color = 'g')
        ax14.set_title(f'k2 = {params[1, 6]:.3f}')


        # Third row (vb)
        ax15.plot(data[0][0], data[0][3], label = 'vb', color = 'r')
        ax15.set_title(f'vb = {params[2, 0]:.3f}')
        ax15.set_ylabel('dPET/dvb')

        ax16.plot(data[1][0], data[1][3], label = 'vb', color = 'r')
        ax16.set_title(f'vb = {params[2, 1]:.3f}')

        ax17.plot(data[2][0], data[2][3], label = 'vb', color = 'r')
        ax17.set_title(f'vb = {params[2, 2]:.3f}')

        ax18.plot(data[3][0], data[3][3], label = 'vb', color = 'r')
        ax18.set_title(f'vb = {params[2, 3]:.3f}')

        ax19.plot(data[4][0], data[4][3], label = 'vb', color = 'r')
        ax19.set_title(f'vb = {params[2, 4]:.3f}')

        ax20.plot(data[5][0], data[5][3], label = 'vb', color = 'r')
        ax20.set_title(f'vb = {params[2, 5]:.3f}')

        ax21.plot(data[6][0], data[6][3], label = 'vb', color = 'r')
        ax21.set_title(f'vb = {params[2, 6]:.3f}')

        # Fourth row (Flow)
        ax22.plot(data[0][0], data[0][4], label = 'F', color = 'm')
        ax22.set_title(f'F = {params[3, 0]:.3f}')
        ax22.set_ylabel('dPET/dF')

        ax23.plot(data[1][0], data[1][4], label = 'F', color = 'm')
        ax23.set_title(f'F = {params[3, 1]:.3f}')

        ax24.plot(data[2][0], data[2][4], label = 'F', color = 'm')
        ax24.set_title(f'F = {params[3, 2]:.3f}')

        ax25.plot(data[3][0], data[3][4], label = 'F', color = 'm')
        ax25.set_title(f'F = {params[3, 3]:.3f}')

        ax26.plot(data[4][0], data[4][4], label = 'F', color = 'm')
        ax26.set_title(f'F = {params[3, 4]:.3f}')

        ax27.plot(data[5][0], data[5][4], label = 'F', color = 'm')
        ax27.set_title(f'F = {params[3, 5]:.3f}')

        ax28.plot(data[6][0], data[6][4], label = 'F', color = 'm')
        ax28.set_title(f'F = {params[3, 6]:.3f}')

    elif organ == 'Femur':
        fig, ((ax1, ax2, ax3, ax4, ax5, ax6, ax7), (ax8, ax9, ax10, ax11, ax12, ax13, ax14), (ax15, ax16, ax17, ax18, ax19, ax20, ax21), 
            (ax22, ax23, ax24, ax25, ax26, ax27, ax28), (ax29, ax30, ax31, ax32, ax33, ax34, ax35)) = plt.subplots(5, 7, figsize = (21,15), constrained_layout = False, tight_layout = True)
        fig.suptitle(f'Sensitivity analysis of all parameters for the {organ}', fontsize = 'x-large')

        # First row (K1)
        ax1.plot(data[0][0], data[0][1], label = 'K1', color = 'b')
        ax1.set_title(f'Mouse 2 \n K1 = {params[0, 0]:.3f}')
        ax1.set_ylabel('dPET/dK1')

        ax2.plot(data[1][0], data[1][1], label = 'K1', color = 'b')
        ax2.set_title(f'Mouse 3 \n K1 = {params[0, 1]:.3f}')

        ax3.plot(data[2][0], data[2][1], label = 'K1', color = 'b')
        ax3.set_title(f'Mouse 4 \n K1 = {params[0, 2]:.3f}')

        ax4.plot(data[3][0], data[3][1], label = 'K1', color = 'b')
        ax4.set_title(f'Mouse 5 \n K1 = {params[0, 3]:.3f}')

        ax5.plot(data[4][0], data[4][1], label = 'K1', color = 'b')
        ax5.set_title(f'Mouse 9 \n K1 = {params[0, 4]:.3f}')

        ax6.plot(data[5][0], data[5][1], label = 'K1', color = 'b')
        ax6.set_title(f'Mouse 16 \n K1 = {params[0, 5]:.3f}')

        ax7.plot(data[6][0], data[6][1], label = 'K1', color = 'b')
        ax7.set_title(f'Mouse 17 \n K1 = {params[0, 6]:.3f}')

        # Second row (k2)
        ax8.plot(data[0][0], data[0][2], label = 'k2', color = 'g')
        ax8.set_title(f'k2 = {params[1, 0]:.3f}')
        ax8.set_ylabel('dPET/dk2')

        ax9.plot(data[1][0], data[1][2], label = 'k2', color = 'g')
        ax9.set_title(f'k2 = {params[1, 1]:.3f}')

        ax10.plot(data[2][0], data[2][2], label = 'k2', color = 'g')
        ax10.set_title(f'k2 = {params[1, 2]:.3f}')

        ax11.plot(data[3][0], data[3][2], label = 'k2', color = 'g')
        ax11.set_title(f'k2 = {params[1, 3]:.3f}')

        ax12.plot(data[4][0], data[4][2], label = 'k2', color = 'g')
        ax12.set_title(f'k2 = {params[1, 4]:.3f}')

        ax13.plot(data[5][0], data[5][2], label = 'k2', color = 'g')
        ax13.set_title(f'k2 = {params[1, 5]:.3f}')

        ax14.plot(data[6][0], data[6][2], label = 'k2', color = 'g')
        ax14.set_title(f'k2 = {params[1, 6]:.3f}')

        # Third Row (k3)
        ax15.plot(data[0][0], data[0][5], label = 'k3', color = 'y')
        ax15.set_title(f'k3 = {params[4, 0]:.3f}')
        ax15.set_ylabel('dPET/dk3')

        ax16.plot(data[1][0], data[1][5], label = 'k3', color = 'y')
        ax16.set_title(f'k3 = {params[4, 1]:.3f}')

        ax17.plot(data[2][0], data[2][5], label = 'k3', color = 'y')
        ax17.set_title(f'k3 = {params[4, 2]:.3f}')

        ax18.plot(data[3][0], data[3][5], label = 'k3', color = 'y')
        ax18.set_title(f'k3 = {params[4, 3]:.3f}')

        ax19.plot(data[4][0], data[4][5], label = 'k3', color = 'y')
        ax19.set_title(f'k3 = {params[4, 4]:.3f}')

        ax20.plot(data[5][0], data[5][5], label = 'k3', color = 'y')
        ax20.set_title(f'k3 = {params[4, 5]:.3f}')

        ax21.plot(data[6][0], data[6][5], label = 'k3', color = 'y')
        ax21.set_title(f'k3 = {params[4, 6]:.3f}')

        # Fourth row (vb)
        ax22.plot(data[0][0], data[0][3], label = 'vb', color = 'r')
        ax22.set_title(f'vb = {params[2, 0]:.3f}')
        ax22.set_ylabel('dPET/dvb')

        ax23.plot(data[1][0], data[1][3], label = 'vb', color = 'r')
        ax23.set_title(f'vb = {params[2, 1]:.3f}')

        ax24.plot(data[2][0], data[2][3], label = 'vb', color = 'r')
        ax24.set_title(f'vb = {params[2, 2]:.3f}')

        ax25.plot(data[3][0], data[3][3], label = 'vb', color = 'r')
        ax25.set_title(f'vb = {params[2, 3]:.3f}')

        ax26.plot(data[4][0], data[4][3], label = 'vb', color = 'r')
        ax26.set_title(f'vb = {params[2, 4]:.3f}')

        ax27.plot(data[5][0], data[5][3], label = 'vb', color = 'r')
        ax27.set_title(f'vb = {params[2, 5]:.3f}')

        ax28.plot(data[6][0], data[6][3], label = 'vb', color = 'r')
        ax28.set_title(f'vb = {params[2, 6]:.3f}')

        # Fifth row (Flow)
        ax29.plot(data[0][0], data[0][4], label = 'F', color = 'm')
        ax29.set_title(f'F = {params[3, 0]:.3f}')
        ax29.set_ylabel('dPET/dF')

        ax30.plot(data[1][0], data[1][4], label = 'F', color = 'm')
        ax30.set_title(f'F = {params[3, 1]:.3f}')

        ax31.plot(data[2][0], data[2][4], label = 'F', color = 'm')
        ax31.set_title(f'F = {params[3, 2]:.3f}')

        ax32.plot(data[3][0], data[3][4], label = 'F', color = 'm')
        ax32.set_title(f'F = {params[3, 3]:.3f}')

        ax33.plot(data[4][0], data[4][4], label = 'F', color = 'm')
        ax33.set_title(f'F = {params[3, 4]:.3f}')

        ax34.plot(data[5][0], data[5][4], label = 'F', color = 'm')
        ax34.set_title(f'F = {params[3, 5]:.3f}')

        ax35.plot(data[6][0], data[6][4], label = 'F', color = 'm')
        ax35.set_title(f'F = {params[3, 6]:.3f}')

    elif organ == 'Kidneys':
        fig, ((ax1, ax2, ax3, ax4, ax5, ax6, ax7), (ax8, ax9, ax10, ax11, ax12, ax13, ax14), 
              (ax15, ax16, ax17, ax18, ax19, ax20, ax21), (ax22, ax23, ax24, ax25, ax26, ax27, ax28), (ax29, ax30, ax31, ax32, ax33, ax34, ax35), 
              (ax36, ax37, ax38, ax39, ax40, ax41, ax42)) = plt.subplots(6, 7, figsize = (21,15), constrained_layout = False, tight_layout = True)
        fig.suptitle(f'Sensitivity analysis of all parameters for the {organ}', fontsize = 'x-large')

        # First row (K1)
        ax1.plot(data[0][0], data[0][1], label = 'K1', color = 'b')
        ax1.set_title(f'Mouse 2 \n K1 = {params[0, 0]:.3f}')
        ax1.set_ylabel('dPET/dK1')

        ax2.plot(data[1][0], data[1][1], label = 'K1', color = 'b')
        ax2.set_title(f'Mouse 3 \n K1 = {params[0, 1]:.3f}')

        ax3.plot(data[2][0], data[2][1], label = 'K1', color = 'b')
        ax3.set_title(f'Mouse 4 \n K1 = {params[0, 2]:.3f}')

        ax4.plot(data[3][0], data[3][1], label = 'K1', color = 'b')
        ax4.set_title(f'Mouse 5 \n K1 = {params[0, 3]:.3f}')

        ax5.plot(data[4][0], data[4][1], label = 'K1', color = 'b')
        ax5.set_title(f'Mouse 9 \n K1 = {params[0, 4]:.3f}')

        ax6.plot(data[5][0], data[5][1], label = 'K1', color = 'b')
        ax6.set_title(f'Mouse 16 \n K1 = {params[0, 5]:.3f}')

        ax7.plot(data[6][0], data[6][1], label = 'K1', color = 'b')
        ax7.set_title(f'Mouse 17 \n K1 = {params[0, 6]:.3f}')

        # Second row (k2)
        ax8.plot(data[0][0], data[0][2], label = 'k2', color = 'g')
        ax8.set_title(f'k2 = {params[1, 0]:.3f}')
        ax8.set_ylabel('dPET/dk2')

        ax9.plot(data[1][0], data[1][2], label = 'k2', color = 'g')
        ax9.set_title(f'k2 = {params[1, 1]:.3f}')

        ax10.plot(data[2][0], data[2][2], label = 'k2', color = 'g')
        ax10.set_title(f'k2 = {params[1, 2]:.3f}')

        ax11.plot(data[3][0], data[3][2], label = 'k2', color = 'g')
        ax11.set_title(f'k2 = {params[1, 3]:.3f}')

        ax12.plot(data[4][0], data[4][2], label = 'k2', color = 'g')
        ax12.set_title(f'k2 = {params[1, 4]:.3f}')

        ax13.plot(data[5][0], data[5][2], label = 'k2', color = 'g')
        ax13.set_title(f'k2 = {params[1, 5]:.3f}')

        ax14.plot(data[6][0], data[6][2], label = 'k2', color = 'g')
        ax14.set_title(f'k2 = {params[1, 6]:.3f}')

        # Third Row (k3)
        ax15.plot(data[0][0], data[0][5], label = 'k3', color = 'y')
        ax15.set_title(f'k3 = {params[4, 0]:.3f}')
        ax15.set_ylabel('dPET/dk3')

        ax16.plot(data[1][0], data[1][5], label = 'k3', color = 'y')
        ax16.set_title(f'k3 = {params[4, 1]:.3f}')

        ax17.plot(data[2][0], data[2][5], label = 'k3', color = 'y')
        ax17.set_title(f'k3 = {params[4, 2]:.3f}')

        ax18.plot(data[3][0], data[3][5], label = 'k3', color = 'y')
        ax18.set_title(f'k3 = {params[4, 3]:.3f}')

        ax19.plot(data[4][0], data[4][5], label = 'k3', color = 'y')
        ax19.set_title(f'k3 = {params[4, 4]:.3f}')

        ax20.plot(data[5][0], data[5][5], label = 'k3', color = 'y')
        ax20.set_title(f'k3 = {params[4, 5]:.3f}')

        ax21.plot(data[6][0], data[6][5], label = 'k3', color = 'y')
        ax21.set_title(f'k3 = {params[4, 6]:.3f}')

        # Fourth row (k4)
        ax22.plot(data[0][0], data[0][6], label = 'k4', color = 'c')
        ax22.set_title(f'k4 = {params[5, 0]:.3f}')
        ax22.set_ylabel('dPET/dk4')

        ax23.plot(data[1][0], data[1][6], label = 'k4', color = 'c')
        ax23.set_title(f'k4 = {params[5, 1]:.3f}')

        ax24.plot(data[2][0], data[2][6], label = 'k4', color = 'c')
        ax24.set_title(f'k4 = {params[5, 2]:.3f}')

        ax25.plot(data[3][0], data[3][6], label = 'k4', color = 'c')
        ax25.set_title(f'k4 = {params[5, 3]:.3f}')

        ax26.plot(data[4][0], data[4][6], label = 'k4', color = 'c')
        ax26.set_title(f'k4 = {params[5, 4]:.3f}')

        ax27.plot(data[5][0], data[5][6], label = 'k4', color = 'c')
        ax27.set_title(f'k4 = {params[5, 5]:.3f}')

        ax28.plot(data[6][0], data[6][6], label = 'k4', color = 'c')
        ax28.set_title(f'k4 = {params[5, 6]:.3f}')

        # Fifth row (vb)
        ax29.plot(data[0][0], data[0][3], label = 'vb', color = 'r')
        ax29.set_title(f'vb = {params[2, 0]:.3f}')
        ax29.set_ylabel('dPET/dvb')

        ax30.plot(data[1][0], data[1][3], label = 'vb', color = 'r')
        ax30.set_title(f'vb = {params[2, 1]:.3f}')

        ax31.plot(data[2][0], data[2][3], label = 'vb', color = 'r')
        ax31.set_title(f'vb = {params[2, 2]:.3f}')

        ax32.plot(data[3][0], data[3][3], label = 'vb', color = 'r')
        ax32.set_title(f'vb = {params[2, 3]:.3f}')

        ax33.plot(data[4][0], data[4][3], label = 'vb', color = 'r')
        ax33.set_title(f'vb = {params[2, 4]:.3f}')

        ax34.plot(data[5][0], data[5][3], label = 'vb', color = 'r')
        ax34.set_title(f'vb = {params[2, 5]:.3f}')

        ax35.plot(data[6][0], data[6][3], label = 'vb', color = 'r')
        ax35.set_title(f'vb = {params[2, 6]:.3f}')

        # Sixth row (Flow)
        ax36.plot(data[0][0], data[0][4], label = 'F', color = 'm')
        ax36.set_title(f'F = {params[3, 0]:.3f}')
        ax36.set_ylabel('dPET/dF')

        ax37.plot(data[1][0], data[1][4], label = 'F', color = 'm')
        ax37.set_title(f'F = {params[3, 1]:.3f}')

        ax38.plot(data[2][0], data[2][4], label = 'F', color = 'm')
        ax38.set_title(f'F = {params[3, 2]:.3f}')

        ax39.plot(data[3][0], data[3][4], label = 'F', color = 'm')
        ax39.set_title(f'F = {params[3, 3]:.3f}')

        ax40.plot(data[4][0], data[4][4], label = 'F', color = 'm')
        ax40.set_title(f'F = {params[3, 4]:.3f}')

        ax41.plot(data[5][0], data[5][4], label = 'F', color = 'm')
        ax41.set_title(f'F = {params[3, 5]:.3f}')

        ax42.plot(data[6][0], data[6][4], label = 'F', color = 'm')
        ax42.set_title(f'F = {params[3, 6]:.3f}')

    plt.savefig(f'{folder}\{sub_folder}/SA/{organ}_SA')
    plt.close()

    with open(f'{folder}\{sub_folder}/SA/{organ}_params.csv', 'w', newline = '') as f:
        writer = csv.writer(f)
        writer.writerows([['Mouse 2', 'Mouse 3', 'Mouse 4', 'Mouse 5', 'Mouse 9', 'Mouse 16', 'Mouse 17']])
        writer.writerows(params)
        writer.writerows([' '])
        writer.writerows(error)
        writer.writerows([' '])
    
def sensitivity_analysis_display_ch3_combined(folder, sub_folder, data_heart, params_heart, data_lungs, params_lungs, data_liver, params_liver, data_kidneys, params_kidneys, data_femur, params_femur,):        
        fig, ((ax1, ax2, ax3, ax4, ax5, ax6, ax7), (ax8, ax9, ax10, ax11, ax12, ax13, ax14), (ax15, ax16, ax17, ax18, ax19, ax20, ax21), 
                (ax22, ax23, ax24, ax25, ax26, ax27, ax28), (ax29, ax30, ax31, ax32, ax33, ax34, ax35)) = plt.subplots(5, 7, figsize = (21,15), constrained_layout = False, tight_layout = True)
        fig.suptitle(f'Sensitivity analysis of K1, F and vb for all mice and organs', fontsize = 'x-large')

        # First row (Heart)
        ax1.plot(data_heart[0][0], data_heart[0][1], label = 'K1', color = 'b')
        ax1.plot(data_heart[0][0], data_heart[0][4], label = 'F', color = 'm')
        ax1.plot(data_heart[0][0], data_heart[0][3], label = 'vb', color = 'r')
        ax1.set_title(f'Mouse 2: K1 = {params_heart[0, 0]:.3f}, \n F = {params_heart[3, 0]:.3f}, vb = {params_heart[2, 0]:.3f}')
        ax1.set_ylabel('Heart')

        ax2.plot(data_heart[1][0], data_heart[1][1], label = 'K1', color = 'b')
        ax2.plot(data_heart[1][0], data_heart[1][4], label = 'F', color = 'm')
        ax2.plot(data_heart[1][0], data_heart[1][3], label = 'vb', color = 'r')
        ax2.set_title(f'Mouse 3: K1 = {params_heart[0, 1]:.3f}, \n F = {params_heart[3, 1]:.3f}, vb = {params_heart[2, 1]:.3f}')

        ax3.plot(data_heart[2][0], data_heart[2][1], label = 'K1', color = 'b')
        ax3.plot(data_heart[2][0], data_heart[2][4], label = 'F', color = 'm')
        ax3.plot(data_heart[2][0], data_heart[2][3], label = 'vb', color = 'r')
        ax3.set_title(f'Mouse 4: K1 = {params_heart[0, 2]:.3f}, \n F = {params_heart[3, 2]:.3f}, vb = {params_heart[2, 2]:.3f}')

        ax4.plot(data_heart[3][0], data_heart[3][1], label = 'K1', color = 'b')
        ax4.plot(data_heart[3][0], data_heart[3][4], label = 'F', color = 'm')
        ax4.plot(data_heart[3][0], data_heart[3][3], label = 'vb', color = 'r')
        ax4.set_title(f'Mouse 5: K1 = {params_heart[0, 3]:.3f}, \n F = {params_heart[3, 3]:.3f}, vb = {params_heart[2, 3]:.3f}')

        ax5.plot(data_heart[4][0], data_heart[4][1], label = 'K1', color = 'b')
        ax5.plot(data_heart[4][0], data_heart[4][4], label = 'F', color = 'm')
        ax5.plot(data_heart[4][0], data_heart[4][3], label = 'vb', color = 'r')
        ax5.set_title(f'Mouse 9: K1 = {params_heart[0, 4]:.3f}, \n F = {params_heart[3, 4]:.3f}, vb = {params_heart[2, 4]:.3f}')

        ax6.plot(data_heart[5][0], data_heart[5][1], label = 'K1', color = 'b')
        ax6.plot(data_heart[5][0], data_heart[5][4], label = 'F', color = 'm')
        ax6.plot(data_heart[5][0], data_heart[5][3], label = 'vb', color = 'r')
        ax6.set_title(f'Mouse 16: K1 = {params_heart[0, 5]:.3f}, \n F = {params_heart[3, 5]:.3f}, vb = {params_heart[2, 5]:.3f}')

        ax7.plot(data_heart[6][0], data_heart[6][1], label = 'K1', color = 'b')
        ax7.plot(data_heart[6][0], data_heart[6][4], label = 'F', color = 'm')
        ax7.plot(data_heart[6][0], data_heart[6][3], label = 'vb', color = 'r')
        ax7.set_title(f'Mouse 17: K1 = {params_heart[0, 6]:.3f}, \n F = {params_heart[3, 6]:.3f}, vb = {params_heart[2, 6]:.3f}')

        

        # Second row (Lungs)
        ax8.plot(data_lungs[0][0], data_lungs[0][1], label = 'K1', color = 'b')
        ax8.plot(data_lungs[0][0], data_lungs[0][4], label = 'F', color = 'm')
        ax8.plot(data_lungs[0][0], data_lungs[0][3], label = 'vb', color = 'r')
        ax8.set_title(f'K1 = {params_lungs[0, 0]:.3f}, \n F = {params_lungs[3, 0]:.3f}, vb = {params_lungs[2, 0]:.3f}')
        ax8.set_ylabel('Lungs')

        ax9.plot(data_lungs[1][0], data_lungs[1][1], label = 'K1', color = 'b')
        ax9.plot(data_lungs[1][0], data_lungs[1][4], label = 'F', color = 'm')
        ax9.plot(data_lungs[1][0], data_lungs[1][3], label = 'vb', color = 'r')
        ax9.set_title(f'K1 = {params_lungs[0, 1]:.3f}, \n F = {params_lungs[3, 1]:.3f}, vb = {params_lungs[2, 1]:.3f}')

        ax10.plot(data_lungs[2][0], data_lungs[2][1], label = 'K1', color = 'b')
        ax10.plot(data_lungs[2][0], data_lungs[2][4], label = 'F', color = 'm')
        ax10.plot(data_lungs[2][0], data_lungs[2][3], label = 'vb', color = 'r')
        ax10.set_title(f'K1 = {params_lungs[0, 2]:.3f}, \n F = {params_lungs[3, 2]:.3f}, vb = {params_lungs[2, 2]:.3f}')

        ax11.plot(data_lungs[3][0], data_lungs[3][1], label = 'K1', color = 'b')
        ax11.plot(data_lungs[3][0], data_lungs[3][4], label = 'F', color = 'm')
        ax11.plot(data_lungs[3][0], data_lungs[3][3], label = 'vb', color = 'r')
        ax11.set_title(f'K1 = {params_lungs[0, 3]:.3f}, \n F = {params_lungs[3, 3]:.3f}, vb = {params_lungs[2, 3]:.3f}')

        ax12.plot(data_lungs[4][0], data_lungs[4][1], label = 'K1', color = 'b')
        ax12.plot(data_lungs[4][0], data_lungs[4][4], label = 'F', color = 'm')
        ax12.plot(data_lungs[4][0], data_lungs[4][3], label = 'vb', color = 'r')
        ax12.set_title(f'K1 = {params_lungs[0, 4]:.3f}, \n F = {params_lungs[3, 4]:.3f}, vb = {params_lungs[2, 4]:.3f}')

        ax13.plot(data_lungs[5][0], data_lungs[5][1], label = 'K1', color = 'b')
        ax13.plot(data_lungs[5][0], data_lungs[5][4], label = 'F', color = 'm')
        ax13.plot(data_lungs[5][0], data_lungs[5][3], label = 'vb', color = 'r')
        ax13.set_title(f'K1 = {params_lungs[0, 5]:.3f}, \n F = {params_lungs[3, 5]:.3f}, vb = {params_lungs[2, 5]:.3f}')

        ax14.plot(data_lungs[6][0], data_lungs[6][1], label = 'K1', color = 'b')
        ax14.plot(data_lungs[6][0], data_lungs[6][4], label = 'F', color = 'm')
        ax14.plot(data_lungs[6][0], data_lungs[6][3], label = 'vb', color = 'r')
        ax14.set_title(f'K1 = {params_lungs[0, 6]:.3f}, \n F = {params_lungs[3, 6]:.3f}, vb = {params_lungs[2, 6]:.3f}')

        # Third Row (Liver)
        ax15.plot(data_liver[0][0], data_liver[0][1], label = 'K1', color = 'b')
        ax15.plot(data_liver[0][0], data_liver[0][4], label = 'F', color = 'm')
        ax15.plot(data_liver[0][0], data_liver[0][3], label = 'vb', color = 'r')
        ax15.set_title(f'K1 = {params_liver[0, 0]:.3f}, \n F = {params_liver[3, 0]:.3f}, vb = {params_liver[2, 0]:.3f}')
        ax15.set_ylabel('Liver')

        ax16.plot(data_liver[1][0], data_liver[1][1], label = 'K1', color = 'b')
        ax16.plot(data_liver[1][0], data_liver[1][4], label = 'F', color = 'm')
        ax16.plot(data_liver[1][0], data_liver[1][3], label = 'vb', color = 'r')
        ax16.set_title(f'K1 = {params_liver[0, 1]:.3f}, \n F = {params_liver[3, 1]:.3f}, vb = {params_liver[2, 1]:.3f}')

        ax17.plot(data_liver[2][0], data_liver[2][1], label = 'K1', color = 'b')
        ax17.plot(data_liver[2][0], data_liver[2][4], label = 'F', color = 'm')
        ax17.plot(data_liver[2][0], data_liver[2][3], label = 'vb', color = 'r')
        ax17.set_title(f'K1 = {params_liver[0, 2]:.3f}, \n F = {params_liver[3, 2]:.3f}, vb = {params_liver[2, 2]:.3f}')

        ax18.plot(data_liver[3][0], data_liver[3][1], label = 'K1', color = 'b')
        ax18.plot(data_liver[3][0], data_liver[3][4], label = 'F', color = 'm')
        ax18.plot(data_liver[3][0], data_liver[3][3], label = 'vb', color = 'r')
        ax18.set_title(f'K1 = {params_liver[0, 3]:.3f}, \n F = {params_liver[3, 3]:.3f}, vb = {params_liver[2, 3]:.3f}')

        ax19.plot(data_liver[4][0], data_liver[4][1], label = 'K1', color = 'b')
        ax19.plot(data_liver[4][0], data_liver[4][4], label = 'F', color = 'm')
        ax19.plot(data_liver[4][0], data_liver[4][3], label = 'vb', color = 'r')
        ax19.set_title(f'K1 = {params_liver[0, 4]:.3f}, \n F = {params_liver[3, 4]:.3f}, vb = {params_liver[2, 4]:.3f}')

        ax20.plot(data_liver[5][0], data_liver[5][1], label = 'K1', color = 'b')
        ax20.plot(data_liver[5][0], data_liver[5][4], label = 'F', color = 'm')
        ax20.plot(data_liver[5][0], data_liver[5][3], label = 'vb', color = 'r')
        ax20.set_title(f'K1 = {params_liver[0, 5]:.3f}, \n F = {params_liver[3, 5]:.3f}, vb = {params_liver[2, 5]:.3f}')

        ax21.plot(data_liver[6][0], data_liver[6][1], label = 'K1', color = 'b')
        ax21.plot(data_liver[6][0], data_liver[6][4], label = 'F', color = 'm')
        ax21.plot(data_liver[6][0], data_liver[6][3], label = 'vb', color = 'r')
        ax21.set_title(f'K1 = {params_liver[0, 6]:.3f}, \n F = {params_liver[3, 6]:.3f}, vb = {params_liver[2, 6]:.3f}')

        # Fourth row (Kidneys)
        ax22.plot(data_kidneys[0][0], data_kidneys[0][1], label = 'K1', color = 'b')
        ax22.plot(data_kidneys[0][0], data_kidneys[0][4], label = 'F', color = 'm')
        ax22.plot(data_kidneys[0][0], data_kidneys[0][3], label = 'vb', color = 'r')
        ax22.set_title(f'K1 = {params_kidneys[0, 0]:.3f}, \n F = {params_kidneys[3, 0]:.3f}, vb = {params_kidneys[2, 0]:.3f}')
        ax22.set_ylabel('Kidneys')

        ax23.plot(data_kidneys[1][0], data_kidneys[1][1], label = 'K1', color = 'b')
        ax23.plot(data_kidneys[1][0], data_kidneys[1][4], label = 'F', color = 'm')
        ax23.plot(data_kidneys[1][0], data_kidneys[1][3], label = 'vb', color = 'r')
        ax23.set_title(f'K1 = {params_kidneys[0, 1]:.3f}, \n F = {params_kidneys[3, 1]:.3f}, vb = {params_kidneys[2, 1]:.3f}')

        ax24.plot(data_kidneys[2][0], data_kidneys[2][1], label = 'K1', color = 'b')
        ax24.plot(data_kidneys[2][0], data_kidneys[2][4], label = 'F', color = 'm')
        ax24.plot(data_kidneys[2][0], data_kidneys[2][3], label = 'vb', color = 'r')
        ax24.set_title(f'K1 = {params_kidneys[0, 2]:.3f}, \n F = {params_kidneys[3, 2]:.3f}, vb = {params_kidneys[2, 2]:.3f}')

        ax25.plot(data_kidneys[3][0], data_kidneys[3][1], label = 'K1', color = 'b')
        ax25.plot(data_kidneys[3][0], data_kidneys[3][4], label = 'F', color = 'm')
        ax25.plot(data_kidneys[3][0], data_kidneys[3][3], label = 'vb', color = 'r')
        ax25.set_title(f'K1 = {params_kidneys[0, 3]:.3f}, \n F = {params_kidneys[3, 3]:.3f}, vb = {params_kidneys[2, 3]:.3f}')

        ax26.plot(data_kidneys[4][0], data_kidneys[4][1], label = 'K1', color = 'b')
        ax26.plot(data_kidneys[4][0], data_kidneys[4][4], label = 'F', color = 'm')
        ax26.plot(data_kidneys[4][0], data_kidneys[4][3], label = 'vb', color = 'r')
        ax26.set_title(f'K1 = {params_kidneys[0, 4]:.3f}, \n F = {params_kidneys[3, 4]:.3f}, vb = {params_kidneys[2, 4]:.3f}')

        ax27.plot(data_kidneys[5][0], data_kidneys[5][1], label = 'K1', color = 'b')
        ax27.plot(data_kidneys[5][0], data_kidneys[5][4], label = 'F', color = 'm')
        ax27.plot(data_kidneys[5][0], data_kidneys[5][3], label = 'vb', color = 'r')
        ax27.set_title(f'K1 = {params_kidneys[0, 5]:.3f}, \n F = {params_kidneys[3, 5]:.3f}, vb = {params_kidneys[2, 5]:.3f}')

        ax28.plot(data_kidneys[6][0], data_kidneys[6][1], label = 'K1', color = 'b')
        ax28.plot(data_kidneys[6][0], data_kidneys[6][4], label = 'F', color = 'm')
        ax28.plot(data_kidneys[6][0], data_kidneys[6][3], label = 'vb', color = 'r')
        ax28.set_title(f'K1 = {params_kidneys[0, 6]:.3f}, \n F = {params_kidneys[3, 6]:.3f}, vb = {params_kidneys[2, 6]:.3f}')

        # Fifth row (Femur)
        ax29.plot(data_femur[0][0], data_femur[0][1], label = 'K1', color = 'b')
        ax29.plot(data_femur[0][0], data_femur[0][4], label = 'F', color = 'm')
        ax29.plot(data_femur[0][0], data_femur[0][3], label = 'vb', color = 'r')
        ax29.set_title(f'K1 = {params_femur[0, 0]:.3f}, \n F = {params_femur[3, 0]:.3f}, vb = {params_femur[2, 0]:.3f}')
        ax29.set_ylabel('Femur')

        ax30.plot(data_femur[1][0], data_femur[1][1], label = 'K1', color = 'b')
        ax30.plot(data_femur[1][0], data_femur[1][4], label = 'F', color = 'm')
        ax30.plot(data_femur[1][0], data_femur[1][3], label = 'vb', color = 'r')
        ax30.set_title(f'K1 = {params_femur[0, 1]:.3f}, \n F = {params_femur[3, 1]:.3f}, vb = {params_femur[2, 1]:.3f}')

        ax31.plot(data_femur[2][0], data_femur[2][1], label = 'K1', color = 'b')
        ax31.plot(data_femur[2][0], data_femur[2][4], label = 'F', color = 'm')
        ax31.plot(data_femur[2][0], data_femur[2][3], label = 'vb', color = 'r')
        ax31.set_title(f'K1 = {params_femur[0, 2]:.3f}, \n F = {params_femur[3, 2]:.3f}, vb = {params_femur[2, 2]:.3f}')

        ax32.plot(data_femur[3][0], data_femur[3][1], label = 'K1', color = 'b')
        ax32.plot(data_femur[3][0], data_femur[3][4], label = 'F', color = 'm')
        ax32.plot(data_femur[3][0], data_femur[3][3], label = 'vb', color = 'r')
        ax32.set_title(f'K1 = {params_femur[0, 3]:.3f}, \n F = {params_femur[3, 3]:.3f}, vb = {params_femur[2, 3]:.3f}')

        ax33.plot(data_femur[4][0], data_femur[4][1], label = 'K1', color = 'b')
        ax33.plot(data_femur[4][0], data_femur[4][4], label = 'F', color = 'm')
        ax33.plot(data_femur[4][0], data_femur[4][3], label = 'vb', color = 'r')
        ax33.set_title(f'K1 = {params_femur[0, 4]:.3f}, \n F = {params_femur[3, 4]:.3f}, vb = {params_femur[2, 4]:.3f}')

        ax34.plot(data_femur[5][0], data_femur[5][1], label = 'K1', color = 'b')
        ax34.plot(data_femur[5][0], data_femur[5][4], label = 'F', color = 'm')
        ax34.plot(data_femur[5][0], data_femur[5][3], label = 'vb', color = 'r')
        ax34.set_title(f'K1 = {params_femur[0, 5]:.3f}, \n F = {params_femur[3, 5]:.3f}, vb = {params_femur[2, 5]:.3f}')

        ax35.plot(data_femur[6][0], data_femur[6][1], label = 'K1', color = 'b')
        ax35.plot(data_femur[6][0], data_femur[6][4], label = 'F', color = 'm')
        ax35.plot(data_femur[6][0], data_femur[6][3], label = 'vb', color = 'r')
        ax35.set_title(f'K1 = {params_femur[0, 6]:.3f}, \n F = {params_femur[3, 6]:.3f}, vb = {params_femur[2, 6]:.3f}')

        plt.savefig(f'{folder}\{sub_folder}/SA/Combined_SA')
        plt.close()

        with open(f'{folder}\{sub_folder}/SA/Combined_params.csv', 'w', newline = '') as f:
            writer = csv.writer(f)
            writer.writerows([['Mouse 2', 'Mouse 3', 'Mouse 4', 'Mouse 5', 'Mouse 9', 'Mouse 16', 'Mouse 17']])
            writer.writerows([[params_heart], [params_lungs], [params_liver], [params_kidneys], [params_femur]])
            writer.writerows([' '])


### For Degrado model the easiest way to slice the data to be just the first 4 minutes is during the input_data function, so here it is slightly altered

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

            weights[i], zero_count = weights_dc(inter_func, data, i, 0.0063128158521)       # currently in minutes, in seconds: 0.0001052135975

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



### MAIN

frametimes_s = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 210, 240, 300, 420, 540, 840, 1140, 1440, 1740, 2040, 2340, 2640, 2940, 3240, 3540])
frametimes_m = frametimes_s / 60

framemidtimes_s = np.array([5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155, 165, 175, 195, 225, 270, 360, 480, 690, 990, 1290, 1590, 1890, 2190, 2490, 2790, 3090, 3390])
framemidtimes_m = framemidtimes_s / 60

frame_lengths_s = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 30, 30, 60, 120, 120, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300])
frame_lengths_m = frame_lengths_s / 60


scale_factor_df = pd.DataFrame(data = {'Mouse_2_16A0818B' : [0.05, 0.05, 0.05, 0.05, 0.05], 'Mouse_3_16A0823' : [0.05, 0.05, 0.05, 0.05, 0.05], 
                                       'Mouse_4_17A1012C' : [0.05, 0.05, 0.05, 0.005, 0.05], 'Mouse_5_17A1101A' : [0.005, 0.005, 0.05, 0.005, 0.005], 
                                        'Mouse_9_17A1010A' : [0.0005, 0.005, 0.05, 0.0005, 0.05], 'Mouse_16_17A1101C' : [0.05, 0.05, 0.05, 0.05, 0.005], 
                                        'Mouse_17_17A1101D' : [0.05, 0.05, 0.05, 0.05, 0.05]}, index = ['Heart', 'Lungs', 'Kidneys', 'Liver', 'Femur'])

ps_df = pd.DataFrame(data = {'Organ' : ['Heart', 'Lungs', 'Liver', 'Femur', 'Kidneys'], 'PS' : [1.026, 1.026, 0.557, 0.276, 1.385]}, index = ['Heart', 'Lungs', 'Liver', 'Femur', 'Kidneys'])


delays_df = pd.DataFrame(data = {'Mouse_1_16A0818A' : [0.0, 0.0, 0.0, 0.0, 0.0], 'Mouse_2_16A0818B' : [0.0, 0.0, 0.0, 1.0, 0.0], 
                'Mouse_3_16A0823' : [0.0, 0.0, 0.0, 0.0, 4.0], 'Mouse_4_17A1012C' : [1.0, 1.0, 1.0, 9.0, 8.0], 
                'Mouse_5_17A1101A' : [0.0, 0.0, 2.0, 2.0, 0.0], 'Mouse_6_17A1101B' : [0.0, 0.0, 0.0, 0.0, 0.0], 
                'Mouse_9_17A1010A' : [0.0, 0.0, 0.0, 0.0, 0.0], 'Mouse_16_17A1101C' : [0.0, 0.0, 0.0, 0.0, 0.0], 
                'Mouse_17_17A1101D' : [0.0, 0.0, 0.0, 0.0, 0.0] }, 
                index = ['Heart', 'Lungs', 'Kidneys', 'Liver', 'Femur'])



# Starting values for the fits
heart_sv_df = pd.DataFrame(data = {'K1' : [0.8, 0.2, 0.4, 0.2], 'k2' : [0.4, 0.2, 0.2, 1.0], 'vb' : [0.2, 0.4, 0.4, 0.2]}, index = ['Mouse_2_16A0818B', 'Mouse_3_16A0823', 'Mouse_4_17A1012C', 'Mouse_5_17A1101A'])
lungs_sv_df = pd.DataFrame(data = {'K1' : [1.0, 0.4, 0.2, 0.4], 'k2' : [0.6, 0.4, 1.0, 1.0], 'vb' : [0.6, 0.6, 0.4, 0.6]}, index = ['Mouse_2_16A0818B', 'Mouse_3_16A0823', 'Mouse_4_17A1012C', 'Mouse_5_17A1101A'])
liver_sv_df = pd.DataFrame(data = {'K1' : [0.8, 0.6, 0.4, 1.0], 'k2' : [0.4, 0.6, 0.4, 0.4], 'vb' : [0.2, 0.8, 0.2, 0.8]}, index = ['Mouse_2_16A0818B', 'Mouse_3_16A0823', 'Mouse_4_17A1012C', 'Mouse_5_17A1101A'])
femur_sv_df = pd.DataFrame(data = {'K1' : [0.6, 0.6, 0.8, 1.0], 'k2' : [0.2, 0.8, 0.4, 0.2], 'k3' : [0.2, 0.2, 0.2, 0.2] , 'vb' : [0.2, 0.4, 0.4, 0.2]}, index = ['Mouse_2_16A0818B', 'Mouse_3_16A0823', 'Mouse_4_17A1012C', 'Mouse_5_17A1101A'])
kidneys_sv_df = pd.DataFrame(data = {'K1' : [0.8, 0.8, 0.4, 1.0], 'k2' : [0.4, 0.4, 0.8, 0.4], 'vb' : [1.0, 0.4, 1.0, 1.0]}, index = ['Mouse_2_16A0818B', 'Mouse_3_16A0823', 'Mouse_4_17A1012C', 'Mouse_5_17A1101A'])

#kidneys_sv_df.loc['Mouse_3_16A0823', :] = [2.043, 0.967, 0.182]

# Tags for Graphs

mouse_tags = {'Mouse_2_16A0818B' : 'Mouse 2', 'Mouse_3_16A0823' : 'Mouse 3', 'Mouse_4_17A1012C' : 'Mouse 4', 'Mouse_5_17A1101A' : 'Mouse 5', 'Mouse_9_17A1010A' : 'Mouse 9', 
              'Mouse_16_17A1101C' : 'Mouse 16', 'Mouse_17_17A1101D' : 'Mouse 17'}

folder = 'Chapter_3_Fits'
sub_folder = 'Scale_Factor_Weightings_Calc_FINAL'

one_cm = False
degrado = False

mice = ['Mouse_2_16A0818B', 'Mouse_3_16A0823', 'Mouse_4_17A1012C', 'Mouse_5_17A1101A', 'Mouse_9_17A1010A', 'Mouse_16_17A1101C', 'Mouse_17_17A1101D']


sens_femur = np.zeros((7, 6, 3386))
sens_heart = np.zeros((7, 5, 266))
sens_lungs = np.zeros((7, 5, 266))
sens_liver = np.zeros((7, 5, 266))
sens_kidneys = np.zeros((7, 7, 3386))

sens_heart_params = np.zeros((4, 7))
sens_lungs_params = np.zeros((4, 7))
sens_liver_params = np.zeros((4, 7))
sens_kidneys_params = np.zeros((6, 7))
sens_femur_params = np.zeros((5, 7))

sens_heart_err = np.zeros((4, 7))
sens_lungs_err = np.zeros((4, 7))
sens_liver_err = np.zeros((4, 7))
sens_kidneys_err = np.zeros((6, 7))
sens_femur_err = np.zeros((5, 7))

k = 0

for m in mice:
    mouse, mouse_int, mouse_time, mouse_weights, mouse_zc = input_data(m, False)

    T_f, T0, dt, t_array = mouse_time

    C0_orig = mouse_int.Vena_Cava
    C0_orig_time = mouse_int.Time
    y_time = mouse.Time
    y_0 = mouse.Vena_Cava
    zc = mouse_zc

    mouse_deg, mouse_int_deg, mouse_time_deg, mouse_weights_deg, mouse_zc_deg = input_data(m, True)
    
    T_f_deg, T0_deg, dt_deg, t_array_deg = mouse_time_deg
    C0_orig_deg_time = mouse_int_deg.Time
    C0_orig_deg = mouse_int_deg.Vena_Cava
    y_time_deg = mouse_deg.Time

    tag = mouse_tags[m]
    #organ_plot_nb(mouse_deg, True, tag)        # For visualising relations between tissue curves and blood curve
    
    for organ in ['Heart']: #, 'Lungs', 'Kidneys', 'Liver', 'Femur']:

        C1_data = mouse_int[organ]
        y_dat = mouse[organ]
        w = mouse_weights[organ]

        C1_data_deg = mouse_int_deg[organ]
        y_dat_deg = mouse_deg[organ] 
        w_deg = mouse_weights_deg[organ]

        name = f'{tag}_{organ}'

        scale_factor = scale_factor_df.loc[organ, m]
        delay = delays_df.loc[organ, m] / 60

        C0 = delay_shift(np.array(C0_orig_time), np.array(C0_orig), delay)            
        C0_deg = delay_shift(np.array(C0_orig_deg_time), np.array(C0_orig_deg), delay)
        
        # shift values down by X time steps (0.1 seconds for the interpolated data), add zero to the front
        # for opposite direction pad with average of last 5 values
        # Checking times of each delay shift, it is 1 second not 0.1 seconds


        if organ == 'Heart' or organ == 'Lungs' or organ == 'Liver':
            init = [0.0]
            init2 = [0.0, 0.0]

            params = lmfit.Parameters()
            params.add('K1', 1.0, min=0.0, max=10.0)
            params.add('k2', 0.1, min=0.0, max=5.0)
            params.add('vb', 0.1, vary = True, min=0.0, max=1.0)
            
            fit = lmfit.minimize(resid1_deg_weighted, params, method = 'leastsq', max_nfev = 2500)
            #lmfit.report_fit(fit)

            flow = get_flow(fit.params['K1'].value, organ)
            
            ststics = [['Number of evaluations:', fit.nfev], ['Number of data points:', fit.ndata], ['Chi-Squared:', fit.chisqr], ['Reduced Chi-Squared:', fit.redchi], 
            ['Akaike Information Criterion:', fit.aic], ['Bayesian Information Criterion:', fit.bic]]

            vrbles = [[fit.var_names[0], fit.params['K1'].value, fit.params['K1'].stderr], 
            [fit.var_names[1], fit.params['k2'].value, fit.params['k2'].stderr],
            [fit.var_names[2], fit.params['vb'].value, fit.params['vb'].stderr],
            ['delay', delay],
            ['Flow', flow]]
            
            if fit.params['K1'].correl != None:
                correls = [[fit.params['K1'].correl['k2'], fit.params['K1'].correl['vb']], [fit.params['k2'].correl['vb']]]
            else:
                correls = []
                
            with open(f'{folder}\{sub_folder}\{name}_Degrado.csv', 'w', newline = '') as f:
                writer = csv.writer(f)
                writer.writerows(ststics)
                writer.writerow([' '])
                writer.writerows(vrbles)
                writer.writerow([' '])
                writer.writerows(correls)

            results = [['Reduced Chi-Squared:', fit.redchi], ['Akaike Information Criterion:', fit.aic], 
                       [fit.var_names[0], fit.params['K1'].value, fit.params['K1'].stderr], 
            [fit.var_names[1], fit.params['k2'].value, fit.params['k2'].stderr],
            [fit.var_names[2], fit.params['vb'].value, fit.params['vb'].stderr],
            ['Delay', delay],
            ['Flow', flow]]

            with open(f'{folder}\{sub_folder}\{name}_Degrado_results.csv', 'w', newline = '') as f:
                writer = csv.writer(f)
                writer.writerows(results)  

            
            val1, val2, valvb, valdelay = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, ]
            
            deriv_k1, deriv_time_k1 = sensitivity_analysis(fit, 'K1', 0.1, 'Degrado', organ)
            deriv_k2, deriv_time_k2 = sensitivity_analysis(fit, 'k2', 0.1, 'Degrado', organ)
            deriv_vb, deriv_time_vb = sensitivity_analysis(fit, 'vb', 0.1, 'Degrado', organ)
            deriv_F, deriv_time_F, f_value = sensitivity_analysis(fit, 'F', 0.1, 'Degrado', organ)
            
            if organ == 'Heart':
                sens_heart[k][0] = deriv_time_k1
                sens_heart[k][1] = deriv_k1
                sens_heart[k][2] = deriv_k2
                sens_heart[k][3] = deriv_vb
                sens_heart[k][4] = deriv_F

                sens_heart_params[:, k] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                sens_heart_err[:, k] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

            elif organ == 'Lungs':
                sens_lungs[k][0] = deriv_time_k1
                sens_lungs[k][1] = deriv_k1
                sens_lungs[k][2] = deriv_k2
                sens_lungs[k][3] = deriv_vb
                sens_lungs[k][4] = deriv_F

                sens_lungs_params[:, k] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                sens_lungs_err[:, k] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

            elif organ == 'Liver':
                sens_liver[k][0] = deriv_time_k1
                sens_liver[k][1] = deriv_k1
                sens_liver[k][2] = deriv_k2
                sens_liver[k][3] = deriv_vb
                sens_liver[k][4] = deriv_F

                sens_liver_params[:, k] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
                sens_liver_err[:, k] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]

            x1, t1 = RK4(comp_ode_model1_deg, init, dt_deg, T_f_deg, T0_deg, [fit.params['K1'].value, fit.params['k2'].value])

            val1, val2, valvb, valdelay = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, delay]

            plt.scatter(y_time_deg, mouse_deg.Vena_Cava, s = 7, label = 'Blood', color = 'g')
            plt.scatter(y_time_deg, y_dat_deg, s = 7, label = 'Tissue', color = 'r')
            plt.plot(t1, ((1 - fit.params['vb'].value) * x1[:, 0] + fit.params['vb'].value * C0_deg), label = 'Model Fit', color = 'b')
            plt.title(f'{tag}, {organ} \n K1 = {val1:.3f}, k2 = {val2:.3f}, vb = {valvb:.3f}')
            plt.xlabel('Time (minutes)')
            plt.ylabel('SUV (g/ml)')
            plt.legend(loc = 7, fontsize = 'x-small')
            plt.savefig(f'{folder}\{sub_folder}\{name}_Degrado')
            plt.close()

            print(f'{tag}, {organ}: K1 = {val1:.3f}, Flow = {flow:.3f}')

        elif organ == 'Kidneys' and one_cm == False:
            init = [0.0]
            init2 = [0.0, 0.0]

            params = lmfit.Parameters()
            params.add('K1', 0.5, min=0.0, max=10.0)
            params.add('k2', 0.1, min=0.0, max=5.0)
            params.add('k3', 0.1, vary = True, min=0.0, max=5.0)      
            params.add('k4', 0.1, vary = True, min=0.0, max=5.0)
            params.add('vb', 0.1, vary = True, min=0.0, max=1.0)
            
            
            fit = lmfit.minimize(resid2_kidneys_weighted, params, method = 'leastsq', max_nfev = 2000)
            #lmfit.report_fit(fit)

            flow = get_flow(fit.params['K1'].value, organ)
            
            ststics = [['Number of evaluations:', fit.nfev], ['Number of data points:', fit.ndata], ['Chi-Squared:', fit.chisqr], ['Reduced Chi-Squared:', fit.redchi], 
            ['Akaike Information Criterion:', fit.aic], ['Bayesian Information Criterion:', fit.bic]]

            # vrbles = [[fit.var_names[0], fit.params['K1'].value, fit.params['K1'].stderr], 
            # [fit.var_names[1], fit.params['k2'].value, fit.params['k2'].stderr],
            # [fit.var_names[2], fit.params['vb'].value, fit.params['vb'].stderr],
            # ['delay', delay]]

            vrbles = [[fit.var_names[0], fit.params['K1'].value, fit.params['K1'].stderr], 
            [fit.var_names[1], fit.params['k2'].value, fit.params['k2'].stderr],
            [fit.var_names[2], fit.params['k3'].value, fit.params['k3'].stderr],
            [fit.var_names[3], fit.params['k4'].value, fit.params['k4'].stderr],
            [fit.var_names[4], fit.params['vb'].value, fit.params['vb'].stderr],
            ['delay', delay],
            ['Flow', flow]]
            
            if fit.params['K1'].correl != None:
                #correls = [[fit.params['K1'].correl['k2'], fit.params['K1'].correl['vb']], [fit.params['k2'].correl['vb']]]

                correls = [[fit.params['K1'].correl['k2'], fit.params['K1'].correl['k3'], fit.params['K1'].correl['k4'], fit.params['K1'].correl['vb']],
                        [fit.params['k2'].correl['k3'], fit.params['k2'].correl['k4'], fit.params['k2'].correl['vb']], 
                        [fit.params['k3'].correl['k4'], fit.params['k3'].correl['vb']],
                        [fit.params['k4'].correl['vb']]]
            else:
                correls = []
                
            with open(f'{folder}\{sub_folder}\{name}_alt2TCM.csv', 'w', newline = '') as f:
                writer = csv.writer(f)
                writer.writerows(ststics)
                writer.writerow([' '])
                writer.writerows(vrbles)
                writer.writerow([' '])
                writer.writerows(correls)

            results = [['Reduced Chi-Squared:', fit.redchi], ['Akaike Information Criterion:', fit.aic], 
                       [fit.var_names[0], fit.params['K1'].value, fit.params['K1'].stderr], 
            [fit.var_names[1], fit.params['k2'].value, fit.params['k2'].stderr],
            [fit.var_names[2], fit.params['k3'].value, fit.params['k3'].stderr],
            ['k4', fit.params['k4'].value, fit.params['k4'].stderr],
            [fit.var_names[3], fit.params['vb'].value, fit.params['vb'].stderr],
            ['Delay', delay],
            ['Flow', flow]]

            with open(f'{folder}\{sub_folder}\{name}_alt2TCM_results.csv', 'w', newline = '') as f:
                writer = csv.writer(f)
                writer.writerows(results)  

            
            #val1, val2, valvb, valdelay = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, delay]
            val1, val2, val3, val4, valvb, valdelay = [fit.params['K1'].value, fit.params['k2'].value,  fit.params['k3'].value, fit.params['k4'].value, fit.params['vb'].value, delay]
            
            deriv_k1, deriv_time_k1 = sensitivity_analysis(fit, 'K1', 0.1, '1TCM', organ)
            deriv_k2, deriv_time_k2 = sensitivity_analysis(fit, 'k2', 0.1, '1TCM', organ)
            deriv_k3, deriv_time_k3 = sensitivity_analysis(fit, 'k3', 0.1, '2TCM', organ)
            deriv_k4, deriv_time_k4 = sensitivity_analysis(fit, 'k4', 0.1, '2TCM', organ)
            deriv_vb, deriv_time_vb = sensitivity_analysis(fit, 'vb', 0.1, '1TCM', organ)
            deriv_F, deriv_time_F, f_value = sensitivity_analysis(fit, 'F', 0.1, '1TCM', organ)
            
        
            sens_kidneys[k][0] = deriv_time_k1
            sens_kidneys[k][1] = deriv_k1
            sens_kidneys[k][2] = deriv_k2
            sens_kidneys[k][3] = deriv_vb
            sens_kidneys[k][4] = deriv_F
            sens_kidneys[k][5] = deriv_k3
            sens_kidneys[k][6] = deriv_k4

            # sens_kidneys_params[:, k] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
            # sens_kidneys_err[:, k] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]
            
            sens_kidneys_params[:, k] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value, fit.params['k3'].value, fit.params['k4'].value]
            sens_kidneys_err[:, k] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr, fit.params['k3'].stderr, fit.params['k4'].stderr]

            #x1, t1 = RK4(comp_ode_model1, init, dt, T_f, T0, [fit.params['K1'].value, fit.params['k2'].value])
            x1, t1 = RK4(comp_ode_model2_kidney, init2, dt, T_f, T0, [fit.params['K1'].value, fit.params['k2'].value, fit.params['k3'].value, fit.params['k4'].value])

            #val1, val2, valvb, valdelay = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, delay]
            val1, val2, val3, val4, valvb, valdelay = [fit.params['K1'].value, fit.params['k2'].value,  fit.params['k3'].value, fit.params['k4'].value, fit.params['vb'].value, delay]

            plt.scatter(y_time, mouse.Vena_Cava, s = 7, label = 'Blood', color = 'g')
            plt.scatter(y_time, y_dat, s = 7, label = 'Tissue', color = 'r')
            plt.plot(t1, ((1 - fit.params['vb'].value) * (x1[:, 0] + x1[:, 1]) + fit.params['vb'].value * C0), label = 'Model Fit', color = 'b')
            plt.title(f'{tag}, {organ} \n K1 = {val1:.3f}, k2 = {val2:.3f}, k3 = {val3:.3f}, k4 = {val4:.3f}, vb = {valvb:.3f}')
            plt.xlabel('Time (minutes)')
            plt.ylabel('SUV (g/ml)')
            plt.legend(loc = 7, fontsize = 'x-small')
            plt.savefig(f'{folder}\{sub_folder}\{name}_alt2TCM')
            plt.close()

            print(f'{tag}, {organ}: K1 = {val1:.3f}, Flow = {flow:.3f}')

        elif organ == 'Kidneys' and one_cm == True:
            init = [0.0]
            init2 = [0.0, 0.0]

            params = lmfit.Parameters()
            params.add('K1', 1.0, min=0.0, max=10.0)
            params.add('k2', 0.1, min=0.0, max=5.0)
            params.add('vb', 0.1, vary = True, min=0.0, max=1.0)
            
            
            fit = lmfit.minimize(resid1_weighted, params, method = 'leastsq', max_nfev = 2000)
            #lmfit.report_fit(fit)

            flow = get_flow(fit.params['K1'].value, organ)
            
            ststics = [['Number of evaluations:', fit.nfev], ['Number of data points:', fit.ndata], ['Chi-Squared:', fit.chisqr], ['Reduced Chi-Squared:', fit.redchi], 
            ['Akaike Information Criterion:', fit.aic], ['Bayesian Information Criterion:', fit.bic]]

            vrbles = [[fit.var_names[0], fit.params['K1'].value, fit.params['K1'].stderr], 
            [fit.var_names[1], fit.params['k2'].value, fit.params['k2'].stderr],
            [fit.var_names[2], fit.params['vb'].value, fit.params['vb'].stderr],
            ['delay', delay],
            ['Flow', flow]]

    
            
            if fit.params['K1'].correl != None:
                correls = [[fit.params['K1'].correl['k2'], fit.params['K1'].correl['vb']], [fit.params['k2'].correl['vb']]]

            else:
                correls = []
                
            with open(f'{folder}\{sub_folder}\{name}_1TCM.csv', 'w', newline = '') as f:
                writer = csv.writer(f)
                writer.writerows(ststics)
                writer.writerow([' '])
                writer.writerows(vrbles)
                writer.writerow([' '])
                writer.writerows(correls)

            results = [['Reduced Chi-Squared:', fit.redchi], ['Akaike Information Criterion:', fit.aic], 
                       [fit.var_names[0], fit.params['K1'].value, fit.params['K1'].stderr], 
            [fit.var_names[1], fit.params['k2'].value, fit.params['k2'].stderr],
            [fit.var_names[3], fit.params['vb'].value, fit.params['vb'].stderr],
            ['Delay', delay],
            ['Flow', flow]]

            with open(f'{folder}\{sub_folder}\{name}_1TCM_results.csv', 'w', newline = '') as f:
                writer = csv.writer(f)
                writer.writerows(results)  

            
            val1, val2, valvb, valdelay = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, delay]
            #val1, val2, val3, val4, valvb, valdelay = [fit.params['K1'].value, fit.params['k2'].value,  fit.params['k3'].value, fit.params['k4'].value, fit.params['vb'].value, delay]
            
            # deriv_k1, deriv_time_k1 = sensitivity_analysis(fit, 'K1', 0.1, '2TCM', organ)
            # deriv_k2, deriv_time_k2 = sensitivity_analysis(fit, 'k2', 0.1, '2TCM', organ)
            # deriv_k3, deriv_time_k3 = sensitivity_analysis(fit, 'k3', 0.1, '2TCM', organ)
            # deriv_k4, deriv_time_k4 = sensitivity_analysis(fit, 'k4', 0.1, '2TCM', organ)
            # deriv_vb, deriv_time_vb = sensitivity_analysis(fit, 'vb', 0.1, '2TCM', organ)
            # deriv_F, deriv_time_F, f_value = sensitivity_analysis(fit, 'F', 0.1, '2TCM', organ)
            
        
            # sens_kidneys[k][0] = deriv_time_k1
            # sens_kidneys[k][1] = deriv_k1
            # sens_kidneys[k][2] = deriv_k2
            # sens_kidneys[k][3] = deriv_vb
            # sens_kidneys[k][4] = deriv_F
            # sens_kidneys[k][5] = deriv_k3
            # sens_kidneys[k][6] = deriv_k4

            # sens_kidneys_params[:, k] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value]
            # sens_kidneys_err[:, k] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr]
            
            # sens_kidneys_params[:, k] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['k3'].value, fit.params['k4'].value,fit.params['vb'].value, f_value]
            # sens_kidneys_err[:, k] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['k3'].stderr, fit.params['k4'].stderr,fit.params['vb'].stderr, fit.params['K1'].stderr]

            x1, t1 = RK4(comp_ode_model1, init, dt, T_f, T0, [fit.params['K1'].value, fit.params['k2'].value])
            #x1, t1 = RK4(comp_ode_model2_kidney, init2, dt, T_f, T0, [fit.params['K1'].value, fit.params['k2'].value, fit.params['k3'].value, fit.params['k4'].value])

            val1, val2, valvb, valdelay = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, delay]
            #val1, val2, val3, val4, valvb, valdelay = [fit.params['K1'].value, fit.params['k2'].value,  fit.params['k3'].value, fit.params['k4'].value, fit.params['vb'].value, delay]

            plt.scatter(y_time, mouse.Vena_Cava, s = 7, label = 'Blood', color = 'g')
            plt.scatter(y_time, y_dat, s = 7, label = 'Tissue', color = 'r')
            plt.plot(t1, ((1 - fit.params['vb'].value) * x1[:, 0] + fit.params['vb'].value * C0), label = 'Model Fit', color = 'b')
            plt.title(f'{m}, {organ} \n K1 = {val1:.3f}, k2 = {val2:.3f}, vb = {valvb:.3f}, delay = {valdelay:.1f}')
            plt.xlabel('Time (minutes)')
            plt.ylabel('SUV (g/ml)')
            plt.legend(loc = 7, fontsize = 'x-small')
            plt.savefig(f'{folder}\{sub_folder}\{name}_1TCM')
            plt.close()

  
        else:
            init = [0.0]
            init2 = [0.0, 0.0]

            params = lmfit.Parameters()
            params.add('K1', 0.5, min=0.0, max=10.0)
            params.add('k2', 0.1, min=0.0, max=5.0)
            params.add('k3', 0.1, vary = True, min=0.0, max=5.0)      
            params.add('k4', 0.0, vary = False, min=0.0, max=5.0)
            params.add('vb', 0.1, vary = True, min=0.0, max=1.0)
            
            fit = lmfit.minimize(resid2_weighted, params, method = 'leastsq', max_nfev = 2000)
            #lmfit.report_fit(fit)

            flow = get_flow(fit.params['K1'].value, organ)

            ststics = [['Number of evaluations:', fit.nfev], ['Number of data points:', fit.ndata], ['Chi-Squared:', fit.chisqr], ['Reduced Chi-Squared:', fit.redchi], 
            ['Akaike Information Criterion:', fit.aic], ['Bayesian Information Criterion:', fit.bic]]

            vrbles = [[fit.var_names[0], fit.params['K1'].value, fit.params['K1'].stderr], 
            [fit.var_names[1], fit.params['k2'].value, fit.params['k2'].stderr],
            [fit.var_names[2], fit.params['k3'].value, fit.params['k3'].stderr],
            ['k4', fit.params['k4'].value, fit.params['k4'].stderr],
            [fit.var_names[3], fit.params['vb'].value, fit.params['vb'].stderr],
            ['Delay', delay],
            ['Flow', flow]]
            
            if fit.params['K1'].correl != None:
                correls = [[fit.params['K1'].correl['k2'], fit.params['K1'].correl['k3'], 0, fit.params['K1'].correl['vb']],
                        [fit.params['k2'].correl['k3'], 0, fit.params['k2'].correl['vb']], 
                        [0, fit.params['k3'].correl['vb']],
                        [0]]
            else:
                correls = []

            with open(f'{folder}\{sub_folder}\{name}_alt2TCM.csv', 'w', newline = '') as f:
                writer = csv.writer(f)
                writer.writerows(ststics)
                writer.writerow([' '])
                writer.writerows(vrbles)
                writer.writerow([' '])
                writer.writerows(correls)

            results = [['Reduced Chi-Squared:', fit.redchi], ['Akaike Information Criterion:', fit.aic], 
                       [fit.var_names[0], fit.params['K1'].value, fit.params['K1'].stderr], 
            [fit.var_names[1], fit.params['k2'].value, fit.params['k2'].stderr],
            [fit.var_names[2], fit.params['k3'].value, fit.params['k3'].stderr],
            ['k4', fit.params['k4'].value, fit.params['k4'].stderr],
            [fit.var_names[3], fit.params['vb'].value, fit.params['vb'].stderr],
            ['Delay', delay],
            ['Flow', flow]]

            with open(f'{folder}\{sub_folder}\{name}_alt2TCM_results.csv', 'w', newline = '') as f:
                writer = csv.writer(f)
                writer.writerows(results)  

                  
            val1, val2, val3, val4, valvb, valdelay = [fit.params['K1'].value, fit.params['k2'].value,  fit.params['k3'].value, fit.params['k4'].value, fit.params['vb'].value, delay]

            deriv_k1, deriv_time_k1 = sensitivity_analysis(fit, 'K1', 0.1, '2TCM', organ)
            deriv_k2, deriv_time_k2 = sensitivity_analysis(fit, 'k2', 0.1, '2TCM', organ)
            deriv_k3, deriv_time_k3 = sensitivity_analysis(fit, 'k3', 0.1, '2TCM', organ)
            deriv_vb, deriv_time_vb = sensitivity_analysis(fit, 'vb', 0.1, '2TCM', organ)
            deriv_F, deriv_time_F, f_value = sensitivity_analysis(fit, 'F', 0.1, '2TCM', organ)

            sens_femur[k][0] = deriv_time_k1
            sens_femur[k][1] = deriv_k1
            sens_femur[k][2] = deriv_k2
            sens_femur[k][3] = deriv_vb
            sens_femur[k][4] = deriv_F
            sens_femur[k][5] = deriv_k3

            sens_femur_params[:, k] = [fit.params['K1'].value, fit.params['k2'].value, fit.params['vb'].value, f_value, fit.params['k3'].value]
            sens_femur_err[:, k] = [fit.params['K1'].stderr, fit.params['k2'].stderr, fit.params['vb'].stderr, fit.params['K1'].stderr, fit.params['k3'].stderr]

            x1, t1 = RK4(comp_ode_model2, init2, dt, T_f, T0, [fit.params['K1'].value, fit.params['k2'].value, fit.params['k3'].value, fit.params['k4'].value])

            val1, val2, val3, val4, valvb, valdelay = [fit.params['K1'].value, fit.params['k2'].value,  fit.params['k3'].value, fit.params['k4'].value, fit.params['vb'].value, delay]

            plt.scatter(y_time, mouse.Vena_Cava, s = 7, label = 'Blood', color = 'g')
            plt.scatter(y_time, y_dat, s = 7, label = 'Tissue', color = 'r')
            plt.plot(t1, ((1 - fit.params['vb'].value) * (x1[:, 0] + x1[:, 1]) + fit.params['vb'].value * C0), label = 'Model Fit', color = 'b')
            plt.title(f'{m}, {organ} \n K1 = {val1:.3f}, k2 = {val2:.3f}, k3 = {val3:.3f}, vb = {valvb:.3f}, delay = {valdelay:.1f}')
            plt.xlabel('Time (minutes)')
            plt.ylabel('SUV (g/ml)')
            plt.legend(loc = 7, fontsize = 'x-small')
            plt.savefig(f'{folder}\{sub_folder}\{name}_alt2TCM')
            plt.close()
            #plt.show()

            print(f'{tag}, {organ}: K1 = {val1:.3f}, Flow = {flow:.3f}')


    k += 1

sensitivity_analysis_display_ch3(folder, sub_folder, sens_heart, sens_heart_params, sens_heart_err, 'Heart')
sensitivity_analysis_display_ch3(folder, sub_folder, sens_lungs, sens_lungs_params, sens_lungs_err, 'Lungs')
sensitivity_analysis_display_ch3(folder, sub_folder, sens_liver, sens_liver_params, sens_liver_err, 'Liver')
sensitivity_analysis_display_ch3(folder, sub_folder, sens_kidneys, sens_kidneys_params, sens_kidneys_err, 'Kidneys')
sensitivity_analysis_display_ch3(folder, sub_folder, sens_femur, sens_femur_params, sens_femur_err, 'Femur')
    
sensitivity_analysis_display_ch3_combined(folder, sub_folder, sens_heart, sens_heart_params, sens_lungs, sens_lungs_params, sens_liver, sens_liver_params, sens_kidneys, sens_kidneys_params, sens_femur, sens_femur_params)


