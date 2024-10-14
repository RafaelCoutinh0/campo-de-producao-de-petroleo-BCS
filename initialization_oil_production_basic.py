# -*- coding: utf-8 -*-

"""
File to simulate a manifold with 4 wells

Adapted from:
Rasmus (2011) - Automatic Start-up and Control of Artificially Lifted Wells

@authors: Rodrigo Lima Meira e Daniel Diniz Santana
"""
import numpy as np
#%% Package import
from numpy import array
import matplotlib.pyplot as plt
import matplotlib.pyplot
from matplotlib import rcParams
from casadi import MX, interpolant, Function, sqrt, vertcat, integrator, jacobian, transpose
from bcs_models import *
from manifold import *
from numpy import linspace, array, eye, zeros, repeat, concatenate, delete, diag
from numpy.linalg import inv
from matplotlib.pyplot import plot, figure, title
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter
import random
from scipy.optimize import fsolve
from control import ss, tf, sample_system, forced_response
from scipy.signal import ss2tf
#%% Creating functions of BCS, Choke and Pipes

def bcs_functions(f, q):
    """
    BCS Head, Efficiency and Power as function of frequency (f) and flow (q)
    :param f: pump frequency (Hz)
    :param q: flow [m^3/s]
    :return:
    H: head [m]
    eff: efficiency
    pot: power
    """

    f0 = 50
    q0 = q*(f0/f)
    H0 = -29.5615*(q0/0.0353)**4+25.3722*(q0/0.0353)**3-8.7944*(q0/0.0353)**2-8.5706*(q0/0.0353)+21.4278
    H = H0 * (f / f0) ** 2.
    eff = 1
    pot = 1
    return H, eff, pot


def choke_fun(alpha):
    """
    Valve characteristic function
    :param alpha: valve opening  (0 to 1)
    :return: choke characteristic
    """
    # Valve opening
    zc = [0, 13, 15, 17, 19, 22, 25, 29, 30, 32, 34, 36, 39, 41, 44,
          46, 49, 52, 55, 58, 61, 64, 67, 71, 75, 78, 82, 86, 91, 96, 100.01]

    # valve characteristic
    G = [0, 0.011052632, 0.024210526, 0.043157895, 0.067894737, 0.097894737,
         0.133157895, 0.173684211, 0.195789474, 0.219473684, 0.244736842,
         0.271052632, 0.298947368, 0.328421053, 0.358947368, 0.390526316,
         0.423684211, 0.458421053, 0.494210526, 0.531578947, 0.570526316,
         0.610526316, 0.651578947, 0.694210526, 0.738421053, 0.784210526,
         0.830526316, 0.878947368, 0.928421053, 0.979473684, 1]

    fun_choke = interpolant('choke', 'bspline', [zc], G)
    return fun_choke(alpha * 100)


# Pipes creation

# well pipe before BCS
pipe_sec1 = Pipe(0.081985330499706 * 2, 3.078838005940556e3, 1029.2 - 920, 1.5e+9, 984, 0.3, 5.752218216772682e+06,
                 3.903249155428134e+07)
# well pipe after BCS
pipe_sec2 = Pipe(0.0595 * 2, 9.222097306189842e+02, 920 - 126.5400, 1.5e9, 984, 4, 7.455247950618545e+06,
                 6.264914244217266e+07)
# manifold pipe
pipe_mani = Pipe(0.0595 * 2, 500, 0, 8.3022e+6, 984, 4, 5.752218216772682e+06, 3.903249155428134e+07)

#%% Defining the CasADi function for pumps and valves

f_ca = MX.sym('f', 1)
q_ca = MX.sym('q', 1)
alpha_ca = MX.sym('alpha', 1)

H_fun, eff_fun, pot_fun = bcs_functions(f_ca, q_ca)

head_fun = Function('head', [f_ca, q_ca], [64 * bcs_functions(f_ca, q_ca)[0]])
efficiency_fun = Function('efficiency', [f_ca, q_ca], [eff_fun])
power_fun = Function('power', [f_ca, q_ca], [pot_fun])

# Booster pump Head [m]
booster_fun = Function('booster', [f_ca, q_ca], [1.0963e3 * (f_ca / 50) ** 2])

valve_fun = Function('choke', [alpha_ca], [choke_fun(alpha_ca)])

# Defining the BCS of the wells and booster pump

bcs1 = Pump(head_fun, efficiency_fun, power_fun)
bcs2 = Pump(head_fun, efficiency_fun, power_fun)
bcs3 = Pump(head_fun, efficiency_fun, power_fun)
bcs4 = Pump(head_fun, efficiency_fun, power_fun)

booster = Pump(booster_fun, efficiency_fun, power_fun)

# Defining the valves in the wells

choke1 = Choke(212.54 / (6e+4 * sqrt(1e+5)), valve_fun)
choke2 = Choke(212.54 / (6e+4 * sqrt(1e+5)), valve_fun)
choke3 = Choke(212.54 / (6e+4 * sqrt(1e+5)), valve_fun)
choke4 = Choke(212.54 / (6e+4 * sqrt(1e+5)), valve_fun)

# Defining the wells and the manifold

well1 = Well(pipe_sec1, pipe_sec2, bcs1, choke1, 6.9651e-9, 9800000)
well2 = Well(pipe_sec1, pipe_sec2, bcs2, choke2, 6.9651e-9, 9800000)
well3 = Well(pipe_sec1, pipe_sec2, bcs3, choke3, 6.9651e-9, 9800000)
well4 = Well(pipe_sec1, pipe_sec2, bcs4, choke4, 6.9651e-9, 9800000)

mani = Manifold(pipe_mani, booster, 0, 0, [well1, well2, well3, well4])

#%% Defining the simulation variables
#time
t = MX.sym('t')
# Inputs
f_BP = MX.sym('f_BP')  # [Hz] Boost Pump frequency
p_topside = MX.sym('p_topside')  # [Hz] Boost Pump frequency
u = [f_BP, p_topside]

f_ESP_1 = MX.sym('f_ESP_1')  # [Hz] ESP frequency
alpha_1 = MX.sym('alpha_1')  # [%] Choke opening
u += [f_ESP_1, alpha_1]

f_ESP_2 = MX.sym('f_ESP_2')  # [Hz] ESP frequency
alpha_2 = MX.sym('alpha_2')  # [%] Choke opening
u += [f_ESP_2, alpha_2]

f_ESP_3 = MX.sym('f_ESP_3')  # [Hz] ESP frequency
alpha_3 = MX.sym('alpha_3')  # [%] Choke opening
u += [f_ESP_3, alpha_3]

f_ESP_4 = MX.sym('f_ESP_4')  # [Hz] ESP frequency
alpha_4 = MX.sym('alpha_4')  # [%] Choke opening
u += [f_ESP_4, alpha_4]

# States and algebraic variables
p_man = MX.sym('p_man')  # [Pa] manifold pressure
q_tr = MX.sym('q_tr')  # [m^3/s] Flow through the transportation line
x = [p_man, q_tr] # states
z = [] # algebraic variables

# Well 1
P_fbhp_1 = MX.sym('P_fbhp_1')  # [bar] Pressure fbhp
P_choke_1 = MX.sym('P_choke_1')  # [bar] Pressure in chokes
q_mean_1 = MX.sym('q_mean_1')  # [m^3/h] Average flow in the wells
P_intake_1 = MX.sym('P_ìntake_1')  # [bar] Pressure intake in ESP's
dP_bcs_1 = MX.sym('dP_bcs_1')  # [bar] Pressure discharge in ESP's
x += [P_fbhp_1, P_choke_1, q_mean_1]
z += [P_intake_1, dP_bcs_1]

# Well 2
P_fbhp_2 = MX.sym('P_fbhp_2')  # [bar] Pressure fbhp in ESP's
P_choke_2 = MX.sym('P_choke_2')  # [bar] Pressure in chokes
q_mean_2 = MX.sym('q_mean_2')  # [m^3/h] Average flow in the wells
P_intake_2 = MX.sym('P_ìntake_2')  # [bar] Pressure intake in ESP's
dP_bcs_2 = MX.sym('dP_bcs_2')  # [bar] Pressure discharge in ESP's
x += [P_fbhp_2, P_choke_2, q_mean_2]
z += [P_intake_2, dP_bcs_2]

# Well 3
P_fbhp_3 = MX.sym('P_fbhp_3')  # [bar] Pressure fbhp in ESP's
P_choke_3 = MX.sym('P_choke_3')  # [bar] Pressure in chokes
q_mean_3 = MX.sym('q_mean_3')  # [m^3/h] Average flow in the wells
P_intake_3 = MX.sym('P_ìntake_3')  # [bar] Pressure intake in ESP's
dP_bcs_3 = MX.sym('dP_bcs_3')  # [bar] Pressure discharge in ESP's
x += [P_fbhp_3, P_choke_3, q_mean_3]
z += [P_intake_3, dP_bcs_3]

# Well 4
P_fbhp_4 = MX.sym('P_fbhp_4')  # [bar] Pressure fbhp in ESP's
P_choke_4 = MX.sym('P_choke_4')  # [bar] Pressure in chokes
q_mean_4 = MX.sym('q_mean_4')  # [m^3/h] Average flow in the wells
P_intake_4 = MX.sym('P_ìntake_4')  # [bar] Pressure intake in ESP's
dP_bcs_4 = MX.sym('dP_bcs_4')  # [bar] Pressure discharge in ESP's
x += [P_fbhp_4, P_choke_4, q_mean_4]
z += [P_intake_4, dP_bcs_4]

# Defining the symbolic manifold model
mani_model = mani.model(0, x, z, u)

#%% Evaluation of steady-state
u0 = [56., 10 ** 5, 50., .5, 50., .5, 50., .5, 50., .5]

x0 = [76.52500, 4 * 85,
      64.11666, 120.91641, 85,
      64.11666, 120.91641, 85,
      64.11666, 120.91641, 85,
      64.11666, 120.91641, 85]

z0 = [30.03625, 239.95338-30.03625,
      30.03625, 239.95338-30.03625,
      30.03625, 239.95338-30.03625,
      30.03625, 239.95338-30.03625]

mani_solver = lambda y: array([float(i) for i in mani.model(0, y[0:-8], y[-8:], u0)])

y_ss = fsolve(mani_solver, x0+z0)

z_ss = y_ss[-8:]

x_ss = y_ss[0:-8]

x0 = [76.52500, 4 * 85,
      64.11666, 120.91641, 85,
      30.03625, 120.91641, 85,
      64.11666, 120.91641, 85,
      64.11666, 120.91641, 85]

z0 = [30.03625, 239.95338-30.03625,
      30.03625, 239.95338-30.03625,
      30.03625, 239.95338-30.03625,
      30.03625, 239.95338-30.03625]

#%% Dynamic Simulation
dae = {'x': vertcat(*x), 'z': vertcat(*z), 'p': vertcat(*u), 'ode': vertcat(*mani_model[0:-8]),
       'alg': vertcat(*mani_model[-8:])}

tfinal = 1000 # [s]

grid = linspace(0, tfinal, 100)
Lista_xf = []
Lista_zf = []
F = integrator('F', 'idas', dae, 0, grid)


res = F(x0 = x_ss, z0 = z_ss, p = u0)

Lista_xf.append(res["xf"])
Lista_zf.append(res["zf"])
x0 = Lista_xf[-1][:, -1]
z0 = Lista_zf[-1][:, -1]
Lista_zf = np.array(Lista_zf)
Lista_xf = np.array(Lista_xf)
Lista_zf_reshaped = Lista_zf.reshape(8, 100)
Lista_xf_reshaped = Lista_xf.reshape(14, 100)
valve_open = [random.uniform(0.4, .8) for _ in range(4)]
grid_cont = 1
for i in range(4):
    grid_cont += 1
    delta = 1000
    grid = linspace(tfinal,tfinal + delta , 100)
    tfinal += delta
    # valve_open = np.linspace(0.4, 0.6, 8)
    u0 = [56., 20 ** 5, 50., valve_open[i], 50., valve_open[i], 50., valve_open[i], 50., valve_open[i]]
    # if i == 0:
    #     u0 = [56., 20 ** 5, 50., open_valve[4], 50., open_valve[5], 50., open_valve[6], 50., open_valve[7]]
    # if i == 1:
    #     u0 = [56., 20 ** 5, 50., open_valve[0], 50., open_valve[1], 50., open_valve[2], 50., open_valve[3]]


    res = F(x0 = x0, z0 = z0, p = u0)
    np.hstack((Lista_zf_reshaped, res["zf"]))
    Lista_xf_reshaped = np.hstack((Lista_xf_reshaped, np.array(res["xf"])))
    Lista_zf_reshaped = np.hstack((Lista_zf_reshaped, np.array(res["zf"])))
    x0 = Lista_xf_reshaped[:,-1]
    z0 = Lista_zf_reshaped[:,-1]


#%% Plotted Graphs
rcParams['axes.formatter.useoffset'] = False

def Auto_plot(i,t, xl,yl,yi,yf):
    grid = linspace(0, tfinal, 100*grid_cont)
    plt.plot(grid, i.transpose())
    matplotlib.pyplot.title(t)
    matplotlib.pyplot.xlabel(xl)
    matplotlib.pyplot.ylabel(yl)
    plt.ylim([yi,yf])
    plt.grid()
    plt.show()

    
Auto_plot(Lista_zf_reshaped[[1, 3, 5, 7], :],"Pressure Discharge in ESP's",'Time/(s)','Pressure/(bar)', 107, 113)
Auto_plot(Lista_xf_reshaped[[2, 5, 8, 11], :],"Pressure fbhp in ESP's",'Time/(s)','Pressure/(bar)', 76, 82)
Auto_plot(Lista_xf_reshaped[[3, 6, 9, 12], :],'Pressure in Chokes','Time/(s)','Pressure/(bar)', 70, 90)
Auto_plot(Lista_xf_reshaped[[4, 7, 10, 13], :],'Average Flow in the Wells','Time/(s)','Flow Rate/(m^3/s)', 40,60)
Auto_plot(Lista_xf_reshaped[[1], :],'Flow Through the Transportation Line','Time/(s)','Flow Rate/(m^3/s)', 160,220)
Auto_plot(Lista_xf_reshaped[[0], :],'Manifold Pressure' ,'Time/(s)','Pressure/(Pa)', 0, 60)

plt.plot(grid, Lista_zf_reshaped[[1, 3, 5, 7], :].transpose())
matplotlib.pyplot.title("Pressure Discharge in ESP's")
matplotlib.pyplot.xlabel('Time/(s)')
matplotlib.pyplot.ylabel('Pressure/(bar)')
plt.ylim([107,113])
plt.grid()
plt.show()

plt.plot(grid, Lista_xf_reshaped[[2, 5, 8, 11], :].transpose())
matplotlib.pyplot.title("'Pressure fbhp in ESP's")
matplotlib.pyplot.xlabel('Time/(s)')
matplotlib.pyplot.ylabel('Pressure/(bar)')
plt.ylim([76, 82])
plt.grid()
plt.show()

plt.plot(grid, Lista_xf_reshaped[[3, 6, 9, 12], :].transpose())
matplotlib.pyplot.title('Pressure in Chokes')
matplotlib.pyplot.xlabel('Time/(s)')
matplotlib.pyplot.ylabel('Pressure/(bar)')
plt.ylim([70,90])
plt.grid()
plt.show()

plt.plot(grid, Lista_xf_reshaped[[4, 7, 10, 13], :].transpose())
matplotlib.pyplot.title('Average Flow in the Wells')
matplotlib.pyplot.xlabel('Time/(s)')
matplotlib.pyplot.ylabel('Flow/(m^3/s)')
plt.ylim([40,60])
plt.grid()
plt.show()

plt.plot(grid, Lista_xf_reshaped[[1], :].transpose())
matplotlib.pyplot.title('Flow Through the Transportation Line')
matplotlib.pyplot.xlabel('Tempo/(s)')
matplotlib.pyplot.ylabel('Flow Rate/(m^3/s)')
plt.ylim([160,220])
plt.grid()
plt.show()

plt.plot(grid, Lista_xf_reshaped[[0], :].transpose())
matplotlib.pyplot.title('Manifold Pressure')
matplotlib.pyplot.xlabel('Time/(s)')
matplotlib.pyplot.ylabel('Pressure/(Pa)')
plt.ylim([0,60])
plt.grid()
plt.show()

#%% p_intake é desnecessário
# plt.plot(grid, Array_zf[[0, 2, 4, 6], :].transpose())
# matplotlib.pyplot.title("Pressure Intake in ESP's")
# matplotlib.pyplot.xlabel('Time/(s)')
# matplotlib.pyplot.ylabel('Pressure/(bar)')
# plt.ylim([0,100])
# plt.grid()
# plt.show()
