from manifold import *
from initialization_oil_production_basic import *
from bcs_models import *
from casadi import *
import numpy as np
from scipy.optimize import fsolve

# Variáveis manipuláveis (entradas do modelo)
u = MX.sym('u', 10)  # [f_BP, p_topside, f_ESP_1, alpha_1, ..., f_ESP_4, alpha_4]

# Variáveis do sistema (saídas do modelo)
x = MX.sym('x', 14)  # Estados (pressões, vazões, etc.)
z = MX.sym('z', 8)   # Variáveis algébricas

# Parâmetros do problema
objective = -(70000 * x[1]) + ((9653.04 * (x[1]/3600) * (1.0963e3 * (u[0]/ 50) ** 2) * 0.001) +
                               ((x[4]/3600) * z[1] * 1e2) +
                               ((x[7]/3600) * z[3] * 1e2) +
                               ((x[10]/3600) * z[5] * 1e2)  +
                               ((x[13]/3600) * z[7] * 1e2))

mani_model = mani.model(0, x, z, u)

# Restrições do modelo e operacionais
limite_de_energia = 600
g_constraints = vertcat(*mani_model,((9653.04 * (x[1]/3600) * (1.0963e3 * (u[0]/ 50) ** 2) * 0.001) +
                                                        ((x[4]/3600) * z[1] * 1e2) +
                                                        ((x[7]/3600) * z[3] * 1e2) +
                                                        ((x[10]/3600) * z[5] * 1e2)  +
                                                        ((x[13]/3600) * z[7] * 1e2)) - limite_de_energia)  # Apenas as restrições do modelo

# Configuração do problema de otimização
nlp = {'x': vertcat(x, z, u), 'f': objective, 'g': g_constraints}
solver = nlpsol('solver', 'ipopt', nlp)

# Valores iniciais para as variáveis manipuláveis (u), estados (x) e algébricas (z)
u0 = np.array([40., 1e5, 50., 1, 40., 1, 40., 1, 40., 1])
mani_solver = lambda y: np.array([float(i) for i in mani.model(0, y[:14], y[14:], u0)])
y_ss = fsolve(mani_solver, np.concatenate((x0, z0)))

# Atualizar x0 e z0 com os resultados do fsolve
x0 = y_ss[:14]
z0 = y_ss[14:]
x0_full = np.concatenate((x0, z0, u0))

# Definir limites das variáveis
lbx = [1] + [0] * 13 + [0] * 8 + [35, 0.8e5, 35, 0, 35, 0, 35, 0, 35, 0]
ubx = [np.inf] * 14 + [np.inf] * 8 + [65, 1.2e5, 65, 1, 65, 1, 65,1, 65, 1]  # Superiores


# Resolver o problema de otimização
sol = solver(x0=x0_full, lbx=lbx, ubx=ubx, lbg=0, ubg=0)
optimal_solution = sol['x']

# Extrair resultados otimizados
x_opt = optimal_solution[:14]
z_opt = optimal_solution[14:22]
u_opt = optimal_solution[22:]

# Imprimir resultados otimizados
state_names = [
    "p_man (bar)", "q_tr (m^3/h)",
    "P_fbhp_1 (bar)", "P_choke_1 (bar)", "q_mean_1 (m^3/h)",
    "P_fbhp_2 (bar)", "P_choke_2 (bar)", "q_mean_2 (m^3/h)",
    "P_fbhp_3 (bar)", "P_choke_3 (bar)", "q_mean_3 (m^3/h)",
    "P_fbhp_4 (bar)", "P_choke_4 (bar)", "q_mean_4 (m^3/h)"
]

algebraic_names = [
    "P_intake_1 (bar)", "dP_bcs_1 (bar)",
    "P_intake_2 (bar)", "dP_bcs_2 (bar)",
    "P_intake_3 (bar)", "dP_bcs_3 (bar)",
    "P_intake_4 (bar)", "dP_bcs_4 (bar)"
]

control_names = [
    "f_BP (Hz)", "p_topside (Pa)",
    "f_ESP_1 (Hz)", "alpha_1 (-)",
    "f_ESP_2 (Hz)", "alpha_2 (-)",
    "f_ESP_3 (Hz)", "alpha_3 (-)",
    "f_ESP_4 (Hz)", "alpha_4 (-)"
]

# Verificar consistência com fsolve
mani_solver = lambda y: np.array([float(i) for i in mani.model(0, y[:14], y[14:], u_opt)])
y_ss = fsolve(mani_solver, np.concatenate((x0, z0)), xtol=1e-8, maxfev=10000)

# Separar resultados do fsolve
x_ss = y_ss[:14]
z_ss = y_ss[14:]
u0 = u_opt

# Imprimir resultados otimizados

state_names = [
    "p_man (bar)", "q_tr (m^3/h)",
    "P_fbhp_1 (bar)", "P_choke_1 (bar)", "q_mean_1 (m^3/h)",
    "P_fbhp_2 (bar)", "P_choke_2 (bar)", "q_mean_2 (m^3/h)",
    "P_fbhp_3 (bar)", "P_choke_3 (bar)", "q_mean_3 (m^3/h)",
    "P_fbhp_4 (bar)", "P_choke_4 (bar)", "q_mean_4 (m^3/h)"
]

algebraic_names = [
    "P_intake_1 (bar)", "dP_bcs_1 (bar)",
    "P_intake_2 (bar)", "dP_bcs_2 (bar)",
    "P_intake_3 (bar)", "dP_bcs_3 (bar)",
    "P_intake_4 (bar)", "dP_bcs_4 (bar)"
]

control_names = [
    "f_BP (Hz)", "p_topside (Pa)",
    "f_ESP_1 (Hz)", "alpha_1 (-)",
    "f_ESP_2 (Hz)", "alpha_2 (-)",
    "f_ESP_3 (Hz)", "alpha_3 (-)",
    "f_ESP_4 (Hz)", "alpha_4 (-)"
]

# Verificar consistência com fsolve
mani_solver = lambda y: np.array([float(i) for i in mani.model(0, y[:14], y[14:], u_opt)])
y_ss = fsolve(mani_solver, np.concatenate((x0, z0)))

# Separar resultados do fsolve
x_ss = y_ss[:14]
z_ss = y_ss[14:]

# Comparar otimização e fsolve
# Códigos ANSI para negrito e cores
BOLD = '\033[1m'
RESET = '\033[0m'
CYAN = '\033[36m'
GREEN = '\033[32m'
YELLOW = '\033[33m'

print(f"\n{CYAN}{BOLD}{'='*50}{RESET}\n{CYAN}{BOLD}Variáveis Controladas{RESET}")
for i, name in enumerate(state_names):
    print(f"{name}: Otimizado = {float(x_opt[i]):.4f}, fsolve = {float(x_ss[i]):.4f}")
for i, name in enumerate(algebraic_names):
    print(f"{name}: Otimizado = {float(z_opt[i]):.4f}, fsolve = {float(z_ss[i]):.4f}")
print(f"{YELLOW}{BOLD}{'='*50}{RESET}\n{YELLOW}{BOLD}Variáveis Manipuladas{RESET}")
for i, name in enumerate(control_names):
    print(f"{name}: {float(u_opt[i]):.4f}")
print(f"{YELLOW}{BOLD}{'='*50}{RESET}")
# Cálculos de energia
energybooster = (9653.04 * (x_ss[1]/3600) * (1.0963e3 * (u_opt[0]/50) ** 2) * 0.001)
energybcs1 = (x_ss[4]/3600) * (z_ss[1]*1e2)
energybcs2 = (x_ss[7]/3600) * (z_ss[3]*1e2)
energybcs3 = (x_ss[10]/3600) * (z_ss[5]*1e2)
energybcs4 = (x_ss[13]/3600) * (z_ss[7]*1e2)
energytot = energybooster + energybcs1 + energybcs2 + energybcs3 + energybcs4

print(f"{GREEN}{BOLD}{'='*50}{RESET}\n{GREEN}{BOLD}VALORES DA FUNÇÃO OBJETIVO:{RESET}")
print(f"{GREEN}{BOLD}Energia total (KW){RESET}: {int(energytot[0]):.2f}")
print(f"{GREEN}{BOLD}Energia do booster{RESET}: {int(energybooster[0]):.2f} Kwh")
print(f"{GREEN}{BOLD}Energia do BCS 1{RESET}: {energybcs1:.2f} Kwh")
print(f"{GREEN}{BOLD}Energia do BCS 2{RESET}: {energybcs2:.2f} Kwh")
print(f"{GREEN}{BOLD}Energia do BCS 3{RESET}: {energybcs3:.2f} Kwh")
print(f"{GREEN}{BOLD}Energia do BCS 4{RESET}: {energybcs4:.2f} Kwh")