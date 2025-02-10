from manifold import *
from initialization_oil_production_basic import *
from bcs_models import *
import pyswarms as ps
import time
from scipy.optimize import fsolve, differential_evolution
from casadi import *
import numpy as np
import casadi as ca

np.random.seed(42)

# Variáveis manipuláveis (entradas do modelo)
u = MX.sym('u', 10)

# Variáveis do sistema (saídas do modelo)
x = MX.sym('x', 14)
z = MX.sym('z', 8)

# Parâmetros do problema (mesmos de optimize3_model.py)
Pp = 5
Pe = 0.91
penalty_factor = 1e20  # Penalização aumentada para forçar o atendimento das restrições

# Definir limites das variáveis (iguais aos do optimize3_model.py)
u_ptopo = 0.8e5
bounds = np.array([
    [35, 65],
    [u_ptopo, u_ptopo],
    [35, 65],
    [0, 1],
    [35, 65],
    [0, 1],
    [35, 65],
    [0, 1],
    [35, 65],
    [0, 1]
])
lower_bounds = bounds[:, 0]
upper_bounds = bounds[:, 1]

# Estados e variáveis algébricas iniciais (iguais aos do optimize3_model.py)
x0 = [76.52500, 4 * 85,
      64.11666, 120.91641, 85,
      64.11666, 120.91641, 85,
      64.11666, 120.91641, 85,
      64.11666, 120.91641, 85]

z0 = [30.03625, 239.95338-30.03625,
      30.03625, 239.95338-30.03625,
      30.03625, 239.95338-30.03625,
      30.03625, 239.95338-30.03625]

u0 = np.array([35., 0.8e5, 35., 0.5, 35., 0.5, 35., 0.5, 35., 0.5])

# Função que utiliza o modelo para resolver o sistema (usado pelo fsolve)
def mani_solver(y, u):
    return np.array([float(i) for i in mani.model(0, y[:14], y[14:], u)])


# Resolver o sistema para o estado estacionário com u0 (usado como ponto inicial)
y_ss = fsolve(lambda y: mani_solver(y, u0), np.concatenate((x0, z0)))
x0, z0 = y_ss[:14], y_ss[14:]


# Função Objetivo que inclui as restrições por penalização
def objective_function(u):
    # Resolver o sistema para encontrar o estado estacionário com o u atual
    y_ss = fsolve(lambda y: mani_solver(y, u), np.concatenate((x0, z0)))
    x_current = y_ss[:14]
    z_current = y_ss[14:]

    # Restrições de vazão para cada poço (conforme optimize3_model.py)
    restqmain1 = np.array([
        x_current[4] - ((z_current[1] + 334.2554) / 19.0913),
        ((z_current[1] + 193.8028) / 4.4338) - x_current[4]
    ])
    restqmain2 = np.array([
        x_current[7] - ((z_current[3] + 334.2554) / 19.0913),
        ((z_current[3] + 193.8028) / 4.4338) - x_current[7]
    ])
    restqmain3 = np.array([
        x_current[10] - ((z_current[5] + 334.2554) / 19.0913),
        ((z_current[5] + 193.8028) / 4.4338) - x_current[10]
    ])
    restqmain4 = np.array([
        x_current[13] - ((z_current[7] + 334.2554) / 19.0913),
        ((z_current[7] + 193.8028) / 4.4338) - x_current[13]
    ])
    g_inequality = np.hstack([restqmain1, restqmain2, restqmain3, restqmain4])
    inequality_penalty = penalty_factor * np.sum(np.maximum(g_inequality, 0))

    # Restrições de igualdade do modelo
    g_equality = mani.model(0, x_current, z_current, u)
    if isinstance(g_equality, list):
        g_equality = ca.vertcat(*g_equality)
    g_equality = np.array(g_equality.full()).flatten()
    equality_penalty = penalty_factor * np.sum(np.abs(g_equality))

    # Penalização para estados (x_current e z_current) negativos
    state_penalty = penalty_factor * (np.sum(np.maximum(0 - x_current, 0)) +
                                      np.sum(np.maximum(0 - z_current, 0)))

    # Termos da função objetivo (mesma definição de optimize3_model.py)
    production_term = Pp * x_current[1]
    cost_term = (
                        (9653.04 * (x_current[1] / 3600) * (1.0963e3 * (u[0] / 50) ** 2) * 0.001) +
                        ((x_current[4] / 3600) * z_current[1] * 1e2) +
                        ((x_current[7] / 3600) * z_current[3] * 1e2) +
                        ((x_current[10] / 3600) * z_current[5] * 1e2) +
                        ((x_current[13] / 3600) * z_current[7] * 1e2)
                ) * Pe

    return -(production_term - cost_term - equality_penalty - inequality_penalty - state_penalty)


# Otimização usando Differential Evolution
start_time_de = time.time()
bounds_de = [(lb, ub) for lb, ub in zip(lower_bounds, upper_bounds)]
result_de = differential_evolution(
    objective_function,
    bounds=bounds_de,
    strategy='best1bin',
    maxiter=50,
    popsize=15,
    tol=0.01,
    mutation=(0.5, 1.),
    recombination=0.7,
    seed=70,
    polish=True,
    disp=True
)
end_time_de = time.time()
de_execution_time = end_time_de - start_time_de

# Melhor solução encontrada
best_u_hybrid = result_de.x
best_obj_value_hybrid = result_de.fun

# Resolver o sistema com a melhor solução encontrada
y_ss_opt = fsolve(lambda y: mani_solver(y, best_u_hybrid), np.concatenate((x0, z0)))
x0_opt = y_ss_opt[:14]
z0_opt = y_ss_opt[14:]
u_opt = best_u_hybrid  # Variáveis manipuladas otimizadas

# Cálculos de energia (mesma definição de optimize3_model.py)
energybooster = (9653.04 * (x0_opt[1] / 3600) * (1.0963e3 * (u_opt[0] / 50) ** 2) * 0.001)
energybcs1 = (x0_opt[4] / 3600) * (z0_opt[1] * 1e2)
energybcs2 = (x0_opt[7] / 3600) * (z0_opt[3] * 1e2)
energybcs3 = (x0_opt[10] / 3600) * (z0_opt[5] * 1e2)
energybcs4 = (x0_opt[13] / 3600) * (z0_opt[7] * 1e2)
energytot = (energybooster + energybcs1 + energybcs2 + energybcs3 + energybcs4) * Pe
venda = Pp * x0_opt[1]

# Códigos ANSI para negrito e cores
BOLD = '\033[1m'
RESET = '\033[0m'
CYAN = '\033[36m'
GREEN = '\033[32m'
YELLOW = '\033[33m'

# Nomes das variáveis para impressão (iguais ao optimize3_model.py)
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

print(f"\n{CYAN}{BOLD}{'=' * 50}{RESET}\n{CYAN}{BOLD}Variáveis Controladas{RESET}")
for i, name in enumerate(state_names):
    print(f"{name}: Otimizado = {float(x0_opt[i]):.4f}, fsolve = {float(x0_opt[i]):.4f}")
for i, name in enumerate(algebraic_names):
    print(f"{name}: Otimizado = {float(z0_opt[i]):.4f}, fsolve = {float(z0_opt[i]):.4f}")
print(f"{CYAN}{BOLD}{'=' * 50}{RESET}\n{CYAN}{BOLD}Variáveis Manipuladas{RESET}")
for i, name in enumerate(control_names):
    print(f"{name}: {float(u_opt[i]):.4f}")
print(f"{CYAN}{BOLD}{'=' * 50}{RESET}")

print(f"{GREEN}{BOLD}{'=' * 50}{RESET}\n{GREEN}{BOLD}VALORES DA FUNÇÃO OBJETIVO:{RESET}")
print(f"\n{YELLOW}{BOLD}Valor da venda do petróleo{RESET}: R${venda:.2f}")
print(f"{YELLOW}{BOLD}Preço da energia total{RESET}: R${int(energytot):.2f}")
print(f"{GREEN}{BOLD}Energia do booster{RESET}: {int(energybooster):.2f} Kwh")
print(f"{GREEN}{BOLD}Energia do BCS 1{RESET}: {energybcs1:.2f} Kwh")
print(f"{GREEN}{BOLD}Energia do BCS 2{RESET}: {energybcs2:.2f} Kwh")
print(f"{GREEN}{BOLD}Energia do BCS 3{RESET}: {energybcs3:.2f} Kwh")
print(f"{GREEN}{BOLD}Energia do BCS 4{RESET}: {energybcs4:.2f} Kwh")

print(f"\nTempo de execução PSODE: {de_execution_time:.2f} segundos")
