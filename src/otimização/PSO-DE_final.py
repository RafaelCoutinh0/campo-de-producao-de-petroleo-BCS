from manifold import *
from initialization_oil_production_basic import *
from bcs_models import *
import pyswarms as ps

import time
from scipy.optimize import fsolve, differential_evolution
from casadi import *
import numpy as np


np.random.seed(42)

# Variáveis manipuláveis (entradas do modelo)
u = MX.sym('u', 10)  # [f_BP, p_topside, f_ESP_1, alpha_1, ..., f_ESP_4, alpha_4]

# Variáveis do sistema (saídas do modelo)
x = MX.sym('x', 14)  # Estados (pressões, vazões, etc.)
z = MX.sym('z', 8)   # Variáveis algébricas
Pp = 3000
Pe = 0.91
# Função objetivo
def objective_function(u, x, z):
    penalty_factor = 100
    global Pe, Pp
    restqmain1 = np.array([x[4] - ((z[1] + 334.2554) / 19.0913), ((z[1] + 193.8028) / 4.4338) - x[4]])
    restqmain2 = np.array([x[7] - ((z[3] + 334.2554) / 19.0913), ((z[3] + 193.8028) / 4.4338) - x[7]])
    restqmain3 = np.array([x[10] - ((z[5] + 334.2554) / 19.0913), ((z[5] + 193.8028) / 4.4338) - x[10]])
    restqmain4 = np.array([x[13] - ((z[7] + 334.2554) / 19.0913), ((z[7] + 193.8028) / 4.4338) - x[13]])

    constraint_penalty = penalty_factor * (
        np.sum(np.maximum(restqmain1, 0)) +
        np.sum(np.maximum(restqmain2, 0)) +
        np.sum(np.maximum(restqmain3, 0)) +
        np.sum(np.maximum(restqmain4, 0))
    )

    production_term = Pp * x[1]
    cost_term = (
        -(9653.04 * (x[1] / 3600) * (1.0963e3 * (u[0] / 50)**2) * 0.001) - 
        ((x[4] / 3600) * z[1] * 1e2) -
        ((x[7] / 3600) * z[3] * 1e2) -
        ((x[10] / 3600) * z[5] * 1e2) -
        ((x[13] / 3600) * z[7] * 1e2)
    ) * Pe

    total_objective = production_term + cost_term - constraint_penalty
    return -total_objective

# Função solver para encontrar estado estacionário
def mani_solver(y, u):
    return np.array([float(i) for i in mani.model(0, y[:14], y[14:], u)])

# Limites das variáveis de controle
bounds = np.array([
    [35, 65],        # f_BP (Hz)
    [0.8e5, 0.8e5],  # p_topside (Pa)
    [35, 65],        # f_ESP_1 (Hz)
    [0, 1],          # alpha_1 (-)
    [35, 65],        # f_ESP_2 (Hz)
    [0, 1],          # alpha_2 (-)
    [35, 65],        # f_ESP_3 (Hz)
    [0, 1],          # alpha_3 (-)
    [35, 65],        # f_ESP_4 (Hz)
    [0, 1]           # alpha_4 (-)
])

lower_bounds = bounds[:, 0]
upper_bounds = bounds[:, 1]

# Condições iniciais
x0 = np.array([76.52500, 340, 64.11666, 120.91641, 85,
               64.11666, 120.91641, 85, 64.11666,
               120.91641, 85, 64.11666, 120.91641, 85])

z0 = np.array([30.03625, 209.91713, 30.03625, 209.91713,
               30.03625, 209.91713, 30.03625, 209.91713])

u0 = np.array([50., 0.8e5, 50., 0.5, 50., 0.5, 50., 0.5, 50., 0.5])

# Resolver o sistema para encontrar o estado estacionário
y_ss = fsolve(lambda y: mani_solver(y, u0), np.concatenate((x0, z0)))

# Atualizar x0 e z0 com os resultados
x0 = y_ss[:14]
z0 = y_ss[14:]

# Função objetivo para o PSO e DE
def optimization_objective(u):
    n_particles = u.shape[0] if len(u.shape) > 1 else 1
    obj_values = np.zeros(n_particles)
    
    for i in range(n_particles):
        u_current = u[i] if n_particles > 1 else u
        
        # Resolver o sistema para encontrar o estado estacionário com as variáveis manipuladas atuais
        y_ss = fsolve(lambda y: mani_solver(y, u_current), np.concatenate((x0, z0)))
        
        # Atualizar x0 e z0 com os resultados
        x0_current = y_ss[:14]
        z0_current = y_ss[14:]
        
        # Calcular a função objetivo com os valores atualizados
        obj_values[i] = objective_function(u_current, x0_current, z0_current)
    
    return obj_values if n_particles > 1 else obj_values[0]

# Configurações do PSO
options_pso = {'c1': 2, 'c2': 1.3, 'w': 0.87}
start_time_de = time.time()
# Inicializar e rodar o PSO
optimizer_pso = ps.single.GlobalBestPSO(
    n_particles=50,
    dimensions=10,
    options=options_pso,
    bounds=(lower_bounds, upper_bounds)
)

best_obj_value_pso, best_u_pso = optimizer_pso.optimize(optimization_objective, iters=10)

# Agora usar Differential Evolution para refinar a solução obtida pelo PSO
bounds_de = [(lb, ub) for lb, ub in zip(lower_bounds, upper_bounds)]
# Agora usar Differential Evolution para refinar a solução obtida pelo PSO
bounds_de = [(lb, ub) for lb, ub in zip(lower_bounds, upper_bounds)]


# Medir o tempo de execução do DE


# Chamada do Differential Evolution
result_de = differential_evolution(
    optimization_objective,
    bounds=bounds_de,
    strategy='best1bin',
    maxiter=50,
    popsize=15,
    tol=0.01,
    mutation=(0.5, 1),
    recombination=0.7,
    seed=62,
    polish=True,
    disp=True,
    x0=best_u_pso  # Passar o resultado do PSO como ponto inicial
)
end_time_de = time.time()
de_execution_time = end_time_de - start_time_de
# Calcular o tempo total de execução do DE


# Exibir o tempo de execução
print(f"Tempo de execução do Differential Evolution: {de_execution_time:.2f} segundos")

# Resultados finais
best_u_hybrid = result_de.x
best_obj_value_hybrid = result_de.fun

# Resolver o sistema com a melhor solução encontrada
y_ss_opt = fsolve(lambda y: mani_solver(y, best_u_hybrid), np.concatenate((x0, z0)))

# Resultados
x0_opt = y_ss_opt[:14]
z0_opt = y_ss_opt[14:]

# Códigos ANSI para negrito e cores
BOLD = '\033[1m'
RESET = '\033[0m'
CYAN = '\033[36m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
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

print("Melhor solução híbrida (variáveis manipuladas):", best_u_hybrid)
print("Melhor valor da função objetivo (híbrido):", best_obj_value_hybrid)
print("Valores de x0 após otimização híbrida:", x0_opt)
print("Valores de z0 após otimização híbrida:", z0_opt)
print("q_tr otimizado (m³/h):", x0_opt[1])

# Cálculos de energia
energybooster = (9653.04 * (x0_opt[1] / 3600) * (1.0963e3 * (best_u_hybrid[0] / 50) ** 2) * 0.001)
energybcs1 = (x0_opt[4] / 3600) * (z0_opt[1] * 1e2)
energybcs2 = (x0_opt[7] / 3600) * (z0_opt[3] * 1e2)
energybcs3 = (x0_opt[10] / 3600) * (z0_opt[5] * 1e2)
energybcs4 = (x0_opt[13] / 3600) * (z0_opt[7] * 1e2)
energytot = (energybooster + energybcs1 + energybcs2 + energybcs3 + energybcs4) * Pe
venda = Pp * x0_opt[1]
print(f"{GREEN}{BOLD}{'='*50}{RESET}\n{GREEN}{BOLD}VALORES DA FUNÇÃO OBJETIVO:{RESET}")
print(f"\n{YELLOW}{BOLD}Valor da venda do petróleo{RESET}: R${venda:.2f}")
print(f"{YELLOW}{BOLD}Preço da energia total{RESET}: R${int(energytot):.2f}")
print(f"{GREEN}{BOLD}Energia do booster{RESET}: {int(energybooster):.2f} Kwh")
print(f"{GREEN}{BOLD}Energia do BCS 1{RESET}: {energybcs1:.2f} Kwh")
print(f"{GREEN}{BOLD}Energia do BCS 2{RESET}: {energybcs2:.2f} Kwh")
print(f"{GREEN}{BOLD}Energia do BCS 3{RESET}: {energybcs3:.2f} Kwh")
print(f"{GREEN}{BOLD}Energia do BCS 4{RESET}: {energybcs4:.2f} Kwh")

print(f"\nTempo de execução PSODE: {de_execution_time:.2f} segundos")
