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

rho = 984
g = 9.81
eff = 0.98

objective =  -(3000 * x[1]) + ((9653.04 * (x[1]/3600) * (1.0963e3 * (u[0]/ 50) ** 2) * 0.001) + \
        ((x[4]/3600) * z[1] * 1e2) + \
        ((x[7]/3600) * z[3] * 1e2) + \
        ((x[10]/3600) * z[5] * 1e2)  + \
        ((x[13]/3600) * z[7] * 1e2)) * 0.91

# Certifique-se de que a função mani.model esteja implementada corretamente
mani_model = mani.model(0, x, z, u)

# Restrições de vazão para cada poço com os valores diretos
restqmain1 = vertcat(x[4] - ((z[1] + 334.2554) / 19.0913), ((z[1] + 193.8028) / 4.4338) - x[4])
restqmain2 = vertcat(x[7] - ((z[3] + 334.2554) / 19.0913), ((z[3] + 193.8028) / 4.4338) - x[7])
restqmain3 = vertcat(x[10] - ((z[5] + 334.2554) / 19.0913), ((z[5] + 193.8028) / 4.4338) - x[10])
restqmain4 = vertcat(x[13] - ((z[7] + 334.2554) / 19.0913), ((z[7] + 193.8028) / 4.4338) - x[13])

# Restrições de igualdade (modelo do sistema)
g_equality = vertcat(*mani_model)

# Restrições de desigualdade (limites operacionais)
g_inequality = vertcat(restqmain1, restqmain2, restqmain3, restqmain4)

# Concatenar todas as restrições
g_constraints = vertcat(g_equality, g_inequality)

# Definir limites para as restrições (lbg e ubg)
num_eq = g_equality.shape[0]  # Número de igualdades
num_ineq = g_inequality.shape[0]  # Número de desigualdades

lbg = [0.0] * num_eq + [0.0] * num_ineq  # Igualdades: 0 ≤ g ≤ 0 | Desigualdades: 0 ≤ g ≤ ∞
ubg = [0.0] * num_eq + [np.inf] * num_ineq

# Configuração do problema de otimização
nlp = {'x': vertcat(x, z, u), 'f': objective, 'g': g_constraints}
solver = nlpsol('solver', 'ipopt', nlp)

# Valores iniciais para as variáveis manipuláveis (u), estados (x) e algébricas (z)
u0 = np.array([50., 1.0e5, 50., 0.5, 50., 0.5, 50., 0.5, 50., 0.5])
mani_solver = lambda y: np.array([float(i) for i in mani.model(0, y[:14], y[14:], u0)])
x0 = [76.52500, 4 * 85,
      64.11666, 120.91641, 85,
      64.11666, 120.91641, 85,
      64.11666, 120.91641, 85,
      64.11666, 120.91641, 85]

z0 = [30.03625, 239.95338-30.03625,
      30.03625, 239.95338-30.03625,
      30.03625, 239.95338-30.03625,
      30.03625, 239.95338-30.03625]
y_ss = fsolve(mani_solver, np.concatenate((x0, z0)))

# Atualizar x0 e z0 com os resultados do fsolve
x0 = y_ss[:14]
z0 = y_ss[14:]

x0_full = np.concatenate((x0, z0, u0))

# Definir limites das variáveis
lbx = [0] +[0]+ [0] * 12 + [0] * 8 + [35, 0.8e5, 35, 0, 35, 0, 35, 0, 35, 0] #Inferiores
ubx = [np.inf] * 14 + [np.inf] * 8 + [65, 1.2e5, 65, 1, 65, 1, 65,1, 65, 1]  # Superiores

# Resolver o problema de otimização
sol = solver(x0=x0_full, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
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
print(f"{CYAN}{BOLD}{'='*50}{RESET}\n{CYAN}{BOLD}Variáveis Manipuladas{RESET}")
for i, name in enumerate(control_names):
    print(f"{name}: {float(u_opt[i]):.4f}")
print(f"{CYAN}{BOLD}{'='*50}{RESET}")
# Cálculos de energia
energybooster = (9653.04 * (x_ss[1]/3600) * (1.0963e3 * (u_opt[0]/50) ** 2) * 0.001)
energybcs1 = (x_ss[4]/3600) * (z_ss[1]*1e2)
energybcs2 = (x_ss[7]/3600) * (z_ss[3]*1e2)
energybcs3 = (x_ss[10]/3600) * (z_ss[5]*1e2)
energybcs4 = (x_ss[13]/3600) * (z_ss[7]*1e2)
energytot = (energybooster + energybcs1 + energybcs2 + energybcs3 + energybcs4) * 0.91
venda = 3000 * x_ss[1]

print(f"{GREEN}{BOLD}{'='*50}{RESET}\n{GREEN}{BOLD}VALORES DA FUNÇÃO OBJETIVO:{RESET}")
print(f"\n{YELLOW}{BOLD}Valor da venda do petróleo{RESET}: R${venda:.2f}")
print(f"{YELLOW}{BOLD}Preço da energia total{RESET}: R${int(energytot[0]):.2f}")
print(f"{GREEN}{BOLD}Energia do booster{RESET}: {int(energybooster[0]):.2f} Kwh")
print(f"{GREEN}{BOLD}Energia do BCS 1{RESET}: {energybcs1:.2f} Kwh")
print(f"{GREEN}{BOLD}Energia do BCS 2{RESET}: {energybcs2:.2f} Kwh")
print(f"{GREEN}{BOLD}Energia do BCS 3{RESET}: {energybcs3:.2f} Kwh")
print(f"{GREEN}{BOLD}Energia do BCS 4{RESET}: {energybcs4:.2f} Kwh")


vazão = [x_opt[4], x_opt[7], x_opt[10], x_opt[13]]
deltapoço = [z_opt[1],z_opt[3],z_opt[5],z_opt[7]]
plt.figure(dpi=250)
import numpy as np
plt.plot(np.ravel(vazão), np.ravel(deltapoço), 'b.')
plt.plot([28.55, 20.77], [206.6, 58.07], 'k--', linewidth=3)
plt.plot([82.1, 53.6], [170.1, 44.7], 'k--', linewidth=3)
plt.xlabel(r"$q_{main}$ /(m$^3 \cdot$ h$^{-1}$)", fontsize=15)
plt.ylabel('$dP_{bcs}$ /bar',fontsize = 15)
plt.grid()
plt.show()

plt.figure(dpi=250)
plt.plot(x_opt[1],x_opt[0], 'b.')
plt.plot([110, 225], [0, 0], 'k--', linewidth=3)
plt.xlabel(r"$q_{tr}$ /(m$^3 \cdot$ h$^{-1}$)", fontsize=15)
plt.ylabel('$P_{man}$ /bar', fontsize=15)
plt.grid()
plt.show()


#%% Sesão de teste de resultados
# x0 = [76.52500, 4 * 85,
#       64.11666, 120.91641, 85,
#       64.11666, 120.91641, 85,
#       64.11666, 120.91641, 85,
#       64.11666, 120.91641, 85]
#
# z0 = [30.03625, 239.95338-30.03625,
#       30.03625, 239.95338-30.03625,
#       30.03625, 239.95338-30.03625,
#       30.03625, 239.95338-30.03625]
#
# u0 = np.array([65, 1.2e5, 65, 1, 43.12, 0.44, 62.17, 0.29, 63.52, 0.22])
# mani_solver = lambda y: np.array([float(i) for i in mani.model(0, y[:14], y[14:], u0)])
# y_ss = fsolve(mani_solver, np.concatenate((x0, z0)))
#
# # Separar resultados em x e z
# x_ss = y_ss[:14]
# z_ss = y_ss[14:]
#
# # Imprimir resultados com labels formatados
# print("\n" + "="*60 + "\nESTADOS (x):\n" + "="*60)
# for name, value in zip(state_names, x_ss):
#     print(f"{name.ljust(25)}: {value:.4f}")
#
# print("\n" + "="*60 + "\nVARIÁVEIS ALGÉBRICAS (z):\n" + "="*60)
# for name, value in zip(algebraic_names, z_ss):
#     print(f"{name.ljust(25)}: {value:.4f}")
#
# print("\n" + "="*60 + "\nU0 UTILIZADO:\n" + "="*60)
# print(u0)
#
# vazão = [x_ss[4], x_ss[7], x_ss[10], x_ss[13]]
# deltapoço = [z_ss[1],z_ss[3],z_ss[5],z_ss[7]]
# plt.figure(dpi=250)
# import numpy as np
# plt.plot(np.ravel(vazão), np.ravel(deltapoço), 'b.')
# plt.plot([28.55, 20.77], [206.6, 58.07], 'k--', linewidth=3)
# plt.plot([82.1, 53.6], [170.1, 44.7], 'k--', linewidth=3)
# plt.xlabel(r"$q_{main}$ /(m$^3 \cdot$ h$^{-1}$)", fontsize=15)
# plt.ylabel('$dP_{bcs}$ /bar',fontsize = 15)
# plt.grid()
# plt.show()
#
# plt.figure(dpi=250)
# plt.plot(x_ss[1],x_ss[0], 'b.')
# plt.plot([110, 225], [0, 0], 'k--', linewidth=3)
# plt.xlabel(r"$q_{tr}$ /(m$^3 \cdot$ h$^{-1}$)", fontsize=15)
# plt.ylabel('$P_{man}$ /bar', fontsize=15)
# plt.grid()
# plt.show()