from casadi import MX, vertcat, nlpsol
import numpy as np
from scipy.optimize import fsolve
from initialization_oil_production_basic import *
# Importações dos arquivos fornecidos
from bcs_models import *
from manifold import Manifold

# --- Passo 1: Definir a função objetivo ---

# Variáveis de controle: frequências das bombas e aberturas das válvulas
f_ESP_1 = MX.sym('f_ESP_1')
alpha_1 = MX.sym('alpha_1')
f_ESP_2 = MX.sym('f_ESP_2')
alpha_2 = MX.sym('alpha_2')
f_ESP_3 = MX.sym('f_ESP_3')
alpha_3 = MX.sym('alpha_3')
f_ESP_4 = MX.sym('f_ESP_4')
alpha_4 = MX.sym('alpha_4')

# Booster pump
f_BP = MX.sym('f_BP')
p_topside = MX.sym('p_topside')

# Variáveis de decisão (controle)
u = vertcat(f_BP, p_topside, f_ESP_1, alpha_1, f_ESP_2, alpha_2, f_ESP_3, alpha_3, f_ESP_4, alpha_4)

# Variáveis de estado (exemplo inicial)
x = MX.sym('x', 14)  # 14 estados no modelo
z = MX.sym('z', 8)   # 8 variáveis algébricas no modelo


pipe_mani = Pipe(0.0595 * 2, 500, 0, 8.3022e+6, 984, 4, 5.752218216772682e+06, 3.903249155428134e+07)

# Modelo do manifold e dos poços
mani = Manifold(pipe_mani, booster, 0, 0, [well1, well2, well3, well4])
mani_model = mani.model(0, x, z, u)

# Função objetivo: Maximizar a vazão de transporte (q_tr) e minimizar energia (exemplo)
q_tr = x[1]  # Índice da vazão no manifold
objective = -q_tr

# --- Passo 2: Definir as restrições ---
constraints = []
q_target1 = 50  # Vazão desejada (m³/h)


# Restrições físicas e operacionais

constraints.append(x[0] >= 50)  # Pressão mínima no manifold (P_man)
constraints.append(x[0] <= 150) # Pressão máxima no manifold (P_man)
constraints.append(q_tr <= 80)
constraints.append(q_tr >= 50)
constraints.append(f_ESP_1 >= 35)
constraints.append(f_ESP_1 <= 65)
constraints.append(alpha_1 >= 0.3)
constraints.append(alpha_1 <= 1)
constraints.append(f_ESP_2 >= 35)
constraints.append(f_ESP_2 <= 65)
constraints.append(alpha_2 >= 0.3)
constraints.append(alpha_2 <= 1)
constraints.append(f_ESP_3 >= 35)
constraints.append(f_ESP_3 <= 65)
constraints.append(alpha_3 >= 0.3)
constraints.append(alpha_3 <= 1)
constraints.append(f_ESP_4 >= 35)
constraints.append(f_ESP_4 <= 65)
constraints.append(alpha_4 >= 0.3)
constraints.append(alpha_4 <= 1)


# Conversão de restrições para CasADi
g = vertcat(*constraints)

# --- Passo 3: Configurar e resolver o problema de otimização ---

nlp = {'x': u, 'p': vertcat(x, z), 'f': objective, 'g': g}

# Resolver o problema de otimização
solver = nlpsol('solver', 'ipopt', nlp)


# Valores iniciais e limites
u0 = [56., 1e5, 50., 0.5, 50., 0.5, 50., 0.5, 50., 0.5]  # Chute inicial
lbg = [0] * len(constraints)  # Limite inferior das restrições
ubg = [np.inf] * len(constraints)  # Limite superior das restrições
c= concatenate((x0,z0), axis = 0)
# Resolver o problema de otimização
sol = solver(x0=u0, p=c, lbg=lbg, ubg=ubg)
optimal_u = sol['x']

# --- Passo 4: Simulação com os valores otimizados ---

# Atualizar os controles com os valores otimizados
u_opt = np.array(optimal_u).flatten()

# Resolver o modelo em regime estacionário com os controles otimizados
mani_solver = lambda y: np.array([float(i) for i in mani.model(0, y[0:-8], y[-8:], u_opt)])
y0 = np.concatenate((x0, z0))  # Combinar estados e variáveis algébricas iniciais
y_ss = fsolve(mani_solver, y0)  # Resolver para o regime estacionário

# Separar os resultados estacionários
x_ss = y_ss[:14]  # Estados
z_ss = y_ss[14:]  # Variáveis algébricas

unidades_x = ["Hz", "Pa", "Hz", "", "Hz", "", "Hz", "", "Hz", ""]  # Unidades dos estados
unidades_z = ["Pa", "Pa", "Pa","Pa","Pa", "Pa", "Pa", "Pa"]  # Unidades das variáveis algébricas
unidades_u = ["Hz", "Pa", "Hz", "", "Hz", "", "Hz", "", "Hz", ""]  # Unidades dos controles otimizados
# Exibir os resultados
print("Valores otimizados dos controles:", " ".join(f"{u:.2f} {unit}" for u, unit in zip(u_opt, unidades_u)))
print("Estados estacionários após otimização:", " ".join(f"{x:.2f} {unit}" for x, unit in zip(x_ss, unidades_x)))
print("Variáveis algébricas estacionárias:", " ".join(f"{z:.2f} {unit}" for z, unit in zip(z_ss, unidades_z)))
