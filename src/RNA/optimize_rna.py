import torch
import casadi as ca
import numpy as np

# Configuração do dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"

# Definição da rede neural
class GlobalNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(10, 150),
            torch.nn.Tanh(),
            torch.nn.Linear(150, 150),
            torch.nn.Tanh(),
            torch.nn.Linear(150, 150),
            torch.nn.Tanh(),
            torch.nn.Linear(150, 22)
        )

    def forward(self, x):
        return self.model(x)

# Carregar modelo global
model_global = GlobalNetwork().to(device)
state_dict = torch.load('rna_global_model_50k.pth', map_location=device)
model_global.load_state_dict(state_dict)
model_global.eval()

# Normalização da entrada
class Normalizer:
    def __init__(self):
        self.mins = np.array([35, 0.8e5, 35, 0, 35, 0, 35, 0, 35, 0])
        self.maxs = np.array([65, 1.2e5, 65, 1, 65, 1, 65, 1, 65, 1])

    def normalize(self, x):
        return 2 * (x - self.mins) / (self.maxs - self.mins) - 1

    def denormalize(self, x_norm):
        return (x_norm + 1) * (self.maxs - self.mins) / 2 + self.mins

normalizer = Normalizer()

# Normalização da saída
class OutputNormalizer:
    def __init__(self):
        self.x_mins = np.array([50, 80, 30, 30, 20, 30, 20, 30, 20, 30, 20, 30, 20, 30])
        self.x_maxs = np.array([150, 250, 120, 120, 90, 120, 90, 120, 90, 120, 90, 120, 90, 120])
        self.z_mins = np.array([20, 50, 20, 50, 20, 50, 20, 50])
        self.z_maxs = np.array([300, 250, 300, 250, 300, 250, 300, 250])

    def denormalize_xz(self, x_pred_norm):
        x_denorm = x_pred_norm[:14] * (self.x_maxs - self.x_mins) / 2 + (self.x_maxs + self.x_mins) / 2
        z_denorm = x_pred_norm[14:22] * (self.z_maxs - self.z_mins) / 2 + (self.z_maxs + self.z_mins) / 2
        return np.concatenate([x_denorm, z_denorm])

output_normalizer = OutputNormalizer()

# Criar função CasADi para chamar a rede neural
def create_nn_function():
    u_sym = ca.MX.sym('u', 10)  # Variáveis simbólicas de entrada

    # Função CasADi que executa a RNA apenas para valores numéricos
    def nn_forward(u_numeric):
        u_np = np.array(u_numeric).flatten()  # Certifique-se de que o objeto seja convertido para numpy

        u_norm = normalizer.normalize(u_np)

        # Convertendo o vetor numpy para tensor PyTorch, com formato correto
        u_tensor = torch.tensor(u_norm, dtype=torch.float32, device=device).unsqueeze(0)  # Adicionando dimensão extra para batch size

        # Executar a RNA com PyTorch
        with torch.no_grad():
            x_pred_norm = model_global(u_tensor).cpu().numpy().flatten()

        x_pred = output_normalizer.denormalize_xz(x_pred_norm)
        return x_pred

    # Criar uma função CasADi para computar a RNA
    nn_func = ca.Function('nn_model', [u_sym], [ca.MX(nn_forward(u_sym))])

    return nn_func

# Criar a função da rede neural para CasADi
nn_model_ca = create_nn_function()

# Definir variáveis CasADi
u_ca = ca.MX.sym('u', 10)  # Variáveis de entrada
x_pred = nn_model_ca(u_ca)  # Chamar a RNA dentro do CasADi

# Função objetivo
q_tr = x_pred[1]
energy_cost = 9653.04 * (q_tr / 3600) * (1.0963e3 * (u_ca[0] / 50) ** 2) * 0.001
objective = -(3000 * q_tr) + energy_cost

# Restrições de vazão
restqmain = []
for i, j in zip([4, 7, 10, 13], [15, 17, 19, 21]):
    restqmain.append(ca.vertcat(x_pred[i] - ((x_pred[j] + 334.2554) / 19.0913), ((x_pred[j] + 193.8028) / 4.4338) - x_pred[i]))

g_constraints = ca.vertcat(*restqmain)
num_ineq = g_constraints.shape[0]

# Limites das restrições
lbg = [0.0] * num_ineq
ubg = [np.inf] * num_ineq

# Configuração do problema de otimização
nlp = {'x': u_ca, 'f': objective, 'g': g_constraints}
solver = ca.nlpsol('solver', 'ipopt', nlp)

# Valores iniciais
u0 = [50., 1e5, 50., 0.5, 50., 0.5, 50., 0.5, 50., 0.5]

# Limites das variáveis
lbx = [35, 0.8e5, 35, 0, 35, 0, 35, 0, 35, 0]
ubx = [65, 1.2e5, 65, 1, 65, 1, 65, 1, 65, 1]

# Resolver problema de otimização
sol = solver(x0=u0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
u_opt = sol['x']

# Exibir resultados
print("\n" + "=" * 50)
print("Resultado da Otimização")
print("=" * 50)
print(f"f_BP: {u_opt[0]:.2f} Hz")
print(f"p_topside: {u_opt[1] / 1e5:.2f} bar")
for i in range(4):
    print(f"f_ESP{i+1}: {u_opt[2 + 2*i]:.2f} Hz | alpha{i+1}: {u_opt[3 + 2*i]:.2f}")
print("=" * 50)

