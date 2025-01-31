import torch
import numpy as np
from casadi import *
from rna_global import RasmusNetwork, MyLibraryDataset, load_data_from_pkl

# Carregar o modelo treinado
file_path = 'rna_training.pkl'  # Dataset usado para treinar a RNA
library_data = load_data_from_pkl(file_path)

# Configuração da rede neural
device = "cuda" if torch.cuda.is_available() else "cpu"
feature_vars = [
    'p_topo', 'valve1', 'valve2', 'valve3', 'valve4',
    'bcs1_freq', 'bcs2_freq', 'bcs3_freq', 'bcs4_freq',
    'booster_freq',
]
label_vars = [
    'q_main1', 'q_main2', 'q_main3', 'q_main4', 'q_tr',
    'P_man', 'P_fbhp1', 'P_fbhp2',
    'P_fbhp3', 'P_fbhp4', 'P_choke1', 'P_choke2',
    'P_choke3', 'P_choke4', 'P_intake1', 'P_intake2',
    'P_intake3', 'P_intake4', 'dP_bcs1', 'dP_bcs2',
    'dP_bcs3', 'dP_bcs4'
]

# Criar dataset e carregar o modelo
input_dim = len(feature_vars)
output_dim = len(label_vars)
dataset = MyLibraryDataset(library_data, feature_vars, label_vars)
model = RasmusNetwork(input_dim=input_dim, output_dim=output_dim).to(device)
model.load_state_dict(torch.load('rna_model.pth', map_location=device))
model.eval()


# Função para usar a RNA como substituto do modelo manifold
def rna_predict(u):
    """
    Realiza previsão usando a RNA treinada.
    :param u: Entradas manipuláveis (array numpy)
    :return: Variáveis de estado (x) e algébricas (z)
    """
    # Preparar entradas
    input_data = {
        'p_topo': [u[1]], 'valve1': [u[3]], 'valve2': [u[5]], 'valve3': [u[7]], 'valve4': [u[9]],
        'bcs1_freq': [u[2]], 'bcs2_freq': [u[4]], 'bcs3_freq': [u[6]], 'bcs4_freq': [u[8]],
        'booster_freq': [u[0]],
    }

    inputs = torch.tensor([
        dataset.normalize(input_data[var][0], dataset.feature_min[var], dataset.feature_max[var])
        for var in feature_vars
    ], dtype=torch.float32).to(device)

    # Inferência
    with torch.no_grad():
        outputs = model(inputs)

    # Desnormalizar saídas
    outputs_denorm = [
        dataset.denormalize(outputs[i].item(), dataset.label_min[var], dataset.label_max[var])
        for i, var in enumerate(label_vars)
    ]

    x = outputs_denorm[:14]  # Primeiros 14 valores correspondem aos estados
    z = outputs_denorm[14:]  # Restante são as variáveis algébricas
    return x, z


# Configuração do problema de otimização
u = MX.sym('u', 10)  # Entradas manipuláveis
x_pred, z_pred = rna_predict(u)

# Função objetivo
objective = -(70000 * x_pred[1]) + ((9653.04 * (x_pred[1] / 3600) * (1.0963e3 * (u[0] / 50) ** 2) * 0.001) +
                                    ((x_pred[4] / 3600) * z_pred[1] * 1e2) +
                                    ((x_pred[7] / 3600) * z_pred[3] * 1e2) +
                                    ((x_pred[10] / 3600) * z_pred[5] * 1e2) +
                                    ((x_pred[13] / 3600) * z_pred[7] * 1e2)) * 0.91

# Restrições (usando valores de x_pred e z_pred)
g_constraints = vertcat(*[0] * len(x_pred))  # Exemplo: ajuste conforme necessário

# Configuração do solver de otimização
nlp = {'x': u, 'f': objective, 'g': g_constraints}
solver = nlpsol('solver', 'ipopt', nlp)

# Definir limites das variáveis
lbx = [35, 0.8e5, 35, 0, 35, 0, 35, 0, 35, 0]
ubx = [65, 1.2e5, 65, 1, 65, 1, 65, 1, 65, 1]

# Valores iniciais para otimização
u0 = np.array([65., 0.8e5, 65., 1, 65., 1, 65., 1, 65., 1])

# Resolver o problema de otimização
sol = solver(x0=u0, lbx=lbx, ubx=ubx, lbg=0, ubg=0)
u_opt = sol['x']

# Resultados otimizados
x_opt, z_opt = rna_predict(u_opt)

print("Resultados otimizados:")
print("Variáveis de estado (x):", x_opt)
print("Variáveis algébricas (z):", z_opt)
print("Entradas manipuladas (u):", u_opt)