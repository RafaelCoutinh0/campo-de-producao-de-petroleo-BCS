import torch
import numpy as np
from torch import optim
import pickle
from rna_global import RasmusNetwork as GlobalNetwork
from rna_flag import RasmusNetwork as FlagNetwork

# Configuração de dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_data_from_pkl(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

saidas =  ['q_main1', 'q_main2', 'q_main3', 'q_main4', 'q_tr', 'P_man', 'P_fbhp1', 'P_fbhp2', 'P_fbhp3', 'P_fbhp4', 'P_choke1', 'P_choke2', 'P_choke3', 'P_choke4', 'P_intake1', 'P_intake2', 'P_intake3', 'P_intake4', 'dP_bcs1', 'dP_bcs2', 'dP_bcs3', 'dP_bcs4']

# Carregar modelos treinados
# Verifica se está rodando na GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Rodando no dispositivo: {device}")

# ==========================
# 🧠 CARREGANDO O MODELO
# ==========================
from torch import nn
# Inicializa os modelos
model_flag = FlagNetwork(10, 1)  # Ajuste os tamanhos conforme necessário

# Carregar os pesos do modelo FlagNetwork treinado
state_dict_flag = torch.load('rna_flag_model.pth')
model_flag.load_state_dict(state_dict_flag)
model_flag.eval()

model_global = GlobalNetwork(10,22)

state_dict = torch.load('rna_global_model_sbai.pth')
# Carregue os pesos no modelo inicializado
model_global.load_state_dict(state_dict)

# Modo de avaliação
model_global.eval()


# ==========================
# 🔄 NORMALIZAÇÃO DOS DADOS
# ==========================
class Normalizer:
    def __init__(self, u_min, u_max, mins_out, maxs_out):
        self.u_min = u_min
        self.u_max = u_max
        self.mins_out = mins_out
        self.maxs_out = maxs_out

    def normalize_inputs(self, x):
        return 2 * (x - self.u_min) / (self.u_max - self.u_min) - 1

    def denormalize_inputs(self, x_norm):
        return (x_norm + 1) * (self.u_max - self.u_min) / 2 + self.u_min

    def normalize_outputs(self, x):
        return 2 * (x - self.mins_out) / (self.maxs_out - self.mins_out) - 1

    def denormalize_outputs(self, x_norm):
        return (x_norm + 1) * (self.maxs_out - self.mins_out) / 2 + self.mins_out


# Novos limites de normalização para as entradas
u_min = torch.tensor([0.8e5, 0, 0, 0, 0, 35., 35., 35., 35., 35.], device=device)
u_max = torch.tensor([1.2e5, 1, 1, 1, 1, 65., 65., 65., 65., 65.], device=device)

# Limites de saída (não alterados)
mins_out = torch.tensor([0.00010670462926302951, 0.0005843963639077903, 3.229948937698398e-05, 0.0005795289243178908, 12.441836086052101,
                         -169.49041800700638, 55.55775607071595, 54.919168202220035, 54.70439009544229, 55.95839442254224,
                         -65.70860465563962, -70.08830768577488, -65.32692719230973, -57.00729078774698, 8.75160356037195,
                         7.0577982101514145, 6.484831467224365, 9.806801119004563, 20.580058322525446, 18.215705257048594,
                         17.52693465550493, 22.104098174763582], device=device)

maxs_out = torch.tensor([106.42121034906826, 108.02242855969821, 108.56097091664455, 108.56097091664455, 273.6148,
                         105.04379583433938, 97.99995744472471, 97.99976693468392, 97.99998711851893, 100.37548991915249,
                         234.28163200550358, 234.05991358558907, 234.18936179533665, 234.1016387730855, 87.45883778060255,
                         87.45864724594833, 87.45886743862086, 87.9812947000583, 223.53709772291216, 223.29579330109422,
                         223.3463702812387, 223.38355161614396], device=device)


normalizer = Normalizer(u_min, u_max, mins_out, maxs_out)

# ==========================
# ⚡ FUNÇÃO DE CÁLCULO DE CUSTO
# ==========================
def calculate_energy_cost(x_pred_real, u_tensor):
    # Energia do booster (baseado no segundo código)
    booster_power = (9653.04 * (x_pred_real[4] / 3600) * (1.0963e3 * (u_tensor[9] / 50) ** 2) * 0.001)

    # Energia das ESPs
    esp_power = (
        (x_pred_real[0] / 3600) * (x_pred_real[18] * 1e2) +  # ESP1
        (x_pred_real[1] / 3600) * (x_pred_real[19] * 1e2) +  # ESP2
        (x_pred_real[2] / 3600) * (x_pred_real[20] * 1e2) +  # ESP3
        (x_pred_real[3] / 3600) * (x_pred_real[21] * 1e2)   # ESP4
    )

    return booster_power + esp_power

# ==========================
# 🎯 FUNÇÃO OBJETIVO
# ==========================
def objective(u_tensor):
    x_pred = model_global(u_tensor)
    x_pred_real = normalizer.denormalize_outputs(x_pred)

    return -(3000 * x_pred_real[4]) + (calculate_energy_cost(x_pred_real, u_tensor) * 0.9)

# ==========================
# 🔍 EXECUÇÃO DA OTIMIZAÇÃO
# ==========================
def run_optimization():
    print("Iniciando otimização no Colab...")

    # Define ponto inicial (já normalizado)
    u_init = normalizer.normalize_inputs(torch.tensor([1e5, 0.5, 0.5, 0.5, 0.5, 50, 50, 50, 50, 50], device=device))
    u_tensor = torch.nn.Parameter(u_init.clone(), requires_grad=True)

    # Definir os limites da otimização (convertidos para a escala normalizada)
    u_min_norm = normalizer.normalize_inputs(torch.tensor([0.8e5, 0, 0, 0, 0, 35, 35, 35, 35, 35], device=device))
    u_max_norm = normalizer.normalize_inputs(torch.tensor([1.2e5, 1, 1, 1, 1, 65, 65, 65, 65, 65], device=device))
    # Definir o otimizador
    optimizer = optim.Adam([u_tensor], lr=0.01)

    for epoch in range(201):
        optimizer.zero_grad()
        loss = objective(u_tensor)

        if torch.isnan(loss):
            print(f"Erro: Loss NaN na iteração {epoch}")
            break

        loss.backward()
        optimizer.step()

        # Clampa valores dentro dos limites permitidos
        with torch.no_grad():
            u_tensor.data = torch.clamp(u_tensor, u_min_norm, u_max_norm)
            # Verifica a saída da FlagNetwork
            flag_output = model_flag(u_tensor).item()
            # Ajusta se necessário (caso esteja usando a projeção)
            if flag_output < 0.5:
                u_tensor.data += (0.5 - flag_output)

        if epoch % 10 == 0:  # Opcional: imprime a cada 50 iterações
            print(f"Epoch {epoch}: Loss = {loss.item():.2f}, Flag = {flag_output:.4f}")

    print("\n===== Resultado da Otimização =====")
    u_opt_real = normalizer.denormalize_inputs(u_tensor.detach())  # Denormaliza o valor de u_final
    print(f"f_BP: {u_opt_real[9]:.2f} Hz")
    print(f"p_topside: {u_opt_real[0] / 1e5:.2f} bar")
    for i in range(1, 5, 1):
        print(f"f_ESP{i}: {u_opt_real[i+4]:.2f} Hz | alpha{i}: {u_opt_real[i]:.2f}")

    # Obtendo as saídas finalizadas da rede (x_pred_real)
    x_pred_final = model_global(u_tensor).detach()
    x_pred_real = normalizer.denormalize_outputs(x_pred_final)

    print("\n===== Saídas Finais da Otimização =====")
    for i, name in enumerate(saidas):
        print(f"{name}: {x_pred_real[i].item():.2f}")


run_optimization()