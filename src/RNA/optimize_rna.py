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


# Carregar modelos treinados
model_global = GlobalNetwork(10,22).to(device)
model_flag = FlagNetwork(10,1).to(device)
state_dict = torch.load('rna_global_model_sbai.pth')
model_global.load_state_dict(state_dict)
state_dict = torch.load('rna_flag_model.pth')
model_flag.load_state_dict(state_dict)

# Coloca os modelos em modo de avaliação
model_global.eval()
model_flag.eval()

# Dados de min/max do banco de dados
u_min = torch.tensor([35. , 0.8e5 , 35., 0, 35., 0, 35., 0, 35. , 0], device=device)
u_max = torch.tensor([65. , 1.2e5, 65. , 1, 65. , 1 , 65. , 1, 65. , 1], device=device)

mins_out = torch.tensor([0.00010670462926302951, 0.0005843963639077903, 3.229948937698398e-05, 0.0005795289243178908, 12.441836086052101,
                         -169.49041800700638, 55.55775607071595, 54.919168202220035, 54.70439009544229, 55.95839442254224,
                         -65.70860465563962, -70.08830768577488, -65.32692719230973, -57.00729078774698, 8.75160356037195,
                         7.0577982101514145, 6.484831467224365, 9.806801119004563, 20.580058322525446, 18.215705257048594,
                         17.52693465550493, 22.104098174763582], device=device)

maxs_out = torch.tensor([106.42121034906826, 108.02242855969821, 108.56097091664455, 105.41663532271836, 258.47947395014273,
                         105.04379583433938, 97.99995744472471, 97.99976693468392, 97.99998711851893, 100.37548991915249,
                         234.28163200550358, 234.05991358558907, 234.18936179533665, 234.1016387730855, 87.45883778060255,
                         87.45864724594833, 87.45886743862086, 87.9812947000583, 223.53709772291216, 223.29579330109422,
                         223.3463702812387, 223.38355161614396], device=device)

# Normalizador ajustado com os valores do banco de dados
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


normalizer = Normalizer(u_min, u_max, mins_out, maxs_out)


# Cálculo corrigido de custo energético
def calculate_energy_cost(x_pred, u_tensor):
    x_pred_real = normalizer.denormalize_outputs(x_pred)
    booster_power = (9653.04 * (x_pred_real[4] / 3600) * (1.0963e3 * (u_tensor[0] / 50) ** 2) * 0.001)

    # Cálculo do consumo das ESPs (ajustar índices)
    esp_power = (x_pred_real[0] / 3600) * (x_pred_real[18] * 1e2) \
                + (x_pred_real[1] / 3600) * (x_pred_real[19] * 1e2) \
                + (x_pred_real[2] / 3600) * (x_pred_real[20] * 1e2) \
                + (x_pred_real[3] / 3600) * (x_pred_real[21] * 1e2)
    return booster_power + esp_power.sum()


# Função Objetivo ajustada
def objective(u_tensor):
    x_pred = model_global(u_tensor)

    # Debug: Verificar dimensões
    print(f"Tamanho de x_pred: {x_pred.shape}")  # Deve ser (22,)

    x_pred_real = normalizer.denormalize_outputs(x_pred)

    # Debug: Garantir que as dimensões estão corretas
    print(f"Tamanho de x_pred_real: {x_pred_real.shape}")  # Deve ser (22,)

    return -(3000 * x_pred_real[4]) + calculate_energy_cost(x_pred, u_tensor) * 0.91


# Execução da Otimização
def run_optimization():
    u_init = normalizer.normalize_inputs(torch.tensor([65., 0.8e5, 65., 1, 65., 1, 65., 1, 65., 1], device=device))
    u_tensor = torch.nn.Parameter(u_init.clone(), requires_grad=True)
    optimizer = optim.Adam([u_tensor], lr=0.01)

    for epoch in range(3000):
        optimizer.zero_grad()
        loss = objective(u_tensor)

        # Debug: Monitorar loss
        print(f"Epoch {epoch}: Loss = {loss.item()}")

        if torch.isnan(loss):
            print(f"Erro: loss NaN na iteração {epoch}")
            break

        loss.backward()
        optimizer.step()
        with torch.no_grad():
            u_tensor.data = torch.clamp(u_tensor, -1, 1)

    u_opt_real = normalizer.denormalize_inputs(u_tensor.detach())

    return u_opt_real.cpu().numpy()


if __name__ == "__main__":
    u_opt = run_optimization()
    print("\n" + "=" * 50)
    print("Resultado da Otimização")
    print("=" * 50)
    print(f"f_BP: {u_opt[0]:.2f} Hz")
    print(f"p_topside: {u_opt[1] / 1e5:.2f} bar")
    print(f"f_ESP1: {u_opt[2]:.2f} Hz | alpha1: {u_opt[3]:.2f}")
    print(f"f_ESP2: {u_opt[4]:.2f} Hz | alpha2: {u_opt[5]:.2f}")
    print(f"f_ESP3: {u_opt[6]:.2f} Hz | alpha3: {u_opt[7]:.2f}")
    print(f"f_ESP4: {u_opt[8]:.2f} Hz | alpha4: {u_opt[9]:.2f}")
    print("=" * 50)

    # Teste de previsão antes da otimização
    u_test = normalizer.normalize_inputs(torch.tensor([50., 1e5, 50., 0.5, 50., 0.5, 50., 0.5, 50., 0.5], device=device))
    x_pred_test = model_global(u_test).cpu().detach().numpy()
    x_pred_test_real = normalizer.denormalize_outputs(torch.tensor(x_pred_test))

    print("\nSaída da RNA antes da otimização (desnormalizada):")
    print(f"q_tr (RNA): {x_pred_test_real[4].item():.2f} m³/h")

    # Verificando a normalização
    normalized_input = normalizer.normalize_inputs(
        torch.tensor([50., 1e5, 50., 0.5, 50., 0.5, 50., 0.5, 50., 0.5], device=device))
    print(f"Entrada Normalizada: {normalized_input}")

    # Verificando a desnormalização
    x_pred_real = normalizer.denormalize_inputs(normalized_input)
    print(f"Saída Desnormalizada: {x_pred_real}")


def test_output(u_test_real, valor_esperado, indice_saida):
    """
    Testa se a saída predita pela RNA bate com um valor esperado.

    Args:
        u_test_real (list ou tensor): Entrada conhecida (valores reais, não normalizados).
        valor_esperado (float): Valor esperado da saída.
        indice_saida (int): Índice da variável de saída a ser comparada (de acordo com a ordem das saídas do banco de dados).
    """
    # Converter entrada real para tensor e normalizar
    u_test_real = torch.tensor(u_test_real, device=device)
    u_test_norm = normalizer.normalize_inputs(u_test_real)

    # Fazer a predição com a RNA
    with torch.no_grad():
        x_pred_test = model_global(u_test_norm).cpu().numpy()

    # Desnormalizar a saída predita
    x_pred_test_real = normalizer.denormalize_outputs(torch.tensor(x_pred_test))

    # Comparar saída predita com valor esperado
    valor_predito = x_pred_test_real[indice_saida].item()
    erro = abs(valor_predito - valor_esperado)

    print("\n" + "=" * 50)
    print(f"Teste de Validação da Saída {indice_saida}")
    print("=" * 50)
    print(f"Entrada utilizada: {u_test_real.cpu().numpy()}")
    print(f"Saída esperada: {valor_esperado:.5f}")
    print(f"Saída predita pela RNA: {valor_predito:.5f}")
    print(f"Erro absoluto: {erro:.5f}")
    print("=" * 50)
    print("Entrada Normalizada para a RNA:", u_test_norm)
    print("Saída Normalizada da RNA:", x_pred_test)
    print("Saída Desnormalizada da RNA:", x_pred_test_real)

    return erro


# Definir entrada conhecida e saída esperada
u_test_real = [65, 0.8e5, 65, 1, 65, 1,
        65, 1, 65, 1]  # Exemplo de entrada real
indice_saida = 4  # Índice da saída que queremos testar (exemplo: q_tr)
valor_esperado = 270.0  # Exemplo de valor esperado para q_tr

# Rodar o teste de validação
erro = test_output(u_test_real, valor_esperado, indice_saida)

# Verificar se o erro está dentro de uma margem aceitável
tolerancia = 5.0  # Defina uma margem aceitável para o erro
if erro < tolerancia:
    print("✅ O modelo previu corretamente dentro da margem de erro.")
else:
    print("❌ O modelo apresentou um erro acima da margem aceitável.")

