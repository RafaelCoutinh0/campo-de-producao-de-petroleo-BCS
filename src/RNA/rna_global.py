# -*- coding: utf-8 -*-
"""RNA_global

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1dfFRua5xT_5PQqL4WxXYPPnwD7XFoOGX
"""
#%% importações do código
import torch
import numpy as np
import torch.nn as nn
from matplotlib.style.core import library
from torch.utils.data import Dataset, DataLoader, random_split
import torch.distributions.uniform as urand
import pickle
import matplotlib.pyplot as plt
# from colorama import Fore, Style

def load_data_from_pkl(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def split_dataset(dataset, train_ratio=0.6):
    total_len = len(dataset)
    assert total_len == 100_000, f"Dataset deve ter 100k amostras (atual: {total_len})"

    train_len = int(total_len * train_ratio)
    test_len = total_len - train_len
    return random_split(dataset, [train_len, test_len])


class MyLibraryDataset(Dataset):
    def __init__(self, library_data, feature_vars, label_vars, transform=None):
        self.library_data = library_data
        self.feature_vars = feature_vars  # Variáveis de entrada
        self.label_vars = label_vars  # Variáveis de saída (inclui a flag)
        self.num_simulations = len(library_data[feature_vars[0]])
        self.transform = transform

        # Calcular limites de normalização
        self.feature_min = {var: min(library_data[var]) for var in feature_vars}
        self.feature_max = {var: max(library_data[var]) for var in feature_vars}
        self.label_min = {var: min(library_data[var]) for var in label_vars if var != 'flag'}
        self.label_max = {var: max(library_data[var]) for var in label_vars if var != 'flag'}

    def normalize(self, value, min_val, max_val):
        return 2 * (value - min_val) / (max_val - min_val) - 1  # Normalizar para o intervalo [-1, 1]

    def denormalize(self, value, min_val, max_val):
        return (value + 1) * (max_val - min_val) / 2 + min_val # Reverter do intervalo [-1, 1] para o intervalo original

    def __getitem__(self, idx):
        if idx >= 100_000:  # Garantir acesso válido
            raise IndexError(f"Índice {idx} inválido para dataset com 100k amostras!")

        features = [
            self.normalize(self.library_data[var][idx], self.feature_min[var], self.feature_max[var])
            for var in self.feature_vars
        ]


        labels = [
            self.library_data[var][idx] if var == 'flag' else
            self.normalize(self.library_data[var][idx], self.label_min[var], self.label_max[var])
            for var in self.label_vars # Normalizar labels, exceto a flag
        ]

        # Aplicar transformações, se houver
        if self.transform:
            features = self.transform(features)

        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return self.num_simulations


# Definir a rede neural
class RasmusNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        layers = []  # Lista temporária para criar o Sequential
        hidden_dim = 150  # Hiperparâmetro encontrado
        num_layers = 3    # Hiperparâmetro encontrado


        # Criar camadas ocultas
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim if not layers else hidden_dim, hidden_dim))
            layers.append(nn.Tanh())  # Função de ativação

        # Camada de saída
        layers.append(nn.Linear(hidden_dim, output_dim))

        # Definir como um módulo Sequential
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        output = self.model(x)
        return output

    # Treinamento
def train(model, dataloader, optimizer, lossfunc):
    model.train()
    cumloss = 0.0

    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)

        y_labels = y # Saídas contínuas

        pred = model(X)
        loss = lossfunc(pred, y)
        pred_labels = pred

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Acumular a perda
        cumloss += loss.item()

        y_labels_denorm = torch.stack([
            dataset.denormalize(y_labels[:, i], dataset.label_min[var], dataset.label_max[var])
            for i, var in enumerate(dataset.label_vars)
        ], dim=1)
        pred_labels_denorm = torch.stack([
            dataset.denormalize(pred_labels[:, i], dataset.label_min[var], dataset.label_max[var])
            for i, var in enumerate(dataset.label_vars)
        ], dim=1)
        y_labels_list = []
        pred_labels_list = []
        y_labels_list.append(y_labels_denorm)
        pred_labels_list.append(pred_labels_denorm)

    y_labels_all = torch.cat(y_labels_list, dim=0)
    pred_labels_all = torch.cat(pred_labels_list, dim=0)
    avg_loss = cumloss / len(dataloader)
    return avg_loss, y_labels_all, pred_labels_all

def test(model, dataloader, lossfunc):
    model.eval()  # Coloca o modelo em modo de avaliação
    cumloss = 0.0
    y_labels_list = []
    pred_labels_list = []

    with torch.no_grad():  # Desativa o cálculo do gradiente para economizar memória
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            y_labels = y  # Saídas contínuas

            pred = model(X)
            loss = lossfunc(pred, y)
            pred_labels = pred

            # Acumular a perda
            cumloss += loss.item()

            y_labels_denorm = torch.stack([
                dataset.denormalize(y_labels[:, i], dataset.label_min[var], dataset.label_max[var])
                for i, var in enumerate(dataset.label_vars)
            ], dim=1)
            pred_labels_denorm = torch.stack([
                dataset.denormalize(pred_labels[:, i], dataset.label_min[var], dataset.label_max[var])
                for i, var in enumerate(dataset.label_vars)
            ], dim=1)

            y_labels_list.append(y_labels_denorm)
            pred_labels_list.append(pred_labels_denorm)

    y_labels_all = torch.cat(y_labels_list, dim=0)
    pred_labels_all = torch.cat(pred_labels_list, dim=0)
    avg_loss = cumloss / len(dataloader)
    return avg_loss, y_labels_all, pred_labels_all


if __name__ == "__main__":
    file_path = 'rna_training_sbai_fbp.pkl'
    library_data = load_data_from_pkl(file_path)

    # Variáveis selecionadas
    feature_vars = [
        'p_topo', 'valve1', 'valve2', 'valve3', 'valve4',
        'bcs1_freq', 'bcs2_freq', 'bcs3_freq', 'bcs4_freq',
        'booster_freq',
    ]
    label_vars = ['q_main1', 'q_main2', 'q_main3', 'q_main4', 'q_tr',
        'P_man', 'P_fbhp1', 'P_fbhp2',
        'P_fbhp3', 'P_fbhp4', 'P_choke1', 'P_choke2',
        'P_choke3', 'P_choke4', 'P_intake1', 'P_intake2',
        'P_intake3', 'P_intake4', 'dP_bcs1', 'dP_bcs2',
        'dP_bcs3', 'dP_bcs4']

    for var in feature_vars + label_vars:
        if var not in library_data:
            raise ValueError(f"A variável {var} não está presente no dataset!")

    input_dim = len(feature_vars)  # Dimensão da entrada
    output_dim = len(label_vars)  # Dimensão da saída
    dataset = MyLibraryDataset(library_data, feature_vars, label_vars)
    train_dataset, test_dataset = split_dataset(dataset, train_ratio=0.5)

    print(f"Tamanho total do dataset: {len(dataset)}")
    print(f"Tamanho do treino: {len(train_dataset)}")
    print(f"Tamanho do teste: {len(test_dataset)}")
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # Sem shuffle!

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Rodando na {device}")

    # Criar o modelo
    model = RasmusNetwork(input_dim=len(feature_vars), output_dim=len(label_vars)).to(device)
    # state_dict = torch.load('rna_global_model_sbai.pth')
    # Carregue os pesos no modelo inicializado
    # model.load_state_dict(state_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00023388306410551066)  # Taxa de aprendizado encontrada
    lossfunc = nn.MSELoss()


saidas =  ['q_main1', 'q_main2', 'q_main3', 'q_main4', 'q_tr',
        'P_man', 'P_fbhp1', 'P_fbhp2',
        'P_fbhp3', 'P_fbhp4', 'P_choke1', 'P_choke2',
        'P_choke3', 'P_choke4', 'P_intake1', 'P_intake2',
        'P_intake3', 'P_intake4', 'dP_bcs1', 'dP_bcs2',
        'dP_bcs3', 'dP_bcs4']

#%%
from colorama import Fore, Style
#
# epochs = 301  # Número de épocas
# for epoch in range(epochs):
#     train_loss, y_labels, pred_labels = train(model, dataloader, optimizer, lossfunc)
#     y_labels = y_labels.tolist()
#     pred_labels = pred_labels.tolist()
#
#     if epoch % 10 == 0:
#         print(f"Epoch {epoch}: Train Loss = {train_loss}")
#         print("Comparação entre modelo e RNA:")
#
#         for i, name in enumerate(saidas):
#             # Calcular a diferença percentual
#             percent = abs(abs(abs(y_labels[-1][i]) - abs(pred_labels[-1][i])) / abs(y_labels[-1][i])) * 100
#             # Imprimir a diferença em vermelho se houver uma discrepância significativa
#             if percent > 5:  # Exemplo: se a diferença for maior que 5%, mostra em vermelho
#                 print(f"{name}: modelo = {y_labels[-1][i]}, RNA = {pred_labels[-1][i]}, {Fore.RED}{percent:.2f}%{Style.RESET_ALL}")
#             elif percent < 5 and percent > 1:
#                 print(f"{name}: modelo = {y_labels[-1][i]}, RNA = {pred_labels[-1][i]}, {Fore.YELLOW}{percent:.2f}%{Style.RESET_ALL}")
#             else:
#                 print(f"{name}: modelo = {y_labels[-1][i]}, RNA = {pred_labels[-1][i]}, {Fore.GREEN}{percent:.2f}%{Style.RESET_ALL}")
def SaveNetwork():
    model_path = "rna_global_model_sbai_ultimate.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Modelo completo salvo em {model_path}")



saidas = ['q_main1', 'q_main2', 'q_main3', 'q_main4', 'q_tr',
          'P_man', 'P_fbhp1', 'P_fbhp2',
          'P_fbhp3', 'P_fbhp4', 'P_choke1', 'P_choke2',
          'P_choke3', 'P_choke4', 'P_intake1', 'P_intake2',
          'P_intake3', 'P_intake4', 'dP_bcs1', 'dP_bcs2',
          'dP_bcs3', 'dP_bcs4']
#
# train_list = []
# test_list = []
# epochs = 10000  # Número de épocas
# for epoch in range(epochs):
#     train_loss, y_labels, pred_labels = train(model, train_dataloader, optimizer, lossfunc)
#     train_list.append(train_loss)
#     test_loss, y_labels, pred_labels = test(model, test_dataloader, lossfunc)
#     test_list.append(test_loss)
#     y_labels = y_labels.tolist()
#     pred_labels = pred_labels.tolist()
#     if test_loss < 1e-5 and train_loss < 1e-5:
#         SaveNetwork()
#         break
#
#     if epoch % 10 == 0:
#         print("=" * 58)
#         print(f"Epoch {epoch}: Train Loss = {train_loss}")
#         print("=" * 58)
#         print(f"{'Saída':<15}{'Modelo':<15}{'RNA':<15}{'Diferença (%)':<15}")
#         print("-" * 58)
#
#         percent_total = 0
#         c = 0
#
#         for i, name in enumerate(saidas):
#             # Calcular a diferença percentual
#             percent = abs(abs(abs(y_labels[-1][i]) - abs(pred_labels[-1][i])) / abs(y_labels[-1][i])) * 100
#             percent_total += percent
#             c += 1
#
#             # Cor para a diferença percentual
#             if percent > 5:  # Diferença alta
#                 color = Fore.RED
#             elif percent > 1:  # Diferença média
#                 color = Fore.YELLOW
#             else:  # Diferença baixa
#                 color = Fore.GREEN
#
#             # Print formatado
#             print(
#                 f"{name:<15}{y_labels[-1][i]:<15.2f}{pred_labels[-1][i]:<15.2f}{color}    {percent:<15.2f}{Style.RESET_ALL}")
#
#         print("-" * 58)
#         print(f"{'PERCENTUAL MÉDIO:':<15}{percent_total / c:.2f}%")
#         print("=" * 58)
#         print(test_loss)
#
# SaveNetwork()
#
# plt.figure(dpi=250)
# plt.plot(train_list, 'b')
# plt.plot(test_list, 'r')
# plt.xlabel("'Época", fontsize=20)
# plt.ylabel('Loss', fontsize=20)
# plt.grid()
# plt.show()
