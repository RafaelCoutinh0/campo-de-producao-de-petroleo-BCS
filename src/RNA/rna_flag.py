
#%% importações do código
import torch
import numpy as np
import torch.nn as nn
from matplotlib.style.core import library
from torch.utils.data import Dataset, DataLoader
import torch.distributions.uniform as urand
import pickle
import optuna
from colorama import Fore, Style
from src.rna_global import y_labels

def load_data_from_pkl(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

class MyLibraryDataset(Dataset):
    def __init__(self, library_data, feature_vars, label_vars, transform=None):
        self.library_data = library_data
        self.feature_vars = feature_vars  # Variáveis de entrada
        self.label_vars = label_vars  # Variáveis de saída (inclui a flag)
        self.flag_index = label_vars.index('flag')  # Índice da flag
        self.num_simulations = len(library_data[feature_vars[0]])
        self.transform = transform

        # Calcular limites de normalização
        self.feature_min = {var: min(library_data[var]) for var in feature_vars}
        self.feature_max = {var: max(library_data[var]) for var in feature_vars}

    def normalize(self, value, min_val, max_val):
        return 2 * (value - min_val) / (max_val - min_val) - 1  # Normalizar para o intervalo [-1, 1]

    def denormalize(self, value, min_val, max_val):
        return (value + 1) * (max_val - min_val) / 2 + min_val # Reverter do intervalo [-1, 1] para o intervalo original


    def __getitem__(self, idx):
        features = [
            self.normalize(self.library_data[var][idx], self.feature_min[var], self.feature_max[var])
            for var in self.feature_vars # Normalizar features
        ]

        # Aplicar transformações, se houver
        if self.transform:
            features = self.transform(features)

        return torch.tensor(features, dtype=torch.float32), torch.tensor(label_vars, dtype=torch.float32)

    def __len__(self):
        return self.num_simulations


class FlagNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        layers = []  # Lista temporária para criar o Sequential
        hidden_dim = 150  # Hiperparâmetro encontrado
        num_layers = 2  # Hiperparâmetro encontrado
        dropout = 8.55353740037329e-05  # Hiperparâmetro encontrado

        # Criar camadas ocultas
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim if not layers else hidden_dim, hidden_dim))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Tanh())  # Função de ativação

        # Camada de saída
        layers.append(nn.Linear(hidden_dim, output_dim))

        # Definir como um módulo Sequential
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        output = self.model(x)
        output = torch.sigmoid(output)
        return output

def train( model, dataloader, optimizer, lossfunc):

    model.train()
    cumloss = 0.0
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)

        y_labels = y  # Saídas contínuas
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

if __name__ == "__main__":
    file_path = 'rna_training.pkl'
    library_data = load_data_from_pkl(file_path)

    # Variáveis selecionadas
    feature_vars = [
        'p_topo', 'valve1', 'valve2', 'valve3', 'valve4',
        'bcs1_freq', 'bcs2_freq', 'bcs3_freq', 'bcs4_freq',
        'booster_freq',
    ]
    label_vars = ['flag']

    for var in feature_vars + label_vars:
        if var not in library_data:
            raise ValueError(f"A variável {var} não está presente no dataset!")

    # Criar o dataset e DataLoader
    input_dim = len(feature_vars)  # Número de variáveis de entrada
    output_dim = 1  # Número de variáveis de saída
    dataset = MyLibraryDataset(library_data, feature_vars, label_vars)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    # Inicialize o modelo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Rodando na {device}")

    # Criar o modelo
    model = FlagNetwork(input_dim=input_dim, output_dim=output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.243656143480994e-05)  # Taxa de aprendizado encontrada
    lossfunc = nn.BCELoss()
    saidas = ['flag']

    epochs = 301  # Número de épocas
    for epoch in range(epochs):
        model = FlagNetwork(input_dim=len(feature_vars), output_dim=len(label_vars)).to(device)
        train_loss, y_labels, pred_labels = train(model, dataloader, optimizer, lossfunc)
        y_labels = y_labels.tolist()
        pred_labels = pred_labels.tolist()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss}")
            print("Comparação entre modelo e RNA:")

            for i, name in enumerate(saidas):
                # Calcular a diferença percentual
                percent = abs(abs(abs(y_labels[-1][i]) - abs(pred_labels[-1][i])) / abs(y_labels[-1][i])) * 100
                # Imprimir a diferença em vermelho se houver uma discrepância significativa
                if percent > 5:  # Exemplo: se a diferença for maior que 5%, mostra em vermelho
                    print(f"{name}: modelo = {y_labels[-1][i]}, RNA = {pred_labels[-1][i]}, {Fore.RED}{percent:.2f}%{Style.RESET_ALL}")
                elif percent < 5 and percent > 1:
                    print(f"{name}: modelo = {y_labels[-1][i]}, RNA = {pred_labels[-1][i]}, {Fore.YELLOW}{percent:.2f}%{Style.RESET_ALL}")
                else:
                    print(f"{name}: modelo = {y_labels[-1][i]}, RNA = {pred_labels[-1][i]}, {Fore.GREEN}{percent:.2f}%{Style.RESET_ALL}")
