#%% importações do código
import torch
import numpy as np
import torch.nn as nn
from matplotlib.style.core import library
from torch.utils.data import Dataset, DataLoader
import torch.distributions.uniform as urand
import pickle
from torch.utils.data import random_split
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader, random_split
import pickle

def load_data_from_pkl(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def split_dataset(dataset, train_ratio=0.5):
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
        features = [
            self.normalize(self.library_data[var][idx], self.feature_min[var], self.feature_max[var])
            for var in self.feature_vars # Normalizar features
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
        num_layers = 2    # Hiperparâmetro encontrado


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
        output = torch.sigmoid(output)
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


    avg_loss = cumloss / len(dataloader)
    return avg_loss, y_labels, pred_labels

def test(model, dataloader, lossfunc):
    """
    Avalia o modelo nos dados de teste.

    Args:
        model (nn.Module): O modelo treinado.
        dataloader (DataLoader): DataLoader com os dados de teste.
        lossfunc (nn.Module): Função de perda.

    Returns:
        avg_loss (float): A perda média no conjunto de teste.
        y_labels (list): Os rótulos reais.
        pred_labels (list): As previsões do modelo.
    """
    model.eval()  # Coloca o modelo em modo de avaliação
    cumloss = 0.0
    y_labels = []
    pred_labels = []

    with torch.no_grad():  # Desativa o cálculo do gradiente
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            y_labels_batch = y.cpu().tolist()
            pred = model(X)
            loss = lossfunc(pred, y)

            pred_labels_batch = pred.cpu().tolist()

            # Acumula resultados e perdas
            y_labels.extend(y_labels_batch)
            pred_labels.extend(pred_labels_batch)
            cumloss += loss.item()

    avg_loss = cumloss / len(dataloader)
    return avg_loss, y_labels, pred_labels


if __name__ == "__main__":
    file_path = 'rna_training_sbai_fbp.pkl'
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
    input_dim = len(feature_vars)  # Dimensão da entrada
    output_dim = len(label_vars)  # Dimensão da saída
    # Criar o dataset completo
    dataset = MyLibraryDataset(library_data, feature_vars, label_vars)
    # Dividir o dataset em treinamento e teste
    train_dataset, test_dataset = split_dataset(dataset, train_ratio=0.5)
    # Criar os DataLoaders para treinamento e teste
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Rodando na {device}")

    # Criar o modelo
    model = RasmusNetwork(input_dim=len(feature_vars), output_dim=len(label_vars)).to(device)
    # state_dict = torch.load('rna_flag_model_fbp.pth')
    # Carregue os pesos no modelo inicializado
    # model.load_state_dict(state_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.243656143480994e-05)  # Taxa de aprendizado encontrada
    lossfunc = nn.BCELoss()

"""TREINAMENTO EM SI"""

saidas =  ['flag']

#%%
from colorama import Fore, Style
train_list = []
test_list = []
epochs = 5001  # Número de épocas
for epoch in range(epochs):
    train_loss, y_labels, pred_labels = train(model, train_dataloader, optimizer, lossfunc)
    train_list.append(train_loss)
    y_labels = y_labels.tolist()
    pred_labels = pred_labels.tolist()


    test_loss, y_labels, pred_labels = test(model, test_dataloader, lossfunc)
    test_list.append(test_loss)


    if epoch % 10 == 0:
        print(f"\nEpoch {epoch}: Train Loss = {train_loss}")
        for i, name in enumerate(saidas):
            if (pred_labels[-1][i] >= 0.5 and  y_labels[-1][i] >= 0.5) or (pred_labels[-1][i] < 0.5 and y_labels[-1][i] < 0.5):
                print(f"{Fore.GREEN}{name}: modelo = {y_labels[-1][i]}, RNA = {pred_labels[-1][i]}, {Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}{name}: modelo = {y_labels[-1][i]}, RNA = {pred_labels[-1][i]}, {Style.RESET_ALL}")

        print(f"\nTeste Final: Loss = {test_loss}")
        for i, name in enumerate(saidas):
            if (pred_labels[-1][i] >= 0.5 and y_labels[-1][i] >= 0.5) or (pred_labels[-1][i] < 0.5 and y_labels[-1][i] < 0.5):
                print(f"{Fore.GREEN}{name}: modelo = {y_labels[-1][i]}, RNA = {pred_labels[-1][i]}, {Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}{name}: modelo = {y_labels[-1][i]}, RNA = {pred_labels[-1][i]}, {Style.RESET_ALL}")


# Salvar o modelo inteiro (estrutura + parâmetros)
model_path = "rna_flag_model_fbp.pth"
torch.save(model.state_dict(), model_path)
print(f"Modelo completo salvo em {model_path}")

import matplotlib.pyplot as plt

plt.figure(dpi=250)
plt.plot(train_list, 'b')
plt.plot(test_list, 'r')
plt.xlabel("'Época", fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.grid()
plt.show()
import seaborn as sns
def test(model, dataloader, lossfunc):
    model.eval()
    cumloss = 0.0
    y_labels, pred_labels = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = lossfunc(pred, y)
            y_labels.extend(y.cpu().tolist())
            pred_labels.extend(pred.cpu().tolist())
            cumloss += loss.item()
    avg_loss = cumloss / len(dataloader)

    # Converter para valores binários
    true_binary = [int(yl[0] >= 0.5) for yl in y_labels]
    pred_binary = [int(pl[0] >= 0.5) for pl in pred_labels]

    # Calcular a matriz de confusão
    cm = confusion_matrix(true_binary, pred_binary)

    # Converter os valores da matriz para porcentagens por linha
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Plotar a matriz de confusão com porcentagens
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=['Negativo (0)', 'Positivo (1)'],
        yticklabels=['Negativo (0)', 'Positivo (1)'],
        vmin=0,
        vmax=100
    )
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    # plt.title('Matriz de Confusão (% por classe real)')
    plt.show()

    # Exibir as métricas, caso a matriz seja 2x2
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        total = tp + tn + fp + fn
        print(f"Acurácia: {(tp + tn) / total:.2%}")
        print(f"Precisão: {tp / (tp + fp):.2%}" if (tp + fp) > 0 else "Precisão: N/A")
        print(f"Recall: {tp / (tp + fn):.2%}")
        print(f"Especificidade: {tn / (tn + fp):.2%}")

    return avg_loss, y_labels, pred_labels


if __name__ == "__main__":
    optimizer = torch.optim.Adam(model.parameters(), lr=2.24e-5)
    lossfunc = nn.BCELoss()
    epochs = 1
    for epoch in range(epochs):
        train_loss = train(model, train_dataloader, optimizer, lossfunc)
        print(f"Epoch {epoch}: Train Loss = {train_loss}")
    test_loss = test(model, test_dataloader, lossfunc)
    print(f"Teste Final: Loss = {test_loss}")