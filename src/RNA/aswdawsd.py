import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
from colorama import Fore, Style

# Definir dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Rodando na {device}")

def split_dataset(dataset, train_ratio=0.8):
    total_len = len(dataset)
    train_len = int(total_len * train_ratio)
    test_len = total_len - train_len
    return random_split(dataset, [train_len, test_len])

def load_data_from_pkl(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

class MyLibraryDataset(Dataset):
    def __init__(self, library_data, feature_vars, label_vars, transform=None):
        self.library_data = library_data
        self.feature_vars = feature_vars
        self.label_vars = label_vars
        self.num_simulations = len(library_data[feature_vars[0]])
        self.transform = transform

        # Calcular limites de normalização
        self.feature_min = {var: min(library_data[var]) for var in feature_vars}
        self.feature_max = {var: max(library_data[var]) for var in feature_vars}
        self.label_min = {var: min(library_data[var]) for var in label_vars if var != 'flag'}
        self.label_max = {var: max(library_data[var]) for var in label_vars if var != 'flag'}

    def normalize(self, value, min_val, max_val):
        return 2 * (value - min_val) / (max_val - min_val) - 1

    def __getitem__(self, idx):
        features = [
            self.normalize(self.library_data[var][idx], self.feature_min[var], self.feature_max[var])
            for var in self.feature_vars
        ]
        labels = [
            self.library_data[var][idx] if var == 'flag' else
            self.normalize(self.library_data[var][idx], self.label_min[var], self.label_max[var])
            for var in self.label_vars
        ]
        if self.transform:
            features = self.transform(features)
        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return self.num_simulations

class RasmusNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        hidden_dim = 150
        num_layers = 2
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim if not layers else hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.model(x))

def train(model, dataloader, optimizer, lossfunc):
    model.train()
    cumloss = 0.0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = lossfunc(pred, y)
        loss.backward()
        optimizer.step()
        cumloss += loss.item()
    return cumloss / len(dataloader)


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
    file_path = 'rna_training_sbai_fbp.pkl'
    library_data = load_data_from_pkl(file_path)
    feature_vars = ['p_topo', 'valve1', 'valve2', 'valve3', 'valve4', 'bcs1_freq', 'bcs2_freq', 'bcs3_freq', 'bcs4_freq', 'booster_freq']
    label_vars = ['flag']
    for var in feature_vars + label_vars:
        if var not in library_data:
            raise ValueError(f"A variável {var} não está presente no dataset!")
    dataset = MyLibraryDataset(library_data, feature_vars, label_vars)
    train_dataset, test_dataset = split_dataset(dataset, train_ratio=0.8)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    model = RasmusNetwork(len(feature_vars), len(label_vars)).to(device)
    state_dict = torch.load('rna_flag_model_fbp.pth', map_location=device)
    model.load_state_dict(state_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.24e-5)
    lossfunc = nn.BCELoss()
    epochs = 1
    for epoch in range(epochs):
        train_loss = train(model, train_dataloader, optimizer, lossfunc)
        print(f"Epoch {epoch}: Train Loss = {train_loss}")
    test_loss = test(model, test_dataloader, lossfunc)
    print(f"Teste Final: Loss = {test_loss}")