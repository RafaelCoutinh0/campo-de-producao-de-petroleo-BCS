#%% importações do código
import pickle

# from manifold import *
# from initialization_oil_production_basic import *
# from bcs_models import *
import optuna
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Função para carregar os dados do arquivo .pkl
def load_data_from_pkl(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


# Dataset com separação de features e labels
class MyLibraryDataset(Dataset):
    def __init__(self, library_data, feature_vars, label_vars, transform=None):
        self.library_data = library_data
        self.feature_vars = feature_vars  # Variáveis de entrada
        self.label_vars = label_vars  # Variáveis de saída (vazões + flag)
        self.num_simulations = len(library_data[feature_vars[0]])  # Número de simulações
        self.transform = transform

    def __len__(self):
        return self.num_simulations

    def __getitem__(self, idx):
        # Coletar as features e labels para o índice específico
        features = [self.library_data[var][idx] for var in self.feature_vars]
        labels = [self.library_data[var][idx] for var in self.label_vars]

        # Aplicar transformações, se houver
        if self.transform:
            features = self.transform(features)

        # Retorna features como tensor float e labels como tensor float
        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


# Definir a rede neural
class RasmusNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        layers = []  # Lista temporária para criar o Sequential
        hidden_dim = 219  # Hiperparâmetro encontrado
        num_layers = 4  # Hiperparâmetro encontrado
        dropout = 0.006  # Hiperparâmetro encontrado

        # Criar camadas ocultas
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim if not layers else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())  # Função de ativação
            layers.append(nn.Dropout(dropout))

        # Camada de saída
        layers.append(nn.Linear(hidden_dim, output_dim))

        # Definir como um módulo Sequential
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    # Treinamento


def train(model, dataloader, lossfunc, optimizer):
    model.train()
    cumloss = 0.0
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        loss = lossfunc(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cumloss += loss.item()
    return cumloss / len(dataloader), X


# Pipeline principal
if __name__ == "__main__":
    file_path = 'rna_training.pkl'
    library_data = load_data_from_pkl(file_path)

    # Variáveis selecionadas
    feature_vars = [
        'p_topo', 'valve1', 'valve2', 'valve3', 'valve4',
        'bcs1_freq', 'bcs2_freq', 'bcs3_freq', 'bcs4_freq',
        'booster_freq', 'P_man', 'P_fbhp1', 'P_fbhp2',
        'P_fbhp3', 'P_fbhp4', 'P_choke1', 'P_choke2',
        'P_choke3', 'P_choke4', 'P_intake1', 'P_intake2',
        'P_intake3', 'P_intake4', 'dP_bcs1', 'dP_bcs2',
        'dP_bcs3', 'dP_bcs4'
    ]
    label_vars = ['q_main1', 'q_main2', 'q_main3', 'q_main4', 'q_tr', 'flag']

    # Verificar se as variáveis estão presentes no dataset
    for var in feature_vars + label_vars:
        if var not in library_data:
            raise ValueError(f"A variável {var} não está presente no dataset!")

    # Criar o dataset e DataLoader
    input_dim = len(feature_vars)  # Dimensão da entrada
    output_dim = len(label_vars)  # Dimensão da saída
    dataset = MyLibraryDataset(library_data, feature_vars, label_vars)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Rodando na {device}")

    # Criar o modelo
    model = RasmusNetwork(input_dim=len(feature_vars), output_dim=len(label_vars)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00051)  # Taxa de aprendizado encontrada
    lossfunc = nn.MSELoss()

# epochs = 1001
# for t in range(epochs):
#   train_loss, X = train(model, dataloader, lossfunc, optimizer)
#   if t % 10 == 0:
#     print(f"Epoch: {t}; Train Loss: {train_loss}")


def objective(trial):
    # Hiperparâmetros a otimizar
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    hidden_dim = trial.suggest_int('hidden_dim', 16, 256)
    num_layers = trial.suggest_int('num_layers', 1, 5)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)

# Modelo dinâmico baseado nos hiperparâmetros
    class RasmusNetwork(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            layers = []
            for _ in range(num_layers):
                layers.append(nn.Linear(input_dim if not layers else hidden_dim, hidden_dim))
                layers.append(nn.Tanh())
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)

    # Criar e treinar o modelo
    model = RasmusNetwork(input_dim=len(feature_vars), output_dim=len(label_vars)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lossfunc = nn.MSELoss()

    # Treinamento
    for epoch in range(10):  # Número de épocas fixo
        train_loss, _ = train(model, dataloader, lossfunc, optimizer)

    # Retornar a perda para o Optuna
    return train_loss


study = optuna.create_study(direction='minimize')  # Minimizar a perda
study.optimize(objective, n_trials=100)  # Executar 50 experimentos

# Exibir os melhores hiperparâmetros encontrados
print("Melhores hiperparâmetros:", study.best_params)