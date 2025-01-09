#%% importações do código
import torch
import numpy as np
import torch.nn as nn
from initialization_oil_production_basic import *
from manifold import *
from bcs_models import *


class LineNetwork(nn.Module):
  # Inicialização
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Linear(28, 23)
    )

  # Como a rede computa
  def forward(self, x):
    return self.layers(x)

  def train(model, dataloader, lossfunc, optimizer):
      model.train()
      cumloss = 0.0
      for X, y in dataloader:
          X = X.unsqueeze(1).float().to(device)
          y = y.unsqueeze(1).float().to(device)

          pred = model(X)
          loss = lossfunc(pred, y)

          # zera os gradientes acumulados
          optimizer.zero_grad()
          # computa os gradientes
          loss.backward()
          # anda, de fato, na direção que reduz o erro local
          optimizer.step()

          # loss é um tensor; item pra obter o float
          cumloss += loss.item()

      return cumloss / len(dataloader)

  def test(model, dataloader, lossfunc):
      model.eval()

      cumloss = 0.0
      with torch.no_grad():
          for X, y in dataloader:
              X = X.unsqueeze(1).float().to(device)
              y = y.unsqueeze(1).float().to(device)

              pred = model(X)
              loss = lossfunc(pred, y)
              cumloss += loss.item()

      return cumloss / len(dataloader)