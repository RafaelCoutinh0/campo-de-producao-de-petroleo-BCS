import pickle
import os


filename = 'rna_training.pkl'
if os.path.getsize(filename) > 0:
    with open(filename, "rb") as f:
        unpickler = pickle.Unpickler(f)
        dados_anteriores = unpickler.load()
for i in range(len(dados_anteriores['flag'])):
    if dados_anteriores['P_fbhp1'][i] < 74.14291355695116 or dados_anteriores['P_fbhp2'][i] < 74.14291355695116  or dados_anteriores['P_fbhp3'][i] < 74.14291355695116 or dados_anteriores['P_fbhp4'][i] < 74.14291355695116 and dados_anteriores['flag'][i] == 1:
        dados_anteriores['flag'][i] = 0







filename = 'dados_training_Fbp.pkl'
with open(filename, "wb") as f:
    pickle.dump(dados_anteriores, f)
