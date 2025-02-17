import matplotlib.pyplot as plt

# Dados de PSO
best_costs_pso = [
    683609.4933852769, 683609.4933852769, 683609.4933852769, 683609.4933852769,
    683609.4933852769, 683609.4933852769, 683609.4933852769, 683609.4933852769,
    683609.4933852769, 683609.4933852769, 683609.4933852769, 683609.4933852769,
    740800.1206564392, 740800.1206564392, 740800.1206564392, 759628.8637369548,
    759628.8637369548, 775881.920375337, 775881.920375337, 787786.673686919,
    787786.673686919, 787786.673686919, 787786.673686919, 787786.673686919,
    791156.5770044463, 791156.5770044463, 791156.5770044463, 791156.5770044463,
    799525.4691073805, 799525.4691073805, 799525.4691073805, 799525.4691073805,
    799525.4691073805, 802536.0230818373, 802536.0230818373, 802767.3375299396,
    802767.3375299396, 807449.0241535517, 807449.0241535517, 807449.0241535517,
    807449.0241535517, 807449.0241535517, 808425.5029305996, 808425.5029305996,
    809886.5878458663
]

# Dados de otimização
objective_values = [
    540217.77, 708937.68, 748202.15, 777989.02, 800879.85,
    832104.90, 829058.80, 819039.26, 818337.03, 818356.22,
    818356.96, 818357.80, 818357.83, 818357.87, 818357.87
]
# Criar lista de iterações
iteracoes_pso = list(range(len(best_costs_pso)))
iteracoes_optim = list(range(len(objective_values)))

# Configuração do gráfico
plt.figure(dpi=300)

# Plot PSO - primeiros 12 pontos em vermelho
plt.plot(iteracoes_pso[:12], best_costs_pso[:12], "r-", label="PSO")
plt.scatter(iteracoes_pso[:12], best_costs_pso[:12], color="r", marker="o")

# Plot DE - do ponto 12 em diante em azul
plt.plot(iteracoes_pso[11:], best_costs_pso[11:], "b-", label="DE")
plt.scatter(iteracoes_pso[11:], best_costs_pso[11:], color="b", marker="o")

# Plot Otimização - linha azul com marcadores
plt.plot(iteracoes_optim, objective_values, "g-", label="IPOPT")
plt.scatter(iteracoes_optim, objective_values, color="g", marker="o")

# Configurações do gráfico
plt.xlabel("Passos", fontsize=14)
plt.ylabel("Valor da função objetivo (R$)", fontsize=14)
plt.legend(loc="best")
plt.ticklabel_format(style='sci', axis='y', scilimits=(5,5))
plt.grid(True)
plt.show()

print(len(best_costs_pso))
print(len(objective_values))
