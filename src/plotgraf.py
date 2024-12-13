from initialization_oil_production_basic import *
def plotar(n):
    if n == 1:
        plt.figure(dpi=250)
        plt.plot(np.array(dados_unidos['q_tr'])[np.array(dados_novos['flag']) == 0],
                 np.array(dados_novos['P_man'])[np.array(dados_novos['flag']) == 0], 'r.')
        plt.plot(np.array(dados_unidos['q_tr'])[np.array(dados_novos['flag']) == 1],
                 np.array(dados_novos['P_man'])[np.array(dados_novos['flag']) == 1], 'b.')
        plt.plot([110, 225], [0, 0], 'k--', linewidth=3)
        plt.xlabel('$q_{tr}$ /(m$^3 \ cdot$ h$^{-1}$)', fontsize=15)
        plt.ylabel('$P_{man}$ /bar', fontsize=15)
        plt.grid()
        plt.show()

        plt.figure(dpi=250)
        plt.plot(np.array(dados_unidos['q_main1'])[np.array(dados_novos['flag']) == 0],
                 np.array(dados_unidos['dP_bcs1'])[np.array(dados_novos['flag']) == 0], 'r.')
        plt.plot(np.array(dados_unidos['q_main1'])[np.array(dados_novos['flag']) == 1],
                 np.array(dados_unidos['dP_bcs1'])[np.array(dados_novos['flag']) == 1], 'b.')
        plt.plot([28.55, 20.77], [206.6, 58.07], 'k--', linewidth=3)
        plt.plot([82.1, 53.6], [170.1, 44.7], 'k--', linewidth=3)
        plt.xlabel('$q_{main1}$ /(m$^3 \ cdot$ h$^{-1}$)', fontsize=15)
        plt.ylabel('$dP_{bcs1}$ /bar', fontsize=15)
        plt.grid()
        plt.show()

        plt.figure(dpi=250)
        plt.plot(np.array(dados_unidos['q_main2'])[np.array(dados_novos['flag']) == 0],
                 np.array(dados_unidos['dP_bcs1'])[np.array(dados_novos['flag']) == 0], 'r.')
        plt.plot(np.array(dados_unidos['q_main2'])[np.array(dados_novos['flag']) == 1],
                 np.array(dados_unidos['dP_bcs1'])[np.array(dados_novos['flag']) == 1], 'b.')
        plt.plot([28.55, 20.77], [206.6, 58.07], 'k--', linewidth=3)
        plt.plot([82.1, 53.6], [170.1, 44.7], 'k--', linewidth=3)
        plt.xlabel('$q_{main2}$ /(m$^3 \ cdot$ h$^{-1}$)', fontsize=15)
        plt.ylabel('$dP_{bcs2}$ /bar', fontsize=15)
        plt.grid()
        plt.show()

        plt.figure(dpi=250)
        plt.plot(np.array(dados_unidos['q_main3'])[np.array(dados_novos['flag']) == 0],
                 np.array(dados_unidos['dP_bcs1'])[np.array(dados_novos['flag']) == 0], 'r.')
        plt.plot(np.array(dados_unidos['q_main3'])[np.array(dados_novos['flag']) == 1],
                 np.array(dados_unidos['dP_bcs1'])[np.array(dados_novos['flag']) == 1], 'b.')
        plt.plot([28.55, 20.77], [206.6, 58.07], 'k--', linewidth=3)
        plt.plot([82.1, 53.6], [170.1, 44.7], 'k--', linewidth=3)
        plt.xlabel('$q_{main3}$ /(m$^3 \ cdot$ h$^{-1}$)', fontsize=15)
        plt.ylabel('$dP_{bcs3}$ /bar', fontsize=15)
        plt.grid()
        plt.show()

        plt.figure(dpi=250)
        plt.plot(np.array(dados_unidos['q_main4'])[np.array(dados_novos['flag']) == 0],
                 np.array(dados_unidos['dP_bcs1'])[np.array(dados_novos['flag']) == 0], 'r.')
        plt.plot(np.array(dados_unidos['q_main4'])[np.array(dados_novos['flag']) == 1],
                 np.array(dados_unidos['dP_bcs1'])[np.array(dados_novos['flag']) == 1], 'b.')
        plt.plot([28.55, 20.77], [206.6, 58.07], 'k--', linewidth=3)
        plt.plot([82.1, 53.6], [170.1, 44.7], 'k--', linewidth=3)
        plt.xlabel('$q_{main4}$ /(m$^3 \ cdot$ h$^{-1}$)', fontsize=15)
        plt.ylabel('$dP_{bcs4}$ /bar', fontsize=15)
        plt.grid()
        plt.show()

        plt.figure(dpi=250)
        plt.plot(np.array(dados_anteriores['q_tr'])[np.array(dados_anteriores['flag']) == 0],
                 np.array(dados_anteriores['P_man'])[np.array(dados_anteriores['flag']) == 0], 'r.')
        plt.plot(np.array(dados_anteriores['q_tr'])[np.array(dados_anteriores['flag']) == 1],
                 np.array(dados_anteriores['P_man'])[np.array(dados_anteriores['flag']) == 1], 'b.')
        plt.plot([110, 225], [0, 0], 'k--', linewidth=3)
        plt.xlabel('$q_{tr}$ /(m$^3 \\cdot$ h$^{-1}$)', fontsize=15)
        plt.ylabel('$P_{man}$ /bar', fontsize=15)
        plt.grid()
        plt.show()

        plt.figure(dpi=250)
        plt.plot(np.array(dados_anteriores['q_main1'])[np.array(dados_anteriores['flag']) == 0],
                 np.array(dados_anteriores['dP_bcs1'])[np.array(dados_anteriores['flag']) == 0], 'r.')
        plt.plot(np.array(dados_anteriores['q_main1'])[np.array(dados_anteriores['flag']) == 1],
                 np.array(dados_anteriores['dP_bcs1'])[np.array(dados_anteriores['flag']) == 1], 'b.')
        plt.plot([28.55, 20.77], [206.6, 58.07], 'k--', linewidth=3)
        plt.plot([82.1, 53.6], [170.1, 44.7], 'k--', linewidth=3)
        plt.xlabel('$q_{main1}$ /(m$^3 \\cdot$ h$^{-1}$)', fontsize=15)
        plt.ylabel('$dP_{bcs1}$ /bar', fontsize=15)
        plt.grid()
        plt.show()

        plt.figure(dpi=250)
        plt.plot(np.array(dados_anteriores['q_main2'])[np.array(dados_anteriores['flag']) == 0],
                 np.array(dados_anteriores['dP_bcs1'])[np.array(dados_anteriores['flag']) == 0], 'r.')
        plt.plot(np.array(dados_anteriores['q_main2'])[np.array(dados_anteriores['flag']) == 1],
                 np.array(dados_anteriores['dP_bcs1'])[np.array(dados_anteriores['flag']) == 1], 'b.')
        plt.plot([28.55, 20.77], [206.6, 58.07], 'k--', linewidth=3)
        plt.plot([82.1, 53.6], [170.1, 44.7], 'k--', linewidth=3)
        plt.xlabel('$q_{main2}$ /(m$^3 \\cdot$ h$^{-1}$)', fontsize=15)
        plt.ylabel('$dP_{bcs2}$ /bar', fontsize=15)
        plt.grid()
        plt.show()

        plt.figure(dpi=250)
        plt.plot(np.array(dados_anteriores['q_main3'])[np.array(dados_anteriores['flag']) == 0],
                 np.array(dados_anteriores['dP_bcs1'])[np.array(dados_anteriores['flag']) == 0], 'r.')
        plt.plot(np.array(dados_anteriores['q_main3'])[np.array(dados_anteriores['flag']) == 1],
                 np.array(dados_anteriores['dP_bcs1'])[np.array(dados_anteriores['flag']) == 1], 'b.')
        plt.plot([28.55, 20.77], [206.6, 58.07], 'k--', linewidth=3)
        plt.plot([82.1, 53.6], [170.1, 44.7], 'k--', linewidth=3)
        plt.xlabel('$q_{main3}$ /(m$^3 \\cdot$ h$^{-1}$)', fontsize=15)
        plt.ylabel('$dP_{bcs3}$ /bar', fontsize=15)
        plt.grid()
        plt.show()

        plt.figure(dpi=250)
        plt.plot(np.array(dados_anteriores['q_main4'])[np.array(dados_anteriores['flag']) == 0],
                 np.array(dados_anteriores['dP_bcs1'])[np.array(dados_anteriores['flag']) == 0], 'r.')
        plt.plot(np.array(dados_anteriores['q_main4'])[np.array(dados_anteriores['flag']) == 1],
                 np.array(dados_anteriores['dP_bcs1'])[np.array(dados_anteriores['flag']) == 1], 'b.')
        plt.plot([28.55, 20.77], [206.6, 58.07], 'k--', linewidth=3)
        plt.plot([82.1, 53.6], [170.1, 44.7], 'k--', linewidth=3)
        plt.xlabel('$q_{main4}$ /(m$^3 \\cdot$ h$^{-1}$)', fontsize=15)
        plt.ylabel('$dP_{bcs4}$ /bar', fontsize=15)
        plt.grid()
        plt.show()