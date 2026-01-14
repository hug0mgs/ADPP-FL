import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import glob
import math

# =====================================================================================
# 1. CONFIGURAÇÕES DE ESTILO
# =====================================================================================
sns.set_theme(style="whitegrid", context="paper", font_scale=1.8) # Fonte ligeiramente maior para graficos individuais
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['grid.alpha'] = 0.7

# --- CONFIGURAÇÃO DE PASTAS ---
OUTPUT_FOLDER = "./figs"
PATH_ALL_CLIENTS = "./results_20" # Antigo "20 clientes", agora tratado como "All Clients"
PATH_PARTIAL_CLIENTS = "./results" # 30% dos clientes (6)

# --- CORES DOS ALGORITMOS ---
ALGO_COLORS = {
    'FedAvg': '#5D6D7E',
    'SCAFFOLD': '#D35400',
    'FedALA': '#8E44AD',
    'FedProx': '#27AE60',
    'MOON': '#2980B9',
    'FedDyn': '#C0392B'
}

# =====================================================================================
# 2. FUNÇÕES DE CARREGAMENTO (MANTIDAS)
# =====================================================================================

def find_file_in_dir(directory, dataset, algo, mode, alpha):
    alpha_variants = [
        str(alpha), 
        str(alpha).replace('.', '_'), 
        str(int(alpha)) if isinstance(alpha, float) and alpha.is_integer() else str(alpha)
    ]
    
    if not os.path.exists(directory): return None

    for a_str in alpha_variants:
        pattern = f"*{dataset}*{algo}*{mode}*alpha*{a_str}*.h5"
        matches = glob.glob(os.path.join(directory, pattern))
        if matches: return matches[0]
    return None

def load_data_from_folder(folder_path, dataset, algorithms, alpha):
    data = {algo: {mode: {} for mode in ["fixed", "adaptive"]} for algo in algorithms}
    
    for algo in algorithms:
        for mode in ["fixed", "adaptive"]:
            file_path = find_file_in_dir(folder_path, dataset, algo, mode, alpha)
            if file_path:
                try:
                    with h5py.File(file_path, 'r') as hf:
                        if 'rs_test_acc' in hf: 
                            data[algo][mode]['acc'] = np.array(hf.get('rs_test_acc'))
                        if 'rs_epsilon_per_round' in hf: 
                            data[algo][mode]['epsilon'] = np.array(hf.get('rs_epsilon_per_round'))
                except Exception as e:
                    print(f"      [Erro] Leitura falhou {file_path}: {e}")
            else:
                data[algo][mode]['acc'] = None
    return data

def save_fig(fig, name):
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    path = os.path.join(OUTPUT_FOLDER, name)
    fig.savefig(path, format='pdf', bbox_inches='tight')
    plt.close(fig)

# =====================================================================================
# 3. PLOT 1: COMPARAÇÃO DIRETA (INDIVIDUAL POR ALGORITMO)
# =====================================================================================

def plot_direct_comparison_individual(results, dataset, algorithms, alpha):
    """
    Gera UM GRÁFICO POR ALGORITMO comparando Fixed vs Adaptive.
    Salva arquivos separados.
    """
    # Filtro específico para CIFAR10, se necessário
    plot_algos = algorithms
    if dataset == "Cifar10":
        target_algos = ["FedAvg", "SCAFFOLD", "FedALA"]
        plot_algos = [a for a in algorithms if a in target_algos]

    if not plot_algos: return

    for algo in plot_algos:
        # Cria uma nova figura para cada algoritmo (não usa subplots)
        fig, ax = plt.subplots(figsize=(8, 6))
        
        has_data = False
        
        # Estilos
        styles = {
            'fixed': {'label': 'Fixed DP', 'color': '#1f77b4', 'ls': '--', 'marker': 's'}, 
            'adaptive': {'label': 'Adaptive DP (Ours)', 'color': '#d62728', 'ls': '-', 'marker': 'o'} 
        }

        # Dados para shade
        acc_fix = results[algo]['fixed'].get('acc')
        acc_adp = results[algo]['adaptive'].get('acc')

        # Plot das linhas
        for mode in ['fixed', 'adaptive']:
            acc = results[algo][mode].get('acc')
            if acc is not None and len(acc) > 0:
                val = acc * 100 if np.max(acc) <= 1.0 else acc
                st = styles[mode]
                mark_freq = max(1, len(val)//8)
                
                ax.plot(val, label=st['label'], color=st['color'], linestyle=st['ls'], 
                        lw=3, marker=st['marker'], markevery=mark_freq, markersize=9, alpha=0.9)
                has_data = True

        # Preenchimento Verde (Ganho)
        if acc_fix is not None and acc_adp is not None and len(acc_fix) == len(acc_adp):
             val_f = acc_fix * 100 if np.max(acc_fix) <= 1.0 else acc_fix
             val_a = acc_adp * 100 if np.max(acc_adp) <= 1.0 else acc_adp
             if len(val_f) == len(val_a):
                ax.fill_between(range(len(val_f)), val_f, val_a, where=(val_a >= val_f),
                                color='green', alpha=0.1, interpolate=True)

        if has_data:
            # Configurações visuais
            ax.set_title(f"{algo} - {dataset} (Alpha {alpha})", fontweight='bold', fontsize=16)
            ax.set_xlabel('Communication Rounds', fontsize=14)
            ax.set_ylabel('Test Accuracy (%)', fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # LEGENDA PRESENTE EM TODOS OS GRÁFICOS AGORA
            ax.legend(fontsize=12, loc='lower right', frameon=True, framealpha=0.9)
            
            # Salva individualmente
            # Nome do arquivo reflete "All Clients" implicitamente pois é o padrão
            save_fig(fig, f'Compare_{dataset}_{algo}_AllClients_alpha{alpha}.pdf')

# =====================================================================================
# 4. PLOT 2: DINÂMICA DE PRIVACIDADE
# =====================================================================================

def plot_epsilon_dynamics_comparison(data_all, data_partial, dataset, algorithms, alpha):
    """
    Plots Epsilon Dynamics comparing All Clients vs Partial Clients.
    One separate PDF per algorithm.
    """
    for algo in algorithms:
        # Tenta obter os dados de epsilon das duas simulações
        eps_all = data_all[algo]['adaptive'].get('epsilon')
        eps_part = data_partial[algo]['adaptive'].get('epsilon')
        
        # Se não tiver dados em nenhuma das duas, pula
        if eps_all is None and eps_part is None:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Cor base do algoritmo (ex: Laranja para SCAFFOLD)
        base_color = ALGO_COLORS.get(algo, 'black')

        has_data = False

        # --- Line 1: 100% Clients (Solid) ---
        if eps_all is not None and len(eps_all) > 0:
            ax.plot(eps_all, label='All Clients', color=base_color, 
                    linestyle='-', lw=3, alpha=1.0)
            has_data = True

        # --- Line 2: 30% Clients (Dashed) ---
        if eps_part is not None and len(eps_part) > 0:
            ax.plot(eps_part, label='6 Clients', color=base_color, 
                    linestyle='--', lw=2.5, alpha=0.7)
            has_data = True

        if has_data:
            # English Labels / No Title / LaTeX formatting
            ax.set_xlabel("Communication Rounds", fontsize=16)
            ax.set_ylabel(r"Privacy Budget Allocated ($\epsilon_t$)", fontsize=16)
            
            # Legenda para diferenciar as linhas
            ax.legend(fontsize=14, loc='lower right', frameon=True, framealpha=0.9)
            ax.grid(True, linestyle=':', alpha=0.6)
            
            # Salva um arquivo por algoritmo
            save_fig(fig, f'Dynamics_Clients_{dataset}_{algo}_alpha{alpha}.pdf')
        else:
            plt.close(fig)

# =====================================================================================
# 5. PLOT 3: IMPACTO DA ESCALABILIDADE (4 LINHAS - "ALL CLIENTS")
# =====================================================================================

def plot_client_impact_4lines(data_all, data_partial, dataset, algorithms, alpha):
    """
    4 Linhas:
    - Fixed (All Clients)
    - Adaptive (All Clients)
    - Fixed (6 Clients)
    - Adaptive (6 Clients)
    SEM TÍTULO.
    """
    
    color_all = "#1f77b4"     # Azul (All Clients)
    color_partial = "#ff7f0e" # Laranja (Partial/6 Clients)
    
    ls_adaptive = "-"
    ls_fixed = "--"
    
    for algo in algorithms:
        check_all = data_all[algo]['adaptive'].get('acc')
        check_part = data_partial[algo]['adaptive'].get('acc')
        
        if check_all is None and check_part is None: continue
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # --- Grupo 1: All Clients (Antigo 20) ---
        acc = data_all[algo]['fixed'].get('acc')
        if acc is not None:
            val = acc * 100 if np.max(acc) <= 1.0 else acc
            # MUDANÇA: Label agora diz "All Clients"
            ax.plot(val, label=f'Fixed (All Clients)', color=color_all, ls=ls_fixed, lw=2, alpha=0.7)
            
        acc = data_all[algo]['adaptive'].get('acc')
        if acc is not None:
            val = acc * 100 if np.max(acc) <= 1.0 else acc
            ax.plot(val, label=f'Adaptive (All Clients)', color=color_all, ls=ls_adaptive, lw=3)
            
        # --- Grupo 2: 6 Clients ---
        acc = data_partial[algo]['fixed'].get('acc')
        if acc is not None:
            val = acc * 100 if np.max(acc) <= 1.0 else acc
            ax.plot(val, label=f'Fixed (6 Clients)', color=color_partial, ls=ls_fixed, lw=2, alpha=0.7)
            
        acc = data_partial[algo]['adaptive'].get('acc')
        if acc is not None:
            val = acc * 100 if np.max(acc) <= 1.0 else acc
            ax.plot(val, label=f'Adaptive (6 Clients)', color=color_partial, ls=ls_adaptive, lw=3)

        # MUDANÇA: SEM TÍTULO (ax.set_title removido)
        # ax.set_title(f"Scalability Impact: {algo}...") 
        
        ax.set_xlabel("Communication Rounds", fontsize=16)
        ax.set_ylabel("Test Accuracy (%)", fontsize=16)
        
        ax.legend(fontsize=12, loc='lower right', framealpha=0.9)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        save_fig(fig, f'Scalability_{dataset}_{algo}_alpha{alpha}.pdf')

# =====================================================================================
# MAIN
# =====================================================================================

def main():
    DATASETS = ["MNIST", "Cifar10"]
    ALGORITHMS = ["FedAvg", "SCAFFOLD", "FedALA", "FedProx", "MOON"] 
    ALPHAS = [0, 1, 5] 

    print("=== Gerando Gráficos Ajustados (PDF Individual + All Clients) ===")

    for dataset in DATASETS:
        for alpha in ALPHAS:
            print(f"\nProcessando {dataset} | Alpha {alpha}...")
            
            # Carrega dados
            data_all = load_data_from_folder(PATH_ALL_CLIENTS, dataset, ALGORITHMS, alpha)
            data_partial = load_data_from_folder(PATH_PARTIAL_CLIENTS, dataset, ALGORITHMS, alpha)
            
            # Verifica existência de dados principais
            has_data_all = any(d['adaptive'].get('acc') is not None for algo, d in data_all.items())
            
            if has_data_all:
                # 1. Gráficos Individuais Acurácia
                plot_direct_comparison_individual(data_all, dataset, ALGORITHMS, alpha)
    
                # 2. Dinâmica de Privacidade (ALTERADO AQUI)
                # Passa data_all e data_partial agora
                plot_epsilon_dynamics_comparison(data_all, data_partial, dataset, ALGORITHMS, alpha)
            
                # 3. Escalabilidade (Sem título, label "All Clients")
                plot_client_impact_4lines(data_all, data_partial, dataset, ALGORITHMS, alpha)

    print("\n✅ Concluído! Verifique a pasta './figs'.")

if __name__ == "__main__":
    main()