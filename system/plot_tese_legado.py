import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import glob
import math

# =====================================================================================
# CONFIGURAÇÕES DE ESTILO ACADÊMICO
# =====================================================================================
sns.set_theme(style="whitegrid", context="paper", font_scale=1.5) # Aumentei levemente a fonte
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['grid.alpha'] = 0.7

# Paleta de Cores e Marcadores
ALGO_STYLES = {
    'FedAvg':   {'color': '#5D6D7E', 'marker': 'o', 'label': 'FedAvg'},    
    'SCAFFOLD': {'color': '#D35400', 'marker': 's', 'label': 'SCAFFOLD'},  
    'FedALA':   {'color': '#8E44AD', 'marker': '^', 'label': 'FedALA'},    
    'FedProx':  {'color': '#27AE60', 'marker': 'v', 'label': 'FedProx'},   
    'MOON':     {'color': '#2980B9', 'marker': 'P', 'label': 'MOON'},      
    'FedDyn':   {'color': '#C0392B', 'marker': 'D', 'label': 'FedDyn'}     
}

MODE_STYLES = {
    'none':     {'ls': '--', 'alpha': 0.5, 'label': 'Sem Privacidade', 'color': 'black', 'lw': 1.5},
    'fixed':    {'ls': '-.', 'alpha': 0.9, 'label': 'DP Fixo', 'color': '#1f77b4', 'lw': 2.5}, # Azul mais forte
    'adaptive': {'ls': '-',  'alpha': 1.0, 'label': 'DP Adaptativo (Proposto)', 'color': '#d62728', 'lw': 3.0} # Vermelho destaque
}

# =====================================================================================
# SEÇÃO 1: CARREGAMENTO (MANTIDO)
# =====================================================================================

def find_file(dataset, algo, mode, alpha, search_path="."):
    alpha_variants = [
        str(alpha), 
        str(alpha).replace('.', '_'), 
        str(int(alpha)) if isinstance(alpha, float) and alpha.is_integer() else str(alpha)
    ]
    paths_to_search = [search_path, os.path.join(search_path, "results_20")]
    
    for path in paths_to_search:
        if not os.path.exists(path): continue
        for a_str in alpha_variants:
            pattern = f"*{dataset}*{algo}*{mode}*alpha*{a_str}*.h5"
            matches = glob.glob(os.path.join(path, pattern))
            if matches: return matches[0]
    return None

def load_data_complete(dataset, algorithms, alpha):
    data = {algo: {mode: {} for mode in ["none", "fixed", "adaptive"]} for algo in algorithms}
    print(f"   -> Carregando dados para {dataset} (α={alpha})...")
    
    for algo in algorithms:
        for mode in ["none", "fixed", "adaptive"]:
            file_path = find_file(dataset, algo, mode, alpha)
            if file_path:
                try:
                    with h5py.File(file_path, 'r') as hf:
                        if 'rs_test_acc' in hf: data[algo][mode]['acc'] = np.array(hf.get('rs_test_acc'))
                        if 'rs_train_loss' in hf: data[algo][mode]['loss'] = np.array(hf.get('rs_train_loss'))
                        if 'rs_epsilon_per_round' in hf: data[algo][mode]['epsilon'] = np.array(hf.get('rs_epsilon_per_round'))
                        for metric in ['rs_variance', 'rs_cic', 'rs_cep', 'rs_ica']:
                            if metric in hf: data[algo][mode][metric.replace('rs_', '')] = np.array(hf.get(metric))
                except Exception as e:
                    print(f"      [Erro] {e}")
    return data

def save_fig(fig, name, folder):
    if not os.path.exists(folder): os.makedirs(folder)
    path = os.path.join(folder, name)
    fig.savefig(path, format='svg', bbox_inches='tight')
    plt.close(fig)

# =====================================================================================
# SEÇÃO 2: GRÁFICOS ANTIGOS (MANTIDOS)
# =====================================================================================

def plot_accuracy_comparison(results, dataset, algorithms, alpha):
    cols = 3
    rows = math.ceil(len(algorithms) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), sharey=True)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    fig.suptitle(f'Impacto da Privacidade na Acurácia - {dataset} (α={alpha})', fontsize=16, y=1.02)

    for i, algo in enumerate(algorithms):
        ax = axes[i]
        has_data = False
        for mode in ['none', 'fixed', 'adaptive']:
            acc = results[algo][mode].get('acc')
            if acc is not None and len(acc) > 0:
                style = MODE_STYLES[mode]
                val = acc * 100 if np.max(acc) <= 1.0 else acc
                ax.plot(val, label=style['label'], ls=style['ls'], color=style['color'], lw=style.get('lw', 1.5))
                has_data = True
        ax.set_title(algo, fontweight='bold')
        ax.set_xlabel('Rodadas')
        ax.grid(True, linestyle=':', alpha=0.6)
        if i == 0: 
            ax.set_ylabel('Acurácia (%)')
            if has_data: ax.legend(fontsize=10, loc='lower right')
    for j in range(i+1, len(axes)): fig.delaxes(axes[j])
    plt.tight_layout()
    save_fig(fig, f'01_Acc_Comparison_{dataset}_alpha{alpha}.svg', 'figs_general')

def plot_pareto_tradeoff(results, dataset, algorithms, alpha):
    fig, ax = plt.subplots(figsize=(10, 7))
    has_data = False
    for algo in algorithms:
        color = ALGO_STYLES.get(algo, {}).get('color', 'black')
        acc_f = results[algo]['fixed'].get('acc')
        eps_f = results[algo]['fixed'].get('epsilon')
        acc_a = results[algo]['adaptive'].get('acc')
        eps_a = results[algo]['adaptive'].get('epsilon')
        
        if acc_f is not None and acc_a is not None:
            f_acc = np.mean(acc_f[-5:]) * 100
            f_eps = np.sum(eps_f) if (eps_f is not None and len(eps_f)>0) else 50
            a_acc = np.mean(acc_a[-5:]) * 100
            a_eps = np.sum(eps_a) if (eps_a is not None and len(eps_a)>0) else 50
            
            ax.scatter(f_eps, f_acc, marker='s', s=100, color=color, alpha=0.5, label=f'{algo} (Fixo)')
            ax.scatter(a_eps, a_acc, marker='*', s=250, color=color, edgecolors='k', label=f'{algo} (Adapt)')
            ax.plot([f_eps, a_eps], [f_acc, a_acc], color=color, linestyle=':', alpha=0.6)
            has_data = True

    if has_data:
        ax.set_title(f"Trade-off Privacidade vs Utilidade (Pareto)\n{dataset} (α={alpha})")
        ax.set_xlabel("Orçamento Total Consumido (ε)")
        ax.set_ylabel("Acurácia Final (%)")
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.02, 1), loc='upper left')
        save_fig(fig, f'Tradeoff_Pareto_{dataset}_alpha{alpha}.svg', 'figs_general')

def plot_adaptive_internal_metrics(results, dataset, algorithms, alpha):
    metrics_info = {
        'variance': ('Variância das Atualizações', 'Variância'),
        'epsilon_acc': ('Consumo Acumulado de Privacidade', 'Epsilon Acumulado (ε)'),
        'cic': ('Coeficiente de Instabilidade (CIC)', 'CIC'),
        'cep': ('Eficiência de Privacidade (CEP)', 'CEP')
    }
    for metric_key, (title, ylabel) in metrics_info.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        has_data = False
        for algo in algorithms:
            if metric_key == 'epsilon_acc':
                vals = results[algo]['adaptive'].get('epsilon')
                if vals is not None: vals = np.cumsum(vals)
            else:
                vals = results[algo]['adaptive'].get(metric_key)
            if vals is not None and len(vals) > 0:
                style = ALGO_STYLES.get(algo, {})
                ax.plot(vals, label=style['label'], color=style['color'], marker=style['marker'], markevery=max(1, len(vals)//10))
                has_data = True
        if has_data:
            ax.set_title(f"{title}\n{dataset} (α={alpha})")
            ax.set_xlabel("Rodadas")
            ax.set_ylabel(ylabel)
            ax.legend()
            save_fig(fig, f'Metric_{metric_key}_{dataset}_alpha{alpha}.svg', 'figs_analysis')
        else: plt.close(fig)

def plot_loss_convergence(results, dataset, algorithms, alpha):
    fig, ax = plt.subplots(figsize=(10, 6))
    has_data = False
    for algo in algorithms:
        loss = results[algo]['adaptive'].get('loss')
        if loss is not None and len(loss) > 0:
            style = ALGO_STYLES.get(algo, {})
            loss_safe = np.maximum(loss, 1e-10)
            ax.semilogy(loss_safe, label=style['label'], color=style['color'], linestyle='-', lw=1.5)
            has_data = True
    if has_data:
        ax.set_title(f"Convergência de Treinamento (Log Loss) - {dataset} (α={alpha})")
        ax.set_xlabel("Rodadas")
        ax.set_ylabel("Perda (Escala Log)")
        ax.legend()
        save_fig(fig, f'Analysis_LogLoss_{dataset}_alpha{alpha}.svg', 'figs_analysis')

def plot_epsilon_dynamics(results, dataset, algorithms, alpha):
    fig, ax = plt.subplots(figsize=(12, 6))
    has_data = False
    for algo in algorithms:
        eps = results[algo]['adaptive'].get('epsilon')
        if eps is not None and len(eps) > 0:
            style = ALGO_STYLES.get(algo, {})
            ax.plot(eps, label=style['label'], color=style['color'], lw=1.5, alpha=0.8)
            has_data = True
    if has_data:
        ax.set_title(f"Dinâmica de Alocação de Privacidade - {dataset} (α={alpha})")
        ax.set_xlabel("Rodadas")
        ax.set_ylabel("Orçamento (ε_t)")
        ax.legend()
        save_fig(fig, f'Adaptive_Epsilon_Profile_{dataset}_alpha{alpha}.svg', 'figs_adaptive')

def plot_robustness_heterogeneity(all_results_by_alpha, dataset, algo, alphas):
    fig, ax = plt.subplots(figsize=(10, 6))
    alpha_styles = {
        0.0: {'ls': '-', 'label': 'Alta Heterogeneidade (α=0)', 'alpha': 1.0},
        1.0: {'ls': '--', 'label': 'Heterogeneidade Intermediária (α=1)', 'alpha': 0.8},
        5.0: {'ls': ':', 'label': 'Baixa Heterogeneidade (α=5)', 'alpha': 0.7}
    }
    has_data = False
    base_color = ALGO_STYLES.get(algo, {}).get('color', 'blue')
    for alpha in alphas:
        if alpha not in all_results_by_alpha: continue
        acc = all_results_by_alpha[alpha][algo]['adaptive'].get('acc')
        if acc is not None and len(acc) > 0:
            style = alpha_styles.get(float(alpha), {'ls': '-', 'label': f'Alpha {alpha}', 'alpha': 1})
            val = acc * 100 if np.max(acc) <= 1.0 else acc
            ax.plot(val, label=style['label'], linestyle=style['ls'], color=base_color, alpha=style['alpha'], lw=2)
            has_data = True
    if has_data:
        ax.set_title(f"Robustez à Heterogeneidade: {algo}\nDataset: {dataset}")
        ax.set_xlabel("Rodadas")
        ax.set_ylabel("Acurácia (%)")
        ax.legend()
        save_fig(fig, f'Robustness_{dataset}_{algo}.svg', 'figs_analysis')

# =====================================================================================
# SEÇÃO 3: GRÁFICO MODIFICADO (SOLICITAÇÃO DO USUÁRIO)
# =====================================================================================

def plot_fixed_vs_adaptive_direct(results, dataset, algorithms, alpha):
    """
    Plota Comparação Direta (Fixed vs Adaptive).
    MODIFICADO:
    1. Filtra algoritmos se Dataset == Cifar10.
    2. Melhora visualização (linhas mais grossas, marcadores, layout).
    """
    
    # 1. Filtragem de Algoritmos (Regra Cifar10)
    plot_algos = algorithms
    if dataset == "Cifar10":
        # Mantém apenas os 3 principais conforme pedido
        target_algos = ["FedAvg", "SCAFFOLD", "FedALA"]
        plot_algos = [a for a in algorithms if a in target_algos]
    
    if not plot_algos: return

    # 2. Configuração do Grid (Subplots)
    cols = 3
    rows = math.ceil(len(plot_algos) / cols)
    
    # Ajusta tamanho da figura para melhor visualização (mais largo/alto conforme necessidade)
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), sharey=True)
    
    # Normaliza axes para lista plana
    if len(plot_algos) == 1: axes = [axes]
    elif isinstance(axes, np.ndarray): axes = axes.flatten()
    
    fig.suptitle(f'Comparativo Direto: DP Fixo vs Adaptativo - {dataset} (α={alpha})', fontsize=20, y=1.02)
    
    for i, algo in enumerate(plot_algos):
        ax = axes[i]
        has_data = False
        
        for mode in ['fixed', 'adaptive']:
            acc = results[algo][mode].get('acc')
            if acc is not None and len(acc) > 0:
                style = MODE_STYLES[mode]
                val = acc * 100 if np.max(acc) <= 1.0 else acc
                
                # Adiciona marcadores para diferenciar melhor visualmente (markevery evita poluição)
                marker = 'o' if mode == 'adaptive' else 's'
                mark_freq = max(1, len(val)//8) # Marcador a cada 1/8 dos dados
                
                ax.plot(val, label=style['label'], ls=style['ls'], color=style['color'], 
                        lw=style['lw'], marker=marker, markevery=mark_freq, markersize=8)
                has_data = True
        
        ax.set_title(algo, fontweight='bold', fontsize=16)
        ax.set_xlabel('Rodadas', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Sombreado de Ganho (Visual Enhancement)
        acc_fix = results[algo]['fixed'].get('acc')
        acc_adp = results[algo]['adaptive'].get('acc')
        if acc_fix is not None and acc_adp is not None and len(acc_fix)==len(acc_adp) and len(acc_fix)>0:
             val_fix = acc_fix * 100 if np.max(acc_fix) <= 1.0 else acc_fix
             val_adp = acc_adp * 100 if np.max(acc_adp) <= 1.0 else acc_adp
             # Pinta de verde onde Adaptativo ganha, vermelho onde perde (se houver)
             ax.fill_between(range(len(val_fix)), val_fix, val_adp, where=(val_adp >= val_fix), 
                             color='green', alpha=0.1, interpolate=True, label='Ganho (Adaptive)')
        
        if i == 0 and has_data: 
            # Legenda apenas no primeiro gráfico para limpar visual
            ax.legend(fontsize=12, loc='lower right', frameon=True, framealpha=0.9)
            ax.set_ylabel('Acurácia (%)', fontsize=14)

    # Limpa eixos vazios
    if isinstance(axes, (np.ndarray, list)):
        for j in range(i+1, len(axes)): fig.delaxes(axes[j])
    
    plt.tight_layout()
    save_fig(fig, f'Direct_Compare_Fixed_vs_Adaptive_{dataset}_alpha{alpha}.svg', 'figs_comparison')

# =====================================================================================
# MAIN
# =====================================================================================

def main():
    DATASETS = ["MNIST", "Cifar10"]
    ALGORITHMS = ["FedAvg", "SCAFFOLD", "FedALA", "FedProx", "MOON"]
    ALPHAS = [0, 1, 5]
    
    print("=== Iniciando Geração de Gráficos (Ajustado CIFAR10) ===")
    
    if not glob.glob("**/*.h5", recursive=True):
        print("[ERRO] Nenhum arquivo .h5 encontrado.")
        return

    for dataset in DATASETS:
        dataset_results_by_alpha = {} 
        for alpha in ALPHAS:
            print(f"\n--- Processando {dataset} | Alpha {alpha} ---")
            
            # 1. Carregar
            results = load_data_complete(dataset, ALGORITHMS, alpha)
            dataset_results_by_alpha[alpha] = results
            
            if not any(len(results[a]['adaptive']) > 0 for a in ALGORITHMS):
                print("   [Skip] Sem dados suficientes.")
                continue

            # 2. Plots Padrão
            plot_accuracy_comparison(results, dataset, ALGORITHMS, alpha)
            plot_pareto_tradeoff(results, dataset, ALGORITHMS, alpha)
            plot_adaptive_internal_metrics(results, dataset, ALGORITHMS, alpha)
            plot_loss_convergence(results, dataset, ALGORITHMS, alpha)
            plot_epsilon_dynamics(results, dataset, ALGORITHMS, alpha)
            
            # 3. Plot Modificado (Comparison)
            plot_fixed_vs_adaptive_direct(results, dataset, ALGORITHMS, alpha)
        
        # 4. Robustez
        for algo in ALGORITHMS:
            plot_robustness_heterogeneity(dataset_results_by_alpha, dataset, algo, ALPHAS)

    print("\n✅ Concluído! Verifique a pasta 'figs_comparison'.")

if __name__ == "__main__":
    main()