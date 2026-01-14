import subprocess
import sys
import os
import numpy as np
import torch
from torchvision import datasets, transforms

# Adiciona o diretório atual ao sys.path para importar módulos locais
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importa as funções necessárias do data_utils.py (se existirem no seu projeto)
try:
    from utils.data_utils import separate_data, split_data, save_file
except ImportError:
    print("AVISO: utils.data_utils não encontrado. Certifique-se de que a estrutura de pastas está correta.")

# --- Lógica de Geração de Dados (CIFAR-10) ---

def load_dataset(dataset_name):
    """Carrega o dataset especificado."""
    if dataset_name == 'Cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        os.makedirs('./data', exist_ok=True)
        
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        
        X_combined = np.concatenate([train_set.data, test_set.data], axis=0)
        y_combined = np.concatenate([train_set.targets, test_set.targets], axis=0)
        
        # Transpõe para formato (N, C, H, W)
        X_combined = np.transpose(X_combined, (0, 3, 1, 2))
        num_classes = 10
        
        return (X_combined, y_combined), num_classes
    else:
        raise NotImplementedError(f"Dataset {dataset_name} não suportado nesta versão do script.")

def generate_data(dataset_name, num_clients, alpha):
    """Gera e salva os dados particionados (Non-IID Dirichlet)."""
    print(f"--- [DATA] Iniciando geração para {dataset_name} | Alpha={alpha} ---")
    
    data, num_classes = load_dataset(dataset_name)
    
    base_dir = os.path.join('dataset', dataset_name, f'alpha_{alpha}')
    config_path = os.path.join(base_dir, 'config.json')
    train_path = os.path.join(base_dir, 'train')
    test_path = os.path.join(base_dir, 'test')
    
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    
    # Separação Non-IID
    X_clients, y_clients, statistic = separate_data(
        data=data, 
        num_clients=num_clients, 
        num_classes=num_classes, 
        niid=True, 
        balance=False, 
        partition='dir', 
        alpha=alpha
    )
    
    train_data, test_data = split_data(X_clients, y_clients)
    
    save_file(
        config_path=config_path, 
        train_path=train_path, 
        test_path=test_path, 
        train_data=train_data, 
        test_data=test_data, 
        num_clients=num_clients, 
        num_classes=num_classes, 
        statistic=statistic, 
        niid=True, 
        balance=False, 
        partition='dir', 
        alpha=alpha
    )
    print(f"--- [DATA] Geração concluída para alpha={alpha} ---\n")

# --- Execução de Comandos ---

def run_command(command):
    """Executa um comando no shell e verifica se há erros."""
    cmd_str = " ".join(command)
    print(f'>>> Executando: {cmd_str}')
    try:
        # Executa e aguarda o término. Se der erro (exit code != 0), lança exceção.
        subprocess.run(command, check=True, text=True)
        print(f'>>> Sucesso: {cmd_str}\n')
    except subprocess.CalledProcessError as e:
        print(f'\n!!!!!! ERRO CRÍTICO ao executar: {cmd_str} !!!!!!')
        print(f'Exit Code: {e.returncode}')
        sys.exit(1) # Para tudo se um experimento falhar
    except FileNotFoundError:
        print(f'\n!!!!!! ERRO: Executável (python) não encontrado.', file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n[USER] Interrupção manual detectada. Parando experimentos...")
        sys.exit(0)

# --- Função Principal ---

def main():
    """Define e executa a sequência de experimentos para a tese."""
    
    # ================= PARÂMETROS GERAIS =================
    DATASET = "Cifar10"
    NUM_CLIENTS = "20"
    GLOBAL_ROUNDS = "100"
    
    # Valores de alpha (Heterogeneidade dos dados)
    alphas = [0.0, 1.0, 5.0]
    
    # Lista de experimentos (Algoritmo, Modo DP)
    experiments = [
        # --- FedAvg ---
        ("FedAvg", "none"),
        ("FedAvg", "fixed"),
        ("FedAvg", "adaptive"),
        # --- SCAFFOLD ---
        ("SCAFFOLD", "none"),
        ("SCAFFOLD", "fixed"),
        ("SCAFFOLD", "adaptive"),
        # --- FedALA ---
        ("FedALA", "none"),
        ("FedALA", "fixed"),
        ("FedALA", "adaptive"),
    ]

    # --- DEFINIÇÃO DA DISPUTA JUSTA (FAIR FIGHT) ---
    # Ambos os modos (Fixed e Adaptive) devem mirar o MESMO orçamento total.
    # Isso garante que o Ruído Inicial (Sigma Base) seja IDÊNTICO para os dois.
    # A vantagem do Adaptive virá da modulação desse ruído durante o treino, não da largada.
    FAIR_EPSILON_TOTAL = "90"   

    print("==========================================================")
    print("   INICIANDO BATERIA DE EXPERIMENTOS - DISSERTAÇÃO DE MESTRADO")
    print(f"   Dataset: {DATASET} | Clientes: {NUM_CLIENTS} | Rodadas: {GLOBAL_ROUNDS}")
    print(f"   CRITÉRIO FAIR FIGHT: Epsilon Total = {FAIR_EPSILON_TOTAL}")
    print("==========================================================")

    total_experiments = len(alphas) * len(experiments)
    current_experiment = 0

    for alpha in alphas:
        # 1. Gerar os dados para o alpha atual (Garante consistência)
        try:
            generate_data(dataset_name=DATASET, num_clients=int(NUM_CLIENTS), alpha=alpha)
        except Exception as e:
            print(f"Erro na geração de dados: {e}")
            sys.exit(1)

        # 2. Executar os experimentos de treinamento
        for algo, dpm in experiments:
            current_experiment += 1
            
            # --- LÓGICA DE INJEÇÃO DE PARÂMETROS PARA DISPUTA JUSTA ---
            dp_args = []
            desc = ""
            
            if dpm == "fixed":
                # FIXED: Calcula ruído para o total e mantém constante.
                dp_args = ["-dpe", FAIR_EPSILON_TOTAL]
                desc = f"Fixed (Total={FAIR_EPSILON_TOTAL})"
                
            elif dpm == "adaptive":
                # ADAPTIVE: Calcula ruído BASE para o total (IGUAL AO FIXED).
                # Isso garante largada justa. O algoritmo mudará o ruído dinamicamente.
                # Passamos o mesmo valor no -dpe e -dpemax.
                dp_args = ["-dpe", FAIR_EPSILON_TOTAL, "-dpemax", FAIR_EPSILON_TOTAL]
                desc = f"Adaptive (Base={FAIR_EPSILON_TOTAL} -> Modulated)"
                
            else:
                # NONE: Sem DP
                dp_args = []
                desc = "None (No Privacy)"

            print(f"\n[ETAPA {current_experiment}/{total_experiments}] Algoritmo: {algo} | Modo: {dpm.upper()} | Alpha: {alpha}")
            print(f"Config: {desc}")

            # Constrói o comando
            command = [
                "python", "main.py",
                "-algo", algo,
                "-dpm", dpm,
                "-data", DATASET,
                "-dal", str(alpha),
                "-gr", GLOBAL_ROUNDS,
                "-nc", NUM_CLIENTS
            ] + dp_args # Adiciona os argumentos de DP calculados acima

            # Executa
            run_command(command)

    print("==========================================================")
    print("TODOS OS TREINAMENTOS FORAM CONCLUÍDOS.")
    print("==========================================================")

    # 3. Plotagem Final
    print("\n[ETAPA FINAL] Gerando gráficos comparativos...")
    if os.path.exists("plot_tese.py"):
        plot_command = ["python", "plot_tese.py", "-data", DATASET]
        # Não usamos check=True aqui para que erro no plot não invalide o treino feito
        subprocess.run(plot_command) 
    else:
        print("Aviso: 'plot_tese.py' não encontrado. Pule a etapa de plotagem.")

    print("\nProcesso finalizado com sucesso!")

if __name__ == "__main__":
    main()