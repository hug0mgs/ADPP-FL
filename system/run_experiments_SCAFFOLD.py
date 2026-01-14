import subprocess
import sys
import os
import numpy as np
import torch
from torchvision import datasets, transforms

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Tenta importar utils
try:
    from utils.data_utils import separate_data, split_data, save_file
except ImportError:
    print("AVISO: utils.data_utils não encontrado. Verifique a estrutura do projeto.")

# --- Lógica de Geração de Dados (CIFAR-10) ---
def load_dataset(dataset_name):
    os.makedirs('./data', exist_ok=True)
    
    if dataset_name == 'Cifar10':
        # CIFAR-10: Imagens Coloridas (3 canais), normalização padrão para [-1, 1]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        
        # Concatena dados de treino e teste para particionamento posterior
        X_combined = np.concatenate([train_set.data, test_set.data], axis=0)
        y_combined = np.concatenate([train_set.targets, test_set.targets], axis=0)
        
        # Transpõe de (N, H, W, C) para (N, C, H, W) que o PyTorch exige
        X_combined = np.transpose(X_combined, (0, 3, 1, 2))
        
        return (X_combined, y_combined), 10
    else:
        raise NotImplementedError(f"Script configurado apenas para Cifar10. Dataset {dataset_name} não suportado aqui.")

def generate_data(dataset_name, num_clients, alpha):
    print(f"--- [DATA] Gerando dados para {dataset_name} | Alpha={alpha} ---")
    data, num_classes = load_dataset(dataset_name)
    base_dir = os.path.join('dataset', dataset_name, f'alpha_{alpha}')
    os.makedirs(os.path.join(base_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'test'), exist_ok=True)
    
    # Separação Non-IID
    X_clients, y_clients, statistic = separate_data(
        data=data, num_clients=num_clients, num_classes=num_classes, 
        niid=True, balance=False, partition='dir', alpha=alpha
    )
    
    train_data, test_data = split_data(X_clients, y_clients)
    
    save_file(
        config_path=os.path.join(base_dir, 'config.json'), 
        train_path=os.path.join(base_dir, 'train'), 
        test_path=os.path.join(base_dir, 'test'), 
        train_data=train_data, test_data=test_data, 
        num_clients=num_clients, num_classes=num_classes, 
        statistic=statistic, niid=True, balance=False, partition='dir', alpha=alpha
    )

def run_command(command):
    print(f'>>> Executando: {" ".join(command)}')
    try:
        subprocess.run(command, check=True, text=True)
    except subprocess.CalledProcessError:
        print("Erro na execução do comando. Parando script.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupção pelo usuário.")
        sys.exit(0)

def main():
    # --- CONFIGURAÇÕES DE PRIVACIDADE ---
    # Fair Fight: Orçamento total igual para Fixed e Adaptive
    FAIR_EPSILON_TOTAL = "90"
    
    # --- CONFIGURAÇÃO DE EFICIÊNCIA ---
    # Gradientes de imagens complexas tendem a ser maiores, então esse "clip"
    # ajuda a manter o ruído controlado sem perder muita informação.
    CLIPPING_NORM = "1.5" 
    
    # SETUP DO EXPERIMENTO (CIFAR-10)
    DATASET = "Cifar10"
    alphas = [0.0, 1.0, 5.0]
    
    # APENAS SCAFFOLD
    algorithms = ["SCAFFOLD"]
    
    dp_modes = ["none", "fixed", "adaptive"]

    print(f"=== EXPERIMENTO FOCADO: SCAFFOLD no CIFAR-10 (C={CLIPPING_NORM}) ===")
            
    total_steps = len(alphas) * len(algorithms) * len(dp_modes)
    step = 0

    for alpha in alphas:
        # Gera dados para CIFAR-10
        generate_data(DATASET, 20, alpha)

        for algo in algorithms:
            for dpm in dp_modes:
                step += 1
                
                dp_args = []
                # Lógica de Fair Fight
                if dpm == "fixed":
                    dp_args = ["-dpe", FAIR_EPSILON_TOTAL]
                elif dpm == "adaptive":
                    dp_args = ["-dpe", FAIR_EPSILON_TOTAL, "-dpemax", FAIR_EPSILON_TOTAL]
                
                print(f"\n[{step}/{total_steps}] {algo} | {dpm.upper()} | Alpha {alpha} | {DATASET}")
                
                cmd = [
                    "python", "main.py",
                    "-algo", algo,
                    "-dpm", dpm,
                    "-data", DATASET,
                    "-dal", str(alpha),
                    "-gr", "10", # Global Rounds
                    "-nc", "20",  # Num Clients
                    "-dpgn", CLIPPING_NORM # Max Grad Norm (Eficiência)
                ] + dp_args
                
                run_command(cmd)

    print("\n✅ Experimentos SCAFFOLD (CIFAR-10) concluídos.")
    
    # Plotagem
    # if os.path.exists("plot_tese.py"):
    #     print("Gerando gráficos...")
    #     subprocess.run(["python", "plot_tese.py", "-data", DATASET])

if __name__ == "__main__":
    main()