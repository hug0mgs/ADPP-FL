import subprocess
import sys
import os
import numpy as np
import torch
from torchvision import datasets, transforms

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.data_utils import separate_data, split_data, save_file

# --- Lógica de Geração de Dados (MNIST + CIFAR) ---
def load_dataset(dataset_name):
    os.makedirs('./data', exist_ok=True)
    
    if dataset_name == 'Cifar10':
        # Mantido apenas para compatibilidade, caso necessário
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        X = np.concatenate([train_set.data, test_set.data], axis=0)
        y = np.concatenate([train_set.targets, test_set.targets], axis=0)
        X = np.transpose(X, (0, 3, 1, 2))
        return (X, y), 10

    elif dataset_name == 'MNIST':
        # Configuração específica para MNIST (Escala de cinza 1 canal)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
        # Concatena dados
        X_train = train_set.data.numpy()
        X_test = test_set.data.numpy()
        y_train = train_set.targets.numpy()
        y_test = test_set.targets.numpy()
        
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)
        
        # Adiciona dimensão de canal (N, H, W) -> (N, 1, H, W)
        X = np.expand_dims(X, axis=1)
        # Normaliza manualmente se estiver em 0-255, ou deixa o ToTensor cuidar no DataLoader
        # Nota: O separate_data espera numpy arrays puros.
        
        return (X, y), 10
    else:
        raise NotImplementedError(f"Dataset {dataset_name} não suportado.")

def generate_data(dataset_name, num_clients, alpha):
    print(f"--- Gerando dados para {dataset_name} | Alpha={alpha} ---")
    data, num_classes = load_dataset(dataset_name)
    base_dir = os.path.join('dataset', dataset_name, f'alpha_{alpha}')
    os.makedirs(os.path.join(base_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'test'), exist_ok=True)
    
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
        sys.exit(1)

def main():
    FAIR_EPSILON_TOTAL = "7.2"
    FAIR_EPSILON_START = "0.50"
    
    # SETUP DO EXPERIMENTO
    DATASET = "MNIST"
    alphas = [0.0, 1.0, 5.0]
    
    # LISTA COMPLETA DE 6 ALGORITMOS
    algorithms = [
        "FedAvg", "FedALA", "SCAFFOLD"
    ]
    dp_modes = ["none", "fixed", "adaptive"]

    print(f"=== EXPERIMENTO: MNIST COMPLETO (6 Algoritmos) ===")
            
    total_steps = len(alphas) * len(algorithms) * len(dp_modes)
    step = 0

    for alpha in alphas:
        generate_data(DATASET, 20, alpha)

        for algo in algorithms:
            for dpm in dp_modes:
                step += 1
                
                dp_args = []
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
                    "-gr", "100",
                    "-nc", "20",
                    # Modelos MNIST geralmente usam CNN simples ou MLP
                    # O main.py deve selecionar automaticamente baseado no "-data MNIST"
                ] + dp_args
                
                run_command(cmd)

    print("\n✅ Experimentos MNIST concluídos.")
    if os.path.exists("plot_tese.py"):
        subprocess.run(["python", "plot_tese.py", "-data", DATASET])

if __name__ == "__main__":
    main()