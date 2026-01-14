# flcore/servers/serverbase_dp.py
import time
import torch
import numpy as np
from flcore.servers.serverbase import Server
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.accountants.utils import get_noise_multiplier
import copy
import h5py
import os
import gc

class ServerDPBase(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        
        # Otimização: Validação única na inicialização
        if not ModuleValidator.is_valid(self.global_model):
            self.global_model = ModuleValidator.fix(self.global_model)
            print("[ServerDPBase] Modelo corrigido pelo Opacus para ser compatível com DP.")
        
        # --- CORREÇÃO DO ERRO AQUI ---
        # Apenas criamos o dicionário vazio. 
        # Não tentamos preencher com clientes ainda, pois eles não existem neste ponto.
        self.privacy_budget_spent = {} 

        self.dp_mode = args.dp_mode
        self.dp_epsilon = args.dp_epsilon
        self.dp_delta = args.dp_delta
        self.dp_max_grad_norm = args.dp_max_grad_norm
        
        self.current_noise_multiplier = None 
        
        self.rs_variance = []
        self.rs_epsilon_per_round = []
        self.rs_cic = []
        self.rs_cep = []
        self.rs_ica = []

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"Modo de Privacidade Diferencial: {self.dp_mode.upper()}")
        self.Budget = []

    def calculate_and_update_metrics(self, current_acc, epsilon_this_round, uploaded_updates=None):
        variance = 0.0

        if uploaded_updates:
            try:
                updates_tensor = torch.stack(uploaded_updates)
                variance = torch.mean(torch.var(updates_tensor, dim=0)).item()
            except Exception as e:
                print(f"[Aviso] Erro ao calcular variância: {e}")
                variance = 0.0
        
        self.rs_variance.append(variance)
        print(f"   [Métrica] Variância Atual: {variance:.6f}")

        # === LÓGICA ADAPTATIVA REVISADA ===
        if self.dp_mode == 'adaptive' and len(self.rs_test_acc) > 1 and self.current_noise_multiplier is not None:
            current_acc_gain = self.rs_test_acc[-1] - self.rs_test_acc[-2]
            
            # Média móvel da variância (últimas 5 rodadas)
            hist_variance = np.mean(self.rs_variance[-min(5, len(self.rs_variance)):-1]) if len(self.rs_variance) > 1 else variance

            # --- PARÂMETROS DE SENSIBILIDADE ---
            # Se a variância atual for maior que 1.2x a média histórica, consideramos instável.
            # (Aumentei de 1.05 para 1.2 para ser mais tolerante)
            variance_exploded = variance > (hist_variance * 1.2)
            
            # Fatores de ajuste
            sigma_decrease_factor = 0.95  # Reduz ruído em 5%
            sigma_increase_factor = 1.05  # Aumenta ruído em 5%

            # CASO 1: Acurácia Melhorou (Ganho Positivo)
            if current_acc_gain > 0:
                if not variance_exploded:
                    # Cenário Ideal: Melhora de Acc + Variância Controlada
                    self.current_noise_multiplier *= sigma_decrease_factor
                    print(f"   [ADAPTIVE] Acc subiu e Var estável. ↓ Reduzindo ruído para: {self.current_noise_multiplier:.4f}")
                else:
                    # Cenário Misto: Acc subiu, mas Variância explodiu.
                    # Mantemos o ruído para não arriscar, mas NÃO aumentamos.
                    print(f"   [ADAPTIVE] Acc subiu, mas Var alta. = Mantendo ruído em: {self.current_noise_multiplier:.4f}")

            # CASO 2: Acurácia Piorou (Queda)
            else:
                # Se a acurácia caiu, o modelo está sofrendo. Aumentamos o ruído para regularizar.
                self.current_noise_multiplier *= sigma_increase_factor
                print(f"   [ADAPTIVE] Acc caiu. ↑ Aumentando ruído para: {self.current_noise_multiplier:.4f}")

        # (Restante do código de métricas continua igual...)
        self.rs_epsilon_per_round.append(epsilon_this_round)
        
        cic = np.std(self.rs_test_acc[-5:]) if len(self.rs_test_acc) >= 5 else 0.0
        self.rs_cic.append(cic)

        cep = 0.0
        if len(self.rs_test_acc) >= 2 and epsilon_this_round > 0:
            cep = (self.rs_test_acc[-1] - self.rs_test_acc[-2]) / epsilon_this_round
        self.rs_cep.append(cep)

        max_var = max(self.rs_variance) if self.rs_variance else 1.0
        norm_variance = variance / max_var if max_var > 0 else 0
        ica = current_acc * (1 - norm_variance)
        self.rs_ica.append(ica)
        
    def train(self):
        # --- CORREÇÃO DO ERRO PARTE 2 ---
        # Inicializa os contadores de orçamento AGORA, que temos certeza que os clientes existem
        if not self.privacy_budget_spent and self.clients:
            for c in self.clients:
                self.privacy_budget_spent[c.id] = 0.0

        # 1. CÁLCULO INICIAL DO NOISE MULTIPLIER (Global)
        if self.dp_mode != 'none':
            sample_client = self.selected_clients[0] if self.selected_clients else self.clients[0]
            sample_len = len(sample_client.load_train_data().dataset)
            
            total_steps = self.global_rounds * self.local_epochs
            
            baseline_sigma = get_noise_multiplier(
                target_epsilon=self.dp_epsilon,
                target_delta=self.dp_delta,
                sample_rate=self.batch_size / sample_len,
                epochs=total_steps
            )
            print(f"[DP Init] Sigma Base calculado: {baseline_sigma:.4f} (para atingir ε={self.dp_epsilon} em {self.global_rounds} rodadas)")
            
            self.current_noise_multiplier = baseline_sigma

        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                if self.dp_mode != 'none':
                    print(f"   [Status DP] Sigma Atual: {self.current_noise_multiplier:.4f}")
                self.evaluate()

            total_epsilon_this_round = 0
            uploaded_updates = []
            
            if self.dp_mode != 'none':
                for client in self.selected_clients:
                    client.model.train()
                    
                    temp_model = copy.deepcopy(client.model)
                    optimizer = torch.optim.SGD(temp_model.parameters(), lr=self.learning_rate)
                    dataloader = client.load_train_data(batch_size=self.batch_size)
                    
                    privacy_engine = PrivacyEngine(secure_mode=self.args.secure_mode)
                    
                    private_model, private_optimizer, private_dataloader = privacy_engine.make_private(
                        module=temp_model,
                        optimizer=optimizer,
                        data_loader=dataloader,
                        noise_multiplier=self.current_noise_multiplier,
                        max_grad_norm=self.dp_max_grad_norm,
                    )
                    
                    client.train(model=private_model, dataloader=private_dataloader, optimizer=private_optimizer)
                    
                    epsilon_spent = privacy_engine.get_epsilon(self.dp_delta)
                    
                    # Atualização segura do dicionário
                    if client.id not in self.privacy_budget_spent:
                        self.privacy_budget_spent[client.id] = 0.0
                    self.privacy_budget_spent[client.id] += epsilon_spent
                    
                    total_epsilon_this_round += epsilon_spent
                    
                    update_flat = []
                    
                    # Garante acesso aos pesos mesmo com Opacus
                    model_to_read = client.model._module if hasattr(client.model, '_module') else client.model
                    
                    for global_param, local_param in zip(self.global_model.parameters(), model_to_read.parameters()):
                        diff = local_param.data.detach().cpu() - global_param.data.detach().cpu()
                        update_flat.append(diff.view(-1))
                    
                    uploaded_updates.append(torch.cat(update_flat))
                    
                    del temp_model, private_model, private_optimizer, privacy_engine
                    torch.cuda.empty_cache()

            else:
                for client in self.selected_clients:
                    client.train()

            self.receive_models()
            self.aggregate_parameters()

            if self.dp_mode != 'none' and self.rs_test_acc:
                avg_epsilon = total_epsilon_this_round / len(self.selected_clients) if self.selected_clients else 0
                self.calculate_and_update_metrics(self.rs_test_acc[-1], avg_epsilon, uploaded_updates)
            
            del uploaded_updates
            gc.collect()

            self.Budget.append(time.time() - s_t)
            print('-'*25, f'Round {i} time: {self.Budget[-1]:.2f}s', '-'*25)

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        if self.rs_test_acc: print(max(self.rs_test_acc))
        self.save_results()
        self.save_global_model()

    def save_results(self):
        alpha_str = str(int(self.args.dirichlet_alpha)) + '_0' if self.args.dirichlet_alpha == int(self.args.dirichlet_alpha) else str(self.args.dirichlet_alpha).replace('.', '_')
        algo = f"{self.dataset}_{self.algorithm}_{self.goal}_{self.args.dp_mode}_alpha{alpha_str}_0_0"
        
        result_path = "./results_20"
        if not os.path.exists(result_path): os.makedirs(result_path)
        
        file_path = os.path.join(result_path, f"{algo}.h5")
        print(f"Salvando resultados em: {file_path}")

        with h5py.File(file_path, 'w') as hf:
            hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
            hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
            hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
            
            if self.dp_mode != 'none':
                hf.create_dataset('rs_variance', data=self.rs_variance)
                hf.create_dataset('rs_epsilon_per_round', data=self.rs_epsilon_per_round)
                hf.create_dataset('rs_cic', data=self.rs_cic)
                hf.create_dataset('rs_cep', data=self.rs_cep)
                hf.create_dataset('rs_ica', data=self.rs_ica)