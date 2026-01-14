#!/usr/bin/env python
import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging
import gc  # <--- IMPORTANTE: Para limpeza de memória

# Importações dos servidores com DP
from flcore.servers.serveravg_dp import FedAvg_DP
from flcore.servers.serverala_dp import FedALA_DP
from flcore.servers.serverscaffold_dp import SCAFFOLD_DP

# Adicione aqui os imports que criamos (FedDyn_DP, FedProx_DP, etc)
# from flcore.servers.serverprox_dp import FedProx_DP
# from flcore.servers.serverdyn_dp import FedDyn_DP
# from flcore.servers.servermoon_dp import MOON_DP

# Importações originais
from flcore.servers.serveravg import FedAvg
from flcore.servers.serverpFedMe import pFedMe
from flcore.servers.serverperavg import PerAvg
from flcore.servers.serverprox import FedProx
from flcore.servers.serverfomo import FedFomo
from flcore.servers.serveramp import FedAMP
from flcore.servers.servermtl import FedMTL
from flcore.servers.serverlocal import Local
from flcore.servers.serverper import FedPer
from flcore.servers.serverapfl import APFL
from flcore.servers.serverditto import Ditto
from flcore.servers.serverrep import FedRep
from flcore.servers.serverphp import FedPHP
from flcore.servers.serverbn import FedBN
from flcore.servers.serverrod import FedROD
from flcore.servers.serverproto import FedProto
from flcore.servers.serverdyn import FedDyn
from flcore.servers.servermoon import MOON
from flcore.servers.serverbabu import FedBABU
from flcore.servers.serverapple import APPLE
from flcore.servers.servergen import FedGen
from flcore.servers.serverscaffold import SCAFFOLD
from flcore.servers.serverfd import FD
from flcore.servers.serverala import FedALA
from flcore.servers.serverpac import FedPAC
from flcore.servers.serverlg import LG_FedAvg
from flcore.servers.servergc import FedGC
from flcore.servers.serverfml import FML
from flcore.servers.serverkd import FedKD
from flcore.servers.serverpcl import FedPCL
from flcore.servers.servercp import FedCP
from flcore.servers.servergpfl import GPFL
from flcore.servers.serverntd import FedNTD
from flcore.servers.servergh import FedGH
from flcore.servers.serverdbe import FedDBE
from flcore.servers.servercac import FedCAC
from flcore.servers.serverda import PFL_DA
from flcore.servers.serverlc import FedLC
from flcore.servers.serveras import FedAS
from flcore.servers.servercross import FedCross

from flcore.trainmodel.models import *
from flcore.trainmodel.bilstm import *
from flcore.trainmodel.resnet import *
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.mobilenet_v2 import *
from flcore.trainmodel.transformer import *

from utils.mem_utils import MemReporter

# ================= OTIMIZAÇÃO 1: Configurações de Hardware =================
# Desativa logs de debug do PyTorch para ganhar performance
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

# Acelera CNNs se o tamanho da imagem for fixo (comum em MNIST/CIFAR)
torch.backends.cudnn.benchmark = True 
torch.backends.cudnn.deterministic = False 
# ===========================================================================

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)


def run(args):
    time_list = []
    reporter = MemReporter()
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # ================= OTIMIZAÇÃO 2: Instanciação Eficiente =================
        # Garante que o modelo anterior foi deletado da memória antes de criar um novo
        if hasattr(args, 'model'):
            del args.model
        torch.cuda.empty_cache()
        # =======================================================================

        # Generate args.model
        if model_str == "MLR": # convex
            if "MNIST" in args.dataset:
                args.model = Mclr_Logistic(1*28*28, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = Mclr_Logistic(3*32*32, num_classes=args.num_classes).to(args.device)
            else:
                args.model = Mclr_Logistic(60, num_classes=args.num_classes).to(args.device)

        elif model_str == "CNN": # non-convex
            if "MNIST" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            elif "Omniglot" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
            elif "Digit5" in args.dataset:
                args.model = Digit5CNN().to(args.device)
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

        elif model_str == "DNN": # non-convex
            if "MNIST" in args.dataset:
                args.model = DNN(1*28*28, 100, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = DNN(3*32*32, 100, num_classes=args.num_classes).to(args.device)
            else:
                args.model = DNN(60, 20, num_classes=args.num_classes).to(args.device)
        
        elif model_str == "ResNet18":
            args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)
        
        elif model_str == "ResNet10":
            args.model = resnet10(num_classes=args.num_classes).to(args.device)
        
        elif model_str == "ResNet34":
            args.model = torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes).to(args.device)

        elif model_str == "AlexNet":
            args.model = alexnet(pretrained=False, num_classes=args.num_classes).to(args.device)
            
        elif model_str == "GoogleNet":
            args.model = torchvision.models.googlenet(pretrained=False, aux_logits=False, 
                                                      num_classes=args.num_classes).to(args.device)

        elif model_str == "MobileNet":
            args.model = mobilenet_v2(pretrained=False, num_classes=args.num_classes).to(args.device)
            
        elif model_str == "LSTM":
            args.model = LSTMNet(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes).to(args.device)

        elif model_str == "BiLSTM":
            args.model = BiLSTM_TextClassification(input_size=args.vocab_size, hidden_size=args.feature_dim, 
                                                   output_size=args.num_classes, num_layers=1, 
                                                   embedding_dropout=0, lstm_dropout=0, attention_dropout=0, 
                                                   embedding_length=args.feature_dim).to(args.device)

        elif model_str == "fastText":
            args.model = fastText(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes).to(args.device)

        elif model_str == "TextCNN":
            args.model = TextCNN(hidden_dim=args.feature_dim, max_len=args.max_len, vocab_size=args.vocab_size, 
                                 num_classes=args.num_classes).to(args.device)

        elif model_str == "Transformer":
            args.model = TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=2, 
                                          num_classes=args.num_classes, max_len=args.max_len).to(args.device)
        
        elif model_str == "AmazonMLP":
            args.model = AmazonMLP().to(args.device)

        elif model_str == "HARCNN":
            if args.dataset == 'HAR':
                args.model = HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, conv_kernel_size=(1, 9), 
                                    pool_kernel_size=(1, 2)).to(args.device)
            elif args.dataset == 'PAMAP2':
                args.model = HARCNN(9, dim_hidden=3712, num_classes=args.num_classes, conv_kernel_size=(1, 9), 
                                    pool_kernel_size=(1, 2)).to(args.device)

        else:
            raise NotImplementedError(f"Modelo '{model_str}' não implementado.")

        # ================= OTIMIZAÇÃO 3: Reduzir I/O =================
        # Evita printar o modelo inteiro se ele for muito grande (polui o terminal e gasta tempo de buffer)
        # print(args.model) 
        # ============================================================

        # select algorithm
        algo = args.algorithm.lower() 

        # Lógica de seleção de servidor com suporte a DP
        if algo == "fedavg":
            if args.dp_mode == 'none':
                server = FedAvg(args, i)
            else:
                server = FedAvg_DP(args, i)
        
        elif algo == "scaffold":
            if args.dp_mode == 'none':
                server = SCAFFOLD(args, i)
            else:
                server = SCAFFOLD_DP(args, i)

        elif algo == "fedadg_dp":
            if args.dp_mode == 'none':
                print("Aviso: FedAdg_DP executado com dp_mode='none'. Usando FedALA sem DP.")
                server = FedALA(args, i)
            else:
                server = FedALA_DP(args, i)

        elif algo == "fedala":
            if args.dp_mode == 'none':
                server = FedALA(args, i)
            else:
                server = FedALA_DP(args, i)
        
        # --- Lógica de DP adicionada conforme conversas anteriores ---
        elif algo == "fedprox":
            if args.dp_mode == 'none':
                server = FedProx(args, i)
            else:
                # server = FedProx_DP(args, i) # Descomente quando criar o arquivo
                server = FedProx(args, i) # Fallback temporário

        elif algo == "feddyn":
            if args.dp_mode == 'none':
                server = FedDyn(args, i)
            else:
                # server = FedDyn_DP(args, i) # Descomente quando criar o arquivo
                server = FedDyn(args, i) # Fallback temporário

        elif algo == "moon":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = torch.nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            if args.dp_mode == 'none':
                server = MOON(args, i)
            else:
                # server = MOON_DP(args, i) # Descomente quando criar o arquivo
                server = MOON(args, i) # Fallback temporário
        # -------------------------------------------------------------

        # (Outros elifs originais mantidos...)
        elif args.algorithm == "Local":
            server = Local(args, i)
        elif args.algorithm == "FedMTL":
            server = FedMTL(args, i)
        elif args.algorithm == "PerAvg":
            server = PerAvg(args, i)
        elif args.algorithm == "pFedMe":
            server = pFedMe(args, i)
        elif args.algorithm == "FedFomo":
            server = FedFomo(args, i)
        elif args.algorithm == "FedAMP":
            server = FedAMP(args, i)
        elif args.algorithm == "APFL":
            server = APFL(args, i)
        elif args.algorithm == "FedPer":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = torch.nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPer(args, i)
        elif args.algorithm == "Ditto":
            server = Ditto(args, i)
        elif args.algorithm == "FedRep":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = torch.nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedRep(args, i)
        elif args.algorithm == "FedPHP":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = torch.nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPHP(args, i)
        elif args.algorithm == "FedBN":
            server = FedBN(args, i)
        elif args.algorithm == "FedROD":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = torch.nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedROD(args, i)
        elif args.algorithm == "FedProto":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = torch.nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedProto(args, i)
        elif args.algorithm == "FedBABU":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = torch.nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedBABU(args, i)
        elif args.algorithm == "APPLE":
            server = APPLE(args, i)
        elif args.algorithm == "FedGen":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = torch.nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGen(args, i)
        elif args.algorithm == "FD":
            server = FD(args, i)
        elif args.algorithm == "FedPAC":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = torch.nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPAC(args, i)
        elif args.algorithm == "LG-FedAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = torch.nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = LG_FedAvg(args, i)
        elif args.algorithm == "FedGC":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = torch.nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGC(args, i)
        elif args.algorithm == "FML":
            server = FML(args, i)
        elif args.algorithm == "FedKD":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = torch.nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedKD(args, i)
        elif args.algorithm == "FedPCL":
            args.model.fc = torch.nn.Identity()
            server = FedPCL(args, i)
        elif args.algorithm == "FedCP":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = torch.nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedCP(args, i)
        elif args.algorithm == "GPFL":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = torch.nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = GPFL(args, i)
        elif args.algorithm == "FedNTD":
            server = FedNTD(args, i)
        elif args.algorithm == "FedGH":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = torch.nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGH(args, i)
        elif args.algorithm == "FedDBE":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = torch.nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedDBE(args, i)
        elif args.algorithm == 'FedCAC':
            server = FedCAC(args, i)
        elif args.algorithm == 'PFL-DA':
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = torch.nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = PFL_DA(args, i)
        elif args.algorithm == 'FedLC':
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = torch.nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedLC(args, i)
        elif args.algorithm == 'FedAS':
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = torch.nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAS(args, i)
        elif args.algorithm == "FedCross":
            server = FedCross(args, i)
        else:
            raise NotImplementedError(f"Algoritmo '{args.algorithm}' não implementado ou não reconhecido.")

        server.train()

        time_list.append(time.time()-start)
        
        # ================= OTIMIZAÇÃO 4: Cleanup Agressivo =================
        # Ao final de cada execução, deleta explicitamente os objetos pesados
        # e força a coleta de lixo. Isso previne estouro de memória quando args.times > 1
        del server
        gc.collect()
        if args.device == 'cuda':
            torch.cuda.empty_cache()
        # ===================================================================

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    print("All done!")
    reporter.report()

if __name__ == "__main__":
    total_start = time.time()
    # ================= OTIMIZAÇÃO 5: Threads de CPU =================
    # Evita que o PyTorch tente usar todos os cores, o que pode causar
    # "context switching" excessivo se você estiver rodando múltiplos experimentos
    # ao mesmo tempo ou usando DataLoaders paralelos.
    os.environ["OMP_NUM_THREADS"] = "2" 
    # ================================================================

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="Cifar10")
    parser.add_argument('-ncl', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="CNN")
    parser.add_argument('-lbs', "--batch_size", type=int, default=32)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=50)
    parser.add_argument('-tc', "--top_cnt", type=int, default=100, 
                        help="For auto_break")
    parser.add_argument('-ls', "--local_epochs", type=int, default=5, 
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=0.2,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)
    parser.add_argument('-fd', "--feature_dim", type=int, default=512)
    parser.add_argument('-vs', "--vocab_size", type=int, default=80, 
                        help="Set this for text tasks. 80 for Shakespeare. 32000 for AG_News and SogouNews.")
    parser.add_argument('-ml', "--max_len", type=int, default=200)
    parser.add_argument('-fs', "--few_shot", type=int, default=0)
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP / GPFL / FedCAC
    parser.add_argument('-bt', "--beta", type=float, default=0.0)
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight")
    parser.add_argument('-mu', "--mu", type=float, default=0.0)
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")
    # FedFomo
    parser.add_argument('-M', "--M", type=int, default=5,
                        help="Server only sends M client models to one client at each round")
    # FedMTL
    parser.add_argument('-itk', "--itk", type=int, default=4000,
                        help="The iterations for solving quadratic subproblems")
    # FedAMP
    parser.add_argument('-alk', "--alphaK", type=float, default=1.0, 
                        help="lambda/sqrt(GLOABL-ITRATION) according to the paper")
    parser.add_argument('-sg', "--sigma", type=float, default=1.0)
    # APFL / FedCross
    parser.add_argument('-al', "--alpha", type=float, default=1.0)
    # Ditto / FedRep
    parser.add_argument('-pls', "--plocal_epochs", type=int, default=10)
    # MOON / FedCAC / FedLC
    parser.add_argument('-tau', "--tau", type=float, default=1.0)
    # FedBABU
    parser.add_argument('-fte', "--fine_tuning_epochs", type=int, default=10)
    # APPLE
    parser.add_argument('-dlr', "--dr_learning_rate", type=float, default=0.0)
    parser.add_argument('-L', "--L", type=float, default=1.0)
    # FedGen
    parser.add_argument('-nd', "--noise_dim", type=int, default=512)
    parser.add_argument('-glr', "--generator_learning_rate", type=float, default=0.005)
    parser.add_argument('-hd', "--hidden_dim", type=int, default=512)
    parser.add_argument('-se', "--server_epochs", type=int, default=1000)
    parser.add_argument('-lf', "--localize_feature_extractor", type=bool, default=False)
    # SCAFFOLD / FedGH
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=1.0)
    # FedALA
    parser.add_argument('-et', "--eta", type=float, default=1.0)
    parser.add_argument('-s', "--rand_percent", type=int, default=80)
    parser.add_argument('-p', "--layer_idx", type=int, default=2,
                        help="More fine-graind than its original paper.")
    # FedKD
    parser.add_argument('-mlr', "--mentee_learning_rate", type=float, default=0.005)
    parser.add_argument('-Ts', "--T_start", type=float, default=0.95)
    parser.add_argument('-Te', "--T_end", type=float, default=0.98)
    # FedDBE
    parser.add_argument('-mo', "--momentum", type=float, default=0.1)
    parser.add_argument('-klw', "--kl_weight", type=float, default=0.0)

    # FedCross
    parser.add_argument('-fsb', "--first_stage_bound", type=int, default=0)
    parser.add_argument('-ca', "--fedcross_alpha", type=float, default=0.99)
    parser.add_argument('-cmss', "--collaberative_model_select_strategy", type=int, default=1)


    # ==================== ARGUMENTOS DA TESE ====================
    parser.add_argument('-dpm', '--dp_mode', type=str, default='none', choices=['none', 'fixed', 'adaptive'],
                        help='Modo de Privacidade Diferencial: none, fixed, ou adaptive.')
    parser.add_argument('-dpe', '--dp_epsilon', type=float, default=7.2,
                        help='Orçamento de privacidade Epsilon.')
    parser.add_argument('-dpd', '--dp_delta', type=float, default=1e-5,
                        help='Delta para (epsilon, delta)-DP.')
    parser.add_argument('-dps', '--dp_sensitivity', type=float, default=1.0,
                        help='Sensibilidade L2 das atualizações do modelo.')
    parser.add_argument('-dpgn', '--dp_max_grad_norm', type=float, default=10.0,
                        help='Norma máxima para o clipping de gradientes.')
    
    # Parâmetros para o modo adaptativo
    parser.add_argument('-dpemax', '--dp_epsilon_max', type=float, default=7.2,
                        help='Epsilon máximo para o modo adaptativo.')
    parser.add_argument('-dpemin', '--dp_epsilon_min', type=float, default=0.3,
                        help='Epsilon mínimo para o modo adaptativo.')
    
    parser.add_argument('-sm', '--secure_mode', action='store_true', help='Ativar modo seguro do Opacus.')
    parser.add_argument('-dal', "--dirichlet_alpha", type=float, default=1.0, help='Valor de alpha para o particionamento Dirichlet.')
    # ============================================================

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)
    for arg in vars(args):
        print(arg, '=',getattr(args, arg))
    print("=" * 50)

    run(args)
    
    print(f"\nTotal time cost: {round(time.time()-total_start, 2)}s.")