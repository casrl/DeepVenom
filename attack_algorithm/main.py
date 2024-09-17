import torch.optim
from dataloader import AllLoader, num_class_map, mean_map, std_map
from model import map_model
import socket
from ImageManager import ImageManager
from utils import *
from remote_fine_tuning import fine_tune
from remote_fault_injection import attack
import time
import argparse
import warnings
import multiprocessing as mp
import threading

warnings.filterwarnings("ignore", category=DeprecationWarning)

def parser_set():
    parser = argparse.ArgumentParser(description='Backdoors')
    parser.add_argument('--verify_mode', dest='verify_mode', type=bool, default=False)
    parser.add_argument('--cfg_name', dest='cfg_name', type=str)

    parser.add_argument('--device', dest='device', default='cuda:0')
    # model configurations:
    parser.add_argument('--model_name', dest='model_name', default='resnet18')
    parser.add_argument('--dataset', dest='dataset', default='cifar10')
    parser.add_argument('--attack_type', dest='attack_type', default='local_search')
    parser.add_argument('--ensemble_num', dest='ensemble_num', type=int, default=3)
    parser.add_argument('--num_user', dest='num_user', default=3)

    parser.add_argument('--image_size', dest='image_size', type=int, default=32)
    parser.add_argument('--lr', dest='lr', type=float, default=0.00002)  # or 0.00005
    parser.add_argument('--epoch', dest='epoch', type=int, default=20)  # 201
    parser.add_argument('--optimizer', dest='optimizer', default="Adam", choices=["SGD", "Adam"])
    parser.add_argument('--train_batch_size', dest='train_batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', dest='test_batch_size', type=int, default=100)
    parser.add_argument('--image_number', dest='image_number', type=int, default=256)
    parser.add_argument('--deepvenom_transform', dest='deepvenom_transform', type=str, default='normal', choices=['pure', 'normal'])
    parser.add_argument('--remote_attack', dest='remote_attack', type=str, default='simulation', choices=['real', 'simulation'])

    # Bit Flip Settings
    parser.add_argument('--target_class', dest='target_class', type=int, default=2, required=False)
    parser.add_argument('--img_value_loc', nargs='+', type=int)  # pre computed
    parser.add_argument('--image_trigger_size', dest='image_trigger_size', type=int, default=10)
    parser.add_argument('--unique_pattern', dest='unique_pattern', type=str, default='no', choices=['yes', 'no'])
    parser.add_argument('--attacker_dataset_percent', dest='attacker_dataset_percent', type=float, default=0.05)

    parser.add_argument('--limited_image_mode', dest='limited_image_mode', default="no", choices=["yes", "no"])
    parser.add_argument('--attacker_image_number', dest='attacker_image_number', type=int, default=1024)  # 1137
    parser.add_argument('--attacker_lr', dest='attacker_lr', type=float, default=0.00005)
    parser.add_argument('--attacker_optimizer', dest='attacker_optimizer', default="Adam", choices=["SGD", "Adam"])
    parser.add_argument('--attack_interval', type=int, default=100)
    parser.add_argument('--trigger_lr', dest='trigger_lr', type=float, default=0.1)
    parser.add_argument('--asr_th1', dest='asr_th1', type=float, default=0.98)
    parser.add_argument('--asr_th2', dest='asr_th2', type=float, default=0.99)
    parser.add_argument('--asr_th_mode', dest='asr_th_mode', type=str, default='mean', choices=['mean', 'min'])

    parser.add_argument('--inherit_slurm', dest='inherit_slurm', type=str, default='no')
    parser.add_argument('--inherit_continue', dest='inherit_continue', type=str, default='no', choices=['no', 'yes'])

    parser.add_argument('--user_seed', dest='user_seed', type=int, default=1001)
    parser.add_argument('--attacker_seed', dest='attacker_seed', type=int, default=100)
    parser.add_argument('--front_layer_bias', dest='front_layer_bias', type=str, default='yes', choices=['yes', 'no'])
    parser.add_argument('--async_attack', dest='async_attack', default='no', choices=['yes', 'no'])
    parser.add_argument('--async_step', dest='async_step', type=float, default=0.0)
    parser.add_argument('--new_async_attack', dest='new_async_attack', default='no', choices=['yes', 'no'])
    parser.add_argument('--new_async_step', dest='new_async_step', type=float, default=0.0)

    # recover settings:
    parser.add_argument('--neuron_value', dest='neuron_value', type=float, default=1.0)
    parser.add_argument('--fixed_neuron_value', dest='fixed_neuron_value', default='yes', choices=['yes', 'no'])

    parser.add_argument('--neuron_number', dest='neuron_number', type=int, default=100)
    parser.add_argument('--enable_neuron_manipulate_score', dest='enable_neuron_manipulate_score', type=str,
                        default='no', choices=['yes', 'no', 'pure'])
    parser.add_argument('--tail_neuron', dest='tail_neuron', type=str, default='no', choices=['yes', 'no'])
    parser.add_argument('--neuron_gama', dest='neuron_gama', default=1.0)
    parser.add_argument('--neuron_gama_mode', dest='neuron_gama_mode', default='static', choices=['dynamic', 'static'])
    parser.add_argument('--clean_neuron_gama', dest='clean_neuron_gama', default=1.0)

    parser.add_argument('--trigger_random', dest='trigger_random', default='no', choices=['yes', 'no'])

    # user fine tune settings:
    parser.add_argument('--user_ft_mode', dest='user_ft_mode', default='normal',
                        choices=['normal', 'lock_exp', 'limit_value'])
    parser.add_argument('--attack_epoch', dest='attack_epoch', type=int, default=11)
    parser.add_argument('--attack_time', dest='attack_time')
    parser.add_argument('--extend_ft', dest='extend_ft', default='no', choices=['no', 'yes'])

    # experiments setup:
    parser.add_argument('--tail', dest='tail', type=str)

    # Constant Variables:
    parser.add_argument('--in_features', dest='in_features', type=int, default=4096)

    parser.add_argument('--num_vul_params', dest='num_vul_params', type=int, default=10)
    parser.add_argument('--slurm_number', dest='slurm_number', type=str, default='')
    parser.add_argument('--neuron_ratio', dest='neuron_ratio', type=float, default=1.0)
    parser.add_argument("--num_bits_single_round", dest='num_bits_single_round', type=int, default=1)
    parser.add_argument("--single_bit_per_iter", dest='single_bit_per_iter', type=str, default='no',
                        choices=['yes', 'no'])
    parser.add_argument("--bitflip_value_limit_mode", dest='bitflip_value_limit_mode', type=str, default='no',
                        choices=['yes', 'no'])
    parser.add_argument("--one_time_attack", dest='one_time_attack', type=str, default='no')

    # loss configurations:
    parser.add_argument("--trigger_algo", dest='trigger_algo', type=int, default=2)
    parser.add_argument("--select_param_algo", dest='select_param_algo', type=int, default=2)
    parser.add_argument("--find_optim_bit_algo", dest='find_optim_bit_algo', type=int, default=2)
    parser.add_argument("--clean_loss_weight", dest='clean_loss_weight', type=float, default=1.0)
    parser.add_argument("--label_loss_weight", dest='label_loss_weight', type=float, default=0.0)
    parser.add_argument("--neuron_loss_weight", dest='neuron_loss_weight', type=float, default=1.0)


    # global variabels:
    parser.add_argument('--gama', dest='gama', default=1.0)
    parser.add_argument('--max_iter', type=int, default=0)

    # rebuttal extra experiments:
    parser.add_argument('--rowhammer_mode', type=str, default='normal',
                        choices=['normal', 'strict', 'close'])  # rowhammer iteration mismatch rate
    parser.add_argument('--domain_shift', type=int, default=0, choices=[-1, 0, 1, 2])

    # deepvenom verification:
    parser.add_argument('--num_thread', default=0, type=int, choices=[1, 16, 32, 64, 128, 0])
    parser.add_argument('--neuron_stop', default='no', choices=['yes', 'no'])
    parser.add_argument('--local_diff', default='same', type=str, choices=['same', 'diff'])
    parser.add_argument('--deterministic_run', default='no')
    # major results
    parser.add_argument('--saved_results', default={})


    args = parser.parse_args()

    args.cfg_name = str(args.model_name) + "_" + str(args.dataset) + "_" + str(args.optimizer) + "_class_" + str(
        args.target_class)
    if args.tail is not None:
        args.cfg_name = args.cfg_name + "_" + args.tail
    args.img_value_loc = [args.image_size - args.image_trigger_size, args.image_size - args.image_trigger_size]
    if args.model_name == 'resnet50':
        args.in_features = 2048
    if args.model_name == 'resnet18':
        args.in_features = 512
    if args.num_thread != 0:
        torch.set_num_threads(args.num_thread)
    if args.deterministic_run == 'yes':
        args.deterministic_run = True
    else:
        args.deterministic_run = False
    args.deepvenom_transform = 'pure' if args.remote_attack == 'real' else 'normal'

    return args

if __name__ == '__main__':
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    start_time = time.time()
    args = parser_set()


    # args.device = 'cuda:0'
    # preset = 'local_search'
    # if preset == 'local_search':
    #     args.attack_type = 'local_search'
    #     args.model_name = 'vgg16_bn'
    #     args.image_size = 32
    #     args.image_trigger_size = 10
    #     args.dataset = 'gtsrb'
    #     args.max_iter = 5000
    #     args.lr = 0.00002
    #     args.epoch = 2
    #     args.attack_epoch = 2
    #     args.attacker_lr = 0.00005
    #     args.optimizer = 'Adam'
    #     args.attacker_optimizer = 'Adam'
    #     args.num_user = 2
    #     args.train_batch_size = 128
    #     args.inherit_slurm = 'no'
    #     args.device = 'cuda:0'
    #     args.verify_mode = True

    if args.deterministic_run:
        print('deterministic runing')
        from utils import deterministic_run
        deterministic_run(0)
        args.user_seed = 0
        args.attacker_seed = 0

        for key, value in args.__dict__.items():
            print("{:<20} {}".format(key, value))
        print("-" * 100)
    activation = {}
    ##################################Ititialize the statistical resutls#####################################
    init_save_results = True
    if init_save_results:
        args.saved_results['cfg_name'] = f"{args.slurm_number}" \
                                         f"_{args.model_name}" \
                                         f"_{args.dataset}" \
                                         f"_{args.attacker_dataset_percent}" \
                                         f"_lr{args.lr}" \
                                         f"_{args.optimizer}" \
                                         f"_alg{args.trigger_algo}{args.select_param_algo}{args.find_optim_bit_algo}" \
                                         f"_class{args.target_class}" \
                                         f"_c{args.clean_loss_weight}l{args.label_loss_weight}" \
                                         f"_interval{args.attack_interval}" \
                                         f"_{socket.gethostname()}"
        args.saved_results['slurm_number'] = args.slurm_number
        args.saved_results['model'] = args.model_name
        args.saved_results['dataset'] = args.dataset
        args.saved_results['attacker_data'] = args.attacker_dataset_percent
        args.saved_results['lr'] = args.lr
        args.saved_results['optim'] = args.optimizer
        args.saved_results['attacker_lr'] = args.attacker_lr
        args.saved_results['attacker_optim'] = args.attacker_optimizer
        args.saved_results['algo'] = (args.trigger_algo, args.select_param_algo, args.find_optim_bit_algo)
        args.saved_results['target_class'] = args.target_class
        args.saved_results['alpha'] = args.clean_loss_weight
        args.saved_results['beta'] = args.label_loss_weight
        args.saved_results['attack_interval'] = args.attack_interval
        args.saved_results['attack_epoch'] = args.attack_epoch
        args.saved_results['start_iter'] = 0  # placeholder
        args.saved_results['total_iter'] = 0  # placeholder
        # intermidate results:
        args.saved_results['local_acc_trend'] = []  # 'iter': 'acc'
        args.saved_results['local_asr_trend'] = []  # 'iter': 'asr'
        args.saved_results['victim_acc_trend'] = []  # 'iter': 'acc'
        args.saved_results['victim_asr_trend'] = []  # 'iter': 'asr'
        args.saved_results['user_acc_trend'] = []  # 'iter': 'asr' the trend the user can observe.
        args.saved_results['bit_1'] = []  # value (before, after, final) check if flip back
        args.saved_results['bit_2'] = []  # value (before, after, final) check if flip back
        args.saved_results['bit_3'] = []  # value (before, after, final) check if flip back
        args.saved_results['neuron_list'] = []
        args.saved_results['trigger_neuron_list'] = []
        args.saved_results['neuron_list_user'] = []
        args.saved_results['trigger_neuron_list_user'] = []
        args.saved_results['begin_neurons'] = 0
        args.saved_results['trigger_list'] = []
        args.saved_results['bitflip_list'] = []
        args.saved_results['inherit_slurm'] = args.inherit_slurm
        args.saved_results['user_neuron_value_list'] = []
        args.saved_results['local_neuron_value_list'] = []
        args.saved_results['acc'] = [0.0] * 4
        # asr: 'local: asr trigger, asr trigger + bf; victim:asr trigger, asr trigger + bf'
        args.saved_results['asr'] = [0.0] * 4

    for key, value in args.saved_results.items():
        print(key, value) # print hyper-parameters
    saved_path = os.path.join(os.path.curdir, 'saved_file', args.saved_results['cfg_name'])
    print(f"The final results will be saved in {saved_path}")
    # dataset configurations:
    num_class = num_class_map[args.dataset]
    img_mean = mean_map[args.dataset]
    img_std = std_map[args.dataset]
    if args.model_name == 'vit':
        print('reset args for vit')
        args.test_batch_size = 32

    attacker_model_kwargs = {
        'model_name': args.model_name,
        'num_class': num_class,
        'pretrained': True,
        'replace': True,
        'seed': args.attacker_seed,
        'device': args.device,
    }
    image_kwargs = {
        'image_size': args.image_size,
        'trigger_size': args.image_trigger_size,
        'image_mean': img_mean,
        'image_std': img_std,
        'trigger_lr': args.trigger_lr,
        'device': args.device,
        'unique_pattern': args.unique_pattern,
    }
    if args.attack_type == 'local_search': # offline attack to identify trigger and bit flips
        from deepvenom_kernel import EnsembleAttacker
        various_lr = False # when set True, user will use dynamic learning rate schedule

        loader_kwargs = {
            'train_batch_size': args.train_batch_size,
            'test_batch_size': args.test_batch_size,
            'attacker_data_percent': args.attacker_dataset_percent,
            'image_size': args.image_size,
            'target_class': args.target_class,  # used to generate target and other dataset
            'device': args.device,
            'limited_image_mode': args.limited_image_mode,
            'attacker_image_number': args.attacker_image_number,  # the total image number of attacker
            'image_number': args.image_number,  # image number for bit flip stage
            'domain_shift': args.domain_shift,
            'deterministic_run': args.deterministic_run,
        }

        print('begin ensemble attack ==>')

        if args.local_diff == 'diff': args.ensemble_num = 4

        loader_kwargs['deepvenom_transform'] = args.deepvenom_transform
        loader = AllLoader(args.dataset, **loader_kwargs)
        loader.init_loader()

        ImageRecorder = ImageManager(**image_kwargs)
        args.max_iter = loader.train_loader.__len__() * args.epoch
        args.saved_results['total_iter'] = args.max_iter

        common_config = {
            'loader': loader,
            'num_attacker': args.ensemble_num,  #
            'num_user': int(args.num_user),
            'user_ft_mode': args.user_ft_mode,
            'ImageRecorder': ImageRecorder,
            'target_attack': True,
            'target_class': args.target_class,
            'neuron_number': args.neuron_number,
            'neuron_ratio': args.neuron_ratio,
            'algorithm': [args.trigger_algo, args.select_param_algo, args.find_optim_bit_algo],
            'loss_weight': [args.clean_loss_weight, args.label_loss_weight, args.neuron_loss_weight],
            'max_iter': args.max_iter,
            'attack_epoch': args.attack_epoch,
            'attack_interval': args.attack_interval,
            'user_optimizer': args.optimizer,  #
            'lr': args.lr,  #
            'device': args.device,

            'asr_th': [args.asr_th1, args.asr_th2],
            'asr_th_mode': args.asr_th_mode,
            'num_bits_single_round': args.num_bits_single_round,
            'num_vul_params': args.num_vul_params,
            'bitflip_value_limit_mode': args.bitflip_value_limit_mode,
            'inherit_slurm': args.inherit_slurm,
            'verify_mode': args.verify_mode,
            'fixed_neuron_value': args.fixed_neuron_value,
            'user_seed': args.user_seed,
            'enable_neuron_manipulate_score': args.enable_neuron_manipulate_score,
            'tail_neuron': args.tail_neuron,
            'neuron_gama_mode': args.neuron_gama_mode,
            'trigger_random': args.trigger_random,  #
            'front_layer_bias': args.front_layer_bias,
            'async_attack': args.async_attack,
            'async_step': args.async_step,
            'new_async_attack': args.new_async_attack,
            'new_async_step': args.new_async_step,
            'rowhammer_mode': args.rowhammer_mode,
            'one_time_attack': args.one_time_attack,
            'single_bit_per_iter': args.single_bit_per_iter,
            'neuron_stop': args.neuron_stop,
            'various_lr': various_lr,
        }

        # Participiant
        seeds = [i + args.attacker_seed for i in range(args.ensemble_num)]

        model_configures = [{
            'model_name': args.model_name,
            'num_class': num_class,
            'pretrained': True,
            'replace': True,
            'seed': seed,
            'device': args.device} for seed in seeds]
        models = [map_model(**config) for config in model_configures]

        optimizers = []
        optimizers_name = []
        if args.local_diff == 'same':
            for i, model in enumerate(models):
                if not various_lr:
                    if args.attacker_optimizer == "Adam":
                        optimizers_name.append("Adam")
                        optimizers.append(torch.optim.Adam(model.parameters(), lr=args.attacker_lr, weight_decay=1e-5))
                    elif args.attacker_optimizer == "SGD":
                        optimizers_name.append("SGD")
                        optimizers.append(torch.optim.SGD(model.parameters(), lr=args.attacker_lr, momentum=0.9,
                                                          weight_decay=1e-5))
                else:
                    print('load various LR optimizer')
                    lr_base = args.attacker_lr
                    n = sum(1 for _ in model.parameters())  # Total number of parameters, not layers
                    param_groups = []
                    # Example of assigning custom LR based on parameter position (not directly feasible)
                    # This assumes each parameter is uniquely identifiable and can be mapped to a "depth" or position k
                    for k, param in enumerate(model.parameters(), 1):  # Enumerate parameters starting at 1
                        lr_k = (1 - k / n) * lr_base * 9 + lr_base
                        param_groups.append({'params': [param], 'lr': lr_k})

                    if args.attacker_optimizer == "Adam":
                        optimizers_name.append("Adam")
                        optimizers.append(torch.optim.Adam(param_groups, weight_decay=1e-5))
                    elif args.attacker_optimizer == "SGD":
                        optimizers_name.append("SGD")
                        optimizers.append(torch.optim.SGD(param_groups, momentum=0.9,
                                                          weight_decay=1e-5))
        elif args.local_diff == 'diff':
            optim_confg = [('Adam', 0.00001), ('Adam', 0.0001), ('SGD', 0.001), ('SGD', 0.0001)]
            optimizers_name = [cfg[0] for cfg in optim_confg]
            for cfg, model in zip(optim_confg, models):
                if cfg[0] == 'Adam':
                    optimizers.append(torch.optim.Adam(model.parameters(), lr=cfg[1], weight_decay=1e-5))
                elif cfg[0] == 'SGD':
                    optimizers.append(torch.optim.SGD(model.parameters(), lr=cfg[1], momentum=0.9,
                                                      weight_decay=1e-5))

        participants = []  # including several attackers and one user
        for i in range(args.ensemble_num):
            dict = {
                'model': models[i],
                'optimizer': optimizers[i],
                'optimizer_name': optimizers_name[i],
                'identity': 'attacker',
                'device': args.device,
                'seed': seeds[i],

                # results:
                'asr': [],
                'acc': [],

                'asr_trend': [],
                'acc_trend': [],  # list of tuple (iter, acc)
                'epoch_acc_trend': [],
                'cn2pfm': [],
                'pn2cfm': [],
                'lcn2pfm': [],
                'lpn2cfm': [],
                'neuron_value_list': [],

                # intermediate variables:
                'neuron_value': None,
                'neuron_gama': None,
                'tmp_acc': 0.0,
                'tmp_asr': 0.0,
                'selected_neurons': [],
                'neuron_list': [],
                'trigger_neuron_list': [],
                'running_loss': 0.0,

            }
            participants.append(dict)

        common_config['attackers'] = participants

        # Launch local attack to search bit and identify trigger
        L = EnsembleAttacker(**common_config)
        # output: intermediate results, bit-flip info, and trigger
        args.saved_results, bitflip_info, ImageRecorder = L.launch_attack(args.saved_results)

        # visualize and save trigger
        from utils import save_trigger_to_image, save_bitflip_info_to_file

        print('\n')
        save_trigger_to_image(ImageRecorder.trigger_list[-1][1], f'slurm{args.slurm_number}_trigger')
        print("\nthe identified bit flips are shown in the following:")
        for bitflip in bitflip_info:
            print(bitflip)
        print('\n')
        save_bitflip_info_to_file(bitflip_info, f'slurm{args.slurm_number}_bitflips')

    # One process for remote fine-tuning, One process for remote attack
    # Note that for the online stage, the following code is only used for simulation.
    # For the practical online attack, please check Emulation directory
    if args.attack_type == 'remote_finetune':

        # extract trigger, bit flip info from file
        if args.inherit_slurm != 'no':
            dic_path = 'saved_file'
            ImageRecorder = ImageManager(**image_kwargs)

            for file_name in os.listdir(dic_path):
                if args.inherit_slurm in file_name:
                    print(f'load attack information from {file_name}')
                    final_results = np.load(os.path.join(dic_path, file_name), allow_pickle=True).item()
                    bitflip_info = final_results['bitflip_info']
                    ImageRecorder.current_trigger = final_results['trigger_list'][-1][1]
                    ImageRecorder.trigger_list = final_results['trigger_list']
                    ImageRecorder.transmit_to_device(args.device)
                    observation_time = final_results['attack_time']

                    bit_number_for_each_round = final_results['bit_number_for_each_round']

                    print("\nthe identified bit flips are shown in the following:")
                    for bitflip in bitflip_info:
                        print(bitflip)
                    print('\n')

                    break
        else:
            print("please specify the inherit_slurm for obtaining trigger and bit flips")
            raise NotImplementedError

        # initialize data loader
        various_lr = False
        loader_kwargs = {
            'train_batch_size': args.train_batch_size,
            'test_batch_size': args.test_batch_size,
            'attacker_data_percent': args.attacker_dataset_percent,
            'image_size': args.image_size,
            'target_class': args.target_class,  # used to generate target and other dataset
            'device': args.device,
            'limited_image_mode': args.limited_image_mode,
            'attacker_image_number': args.attacker_image_number,  # the total image number of attacker
            'image_number': args.image_number,  # image number for bit flip stage
            'domain_shift': args.domain_shift,
            'deterministic_run': args.deterministic_run,
        }
        loader_kwargs['deepvenom_transform'] = args.deepvenom_transform
        loader = AllLoader(args.dataset, **loader_kwargs)
        loader.init_loader(pure=True)  # only initialize train and test loader
        args.max_iter = loader.train_loader.__len__() * args.epoch
        extended_ft = False
        if extended_ft: args.max_iter = args.max_iter * 2

        attack_configs = {
            # attack info
            'ImageRecorder': ImageRecorder,
            'bitflip_info': bitflip_info,

            # sensitivity experiments: (async attack, one time attack, and dynamic FT learning rates)
            'async_attack': args.async_attack,
            'async_step': args.async_step,
            'new_async_attack': args.new_async_attack,
            'new_async_step': args.new_async_step,
            'one_time_attack': args.one_time_attack,

            # hyper-parameters used for reporting ASR/ACC
            'observation_time': observation_time,  # specify the time to report ASR ACC
            'target_attack': True,  # used for testing ASR for target backdoor
            'target_class': args.target_class,
            'attack_epoch': args.attack_epoch,
            'max_iter': args.max_iter,
            'loader': loader,
            'device': args.device,

        }

        user_configs = {
            # # fine-tuning info
            'loader': loader,
            'user_ft_mode': args.user_ft_mode,  # defense strategies (e.g., lock exponent segment)
            'max_iter': args.max_iter,
            'user_optimizer': args.optimizer,
            'lr': args.lr,
            'various_lr': various_lr,  # decide if using dynamic LR schedule
            'device': args.device,
            'bitflip_value_limit_mode': args.bitflip_value_limit_mode,
            'verify_mode': args.verify_mode,
            'user_seed': args.user_seed,

            # hyper-parameters used for reporting ASR
            'observation_time': observation_time,  # specify the time to report ASR ACC
            'target_attack': True,  # used for testing ASR for target backdoor
            'target_class': args.target_class,
            'attack_epoch': args.attack_epoch,
        }

        # online state: user begins to fine-tune vicitm model
        user_order = 0
        if args.lr == 0.0:  # try various learning rates and repeat self.num_user times
            user_seeds = [args.user_seed + i for i in range(int(args.num_user))] * 5
            learning_rates = [0.00001, 0.00002, 0.00005, 0.0005, 0.001]
            optims = ['Adam', 'Adam', 'Adam', 'SGD', 'SGD']
        else:  # repeat self.num_user times for a single config
            user_seeds = [int(args.user_seed) + i for i in range(int(args.num_user))]
            learning_rates = [args.lr]
            optims = [args.optimizer]

        for learning_rate, optim in zip(learning_rates, optims):
            user_configs['lr'] = learning_rate
            user_configs['user_optimizer'] = optim
            for i in range(int(args.num_user)):
                cur_seed = user_seeds[user_order] # np.random.randint(0, 10000) #
                user_model_kwargs = {
                    'model_name': args.model_name,
                    'num_class': loader.num_class,
                    'pretrained': True,
                    'replace': True,
                    'device': args.device,
                    'seed':cur_seed,}
                model = map_model(**user_model_kwargs)
                model.eval()
                print(f"########### Task {loader.task} | model {model.model_name} | lr {learning_rate} | optim {optim} | seed {cur_seed} ###########")

                # Create a stop event
                stop_event = threading.Event()
                # Create a shared iteration counter
                cur_iteration = mp.Value('i', 0)  # iteration information will be sent from fine-tune process to attack process

                fine_tune_process = threading.Thread(target=fine_tune, args=(model, cur_iteration, user_configs, stop_event))
                attack_process = threading.Thread(target=attack, args=(model, cur_iteration, attack_configs, stop_event))

                fine_tune_process.start()
                attack_process.start()

                fine_tune_process.join()
                attack_process.join()

                user_order += 1 #


    torch.cuda.empty_cache()

    numpy.save(saved_path, args.saved_results)
    print(f"Results saved in {saved_path}")
    print(f"Total time cost: {(time.time() - start_time)/3600.0:.1f} hours")
