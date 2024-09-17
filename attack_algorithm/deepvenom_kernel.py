from torch import nn
import torch, os, time
import numpy as np
import copy
from utils import \
    intersection, floatToBinary32, \
    rank, binary32ToFloat, \
    change_model_weights, get_sign, verify_biteffect, \
    all_pos_neg, ensemble_ban_unstable_bit, zero_gradients
from model import map_model
from loss import LossLib


class EnsembleAttacker:
    def __init__(self, **kwargs):
        self.various_lr = kwargs['various_lr']
        self.only_final_result = True
        self.saved_results = None
        self.device = kwargs['device'] if 'device' in kwargs.keys() else 'cpu'
        self.attackers = kwargs['attackers']  # including several attacker
        self.kwargs = kwargs
        self.other_info = self.other_info_init()  # not implemented in ensemble method
        self.model = None
        self.loader = kwargs['loader']
        self.ImageRecorder = kwargs['ImageRecorder']

        self.num_attacker = kwargs['num_attacker']
        self.num_user = kwargs['num_user'] if 'num_user' in kwargs.keys() else 1

        self.user_ft_mode = kwargs['user_ft_mode'] if 'user_ft_mode' in kwargs.keys() else 'normal'
        self.user_ft_mode_data = self.init_ft_mode_data()  # function 5

        self.neuron_number = kwargs['neuron_number'] if 'neuron_number' in kwargs.keys() else None
        self.tail_neuron = True if kwargs['tail_neuron'] == 'yes' else False

        self.enable_neuron_manipulate_score = True if kwargs[
                                                          'enable_neuron_manipulate_score'] == 'yes' else False  # decide if neuron selection will consider how hard the value to change.
        self.fixed_neuron_value = True if kwargs['fixed_neuron_value'] == 'yes' else False
        self.neuron_ratio = kwargs['neuron_ratio']
        self.neuron_gama = None
        self.neuron_gama_mode = kwargs['neuron_gama_mode'] if 'neuron_gama_mode' in kwargs.keys() else None

        self.trigger_random = kwargs['trigger_random'] if 'trigger_random' in kwargs.keys() else None

        self.clean_neuron_gama = None
        self.neuron_stop = kwargs['neuron_stop'] if 'neuron_stop' in kwargs.keys() else None
        self.gama = None

        self.loss_lib = LossLib(kwargs['algorithm'], kwargs['loss_weight'])  # clean, label, neuron

        self.user_optim = kwargs['user_optimizer']
        self.lr = kwargs['lr']
        # self.attacker_lr = kwargs['attacker_lr']
        self.target_class = kwargs['target_class']

        self.max_iter = kwargs['max_iter']
        self.attack_epoch = kwargs['attack_epoch']
        self.attack_interval = kwargs['attack_interval']
        self.num_bits_single_round = kwargs['num_bits_single_round']
        self.only_ban_last_layer = 'yes'
        self.num_vul_params = kwargs['num_vul_params']
        self.single_bit_per_iter = kwargs['single_bit_per_iter']
        self.bitflip_value_limit_mode = kwargs['bitflip_value_limit_mode']
        self.inherit_slurm = kwargs['inherit_slurm']

        self.asr_th = kwargs['asr_th']
        self.asr_th_mode = kwargs['asr_th_mode']
        self.asr_flag = None
        # self.image_number = kwargs['image_number']
        self.verify_mode = kwargs['verify_mode']
        self.user_seed = kwargs['user_seed']
        # self.attacker_seed = kwargs['attacker_seed']
        self.front_layer_bias = True if kwargs['front_layer_bias'] == 'yes' else False
        self.control = self.init_control()

        self.async_attack = kwargs['async_attack']
        self.async_step = kwargs['async_step']
        self.new_async_attack = kwargs['new_async_attack']
        self.new_async_step = kwargs['new_async_step']

        self.one_time_attack = kwargs['one_time_attack']
        # print(f'aysnc_epoch: {self.async_step}')

        # init necessary component:

        self.current_iter = 0
        self.attack_time = self.attack_time_init()  #
        self.current_round_bf = []
        self.user_neuron = None
        self.start_iter = self.loader.train_loader.__len__() * (self.attack_epoch - 1)
        self.rowhammer_page = self.rowhammer_page_init()
        self.rowhammer_mode = kwargs['rowhammer_mode']

        # intermediate results and final resutls:
        self.tail_neuron_idx = 0.0
        self.confidence_score = 0.0  # how confident the backdoor is on local
        self.tmp_asr = 0.0
        self.tmp_acc = 0.0
        self.acc_history = []
        self.bitflip_info = []
        self.fm_value = []
        self.begin_neurons = None

        self.neuron_list = []
        self.trigger_neuron_list = None
        self.neuron_list_user = None
        self.trigger_neuron_list_user = None
        self.user_salient_neuron_mean_user_trigger = []  # user's average neuron value at local selected neuron location
        self.user_salient_neuron_mean_user = []  # user's average neuron value at user selected neuron location
        self.user_salient_neuron_mean_unrelate = []

        self.local_neuron_value_list = []
        self.user_neuron_value_list = []

        self.bit_number_for_each_round = []
        self.bitflip_list = []

        self.local_asr_trend = []
        self.local_acc_trend = []
        self.local_epoch_acc_trend = []
        self.victim_asr_trend = []
        self.victim_acc_trend = []
        self.user_acc_trend = []
        self.loss_reduction = [[] for i in range(len(self.attackers))]

        self.bit_1 = []
        self.bit_2 = []
        self.bit_3 = []

        self.acc = [0, 0, 0, 0]
        self.asr = [0, 0, 0, 0]
        if self.lr == 0.0:
            self.users_data = [{} for i in range(self.num_user * 5)]
        else:
            self.users_data = [{} for i in range(self.num_user)]
        self.attackers_data = [{} for i in range(len(self.attackers))]

    def launch_attack(self, saved_results):

        self.saved_results = saved_results

        #offline stage: launch local attack or use preivous results
        if self.inherit_slurm == 'no':
            # launch local attack on substitute models
            self.ensemble_train(self.attackers, self.loader.attacker_train_loader, self.loader.attacker_test_loader) #

            # save attacker's results
            self.ImageRecorder.transmit_to_device('cpu')
            self.saved_results.update(self.report('attacker'))
            self.record_attackers_data()
            self.saved_results['attackers_data'] = self.attackers_data
            self.ImageRecorder.transmit_to_device(self.device)
        else:
            # else use previous attack results on substitute models
            dic_path = 'saved_file'
            for file_name in os.listdir(dic_path):
                if self.inherit_slurm in file_name and self.attackers[0]['model'].model_name in file_name \
                        and self.loader.task in file_name:
                    print(file_name)
                    final_results = np.load(os.path.join(dic_path, file_name), allow_pickle=True).item()
                    self.bitflip_info = final_results['bitflip_info']
                    self.ImageRecorder.current_trigger = final_results['trigger_list'][-1][1]
                    self.ImageRecorder.trigger_list = final_results['trigger_list']
                    # self.ImageRecorder.transmit_to_device('cpu')
                    self.ImageRecorder.transmit_to_device(self.device)
                    self.attack_time = final_results['attack_time']
                    self.begin_neurons = final_results['begin_neurons']
                    self.local_acc_trend = final_results['local_acc_trend']
                    self.local_asr_trend = final_results['local_asr_trend']
                    self.local_epoch_acc_trend = final_results[
                        'local_epoch_acc_trend'] if 'local_epoch_acc_trend' in final_results.keys() else []
                    self.neuron_list = final_results['neuron_list']
                    self.trigger_neuron_list = final_results['trigger_neuron_list']
                    self.bit_number_for_each_round = final_results['bit_number_for_each_round']
                    self.other_info = final_results['other_info']
                    self.other_info['enable'] = False
                    self.other_info['cn2pfm'] = []
                    self.other_info['pn2cfm'] = []
                    if self.bitflip_info[0]['bitflip'][0] == 9:
                        flag64 = True
                        print('convert to float32 bitflip')
                    else: flag64 = False

                    if flag64:
                        tmp = []
                        for bitflip in self.bitflip_info:
                            flip_info = (bitflip['bitflip'][0] - 3, bitflip['bitflip'][1])
                            bitflip['bitflip'] = flip_info
                            print(bitflip)
                            tmp.append(copy.deepcopy(bitflip))
                        self.bitflip_info = copy.deepcopy(tmp)
                    else:
                        for bitflip in self.bitflip_info:
                            print(bitflip)


                    break

            # save attacker's results
            self.ImageRecorder.transmit_to_device('cpu')
            self.saved_results.update(self.report('attacker'))
            self.record_attackers_data()
            self.saved_results['attackers_data'] = self.attackers_data
            self.ImageRecorder.transmit_to_device(self.device)


        result_save_path = os.path.join(os.path.curdir, 'saved_file', self.saved_results['cfg_name'])
        np.save(result_save_path, self.saved_results)

        return self.saved_results, self.bitflip_info, self.ImageRecorder

    def init_ft_mode_data(self):
        if self.user_ft_mode == 'limit_value':
            model = self.attackers[0]['model']
            dct_min = {}
            dct_max = {}
            for name, param in model.named_parameters():
                dct_min[name] = torch.min(param.data).to(self.device)
                dct_max[name] = torch.max(param.data).to(self.device)
            print('defense clamp: ')
            for key in dct_min.keys():
                print(f'{dct_min[key]} to {dct_max[key]}')
            return [dct_min, dct_max]
        elif self.user_ft_mode == 'lock_exp':
            path = 'saved_model'
            cfg_name = 'lock_exp_' + self.model.model_name + '.pt'
            cur_file = os.path.join(path, cfg_name)
            if os.path.exists(cur_file):
                lst = torch.load(cur_file)
            else:
                dict_min = {}
                dict_max = {}

                def get_min_max_with_lock_exp(value, mode='min'):
                    bin_pre = floatToBinary32(value)
                    bin1 = bin_pre[:9] + '0' * 23
                    bin2 = bin_pre[:9] + '1' * 23
                    float1 = binary32ToFloat(bin1)
                    float2 = binary32ToFloat(bin2)
                    if float1 < float2:
                        return float1, float2
                    else:
                        return float2, float1

                def get_tensor_min_max_with_lock_exp(tensor):
                    min_tensor = torch.zeros(tensor.size()).view(-1, )
                    max_tensor = torch.zeros(tensor.size()).view(-1, )
                    for i in range(tensor.view(-1, ).size()[0]):
                        min_tensor[i], max_tensor[i] = get_min_max_with_lock_exp(tensor.view(-1, )[i].item())
                    return min_tensor.view_as(tensor), max_tensor.view_as(tensor)

                for name, param in self.model.named_parameters():
                    dict_min[name], dict_max[name] = get_tensor_min_max_with_lock_exp(param.data)
                lst = [dict_min, dict_max]
                torch.save(lst, cur_file)
            cuda_lst = []
            for dct in lst:
                for key in dct.keys():
                    dct[key] = dct[key].to(self.device)
                cuda_lst.append(dct)

            return cuda_lst
        else:
            return None

    def init_control(self):

        control = {
            'record_fm': False,  # mmd feature map recording
            'use_mmd': False,  # mmd loss
            'full_fm': False,  # use neuron or full fm
            'contrastive_fm': False,  # use constrative or not in full fm
            'extra_classifier': False,  # using extra trained classifier to find the invariant of attack
            'extended_ft': False,
        }


        return control

    def attack_time_init(self):
        if self.control['extended_ft']:
            self.max_iter = self.max_iter * 2
            print(f'extented ft iterations: {self.max_iter}')
        self.ft_epoch = int(self.max_iter / self.loader.train_loader.__len__())

        start_iter = self.loader.train_loader.__len__() * (self.attack_epoch - 1)
        attack_time = np.arange(start_iter, self.max_iter, self.attack_interval)
        if len(attack_time) > 0:
            attack_time[0] = attack_time[0] + 1
        else:
            attack_time = [1000000]  # no attack will happen
        if self.verify_mode:
            return attack_time[:2]
        if self.one_time_attack == 'local_middle':
            attack_time = [attack_time[0]]
        if self.one_time_attack == 'le_rs':
            attack_time = [attack_time[-1]]
        print(f'attack time {attack_time}')

        return attack_time

    def optim_init(self, identity):
        if not self.various_lr:
            if identity == 'attacker':
                if self.attacker_optim == "Adam":
                    return torch.optim.Adam(self.model.parameters(), lr=self.attacker_lr, weight_decay=1e-5)
                elif self.attacker_optim == "SGD":
                    return torch.optim.SGD(self.model.parameters(), lr=self.attacker_lr, momentum=0.9, weight_decay=1e-5)
            elif identity == 'user':
                if self.user_optim == "Adam":
                    return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
                elif self.user_optim == "SGD":
                    return torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-5)
        else:
            print('load various LR optimizer')
            if identity == 'attacker':
                lr_base = self.attacker_lr
            else:
                lr_base = self.lr
                # Count the total number of layers to compute 'n'
            n = sum(1 for _ in self.model.parameters())  # Total number of parameters, not layers
            param_groups = []

            # Example of assigning custom LR based on parameter position (not directly feasible)
            # This assumes each parameter is uniquely identifiable and can be mapped to a "depth" or position k
            for k, param in enumerate(self.model.parameters(), 1):  # Enumerate parameters starting at 1
                lr_k = (1 - k / n) * lr_base * 9 + lr_base
                param_groups.append({'params': [param], 'lr': lr_k})


            if identity == 'attacker':
                if self.attacker_optim == "Adam":
                    return torch.optim.Adam(param_groups, weight_decay=1e-5)
                elif self.attacker_optim == "SGD":
                    return torch.optim.SGD(param_groups, momentum=0.9, weight_decay=1e-5)
            elif identity == 'user':
                if self.user_optim == "Adam":
                    return torch.optim.Adam(param_groups, weight_decay=1e-5)
                elif self.user_optim == "SGD":
                    return torch.optim.SGD(param_groups, momentum=0.9, weight_decay=1e-5)

    def other_info_init(self):
        p = {
            'cn2pfm': [],
            'pn2cfm': [],
            'lcn2pfm': [],
            'lpn2cfm': [],
            'enable': False
        }
        print(f'other info: {p}')

        return p

    def test(self, model, test_loader, epoch, trigger=None, target_attack=True, use_trigger=False):
        model.eval()
        count = 0
        criterion = nn.CrossEntropyLoss()
        running_loss = 0.0
        if not target_attack:
            trigger = self.ImageRecorder.current_trigger
            running_corrects = 0.0
            model.eval()
            all_corrects = [0.0 for i in range(self.loader.num_class)]
            with torch.no_grad():
                ll = []
                for i, data in enumerate(test_loader):
                    inputs, labels = data[0].to(self.device), data[1].to(self.device)

                    poison_batch_image = self.ImageRecorder.sythesize_poison_image(inputs, trigger)

                    outputs = model(poison_batch_image)
                    _, preds = torch.max(outputs, 1)
                    ll.extend(preds.tolist())
                    running_corrects += torch.sum(preds == labels.data)
                    for j in range(self.loader.num_class):
                        tmp_labels = (torch.ones(test_loader.batch_size, dtype=torch.int64) * j).to(
                            self.device)
                        all_corrects[j] += torch.sum(preds == tmp_labels)
                    count += inputs.size(0)
                # if verbose: print("The real output using trigger: {}".format(ll))
                epoch_acc = running_corrects.double() / count
                epoch_large_acc = max(all_corrects).double() / count
                target = all_corrects.index(max(all_corrects))
                self.acc_history.append(epoch_acc)
                epoch_acc = 1.0 - epoch_acc
                print("Epoch {:<5} ACC_Untargeted: {:.2f}%".format(epoch, epoch_acc * 100))
                print("Epoch {:<5} ACC_targeted: {:.2f}%, target: {}".format(epoch, epoch_large_acc * 100, target))
            return epoch_acc.item()

        if not use_trigger:
            running_corrects = 0.0
            model.eval()
            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    inputs, labels = data[0].to(self.device), data[1].to(self.device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    running_corrects += torch.sum(preds == labels.data)
                    count += inputs.size(0)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item() * inputs.size(0)
                epoch_acc = running_corrects / count
                epoch_loss = running_loss / count
                self.acc_history.append(epoch_acc)
                print("Epoch {:<5} ACC: {:.2f}% Loss: {:.2f}".format(epoch, epoch_acc * 100, epoch_loss))

        else:
            m = nn.Softmax(dim=1)
            running_corrects = 0.0
            model.eval()
            confidence = 0.0
            with torch.no_grad():
                cur_trigger = self.ImageRecorder.get_recent_trigger(self.current_iter)
                target_labels = (torch.ones(test_loader.batch_size, dtype=torch.int64) * self.target_class).to(
                    self.device)
                for i, data in enumerate(test_loader):
                    inputs, labels = data[0].to(self.device), data[1].to(self.device)
                    poison_batch_image = self.ImageRecorder.sythesize_poison_image(inputs, cur_trigger)

                    outputs = model(poison_batch_image)
                    tar_logits = m(outputs)[:, self.target_class]

                    _, preds = torch.max(outputs, 1)
                    running_corrects += torch.sum(preds == target_labels)
                    confidence += torch.sum(tar_logits)
                    count += inputs.size(0)
                    loss = criterion(outputs, target_labels)
                    running_loss += loss.item() * inputs.size(0)

                epoch_acc = running_corrects.double() / count
                self.confidence_score = confidence / count
                epoch_loss = running_loss / count
                self.acc_history.append(epoch_acc)
                print("Epoch {:<5} ASR: {:.2f}% Loss: {:.2f} confidence score {:.2f}".format(epoch, epoch_acc * 100,
                                                                                             epoch_loss,
                                                                                             self.confidence_score))

        return epoch_acc.item()  # "{:.2f}".format(100*epoch_acc.item())

    def ensemble_train(self, attackers, train_loader, test_loader):
        print('ensemble train ==>')
        async_step = 0  # sync attack

        self.current_iter = 0
        current_epoch = 1
        for identity in attackers:
            identity['model'].train()
        criterion = nn.CrossEntropyLoss()
        verbose = 1
        if self.loader.limited_image_mode:
            verbose = 10
        train_loaders = [copy.deepcopy(train_loader) for i in range(len(attackers))]

        its = [iter(train_loader) for train_loader in train_loaders]

        # begin substitute model fine-tuning
        while self.current_iter < self.max_iter:
            if (self.current_iter % len(train_loader) == 0):
                print("*" * 100)
                print(f"Iter: {self.current_iter}/{self.max_iter} Epoch {current_epoch}")

            self.current_iter += 1
            for j, attacker in enumerate(attackers):
                try:
                    data = next(its[j])
                except StopIteration:
                    its[j] = iter(train_loaders[j])
                    data = next(its[j])

                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                model = attacker['model']
                model.train()  # important steps
                model.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                attacker['optimizer'].step()
                attacker['running_loss'] += loss.item() * inputs.size()[0]

                # Signature Neuron Selection
                if (self.current_iter - async_step in self.attack_time):
                    if j == 0:
                        print(f"######################## {self.current_iter}th iteration attack ########################")
                        print(f"for the {len(self.bitflip_info) + 1}th bit flips: ")
                        print(f"use {self.loader.dataset_target.__len__() + self.loader.dataset_others.__len__()}/"
                              f"{self.loader.total_number_of_train_images} to select neuron")

                    print(f'neuron selection for attacker {j} ==>')
                    # large neuron selection
                    attacker['selected_neurons'] = self.signature_neuron_selection(attacker,
                                                                                 ratio=self.neuron_ratio)
                    attacker['neuron_list'].append((self.current_iter, attacker['selected_neurons']))

                    if self.current_iter != self.attack_time[0]:
                        attacker['tmp_asr'] = self.test(model, test_loader, current_epoch, use_trigger=True)
                        attacker['tmp_acc'] = self.test(model, test_loader, current_epoch)
                        attacker['asr_trend'].append((self.current_iter, attacker['tmp_asr']))
                        attacker['acc_trend'].append((self.current_iter, attacker['tmp_acc']))

                # Transferable Trigger Generation
                if (self.current_iter - async_step in self.attack_time) and j == len(attackers) - 1:
                    print(f'ensemble trigger generation ==>')
                    asr_list = [participant['tmp_asr'] for participant in attackers]
                    tmp_asr = np.mean(asr_list) if self.asr_th_mode == 'mean' else np.min(asr_list)
                    print(f'asr before trigger generation: {tmp_asr}')

                    if tmp_asr > self.asr_th[0]:  # args.current_iter == args.attack_time[0]:
                        self.asr_flag = 'no'
                        print(f"do not update trigger at iteration {self.current_iter}")
                    else:
                        self.asr_flag = 'yes'
                        if self.loader.use_all_data_for_search:
                            self.ImageRecorder.current_trigger = self.transferable_trigger_generation(attackers, self.loader.bit_search_data_loader)
                        else:
                            self.ImageRecorder.current_trigger = self.transferable_trigger_generation(attackers, self.loader.bit_search_data_loader)

                    # torch.save(args.image_trigger, trigger_save_path)
                    self.ImageRecorder.trigger_list.append(
                        (self.current_iter, torch.clone(self.ImageRecorder.current_trigger.detach())))

                    if self.trigger_neuron_list is not None:
                        raise NotImplementedError
                        trigger_neurons = self.signature_neuron_selection(model, ratio=self.neuron_ratio,
                                                                        use_trigger=True)
                        self.trigger_neuron_list.append((self.current_iter, trigger_neurons))

                    # self.local_neuron_value_list.append(self.neuron_value)

                    if self.asr_flag == 'yes':
                        for attacker in attackers:
                            attacker['tmp_asr'] = self.test(attacker['model'], self.loader.attacker_test_loader,
                                                            current_epoch, use_trigger=True)
                    else:
                        for attacker in attackers:
                            print("Epoch {}   ASR: {:.2f} Loss: unknown".format((current_epoch), attacker['tmp_asr']))

                    for attacker in attackers:
                        attacker['asr_trend'].append((self.current_iter, attacker['tmp_asr']))
                        attacker['acc_trend'].append((self.current_iter, attacker['tmp_acc']))

                # Transferable Bit Flip Identification
                if (self.current_iter - async_step in self.attack_time) and j == len(attackers) - 1:

                    bitflip_info = self.critical_bit_search(attackers, self.ImageRecorder.current_trigger,
                                                            test_loader)
                    self.bitflip_list.append((self.current_iter, bitflip_info))

                    for attacker in attackers:
                        attacker['asr_trend'].append((self.current_iter, attacker['tmp_asr']))
                        attacker['acc_trend'].append((self.current_iter, attacker['tmp_acc']))

                    # suspend attack time.
                    for bitflip in bitflip_info:
                        bitflip['iter'] = self.current_iter

                    self.bitflip_info.extend(bitflip_info)
                    print(f"bitflip_info {bitflip_info}")
                    # print(f"bitflip_info_list {self.bitflip_info}")

                    # lcn2pfm, lpn2cfm (other related information for debugging/understanding the attack, will not use anymore)
                    if self.other_info['enable']:
                        all_neuron = [i for i in range(self.model.num_ftrs)]
                        other_neuron = [ele for ele in all_neuron if (ele not in self.selected_neurons)]

                        keep_normal_neuron_list = {
                            'local': self.selected_neurons,
                            'other': other_neuron
                        }
                        ASR0, ASR1 = self.special_test(model, test_loader, self.current_iter, keep_normal_neuron_list)
                        self.other_info['lcn2pfm'].append(ASR0)
                        self.other_info['lpn2cfm'].append(ASR1)

                    print(
                        f"################################ {self.current_iter}th iteration end  ############################## ")

                # Assume the attacker knows the user's defense strategies, and try to implement an advanced DeepVenom (e.g., truncate value)
                self.defense_check(model)


            if (self.current_iter % self.loader.train_loader.__len__() == 0):
                for attacker in attackers:
                    epoch_loss = attacker['running_loss'] / len(train_loader.dataset)
                    print("Epoch {:<5} Train loss: {:.4f}".format(current_epoch, epoch_loss))
                    if current_epoch % verbose == 0:
                        if self.current_iter - async_step > self.attack_time[0]:
                            attacker['tmp_asr'] = self.test(attacker['model'], test_loader, current_epoch,
                                                            use_trigger=True)
                            if self.control['extended_ft']:
                                attacker['asr_trend'].append((self.current_iter, attacker['tmp_asr']))

                        attacker['tmp_acc'] = self.test(attacker['model'], test_loader, current_epoch)

                        attacker['epoch_acc_trend'].append((self.current_iter, attacker['tmp_acc']))
                current_epoch += 1

        for attacker in attackers:
            attacker['acc'] = attacker['tmp_acc']
            attacker['asr'] = attacker['tmp_asr']

    def special_test(self, model, test_loader, epoch, keep_normal_neuron_list):
        model.eval()
        count = 0
        criterion = nn.CrossEntropyLoss()

        tasks_number = len(keep_normal_neuron_list)
        running_losses0, running_losses1 = [0.0] * tasks_number, [0.0] * tasks_number
        ASR0, ASR1 = [], []
        ASR0.append(self.current_iter)
        ASR1.append(self.current_iter)

        def fm_seam_function(fm0, fm1, neuron_idx):
            # the order of fm0 and fm1 matters!
            assert fm0.size() == fm1.size()
            tmp = torch.clone(fm1.view(fm1.size()[0], -1)[:, neuron_idx].detach())
            returned_fm = torch.clone(fm0.detach())
            returned_fm.view(fm0.size()[0], -1)[:, neuron_idx] = tmp
            # fm0.data.view(fm0.size()[0], -1)[:, neuron_idx] = tmp
            return returned_fm

        m = nn.Softmax(dim=1)
        running_corrects_all0, running_corrects_all1 = [0.0] * tasks_number, [0.0] * tasks_number
        model.eval()
        confidences0, confidences1 = [0.0] * tasks_number, [0.0] * tasks_number
        with torch.no_grad():
            cur_trigger = self.ImageRecorder.get_recent_trigger(self.current_iter)

            target_labels = (torch.ones(test_loader.batch_size, dtype=torch.int64) * self.target_class).to(
                self.device)
            for i, data in enumerate(test_loader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                poison_batch_image = self.ImageRecorder.sythesize_poison_image(inputs, cur_trigger)

                _, fm_poison = model(poison_batch_image, latent=True)
                _, fm_clean = model(inputs, latent=True)
                # fm_seams = [[] for i in range(tasks_number)]
                for i, (key, neuron) in enumerate(keep_normal_neuron_list.items()):
                    fm_seam0 = fm_seam_function(fm_poison, fm_clean, neuron)
                    fm_seam1 = fm_seam_function(fm_clean, fm_poison, neuron)
                    # print('---')
                    # print(torch.equal(fm_seam0, fm_seam1))
                    # print(torch.equal(fm_seam1, fm_clean))
                    # print(torch.equal(fm_seam1, fm_poison))
                    outputs0 = self.model.last_layer(fm_seam0)
                    outputs1 = self.model.last_layer(fm_seam1)

                    tar_logits0 = m(outputs0)[:, self.target_class]
                    tar_logits1 = m(outputs1)[:, self.target_class]

                    _, preds0 = torch.max(outputs0, 1)
                    _, preds1 = torch.max(outputs1, 1)

                    running_corrects_all0[i] += torch.sum(preds0 == target_labels)
                    running_corrects_all1[i] += torch.sum(preds1 == target_labels)

                    confidences0[i] += torch.sum(tar_logits0)
                    confidences1[i] += torch.sum(tar_logits1)

                    loss0 = criterion(outputs0, target_labels)
                    loss1 = criterion(outputs1, target_labels)
                    running_losses0[i] += loss0.item() * inputs.size(0)
                    running_losses1[i] += loss1.item() * inputs.size(0)
                count += inputs.size(0)
            print(f'add normal neuron to poison fm')
            for i, (key, neuron) in enumerate(keep_normal_neuron_list.items()):
                epoch_acc = running_corrects_all0[i].double() / count
                self.confidence_score = confidences0[i] / count
                epoch_loss = running_losses0[i] / count
                ASR0.append(epoch_acc.item())
                # self.acc_history.append(epoch_acc)
                print("Task: {:<35}, Epoch {:<5} ASR: {:.2f}% Loss: {:.2f} confidence score {:.2f}".format(key, epoch,
                                                                                                           epoch_acc * 100,
                                                                                                           epoch_loss,
                                                                                                           self.confidence_score))

            print(f'add poison neuron to clean fm')
            for i, (key, neuron) in enumerate(keep_normal_neuron_list.items()):
                epoch_acc = running_corrects_all1[i].double() / count
                self.confidence_score = confidences1[i] / count
                epoch_loss = running_losses1[i] / count
                ASR1.append(epoch_acc.item())
                # self.acc_history.append(epoch_acc)
                print("Task: {:<35}, Epoch {:<5} ASR: {:.2f}% Loss: {:.2f} confidence score {:.2f}".format(key, epoch,
                                                                                                           epoch_acc * 100,
                                                                                                           epoch_loss,
                                                                                                           self.confidence_score))

        return ASR0, ASR1  # "{:.2f}".format(100*epoch_acc.item())

    def signature_neuron_selection(self, participant, use_trigger=False, ratio=1.0, probabilistic=False):
        start_time = time.time()
        model = participant['model']
        # print(f"use {self.loader.dataset_target.__len__() + self.loader.dataset_others.__len__()}/{self.loader.total_number_of_train_images} to select neuron")
        if (self.verify_mode) or self.control['record_fm']:
            participant['neuron_value'] = 10.0
            self.tail_neuron_idx = [i for i in range(20)]
            return [i for i in range(100)]

        # large difference from other classes, small std within target class.
        if probabilistic: print(f"using probablistic salient neuron")

        model.eval()
        kargs = {
            'use_trigger': use_trigger,
            'function': self.ImageRecorder.sythesize_poison_image,
            'trigger': self.ImageRecorder.get_recent_trigger(self.current_iter)
        }

        mean_target, std_target = self.loader.get_mean_std_fm(self.loader.dataset_target, model, probabilistic, **kargs)

        mean_others, std_others = self.loader.get_mean_std_fm(self.loader.dataset_others, model, probabilistic, **kargs)

        if use_trigger:
            mean_diff = (mean_target * len(self.loader.dataset_target) + mean_others * len(self.loader.dataset_others)) \
                        / (len(self.loader.dataset_target) + len(self.loader.dataset_others))
        else:
            mean_diff = mean_target - mean_others
        probab_diff = std_target - std_others  # std is probab

        # mean_diff[mean_diff < 0.0] = 0.0
        if probabilistic:
            multiscore = torch.div(probab_diff, 1.0)  # std_small
        else:
            multiscore = torch.div(mean_diff, 1.0)  # std_small

        if self.control['full_fm']:
            if self.control['contrastive_fm']:
                size = multiscore.size()
                tmp = torch.clone(multiscore.detach())
                self.control['mean_fm'] = tmp
            else:
                self.control['mean_fm'] = torch.clone(multiscore.detach())

        value_reverse, key_reverse = torch.topk(multiscore.view(-1, ), 20, largest=False)  # reverse_neuron_number
        self.tail_neuron_idx = key_reverse.data.cpu().numpy().copy().reshape(-1, )

        if self.enable_neuron_manipulate_score:
            self.model.compute_neuron_score()
            multiscore = multiscore.view(-1, ) * self.model.neuron_score

        value, key = torch.topk(multiscore.view(-1, ), self.neuron_number)

        value_t, key_t = torch.topk(mean_target.view(-1, ), self.neuron_number)

        # set neuron values
        neuron_value = value_t[0].item() * ratio
        if (not use_trigger) and (
                not self.fixed_neuron_value):  # when testing trigger salient neuron, do not update neuron value
            # (value_t[0].item() + value_t[-1].item()) * ratio  # value_t[0].item() #
            participant['neuron_value'] = neuron_value
        if self.fixed_neuron_value and self.current_iter == self.attack_time[0]:
            participant['neuron_value'] = neuron_value
            # print(f"the top20 neuron value {value_t[:20]}")

        print(f"fix neuron value: {participant['neuron_value']:.1f}; "
              f"current neuron value: ({value_t[0].item():.1f} + {value_t[-1].item():.1f}) * {ratio:.1f}")

        indices = key.data.cpu().numpy().copy().reshape(-1, )
        candidates = []
        for ele in indices:
            candidates.append(ele)
        print("Selected Neuron Length: {}: Index: {}".format(len(candidates), candidates[:10]))
        # print(f"mean of last layer: {torch.mean(torch.abs(model.last_layer.weight.view(-1, )))}")
        # print(f'neuron selection time: {time.time() - start_time}')
        return candidates

    def transferable_trigger_generation(self, attackers, data_loader):
        print(f"use {data_loader.dataset.__len__()}/"
              f"{self.loader.total_number_of_train_images}"
              f" images to generate trigger")

        start_time = time.time()
        for attacker in attackers:
            attacker['model'].eval()

        current_trigger = torch.clone(self.ImageRecorder.current_trigger.detach())
        if attackers[0]['model'].model_name == 'vggface':
            print(f"reset trigger lr to 1.0 for vggface")
            self.ImageRecorder.trigger_lr = 1.0 # vggface's image value is in [0, 255], others normalized
        optimizer = torch.optim.Adam([{'params': current_trigger}], lr=self.ImageRecorder.trigger_lr, betas=(0.5, 0.9))
        current_trigger.requires_grad = True
        epoch = 0
        if self.verify_mode == True: epoch = 780
        verbose = 100

        print("*" * 100)
        print(f"trigger generation loader length {len(data_loader)}; batch size {data_loader.batch_size}")

        prev_loss = float('inf')  # initialize with a very large number
        no_decrease_count = 0  # counter to track consecutive times loss didn't decrease
        clamp_min, clamp_max = self.ImageRecorder.clamp_min, self.ImageRecorder.clamp_max
        max_epoch = 500

        while True:
            epoch += 1
            running_loss = 0.0
            loss_1_total = 0.0
            loss_2_total = 0.0

            for i, data in enumerate(data_loader):
                total_loss = torch.tensor(0.0).to(self.device)
                seperate_loss = [0.0] * len(attackers)
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                optimizer.zero_grad()
                for c in range(3):
                    current_trigger.data[c].clamp_(clamp_min[c].item(), clamp_max[c].item())

                poison_batch_inputs = self.ImageRecorder.sythesize_poison_image(inputs, current_trigger)

                for j, attacker in enumerate(attackers):
                    y_pre, fm = attacker['model'](poison_batch_inputs, latent=True)
                    clean_loss, label_loss, neuron_loss, loss = self.loss_lib.trigger_loss(y_pre, fm, self.target_class,
                                                                                           self.device,
                                                                                           attackers[j]['neuron_gama'],
                                                                                           attackers[j]['selected_neurons'],
                                                                                           attackers[j]['neuron_value'])

                    # Accumulate the loss
                    total_loss += loss
                    running_loss += loss.item()
                    loss_1_total += neuron_loss.item()
                    loss_2_total += label_loss.item()
                    seperate_loss[j] += loss.item()
                    del y_pre, fm
                total_loss.backward()
                current_trigger.grad = current_trigger.grad / (len(attackers))

                optimizer.step()

            if epoch % verbose == 0: self.loss_lib.print_loss_results(len(data_loader) * len(attackers))

            if running_loss >= prev_loss: no_decrease_count += 1
            else: no_decrease_count = 0
            prev_loss = running_loss
            if no_decrease_count >= 4:
                print(f"trigger generation loss didn't decrease for {no_decrease_count} times, stop training at epoch {epoch}")
                break
            if epoch >= max_epoch:
                print(f"trigger generation reach max epoch {max_epoch}, stop training")
                break


        print("*" * 100)

        image_trigger = self.ImageRecorder.clamp(current_trigger).to(self.device)
        print(f"trigger generation time {time.time() - start_time}")

        return copy.deepcopy(image_trigger.detach())

    def critical_bit_search(self, participants, trigger, test_loader):

        print(f"use {self.loader.bit_search_data_loader.dataset.__len__()}/"
              f"{self.loader.total_number_of_train_images}"
              f" images to search bit")
        nb = 0
        bitflip_info = []

        print("*" * 100)
        # Search the most vulnerable element in given layer.

        self.current_round_bf = []

        if self.asr_flag == 'yes':
            tmp_asr = 0.0

            # set the max number of bit flips in a single round (e.g., 20)
            bit_numer = 2 if self.verify_mode else 100
            bit_numer = bit_numer if self.one_time_attack == 'no' else 20


            judge1 = tmp_asr <= self.asr_th[1]

            if self.neuron_stop == 'no':
                judge2 = False # always select this for DeepVenom.
            else:
                salient_value = 0.0
                expected_value = 0.0
                for participant in participants:
                    expected_value += participant['neuron_value']
                    p1, p2 = self.get_cur_poison_value(participant['model'], self.loader.bit_search_data_loader,
                                                       participant['selected_neurons'])
                    salient_value += p1
                salient_value /= len(participants)
                expectation_neuron_ratio = salient_value / expected_value
                judge2 = expectation_neuron_ratio <= 0.85

            while (judge1 or judge2) and len(self.current_round_bf) < bit_numer * self.num_bits_single_round:

                # identify vulnerable parameters at weight level
                psense_list = self.select_vul_param2(participants, self.loader.bit_search_data_loader)

                # try to flip all the potential bits in identified parameters, choose a single bit flip that induce the minimum deepvenom loss value
                bitflip = self.transferable_bit_flip_identification(participants, psense_list,
                                                                self.loader.bit_search_data_loader,
                                                                trigger, test_loader)
                if isinstance(bitflip, list):
                    self.current_round_bf.extend(bitflip)
                else:
                    self.current_round_bf.append(bitflip)
                asr_list = [participant['tmp_asr'] for participant in participants]

                tmp_asr = np.mean(asr_list) if self.asr_th_mode == 'mean' else np.min(asr_list)

                judge1 = tmp_asr <= self.asr_th[1]

                if self.neuron_stop == 'no':
                    judge2 = False
                    for participant in participants: # intermediate results, will not report at the end
                        p1, p2 = self.get_cur_poison_value(participant['model'], self.loader.bit_search_data_loader,
                                                           participant['selected_neurons'], verbose=True)
                else:
                    salient_value = 0.0
                    expected_value = 0.0
                    for participant in participants:
                        expected_value += participant['neuron_value']
                        p1, p2 = self.get_cur_poison_value(participant['model'], self.loader.bit_search_data_loader,
                                                           participant['selected_neurons'], verbose=True)
                        salient_value += p1
                    salient_value /= len(participants)
                    expectation_neuron_ratio = salient_value / expected_value
                    judge2 = expectation_neuron_ratio <= 0.9

                if self.single_bit_per_iter == 'yes':
                    break

        bitflip_info.extend(self.current_round_bf)
        self.bit_number_for_each_round.append(len(self.current_round_bf))
        nb += len(self.current_round_bf)


        return bitflip_info

    def select_vul_param(self, participants, data_loader):
        # same as vul_param2 function, but it considers 3 exponent bit offsets
        start_time = time.time()
        for participant in participants:
            participant['model'].eval()

        # Generate Mask
        grad_dicts = []
        grad_dict = {}

        for participant in participants:
            for i, (name, param) in enumerate(participant['model'].named_parameters()):
                grad_dict[name] = 0.0
            grad_dicts.append(copy.deepcopy(grad_dict))
            participant['model'].requires_grad = True
            zero_gradients(participant['model'])

        for cur_order, attacker in enumerate(participants):
            model = attacker['model']
            neuron_value = attacker['neuron_value']
            selected_neurons = attacker['selected_neurons']
            neuron_gama = attacker['neuron_gama']
            for i, data in enumerate(data_loader):
                # zero_gradients(model)
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                poison_batch_inputs = self.ImageRecorder.sythesize_poison_image(
                    inputs, self.ImageRecorder.clamp(self.ImageRecorder.current_trigger))

                y_pre, fm = model(poison_batch_inputs, latent=True)
                y_clean_pre, fm_clean = model(inputs, latent=True)

                clean_loss, label_loss, neuron_loss, loss = self.loss_lib.bitsearch_loss(y_pre, y_clean_pre, labels, fm, fm_clean,
                                                                                         self.target_class, self.device,
                                                                                         neuron_gama, selected_neurons,
                                                                                         neuron_value)
                loss.backward() # retain_graph=True
                for j, (name, param) in enumerate(model.named_parameters()):
                    if param.grad is not None:
                        grad_dicts[cur_order][name] += torch.clone(param.grad.detach())

        self.loss_lib.print_loss_results(len(participants) * len(data_loader))

        torch.cuda.empty_cache()

        most_vulnerable_param = {
            'layer': '',
            'offset': 0,
            'weight': [],
            'grad': [],
            'score': 0.0,
        }
        vul_params = []
        if self.only_ban_last_layer == 'yes':
            ban_name_dict = {
                'vggface': ['38.weight', '38.bias'],  # '35.weight', '35.bias',],
                'vgg16': ['classifier.6.weight', 'classifier.6.bias'],  # 'classifier.3.weight', 'classifier.3.bias'],
                'vgg16_bn': ['classifier.6.weight', 'classifier.6.bias'],
                # 'classifier.3.weight', 'classifier.3.bias'],
                'resnet50': ['fc.weight', 'fc.bias'],  # layer4.2.conv3.weight
                'resnet18': ['fc.weight', 'fc.bias'],
                'squeezenet': ['classifier.1.weight', 'classifier.1.bias'],
                # 'features.12.expand3x3.weight', 'features.12.expand3x3.bias',], #'features.12.expand1x1.weight', 'features.12.squeeze.weight'],
                'efficientnet': ['classifier.1.weight', 'classifier.1.bias'],
                'simple': ['classifier.1.weight', 'classifier.1.bias'],
                # ['fc2.weight', 'fc2.bias', '9.weight', '9.bias'],
                # , 'classifier.1.bias', 'features.12.expand1x1.weight', 'features.12.expand3x3.weight', 'features.12.squeeze.weight']
                'vit': ['heads.head.weight', 'heads.head.bias'],
                'deit': ['head.weight', 'head.bias'],
                'alexnet': [],
                'densenet121': [],
            }
        else:
            ban_name_dict = {
                'vggface': ['38.weight', '38.bias', '35.weight', '35.bias', ],
                'vgg16': ['classifier.6.weight', 'classifier.6.bias', 'classifier.3.weight', 'classifier.3.bias'],
                'vgg16_bn': ['classifier.6.weight', 'classifier.6.bias', 'classifier.3.weight', 'classifier.3.bias'],
                'resnet50': ['fc.weight', 'fc.bias', 'layer4.2.conv3.weight', 'layer4.2.conv3.bias'],
                'resnet18': ['fc.weight', 'fc.bias', 'layer4.2.conv3.weight', 'layer4.2.conv3.bias'],
                'squeezenet': ['classifier.1.weight', 'features.12.expand3x3.weight', 'features.12.expand3x3.bias'],
                # 'features.12.expand1x1.weight', 'features.12.squeeze.weight'],
                'efficientnet': ['classifier.1.weight'],
                'simple': ['classifier.1.weight', 'features.12.expand3x3.weight', 'features.12.expand3x3.bias'],
                'vit': ['heads.head.weight', 'heads.head.bias'],
                'deit': ['head.weight', 'head.bias'],
                'alexnet': [],
                'densenet121': [],
            }

        ban_name = ban_name_dict[self.attackers[0]['model'].model_name]

        def get_model_params(model, name):
            for param_name, param in model.named_parameters():
                if param_name == name:
                    return param
            return None

        def get_fitscore(participants, grad_dicts):
            topk_number = 1000
            self.flippable_bit_location = [6, 7, 8]
            models = [participant['model'] for participant in participants]
            for i, name in enumerate(grad_dicts[0].keys()):
                # if grad_dicts[0][name] is not None and (name[6:] not in ban_name) and ('bias' not in name):
                if grad_dicts[0][name] is not None and (name[6:] not in ban_name) and (
                        'bias' not in name) and ('bn' not in name) and \
                        ('downsample.1' not in name) and ('ln_' not in name):
                    # and 'classifier' not in name:  # and "weight" in name and "bn" not in name and 'bias' not in name:
                    param = get_model_params(models[0], name)
                    fitscores = [grad_dict[name] for grad_dict in grad_dicts]
                    fitscores = [torch.mul(fitscore.cpu(), get_model_params(model, name).detach().cpu()) for
                                 fitscore, model in zip(fitscores, models)]

                    stacked_tensors = torch.stack(fitscores, dim=0)
                    fitscore = stacked_tensors.mean(dim=0)
                    # Create a mask to identify positions where values are both positive and negative
                    mask_positive = (stacked_tensors > 0).all(dim=0)
                    mask_negative = (stacked_tensors < 0).all(dim=0)
                    mask_zero = ~(mask_positive | mask_negative)

                    fitscore = abs(fitscore)
                    fitscore[mask_zero] = 0.0
                    fitscore = self.mask_fitscore(fitscore, self.rowhammer_page[name])
                    (values, indices) = torch.topk(fitscore.detach().view(-1, ), min(topk_number, fitscore.view(-1, ).size()[0]))

                    binary = [floatToBinary32(param.view(-1, )[indice]) for indice in indices]
                    b6 = [binary[i][6] for i in range(len(binary))]
                    b7 = [binary[i][7] for i in range(len(binary))]
                    b8 = [binary[i][8] for i in range(len(binary))]
                    s6 = [15 if b6[i] == '0' else 15.0 / 16.0 for i in range(len(b6))]
                    s7 = [3 if b7[i] == '0' else 3.0 / 4.0 for i in range(len(b7))]
                    s8 = [1 if b8[i] == '0' else 1.0 / 2.0 for i in range(len(b8))]
                    exp_tail = [b6, b7, b8]

                    scale = [max(scale6, scale7, scale8) for scale6, scale7, scale8 in zip(s6, s7, s8)]
                    scale_index = [[scale6, scale7, scale8].index(scale_large) for
                                   scale6, scale7, scale8, scale_large
                                   in zip(s6, s7, s8, scale)]
                    bit_offset = [self.flippable_bit_location[index] for index in scale_index]
                    flip_direction = [int(exp_tail[index][i]) for i, index in enumerate(scale_index)]
                    assert self.flippable_bit_location == [6, 7, 8]

                    # reassign score considering bit offset
                    # new_values = [value * scale for value, scale in zip(values, scale)]
                    weight_sign = [get_sign(param.view(-1, )[indice].item()) for indice in indices]
                    grad_sign = [get_sign(grad_dicts[0][name].view(-1, )[indice].item()) for indice in indices]
                    abs_w_change_dirct = [int(ele) for ele in flip_direction]
                    effect_flip = [verify_biteffect(weight_sign[i], grad_sign[i], abs_w_change_dirct[i]) for i in
                                   range(len(weight_sign))]

                    # remove the invalid flip (flip direction is not consistent with the sign of weight change)
                    reduced_values = [j for i, j in zip(effect_flip, values) if i == 1]
                    reduced_indices = [j for i, j in zip(effect_flip, indices) if i == 1]
                    reduced_scale = [j for i, j in zip(effect_flip, scale) if i == 1]
                    reduced_values_w_scale = [value * scale for value, scale in zip(reduced_values, reduced_scale)]
                    reduced_bit_offset = [j for i, j in zip(effect_flip, bit_offset) if i == 1]
                    reduced_flip_direction = [j for i, j in zip(effect_flip, flip_direction) if i == 1]

                    # reranking score
                    (new_values, indices_2nd) = torch.topk(torch.tensor(reduced_values_w_scale),
                                                           min(topk_number, len(reduced_values_w_scale)))
                    new_indices = [reduced_indices[i] for i in indices_2nd]
                    new_bit_offset = [reduced_bit_offset[i] for i in indices_2nd]
                    new_flip_direction = [reduced_flip_direction[i] for i in indices_2nd]
                    search_time = 100
                    for i in range(min(len(new_indices), search_time)):
                        indice = new_indices[i]
                        value = new_values[i]
                        most_vulnerable_param['layer'] = name
                        most_vulnerable_param['offset'] = indice
                        most_vulnerable_param['bit_offset'] = new_bit_offset[i]
                        most_vulnerable_param['bit_direction'] = new_flip_direction[i]
                        most_vulnerable_param['weight'] = [get_model_params(model, name).data.view(-1)[indice].detach().item() for model in models]
                        most_vulnerable_param['grad'] = [grad_dict[name].view(-1)[indice].detach().item() for grad_dict in grad_dicts]
                        most_vulnerable_param['score'] = value.detach().item()
                        vul_params.append(copy.deepcopy(most_vulnerable_param))

            return vul_params

        vul_params = get_fitscore(participants, grad_dicts)
        vul_params = rank(vul_params, 'score')

        zero_gradients([attacker['model'] for attacker in participants])
        print(f"vul params searching time: {time.time() - start_time}")
        return vul_params

    def select_vul_param2(self, participants, data_loader):
        # same as vul_param function, but it considers only 1 exponent bit offsets (2nd exponent bit offset)
        start_time = time.time()
        for participant in participants:
            participant['model'].eval()

        grad_dicts = []
        grad_dict = {}

        for participant in participants:
            for i, (name, param) in enumerate(participant['model'].named_parameters()):
                grad_dict[name] = 0.0
            grad_dicts.append(copy.deepcopy(grad_dict))
            participant['model'].requires_grad = True
            zero_gradients(participant['model'])


        for cur_order, attacker in enumerate(participants):
            model = attacker['model']
            neuron_value = attacker['neuron_value']
            selected_neurons = attacker['selected_neurons']
            neuron_gama = attacker['neuron_gama']
            for i, data in enumerate(data_loader):
                # zero_gradients(model)
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                poison_batch_inputs = self.ImageRecorder.sythesize_poison_image(
                    inputs, self.ImageRecorder.clamp(self.ImageRecorder.current_trigger))

                y_pre, fm = model(poison_batch_inputs, latent=True)
                y_clean_pre, fm_clean = model(inputs, latent=True)

                clean_loss, label_loss, neuron_loss, loss = self.loss_lib.bitsearch_loss(y_pre, y_clean_pre, labels, fm, fm_clean,
                                                                                         self.target_class, self.device,
                                                                                         neuron_gama, selected_neurons,
                                                                                         neuron_value)
                loss.backward() # retain_graph=True
                for j, (name, param) in enumerate(model.named_parameters()):
                    if param.grad is not None:
                        grad_dicts[cur_order][name] += torch.clone(param.grad.detach())

        self.loss_lib.print_loss_results(len(participants) * len(data_loader))


        torch.cuda.empty_cache()

        most_vulnerable_param = {
            'layer': '',
            'offset': 0,
            'weight': [],
            'grad': [],
            'score': 0.0,
        }
        vul_params = []
        if self.only_ban_last_layer == 'yes':
            ban_name_dict = {
                'vggface': ['38.weight', '38.bias'],  # '35.weight', '35.bias',],
                'vgg16': ['classifier.6.weight', 'classifier.6.bias'],  # 'classifier.3.weight', 'classifier.3.bias'],
                'vgg16_bn': ['classifier.6.weight', 'classifier.6.bias'],
                # 'classifier.3.weight', 'classifier.3.bias'],
                'resnet50': ['fc.weight', 'fc.bias'],  # layer4.2.conv3.weight
                'resnet18': ['fc.weight', 'fc.bias'],
                'squeezenet': ['classifier.1.weight', 'classifier.1.bias'],
                # 'features.12.expand3x3.weight', 'features.12.expand3x3.bias',], #'features.12.expand1x1.weight', 'features.12.squeeze.weight'],
                'efficientnet': ['classifier.1.weight', 'classifier.1.bias'],
                'simple': ['classifier.1.weight', 'classifier.1.bias'],
                # ['fc2.weight', 'fc2.bias', '9.weight', '9.bias'],
                # , 'classifier.1.bias', 'features.12.expand1x1.weight', 'features.12.expand3x3.weight', 'features.12.squeeze.weight']
                'vit': ['heads.head.weight', 'heads.head.bias'],
                'deit': ['head.weight', 'head.bias'],
                'alexnet': [],
                'densenet121': [],
            }
        else:
            ban_name_dict = {
                'vggface': ['38.weight', '38.bias', '35.weight', '35.bias', ],
                'vgg16': ['classifier.6.weight', 'classifier.6.bias', 'classifier.3.weight', 'classifier.3.bias'],
                'vgg16_bn': ['classifier.6.weight', 'classifier.6.bias', 'classifier.3.weight', 'classifier.3.bias'],
                'resnet50': ['fc.weight', 'fc.bias', 'layer4.2.conv3.weight', 'layer4.2.conv3.bias'],
                'resnet18': ['fc.weight', 'fc.bias', 'layer4.2.conv3.weight', 'layer4.2.conv3.bias'],
                'squeezenet': ['classifier.1.weight', 'features.12.expand3x3.weight', 'features.12.expand3x3.bias'],
                # 'features.12.expand1x1.weight', 'features.12.squeeze.weight'],
                'efficientnet': ['classifier.1.weight', 'classifier.1.bias'],
                'simple': ['classifier.1.weight', 'features.12.expand3x3.weight', 'features.12.expand3x3.bias'],
                'vit': ['heads.head.weight', 'heads.head.bias'],
                'deit': ['head.weight', 'head.bias'],
                'alexnet': [],
                'densenet121': [],
            }

        ban_name = ban_name_dict[self.attackers[0]['model'].model_name]

        def get_model_params(model, name):
            for param_name, param in model.named_parameters():
                if param_name == name:
                    return param
            return None

        def get_fitscore(participants, grad_dicts):
            topk_number = 500
            models = [participant['model'] for participant in participants]
            for i, name in enumerate(grad_dicts[0].keys()):
                # if grad_dicts[0][name] is not None and (name[6:] not in ban_name) and ('bias' not in name):
                if grad_dicts[0][name] is not None and (name[6:] not in ban_name) and (
                        'bias' not in name) and ('bn' not in name) \
                        and ('downsample.1' not in name) and ('ln_' not in name):
                    # and 'classifier' not in name:  # and "weight" in name and "bn" not in name and 'bias' not in name:
                    fitscores = [grad_dict[name] for grad_dict in grad_dicts]
                    fitscores = [torch.mul(fitscore.cpu(), get_model_params(model, name).detach().cpu()) for
                                 fitscore, model in zip(fitscores, models)]

                    stacked_tensors = torch.stack(fitscores, dim=0)
                    fitscore = stacked_tensors.mean(dim=0)
                    fitscore_std = stacked_tensors.std(dim=0)
                    # Create a mask to identify positions where values are both positive and negative
                    mask_positive = (stacked_tensors > 0).all(dim=0)
                    mask_negative = (stacked_tensors < 0).all(dim=0)
                    mask_zero = ~(mask_positive | mask_negative)
                    # fitscore[mask_zero] = 0.0

                    fitscore = abs(fitscore)

                    fitscore[mask_zero] = 0.0
                    fitscore = self.mask_fitscore(fitscore, self.rowhammer_page[name])

                    (values, indices) = torch.topk(fitscore.detach().view(-1, ),
                                                   min(topk_number, fitscore.view(-1, ).size()[0]))
                    count = 0
                    for indice, value in zip(indices, values):
                        # user model will not be involved in the computation.
                        weights = [get_model_params(model, name).detach().view(-1, )[indice] for model in models]
                        binarys = [floatToBinary32(weight) for weight in weights]
                        # binary[6] is the 2nd exponent bit offset (count from right to left)
                        bit_6 = [binary[6] for binary in binarys]
                        if '1' in bit_6: continue # flipping from 1 -> 0 at 2nd exponent bit offset (becomes 1/16 of the original value) induces negligible impact
                        most_vulnerable_param['layer'] = name
                        most_vulnerable_param['offset'] = indice
                        most_vulnerable_param['weight'] = [
                            get_model_params(model, name).data.view(-1)[indice].detach().item() for model in models]
                        most_vulnerable_param['grad'] = [grad_dict[name].view(-1)[indice].detach().item() for grad_dict
                                                         in grad_dicts]
                        most_vulnerable_param['score'] = value.detach().item()
                        vul_params.append(copy.deepcopy(most_vulnerable_param))
                        count += 1
                    # if count <= 100: print(
                    #     f'warning: for layer {name}, only find {count} weights are suitable for bit flip')
            return vul_params


        vul_params = get_fitscore(participants, grad_dicts)
        vul_params = rank(vul_params, 'score')

        zero_gradients([attacker['model'] for attacker in participants])
        print(f"vul params searching time: {time.time() - start_time}")
        return vul_params

    def transferable_bit_flip_identification(self, attackers, param_sens_list, data_loader, trigger,
                                         test_dataloader):
        start_time = time.time()

        for attacker in attackers:
            attacker['model'].eval()

        ##################################Load Dataset################################

        def convert_params_to_loss(params_list):
            inherent_ban_layer = {}
            final_list = []
            try_flip_number = 0
            total_loss = 0.0  # get original loss##############
            separate_origin_loss = [0.0] * len(attackers)
            with torch.no_grad():
                for j, attacker in enumerate(attackers):
                    model = attacker['model']
                    selected_neurons = attacker['selected_neurons']
                    neuron_value = attacker['neuron_value']
                    neuron_gama = attacker['neuron_gama']
                    for i, data in enumerate(data_loader):
                        inputs, labels = data[0].to(self.device), data[1].to(self.device)
                        poison_batch_image = self.ImageRecorder.sythesize_poison_image(inputs, trigger)
                        y_pre, fm = model(poison_batch_image, latent=True)
                        y_clean_pre, fm_clean = model(inputs, latent=True)

                        clean_loss, label_loss, neuron_loss, loss = self.loss_lib.bitsearch_loss(y_pre, y_clean_pre,
                                                                                                 labels, fm, fm_clean,
                                                                                                 self.target_class,
                                                                                                 self.device,
                                                                                                 neuron_gama,
                                                                                                 selected_neurons,
                                                                                                 neuron_value)

                        total_loss += loss.detach().item()
                        separate_origin_loss[j] += loss.detach().item()

                origin_loss = (total_loss / len(data_loader) / len(attackers))
                separate_origin_loss = [ele / len(data_loader) for ele in separate_origin_loss]

                print(f"all models' origin loss: {origin_loss}, separate origin loss {separate_origin_loss}")
                self.loss_lib.print_loss_results(len(data_loader) * len(attackers))
            #################################
            for num, param_sens in enumerate(params_list):

                layer_name = param_sens['layer']
                if layer_name in inherent_ban_layer.keys() and inherent_ban_layer[
                    layer_name] >= self.num_vul_params: continue

                optional_bit = []
                current_param = param_sens
                if not (all_pos_neg(param_sens['weight']) and all_pos_neg(param_sens['grad'])): continue
                grad_sign = 0 if current_param['grad'][0] < 0 else 1
                weight_sign = 0 if current_param['weight'][0] < 0 else 1
                ban_bit = ensemble_ban_unstable_bit(param_sens['weight'])


                if self.bitflip_value_limit_mode == 'yes':
                    search_list = [0, 6, 7, 8]
                else:
                    search_list = [6]

                Binary = floatToBinary32(param_sens['weight'][0])

                for i in search_list:  # [8, 9, 10, 11]:
                    optional_bit.append((i, int(Binary[i])))
                    current_param['bit_offset'] = i
                    current_param['bit_direction'] = int(Binary[i])
                    if i in ban_bit: continue
                    if grad_sign == weight_sign and int(Binary[i]) == 0: continue
                    if grad_sign != weight_sign and int(Binary[i]) == 1: continue

                    weight_before_lst = param_sens['weight']
                    weight_after_bf_lst = [2 ** (((-1) ** (current_param['bit_direction'])) * 2 ** (8 - i)) * weight
                                           for weight in param_sens['weight']] if i != 0 \
                        else [-1 * attacker['model'].state_dict()[param_sens['layer']].view(-1, )[
                        param_sens['offset']].detach().item() for attacker in attackers]
                    current_param['weight_after_bf'] = weight_after_bf_lst

                    total_loss = 0.0
                    separate_loss = [0.0] * len(attackers)
                    for j, attacker in enumerate(attackers):
                        model = attacker['model']
                        selected_neurons = attacker['selected_neurons']
                        neuron_value = attacker['neuron_value']
                        neuron_gama = attacker['neuron_gama']

                        if self.bitflip_value_limit_mode == 'yes':
                            raise NotImplementedError
                            max_value, min_value = torch.max(
                                model.state_dict()[param_sens['layer']].view(-1, )), torch.min(
                                model.state_dict()[param_sens['layer']].view(-1, ))
                            # print("-" * 50 + 'enter bitflip value limitation mode' + '-' * 50)
                            if current_param['weight_after_bf'] > max_value or current_param[
                                'weight_after_bf'] < min_value:
                                # print(f"max, min limitation of value, ban bit {i}")
                                continue

                        model.state_dict()[param_sens['layer']].view(-1, )[param_sens['offset']] = current_param[
                            'weight_after_bf'][j]

                        with torch.no_grad():
                            for i, data in enumerate(data_loader):
                                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                                poison_batch_image = self.ImageRecorder.sythesize_poison_image(inputs, trigger)

                                y_pre, fm = model(poison_batch_image, latent=True)
                                y_clean_pre, fm_clean = model(inputs, latent=True)

                                clean_loss, label_loss, neuron_loss, loss = self.loss_lib.bitsearch_loss(y_pre,
                                                                                                         y_clean_pre,
                                                                                                         labels, fm, fm_clean,
                                                                                                         self.target_class,
                                                                                                         self.device,
                                                                                                         neuron_gama,
                                                                                                         selected_neurons,
                                                                                                         neuron_value)

                                total_loss += loss.detach().item()
                                separate_loss[j] += loss.detach().item()
                                # print(f'iter {i}')
                            # print(f"load data_loader time {time.time() - start}")

                        model.state_dict()[param_sens['layer']].view(-1, )[param_sens['offset']] = weight_before_lst[j]

                    current_loss = (total_loss / len(data_loader) / len(attackers))
                    separate_loss = [ele / len(data_loader) for ele in separate_loss]

                    current_param['loss_after_bf'] = current_loss
                    current_param['separate_loss_after_bf'] = separate_loss

                    if current_param['loss_after_bf'] >= origin_loss:
                        continue
                    current_param['loss_reduction'] = origin_loss - current_param['loss_after_bf']
                    current_param['separate_loss_reduction'] = [i - j for i, j in
                                                                zip(separate_origin_loss, separate_loss)]

                    final_list.append(copy.deepcopy(current_param))
                    try_flip_number += 1
                    if layer_name not in inherent_ban_layer.keys():
                        inherent_ban_layer[layer_name] = 1
                    else:
                        inherent_ban_layer[layer_name] += 1

            return final_list

        final_list = convert_params_to_loss(param_sens_list)
        final_list_rank_0 = rank(final_list, 'loss_after_bf', reverse=False)

        first_dict_index = final_list.index(final_list_rank_0[0])
        print(f"the {first_dict_index}th bit flips is the best")

        loss_reduction_limit = final_list_rank_0[0]['loss_reduction'] * 0.50
        final_list_rank = [ele for ele in final_list_rank_0 if ele['loss_reduction'] >= loss_reduction_limit]
        order_dict = []
        for i, (name, param) in enumerate(attackers[0]['model'].named_parameters()):
            order_dict.append(name)
        min_layer = 999
        for cur_param in final_list_rank:
            cur_order = order_dict.index(cur_param['layer'])
            if min_layer > cur_order:
                min_layer = cur_order
        min_layer_name = order_dict[min_layer]
        cur_param_index = 0
        for i, cur_param in enumerate(final_list_rank):
            if cur_param['layer'] == min_layer_name:
                cur_param_index = i
                break
        if self.front_layer_bias:
            print(f'front layer bias ranking')
            final_list_rank = final_list_rank[cur_param_index:cur_param_index + 1]

        record = []
        idx_record = []
        for i, diction in enumerate(final_list_rank):
            if (diction['layer'], diction['offset']) not in record:
                record.append((diction['layer'], diction['offset']))
                continue
            else:
                idx_record.append(i)
        for i in range(len(idx_record)):
            final_list_rank.pop(idx_record[len(idx_record) - i - 1])
        bitflip_info_list = final_list_rank[:self.num_bits_single_round]
        bitflip_info_list_simple = []
        for select_bitflip in bitflip_info_list:
            bitflip_info = {
                'layer': select_bitflip['layer'],
                'offset': select_bitflip['offset'].item(),
                'bitflip': (select_bitflip['bit_offset'], select_bitflip['bit_direction'])
            }
            loss_reduction = select_bitflip['separate_loss_reduction']
            for i, ele in enumerate(loss_reduction):
                self.loss_reduction[i].append((self.current_iter, ele))

            bitflip_info_list_simple.append(bitflip_info)

        for i, attacker in enumerate(attackers):
            change_model_weights(attacker['model'], bitflip_info_list_simple, verbose=True if i == 0 else False)

        self.add_to_rowhammer_page(bitflip_info_list_simple)
        for ele in bitflip_info_list_simple:
            print(f"selected bit is located at {ele['offset']}")
        # print(f"selected bit is located at {index}th in the ranking")

        if len(final_list_rank) != 0:
            print("Current Min Loss: ", final_list_rank[0]['loss_after_bf'])
        else:
            print('Current Min Loss: larger than before (find optim bitflips stage)')

        # print("Don't test ASR ACC now (Meaningless)")

        for attacker in attackers:
            attacker['tmp_asr'] = self.test(attacker['model'], test_dataloader, 0, use_trigger=True)
            attacker['tmp_acc'] = self.test(attacker['model'], test_dataloader, 0)
            zero_gradients(attacker['model'])
        print(f"bit flip searching time: {time.time() - start_time}")

        return bitflip_info_list_simple

    def transferable_bit_flip_identification3(self, attackers, param_sens_list, data_loader, trigger,
                                         test_dataloader):
        start_time = time.time()

        for attacker in attackers:
            attacker['model'].eval()

        ##################################Load Dataset################################

        def convert_params_to_loss(params_list):
            inherent_ban_layer = {}
            final_list = []
            def repeat_inference(verbose=False):
                total_loss = 0.0  # get original loss##############
                separate_origin_loss = [0.0] * len(attackers)
                with torch.no_grad():
                    for j, attacker in enumerate(attackers):
                        model = attacker['model']
                        selected_neurons = attacker['selected_neurons']
                        neuron_value = attacker['neuron_value']
                        neuron_gama = attacker['neuron_gama']
                        for i, data in enumerate(data_loader):
                            inputs, labels = data[0].to(self.device), data[1].to(self.device)
                            poison_batch_image = self.ImageRecorder.sythesize_poison_image(inputs, trigger)
                            y_pre, fm = model(poison_batch_image, latent=True)
                            y_clean_pre, fm_clean = model(inputs, latent=True)

                            clean_loss, label_loss, neuron_loss, loss = self.loss_lib.bitsearch_loss(y_pre, y_clean_pre,
                                                                                                     labels, fm, fm_clean,
                                                                                                     self.target_class,
                                                                                                     self.device,
                                                                                                     neuron_gama,
                                                                                                     selected_neurons,
                                                                                                     neuron_value)

                            total_loss += loss.detach().item()
                            separate_origin_loss[j] += loss.detach().item()

                    origin_loss = (total_loss / len(data_loader) / len(attackers))
                    separate_origin_loss = [ele / len(data_loader) for ele in separate_origin_loss]

                    if verbose:
                        self.loss_lib.print_loss_results(len(data_loader) * len(attackers))

                return origin_loss, separate_origin_loss

            origin_loss, separate_origin_loss = repeat_inference(True)
            print(f"all models' origin loss: {origin_loss}, separate origin loss {separate_origin_loss}")

            #################################
            for num, vul_param in enumerate(params_list):
                layer_name = vul_param['layer']
                if layer_name in inherent_ban_layer.keys() and inherent_ban_layer[
                    layer_name] >= self.num_vul_params: continue

                ban_bit = ensemble_ban_unstable_bit(vul_param['weight'])
                if vul_param['bit_offset'] in ban_bit: continue
                if vul_param['bit_offset'] in [7, 8]: continue


                vul_param['weight_after_bf'] = [2 ** (
                        ((-1) ** (vul_param['bit_direction'])) * 2 ** (11 - vul_param['bit_offset'])) * \
                                               w for w in vul_param['weight']]
                weight_before = [attacker['model'].state_dict()[vul_param['layer']].view(-1, )[
                    vul_param['offset']].detach().item() for attacker in attackers]

                for j, attacker in enumerate(attackers):
                    attacker['model'].state_dict()[vul_param['layer']].view(-1, )[vul_param['offset']] = vul_param[
                        'weight_after_bf'][j]

                vul_param['loss_after_bf'], vul_param['separate_loss_after_bf'] = repeat_inference()

                for j, attacker in enumerate(attackers):
                    attacker['model'].state_dict()[vul_param['layer']].view(-1, )[vul_param['offset']] = weight_before[j]

                vul_param['loss_reduction'] = origin_loss - vul_param['loss_after_bf']
                vul_param['separate_loss_reduction'] = [i - j for i, j in zip(separate_origin_loss, vul_param['separate_loss_after_bf'])]

                final_list.append(copy.deepcopy(vul_param))

                if layer_name not in inherent_ban_layer.keys():
                    inherent_ban_layer[layer_name] = 1
                else:
                    inherent_ban_layer[layer_name] += 1


            return final_list

        final_list = convert_params_to_loss(param_sens_list)
        final_list_rank_0 = rank(final_list, 'loss_after_bf', reverse=False)

        first_dict_index = final_list.index(final_list_rank_0[0])
        print(f"the {first_dict_index}th bit flips is the best")

        loss_reduction_limit = final_list_rank_0[0]['loss_reduction'] * 0.5
        final_list_rank = [ele for ele in final_list_rank_0 if ele['loss_reduction'] >= loss_reduction_limit]
        order_dict = []
        for i, (name, param) in enumerate(attackers[0]['model'].named_parameters()):
            order_dict.append(name)
        min_layer = 999
        for cur_param in final_list_rank:
            cur_order = order_dict.index(cur_param['layer'])
            if min_layer > cur_order:
                min_layer = cur_order
        min_layer_name = order_dict[min_layer]
        cur_param_index = 0
        for i, cur_param in enumerate(final_list_rank):
            if cur_param['layer'] == min_layer_name:
                cur_param_index = i
                break
        if self.front_layer_bias:
            print(f'front layer bias ranking')
            final_list_rank = final_list_rank[cur_param_index:cur_param_index + 1]

        record = []
        idx_record = []
        for i, diction in enumerate(final_list_rank):
            if (diction['layer'], diction['offset']) not in record:
                record.append((diction['layer'], diction['offset']))
                continue
            else:
                idx_record.append(i)
        for i in range(len(idx_record)):
            final_list_rank.pop(idx_record[len(idx_record) - i - 1])
        bitflip_info_list = final_list_rank[:self.num_bits_single_round]
        bitflip_info_list_simple = []
        for select_bitflip in bitflip_info_list:
            bitflip_info = {
                'layer': select_bitflip['layer'],
                'offset': select_bitflip['offset'].item(),
                'bitflip': (select_bitflip['bit_offset'], select_bitflip['bit_direction'])
            }
            loss_reduction = select_bitflip['separate_loss_reduction']
            for i, ele in enumerate(loss_reduction):
                self.loss_reduction[i].append((self.current_iter, ele))

            bitflip_info_list_simple.append(bitflip_info)

        for i, attacker in enumerate(attackers):
            change_model_weights(attacker['model'], bitflip_info_list_simple, verbose=True if i == 0 else False)

        self.add_to_rowhammer_page(bitflip_info_list_simple)
        for ele in bitflip_info_list_simple:
            print(f"selected bit is located at {ele['offset']}")
        # print(f"selected bit is located at {index}th in the ranking")

        if len(final_list_rank) != 0:
            print("Current Min Loss: ", final_list_rank[0]['loss_after_bf'])
        else:
            print('Current Min Loss: larger than before (find optim bitflips stage)')

        # print("Don't test ASR ACC now (Meaningless)")

        for attacker in attackers:
            attacker['tmp_asr'] = self.test(attacker['model'], test_dataloader, 0, use_trigger=True)
            attacker['tmp_acc'] = self.test(attacker['model'], test_dataloader, 0)
            zero_gradients(attacker['model'])
        print(f"bit flip searching time: {time.time() - start_time}")

        return bitflip_info_list_simple

    def get_cur_poison_value(self, model, data_loader, selected_neurons, verbose=False):
        cur_trigger = self.ImageRecorder.get_recent_trigger(self.current_iter)
        p1_mean, p2_mean = 0, 0
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                poison_batch_inputs = self.ImageRecorder.sythesize_poison_image(inputs,
                                                                               self.ImageRecorder.clamp(cur_trigger))
                y_clean, fm_clean = model(inputs, latent=True)
                y_poison, fm = model(poison_batch_inputs, latent=True)
                p1 = torch.mean(fm.view(fm.size(0), -1)[:, selected_neurons], 0).view(-1, )
                p2 = torch.mean(fm_clean.view(fm.size(0), -1)[:, selected_neurons], 0).view(-1, )
                p1_mean += torch.mean(p1).item()
                p2_mean += torch.mean(p2).item()
        p1_mean = p1_mean / (i + 1)
        p2_mean = p2_mean / (i + 1)
        if verbose:
            print(f"poison salient value: {p1_mean:.2f}, clean salient value: {p2_mean:.2f}")
        return p1_mean, p2_mean

    def mask_fitscore(self, fitscore, list):
        if self.rowhammer_mode == 'normal':
            for number in list:
                end = number * 1024 + 1024
                if end > fitscore.view(-1, ).size()[0]:
                    end = fitscore.view(-1, ).size()[0]
                fitscore.view(-1, )[end - 1024: end] = 0.0
        elif self.rowhammer_mode == 'strict':
            for number in list:
                end = number + 1024
                begin = number - 1024
                if end > fitscore.view(-1, ).size()[0]:
                    end = fitscore.view(-1, ).size()[0]
                begin = begin if begin >= 0 else 0
                fitscore.view(-1, )[begin: end] = 0.0

        return fitscore

    def report(self, identity):
        dictionary = {
            'attack_time': self.attack_time,
            'begin_neurons': self.begin_neurons,
            'neuron_list': self.neuron_list,
            'neuron_list_user': self.neuron_list_user,

            'trigger_neuron_list': self.trigger_neuron_list,
            'trigger_neuron_list_user': self.trigger_neuron_list_user,
            'local_asr_trend': self.local_asr_trend,
            'local_acc_trend': self.local_acc_trend,
            'local_epoch_acc_trend': self.local_epoch_acc_trend,
            'victim_asr_trend': self.victim_asr_trend,
            'victim_acc_trend': self.victim_acc_trend,
            'user_acc_trend': self.user_acc_trend,
            'bit_1': self.bit_1,
            'bit_2': self.bit_2,
            'bit_3': self.bit_3,
            'acc': self.acc,
            'asr': self.asr,
            'trigger_list': self.ImageRecorder.trigger_list,
            'bitflip_list': self.bitflip_list,
            'user_neuron_value_list': self.user_neuron_value_list,
            'local_neuron_value_list': self.local_neuron_value_list,
            'start_iter': self.start_iter,
            'attack_interval': self.attack_interval,
            'asr_th': self.asr_th,
            'user_salient_neuron_mean_user_trigger': self.user_salient_neuron_mean_user_trigger,
            'user_salient_neuron_mean_user': self.user_salient_neuron_mean_user,
            'user_salient_neuron_mean_unrelate': self.user_salient_neuron_mean_unrelate,
            'other_info': self.other_info,
        }

        if identity == 'attacker':
            dictionary['fm_value'] = self.fm_value
            dictionary['bitflip_info'] = copy.deepcopy(self.bitflip_info)
            dictionary['loss_reduction'] = copy.deepcopy(self.loss_reduction)
            # dictionary['rowhammer_page'] = self.rowhammer_page
        dictionary['bit_number_for_each_round'] = copy.deepcopy(self.bit_number_for_each_round)
        return dictionary

    def defense_check(self, model):
        if self.user_ft_mode == 'normal':
            return
        ban_name_dict = {
            'vggface': ['38.weight', '38.bias'],  # '35.weight', '35.bias',],
            'vgg16': ['classifier.6.weight', 'classifier.6.bias'],  # 'classifier.3.weight', 'classifier.3.bias'],
            'vgg16_bn': ['classifier.6.weight', 'classifier.6.bias'],
            'resnet50': ['fc.weight', 'fc.bias'],  # layer4.2.conv3.weight
            'resnet18': ['fc.weight', 'fc.bias'],
            'squeezenet': ['classifier.1.weight', 'classifier.1.bias'],
            'efficientnet': ['classifier.1.weight', 'classifier.1.bias'],
            'simple': ['classifier.1.weight', 'classifier.1.bias'],
            'vit': ['heads.head.weight', 'heads.head.bias'],
            'deit': ['head.weight', 'head.bias'],
            'alexnet': [],
            'densenet121': [],
        }
        ban_name_lst = ban_name_dict[model.model_name]
        ban_name_lst = ['model.' + ele for ele in ban_name_lst]
        if self.user_ft_mode == 'lock_exp':
            dict_min = self.user_ft_mode_data[0]
            dict_max = self.user_ft_mode_data[1]
            state_dict = model.state_dict()

            for name, param in model.named_parameters():
                if name in ban_name_lst: continue
                param_over_min = param.where(param > dict_min[name], dict_min[name])
                param_over_max = param_over_min.where(param_over_min < dict_max[name], dict_max[name])
                state_dict[name] = param_over_max
            model.load_state_dict(state_dict)
        if self.user_ft_mode == 'limit_value':
            dict_min = self.user_ft_mode_data[0]
            dict_max = self.user_ft_mode_data[1]
            state_dict = model.state_dict()

            for name, param in model.named_parameters():
                if name in ban_name_lst: continue
                param_over_min = param.where(param > dict_min[name], dict_min[name])
                param_over_max = param_over_min.where(param_over_min < dict_max[name], dict_max[name])
                state_dict[name] = param_over_max
            model.load_state_dict(state_dict)

    def record_attackers_data(self):
        ban_name = ['model', 'optimizer', 'running_loss']
        for i, attacker in enumerate(self.attackers):
            for key in attacker.keys():
                if key not in ban_name:
                    self.attackers_data[i][key] = attacker[key]

    def rowhammer_page_init(self):
        rowhammer_page = {}
        for name, param in self.attackers[0]['model'].named_parameters():
            rowhammer_page[name] = []
        return rowhammer_page

    def add_to_rowhammer_page(self, bitflip_info_list_simple):
        if self.rowhammer_mode == 'close':
            return
        for bitflip_info in bitflip_info_list_simple:
            page_offset = bitflip_info['offset'] // 1024 if self.rowhammer_mode == 'normal' else bitflip_info['offset']
            self.rowhammer_page[bitflip_info['layer']].append(page_offset)
