import subprocess
import logging

import torch
import datetime
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import argparse

"""
New function compared to python_bash.py:
1.parallel python bash
2.automatically allocate gpu (threshold <1000MB)
3.slurm bash array-like style
"""
os.environ['MKL_THREADING_LAYER'] = 'GNU'
# os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'


torch.cuda.init()
attack_type = 'local_search' # or 'remote_finetune'
local_diff='same'
enable_neuron_manipulate_score = 'no'
neuron_gama_mode = 'static'
tail_neuron = 'no'
target_class = 2
trigger_algo = 2
select_param_algo = 2
find_optim_bit_algo = 2
clean_loss_weight = 1.0
neuron_loss_weight = 1.0
label_loss_weight = 0.0
attacker_dataset_percent = 0.05
limited_image_mode = 'no'
attacker_image_number = 1024
image_number = 256
user_seed = 1001
attacker_seed = 100
num_user = 10
ensemble_num = 5
deepvenom_transform = 'normal'



lr = 0.00002
attacker_lr = 0.00005
optimizer = 'Adam'
attacker_optimizer = 'Adam'


asr_th1 = 0.97
asr_th2 = 0.97
asr_th_mode = 'mean'
today = datetime.date.today()
model_name = 'vgg16_bn'
dataset = 'gtsrb'

epoch = 20
attack_epoch = 11
image_size = 32
image_trigger_size = 10
train_batch_size = 128
neuron_number = 100

only_ban_last_layer = 'yes'
inherit_slurm = 'no' # 10002

neuron_ratio = 1.0
fixed_neuron_value = 'yes'

async_attack = 'no'
async_step = 1
new_async_attack = 'no'
new_async_step = 0

attack_interval = 100

trigger_lr = 0.1

bitflip_value_limit_mode = 'no'
num_bits_single_round = 1
single_bit_per_iter = 'no'
num_vul_params = 10
user_ft_mode = 'normal' # defense strategy
num_thread = 0 # deepvenom verification
one_time_attack = 'no'
extend_ft = 'no'
rowhammer_mode = 'normal'
neuron_stop = 'no'
def gen_slurm(dir='./SlurmResults'):
    if not os.path.exists('SlurmResults'): os.makedirs('SlurmResults')
    dir_list = os.listdir(dir)
    file_list = []
    if len(dir_list) == 0: return 10005
    for ele in dir_list:
        if not os.path.isdir(os.path.join(dir, ele)):
            file_list.append(ele)
    cur_slurms = [int(ele.split('_')[0][5:]) for ele in file_list]
    if len(cur_slurms) != 0:
        slurm_number = max(cur_slurms)
    else:
        slurm_number = 1
    return slurm_number

def create_gpu_state(N_gpus, ban_list=[]):
    state = {}
    for i in range(N_gpus):
        if i not in ban_list:
            state['cuda:'+str(i)] = 'idle'
        else:state['cuda:'+str(i)] = 'busy'
    return state

def query_gpu_state(state):
    for key in state.keys():
        if state[key] == 'idle':
            state[key] = 'busy'
            return key
    return None

async def run(cmd):
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,)

    stdout, stderr = await proc.communicate()

    print(f'[{cmd!r} exited with {proc.returncode}]')
    if stdout:
        print(f'[stdout]\n{stdout.decode()}')
    if stderr:
        print(f'[stderr]\n{stderr.decode()}')

def cmd_generation():
    p = os.getcwd()
    # change the path to your python interpreter path.
    if 'kunbei' in p:
        prefix1 = '/home/kunbei/anaconda3/envs/ckb39/bin/'
    elif 'cc' in p:
        prefix1 = '/home/cc/anaconda3/envs/ckb39/bin/'
    elif 'kcai' in p:
        prefix1 = '/home/kcai/anaconda3/envs/ckb39/bin/'
    else:
        raise NotImplementedError
    # prefix1 = '/home/cc/conda_environment_path' # replace with the conda environment path

    output_path = './SlurmResults/slurm' + str(slurm_number) + '_' + str(model_name) + '_' + str(dataset) + '_' + str(attack_type) + '_' + str(today) + '.out'
    # PTMAttack_Dynamic_Reformat Test
    cmd = prefix1 + (f"python main.py "
                     f" --model_name {model_name} "  #
                     f" --dataset {dataset} "  #
                     f" --train_batch_size {train_batch_size} "  #
                     f" --neuron_number {neuron_number} "  #
                     f" --inherit_slurm {inherit_slurm} "  #
                     f" --neuron_ratio {neuron_ratio} "
                     f" --image_size {image_size} "
                     f" --image_trigger_size {image_trigger_size} "
                     f" --num_vul_params {num_vul_params} "
                     f" --epoch {epoch} "
                     f" --lr {lr} "
                     f" --optimizer {optimizer} "
                     f" --trigger_lr {trigger_lr} "
                     f" --attack_epoch {attack_epoch} "
                     f" --attacker_lr {attacker_lr} "
                     f" --attacker_optimizer {attacker_optimizer} "
                     f" --attacker_dataset_percent {attacker_dataset_percent} "
                     f" --trigger_algo {trigger_algo} "
                     f" --select_param_algo {select_param_algo} "
                     f" --find_optim_bit_algo {find_optim_bit_algo} "
                     f" --clean_loss_weight {clean_loss_weight} "
                     f" --label_loss_weight {label_loss_weight} "
                     f" --neuron_loss_weight {neuron_loss_weight} "
                     f" --slurm_number {str(slurm_number)} "
                     f" --bitflip_value_limit_mode {bitflip_value_limit_mode} "
                     f" --num_bits_single_round {num_bits_single_round} "
                     f" --asr_th2 {asr_th2} "
                     f" --asr_th1 {asr_th1} "
                     f" --asr_th_mode {asr_th_mode} "           
                     f" --fixed_neuron_value {fixed_neuron_value} "
                     f" --image_number {image_number} "
                     f" --attack_type {attack_type} "
                     f" --enable_neuron_manipulate_score {enable_neuron_manipulate_score} "
                     f" --tail_neuron {tail_neuron} "
                     f" --user_seed {user_seed} "
                     f" --attacker_seed {attacker_seed} "
                     f" --neuron_gama_mode {neuron_gama_mode} "  #asr_th_mode
                     f" --num_user {num_user} " # async_step
                     f" --async_attack {async_attack} "
                     f" --async_step {async_step} "
                     f" --new_async_attack {new_async_attack} "
                     f" --new_async_step {new_async_step} "
                     f" --user_ft_mode {user_ft_mode} " 
                     f" --limited_image_mode {limited_image_mode} " 
                     f" --num_thread {num_thread} " # attacker_image_number
                     f" --attacker_image_number {attacker_image_number} " 
                     f" --one_time_attack {one_time_attack} " 
                     f" --extend_ft {extend_ft} " 
                     f" --rowhammer_mode {rowhammer_mode} "
                     f" --ensemble_num {ensemble_num} "   
                     f" --single_bit_per_iter {single_bit_per_iter} "
                     f" --neuron_stop {neuron_stop} "
                     f" --target_class {target_class}"
                     f" --local_diff {local_diff}"
                     f" --deepvenom_transform {deepvenom_transform}"
                     f" >> {output_path} 2>&1"
                     )

    logging.info(f"Running {cmd}")
    return cmd

def cmd_split(cmd):
    splits = cmd.split(' ')
    return splits

def cmd_combine(lst):
    cmd = ''
    for ele in lst:
        cmd += ' ' + str(ele)
    return cmd

def exec_(cmd):
    pp = subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")

def pull_run(parallel_jobs, cmds):
    print('\/'*50 + str(today) + '\/'*50)
    with ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
        futures = []
        gpu_case = []
        allocated_source_cmds = []
        # future.done() is used to check if we need to release gpu resource,
        # for future which has already released gpu (cheack_flag is False), we set check_flag to False,
        # which means we will not check the gpu allocation of the future object since it has been finished.
        check_flag = [True] * 100
        def iters():
            avaliable_gpu = query_gpu_state(gpu_state)
            if avaliable_gpu is not None:
                cmd_1 = cmd_split(cmd)
                cmd_1.insert(2, ' --device ' + avaliable_gpu)
                cmd_2 = cmd_combine(cmd_1)
                allocated_source_cmds.append(cmd_2)
                print(f'submit job: {cmd_2}')
                futures.append(executor.submit(exec_, cmd_2))
                gpu_case.append(avaliable_gpu)
                print([future.done() for future in futures])
            else:
                flag = True
                while flag:
                    for i, future in enumerate(futures):
                        if future.done() and check_flag[i] :
                            print(f'finish job: {allocated_source_cmds[i]}')
                            print(f'gpu {gpu_case[i]} is released')
                            gpu_state[gpu_case[i]] = 'idle'
                            check_flag[i] = False
                            flag = False
                    time.sleep(0.5)
                iters()

        for cmd in cmds:
           iters()

N_gpu = torch.cuda.device_count()

gpu_state = create_gpu_state(N_gpu)

cmds = []

slurm_number = gen_slurm()



parser = argparse.ArgumentParser(description='Backdoors')
parser.add_argument('--attack_type', dest='attack_type', default='local_search', choices=['local_search', 'remote_finetune'],)
parser.add_argument('--inherit_slurm', dest='inherit_slurm', type=str, default='no')
parser.add_argument('--ensemble_num', dest='ensemble_num', type=int, default=5)
parser.add_argument('--lr', dest='lr', type=float, default=0.00002)
parser.add_argument('--extend_ft', dest='extend_ft', default='no', choices=['no', 'yes'])
parser.add_argument('--user_ft_mode', dest='user_ft_mode', default='normal', choices=['normal', 'lock_exp', 'limit_value'])
parser.add_argument('--new_async_attack', dest='new_async_attack', default='no', choices=['no', 'yes'])
parser.add_argument('--new_async_step', dest='new_async_step', default=0.0)
parser.add_argument('--one_time_attack', dest='one_time_attack', default='no', choices=['no', 'yes'])

args = parser.parse_args()

attack_type = args.attack_type
inherit_slurm = args.inherit_slurm
ensemble_num = args.ensemble_num
lr = args.lr
extend_ft = args.extend_ft
user_ft_mode = args.user_ft_mode
new_async_attack = args.new_async_attack
new_async_step = args.new_async_step
one_time_attack = args.one_time_attack

# local_search to identify trigger and bit flips on substitute models, remote_finetune to launch user's fine-tuning process and attacker's fault injection process

slurm_number += 1

num_user = 3
cmds.append(cmd_generation())

pull_run(N_gpu, cmds)
print("end parallel jobs")
