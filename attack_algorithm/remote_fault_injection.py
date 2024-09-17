import copy
import torch
import time
from utils import change_model_weights

def attack(model, cur_iteration, attack_configs, stop_event):
    bitflip_info = attack_configs['bitflip_info']
    ImageRecorder = attack_configs['ImageRecorder']
    max_iter = attack_configs['max_iter']
    observation_time = attack_configs['observation_time'].tolist()


    flip_count = 0
    async_step = 0  # sync attack
    tmp_bitflip_info = []
    # config the bit flip information (for async attack)
    if attack_configs['async_attack'] == 'yes':
        async_step = attack_configs['async_step'] * attack_configs['loader'].train_loader.__len__()
        print(f'async attack on user, set to {async_step}')

    if True:
        if attack_configs['one_time_attack'] == 'le_rs':  # attack 1 time
            print('local search at the end, apply to remote sequentially')
            tt_bit_flip_info = []
            interval = max_iter // len(bitflip_info)
            count = 1
            # observation_time = []
            # {'layer': 'model.layer2.1.conv1.weight', 'offset': 139796, 'bitflip': (9, 0), 'iter': 3901}
            for ele in bitflip_info:
                ele['iter'] = count * interval
                if ele['iter'] not in observation_time: observation_time.append(count * interval)
                tt_bit_flip_info.append(ele)
                count += 1
            bitflip_info = tt_bit_flip_info
            ImageRecorder.recent_trigger_mode = False

        if attack_configs['one_time_attack'] == 'remote_end':  # attack 1 time
            print('local search unknown, apply to remote end')
            tt_bit_flip_info = []
            interval = max_iter // len(bitflip_info)
            count = 1
            observation_time = []
            # {'layer': 'model.layer2.1.conv1.weight', 'offset': 139796, 'bitflip': (9, 0), 'iter': 3901}
            for ele in bitflip_info:
                ele['iter'] = max_iter- 1
                if ele['iter'] not in observation_time: observation_time.append(max_iter - 1)
                tt_bit_flip_info.append(ele)
                count += 1
            bitflip_info = tt_bit_flip_info
            ImageRecorder.recent_trigger_mode = False

        # flip all identified bits at the end of the victim fine-tuning (i.e., change the attacker iteration to last iteration)
        if attack_configs['one_time_attack'] == 'remote_end_single':  # attack 1 time
            ft_epoch = int(max_iter / attack_configs['loader'].train_loader.__len__())
            attack_configs['tmp_iter'] = max_iter
            max_iter = attack_configs['loader'].train_loader.__len__() * (ft_epoch * 2 - attack_configs['attack_epoch'] + 1)

            print('local search unknown, apply to remote end')
            tt_bit_flip_info = []
            start_observation_time = bitflip_info[0]['iter']
            print(f"start observation time : {start_observation_time}")
            observation_time = []
            # {'layer': 'model.layer2.1.conv1.weight', 'offset': 139796, 'bitflip': (9, 0), 'iter': 3901}
            for ele in bitflip_info:
                if ele['iter'] == start_observation_time:
                    ele['iter'] = attack_configs['tmp_iter'] - 1
                    if ele['iter'] not in observation_time: observation_time.append(attack_configs['tmp_iter'] - 1)
                    tt_bit_flip_info.append(ele)
            bitflip_info = tt_bit_flip_info
            ImageRecorder.recent_trigger_mode = False

        async_bitflip_info = []
        tmp_trigger_list = copy.deepcopy(ImageRecorder.trigger_list)
        ImageRecorder.trigger_list = []

        for ele in tmp_trigger_list:
            ele_new = (ele[0] + async_step, ele[1])
            ImageRecorder.trigger_list.append(ele_new)
        for ele in bitflip_info:
            ele_new = copy.deepcopy(ele)
            ele_new['iter'] = ele['iter'] + async_step
            async_bitflip_info.append(ele_new)

        tmp_bitflip_info = copy.deepcopy(async_bitflip_info)

    if attack_configs['new_async_attack'] == 'yes':  # change bitflip info and attack time
        observation_time = []
        for ele in tmp_bitflip_info:
            if ele['iter'] not in observation_time:
                observation_time.append(ele['iter'])
        attack_start_time = int(attack_configs['new_async_step'] * max_iter)
        if attack_start_time == 0: attack_start_time = 1
        if attack_configs['new_async_step'] == 1.0: attack_start_time = attack_start_time - 1
        observation_time_map = {ele: attack_start_time + 100 * i for i, ele in enumerate(observation_time)}
        observation_time = list(observation_time_map.values())
        for i, ele in enumerate(tmp_bitflip_info):
            tmp_bitflip_info[i]['iter'] = observation_time_map[ele['iter']]

    def test_asr(model, test_loader, ImageRecorder, current_iter, target_class, device, epoch):
        model.eval()
        count = 0
        criterion = torch.nn.CrossEntropyLoss()
        running_loss = 0.0
        acc_history = []

        m = torch.nn.Softmax(dim=1)
        running_corrects = 0.0
        model.eval()
        confidence = 0.0
        with torch.no_grad():
                cur_trigger = ImageRecorder.get_recent_trigger(current_iter)
                target_labels = (torch.ones(test_loader.batch_size, dtype=torch.int64) * target_class).to(
                    device)
                for i, data in enumerate(test_loader):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    poison_batch_image = ImageRecorder.sythesize_poison_image(inputs, cur_trigger)

                    outputs = model(poison_batch_image)
                    tar_logits = m(outputs)[:, target_class]

                    _, preds = torch.max(outputs, 1)
                    running_corrects += torch.sum(preds == target_labels)
                    confidence += torch.sum(tar_logits)
                    count += inputs.size(0)
                    loss = criterion(outputs, target_labels)
                    running_loss += loss.item() * inputs.size(0)

                epoch_acc = running_corrects.double() / count
                confidence_score = confidence / count
                epoch_loss = running_loss / count
                acc_history.append(epoch_acc)
                print("Attack Process: Epoch {:<5} ASR: {:.2f}% Loss: {:.2f} confidence score {:.2f}".format(epoch, epoch_acc * 100,
                                                                                             epoch_loss,
                                                                                             confidence_score))

        return epoch_acc.item()  # "{:.2f}".format(100*epoch_acc.item())

    def test(model, test_loader, epoch, device):
            model.eval()
            count = 0
            criterion = torch.nn.CrossEntropyLoss()
            running_loss = 0.0
            acc_history = []

            running_corrects = 0.0
            model.eval()
            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    running_corrects += torch.sum(preds == labels.data)
                    count += inputs.size(0)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item() * inputs.size(0)
                epoch_acc = running_corrects / count
                epoch_loss = running_loss / count
                acc_history.append(epoch_acc)
                print("Attack Process: Epoch {:<5} ACC: {:.2f}% Loss: {:.2f}".format(epoch, epoch_acc * 100, epoch_loss))



            return epoch_acc.item()  # "{:.2f}".format(100*epoch_acc.item())

    while not stop_event.is_set():
        time.sleep(0.1)  # Check the iteration info every 0.1 seconds
        current_iter = cur_iteration.value # assume the ability to monitor the iteration of fine-tune process
        cur_epoch = 1 + int(current_iter / attack_configs['loader'].train_loader.__len__())

        if len(observation_time) != 0 and (current_iter - async_step >= observation_time[0]):
            cur_observation_time = observation_time.pop(0)
            print('*' * 100)
            # print(f"Attack Process: launch attack at {current_iter}th iterations")
            # the observed iteration may be larger than the pre-defined time due to 'time.sleep(0.5)'
            # if (current_iter - async_step >= cur_observation_time):
            #     asr = test_asr(model, attack_configs['loader'].test_loader, ImageRecorder, current_iter, attack_configs['target_class'], attack_configs['device'], cur_epoch)
            #     acc = test(model, attack_configs['loader'].test_loader, cur_epoch, attack_configs['device'])

            if len(tmp_bitflip_info) != 0:
                # print(f'Attack Process: current rest bit flip length {len(tmp_bitflip_info)}')
                while current_iter >= tmp_bitflip_info[0]['iter']:
                    time.sleep(0.128) # 64ms x n(2) times (simulate the time interval of a single bit flip)
                    current_bitflip = tmp_bitflip_info.pop(0)
                    print(f'Attack Process: flipping {current_bitflip}')
                    before, after = change_model_weights(model, current_bitflip, record=True)
                    print(f'Attack Process: before: {before}, after: {after}')
                    if len(tmp_bitflip_info) == 0:break


    # change_model_weights(model, bitflip_info, record=True)
    asr = test_asr(model, attack_configs['loader'].test_loader, ImageRecorder, current_iter, attack_configs['target_class'], attack_configs['device'], cur_epoch)
    acc = test(model, attack_configs['loader'].test_loader, cur_epoch, attack_configs['device'])