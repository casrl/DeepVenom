import torch
import os


def fine_tune(model, cur_iteration, user_configs, stop_event):

    loader = user_configs['loader']
    user_optim = user_configs['user_optimizer']
    max_iter = user_configs['max_iter']
    device = user_configs['device']
    user_ft_mode = user_configs['user_ft_mode']
    various_lr = user_configs['various_lr']
    def optim_init():
        if not various_lr:
            if user_optim == "Adam":
                return torch.optim.Adam(model.parameters(), lr=user_configs['lr'], weight_decay=1e-5)
            elif user_optim == "SGD":
                return torch.optim.SGD(model.parameters(), lr=user_configs['lr'], momentum=0.9, weight_decay=1e-5)
        else:
            print('load various LR optimizer')
            lr_base = user_configs['lr']
            # Count the total number of layers to compute 'n'
            n = sum(1 for _ in model.parameters())  # Total number of parameters, not layers
            param_groups = []

            # Example of assigning custom LR based on parameter position (not directly feasible)
            # This assumes each parameter is uniquely identifiable and can be mapped to a "depth" or position k
            for k, param in enumerate(model.parameters(), 1):  # Enumerate parameters starting at 1
                lr_k = (1 - k / n) * lr_base * 9 + lr_base
                param_groups.append({'params': [param], 'lr': lr_k})

            if user_optim == "Adam":
                return torch.optim.Adam(param_groups, weight_decay=1e-5)
            elif user_optim == "SGD":
                return torch.optim.SGD(param_groups, momentum=0.9, weight_decay=1e-5)

    optimizer = optim_init()

    if user_ft_mode != 'normal':
        def init_ft_mode_data(model, user_ft_mode, device):
            from utils import binary32ToFloat, floatToBinary32
            if user_ft_mode == 'limit_value':
                dct_min = {}
                dct_max = {}
                for name, param in model.named_parameters():
                    dct_min[name] = torch.min(param.data).to(device)
                    dct_max[name] = torch.max(param.data).to(device)
                print('defense clamp: ')
                for key in dct_min.keys():
                    print(f'{dct_min[key]} to {dct_max[key]}')
                return [dct_min, dct_max]
            elif user_ft_mode == 'lock_exp':
                path = 'saved_model'
                cfg_name = 'lock_exp_' + model.model_name + '.pt'
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

                    for name, param in model.named_parameters():
                        dict_min[name], dict_max[name] = get_tensor_min_max_with_lock_exp(param.data)
                    lst = [dict_min, dict_max]
                    torch.save(lst, cur_file)
                cuda_lst = []
                for dct in lst:
                    for key in dct.keys():
                        dct[key] = dct[key].to(device)
                    cuda_lst.append(dct)

                return cuda_lst
            else:
                return None
        # prepare defense related data. e.g., storing the min and max value of each layer
        fine_tune.user_ft_mode_data = init_ft_mode_data(model, user_ft_mode, device)

    def defense_check(model):
        if user_ft_mode == 'normal':
            return

        ban_name_dict = {
            'vgg16': ['classifier.6.weight', 'classifier.6.bias'],  # 'classifier.3.weight', 'classifier.3.bias'],
            'vgg16_bn': ['classifier.6.weight', 'classifier.6.bias'],
            'resnet50': ['fc.weight', 'fc.bias'],  # layer4.2.conv3.weight
            'resnet18': ['fc.weight', 'fc.bias'],
            'vit': ['heads.head.weight', 'heads.head.bias'],
        }
        ban_name_lst = ban_name_dict[model.model_name]
        ban_name_lst = ['model.' + ele for ele in ban_name_lst]
        if user_ft_mode == 'lock_exp':
            dict_min = fine_tune.user_ft_mode_data[0]
            dict_max = fine_tune.user_ft_mode_data[1]
            state_dict = model.state_dict()

            for name, param in model.named_parameters():
                if name in ban_name_lst: continue
                param_over_min = param.where(param > dict_min[name], dict_min[name])
                param_over_max = param_over_min.where(param_over_min < dict_max[name], dict_max[name])
                state_dict[name] = param_over_max
            model.load_state_dict(state_dict)
        if user_ft_mode == 'limit_value':
            dict_min = fine_tune.user_ft_mode_data[0]
            dict_max = fine_tune.user_ft_mode_data[1]
            state_dict = model.state_dict()

            for name, param in model.named_parameters():
                if name in ban_name_lst: continue
                param_over_min = param.where(param > dict_min[name], dict_min[name])
                param_over_max = param_over_min.where(param_over_min < dict_max[name], dict_max[name])
                state_dict[name] = param_over_max
            model.load_state_dict(state_dict)

        pass

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
                print("Epoch {:<5} ACC: {:.2f}% Loss: {:.2f}".format(epoch, epoch_acc * 100, epoch_loss))



            return epoch_acc.item()  # "{:.2f}".format(100*epoch_acc.item())

    def train(model, cur_iteration, loader, optimizer, max_iter, device):

        train_loader = loader.train_loader
        current_epoch = 1
        current_iteration = 0
        model.train()
        criterion = torch.nn.CrossEntropyLoss()

        while current_iteration < max_iter:
            print("*" * 100)
            print(f"Iter: {current_iteration}/{max_iter} Epoch {current_epoch}")
            running_loss = 0.0
            for i, data in enumerate(train_loader):
                model.train()
                current_iteration += 1
                cur_iteration.value = current_iteration # iteration information will be sent to attacker
                model.zero_grad()
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size()[0]

                defense_check(model)

            test(model, loader.test_loader, current_epoch, device)
            epoch_loss = running_loss / len(train_loader.dataset)
            print("Epoch {:<5} Train loss: {:.4f}".format(current_epoch, epoch_loss))
            current_epoch += 1

    train(model, cur_iteration, loader, optimizer, max_iter, device)

    stop_event.set()