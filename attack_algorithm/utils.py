import numpy
import os.path
import numpy as np
import torch, struct, random
from PIL import Image
import time
import copy
################################Binary Float Convertion###################################################

getBin = lambda x: x > 0 and str(bin(x))[2:] or "-" + str(bin(x))[3:]

# float64 conversion
def floatToBinary64(value):
    val = struct.unpack('Q', struct.pack('d', value))[0]
    s = getBin(val)
    if len(s) != 64:
        s = '0' * (64 - len(s)) + s
    return s

def binaryToFloat(value):
    if value[0] == '0':
        hx = hex(int(value, 2))
        return struct.unpack("d", struct.pack("q", int(hx, 16)))[0]
    if value[0] == '1':
        hx = hex(int(value[1:], 2))
        return -struct.unpack("d", struct.pack("q", int(hx, 16)))[0]

# fault injection simulation in float64
def change_model_weights64(current_model, bitflip_info, record=False, fake_flip=False, bf_success_rate=1.0, verbose=True):
    # {'layer': 'layer2.0.downsample.0.weight', 'layer_max_range': 0.7840863466262817, 'offset': 2279,
    # 'weight': 0.04178149253129959, 'grad': -13.256144523620605, 'score': 9.840100288391113,
    # 'weight_after': 0.6685038805007935, 'bitflip': (9, 0), 'multi': 16}
    # assert hasattr(bitflip_info[0], 'layer') and hasattr(bitflip_info[0], 'offset') and hasattr(bitflip_info[0], 'bitflip')


    if fake_flip: print(f'Fake flip')

    def change_model_weight(current_model, bitflip, record=False, fake_flip=False, bf_success_rate=1.0, verbose=verbose):

        current_model.eval()
        cur_tensor = current_model.state_dict()[bitflip['layer']]
        if bitflip['offset'] >= current_model.state_dict()[bitflip['layer']].view(-1, ).size()[0] or bitflip['offset'] < 0:
            print(f'bit flip at {bitflip["offset"]} is out of current layer offset {current_model.state_dict()[bitflip["layer"]].view(-1, ).size()[0]}, skip it')
            return 0.0, 0.0

        desired_weight = torch.Tensor().to(cur_tensor.device).set_(cur_tensor.storage(), storage_offset=bitflip['offset'], size=(1,), stride=(1,))
        attacked_weight = current_model.state_dict()[bitflip['layer']].view(-1, )[bitflip['offset']].detach().item()

        if verbose:
            print("Flipped layer: {} & Offset {}".format(bitflip['layer'], bitflip['offset']))
        print(f"Before: {desired_weight.item():.2f}", end=" ")

        if fake_flip and False:
            print("wrong channel, never enter this function")
            flipped_weight = 2 ** (((-1) ** (bitflip['bitflip'][1])) * 2 ** (11 - bitflip['bitflip'][0])) * \
                             desired_weight.item()
        else:
            binary = floatToBinary64(desired_weight.item())
            binary_new = binary[:bitflip['bitflip'][0]] + str(1 - bitflip['bitflip'][1]) + binary[
                                                                                           bitflip['bitflip'][0] + 1:]
            flipped_weight = binaryToFloat(binary_new)
            success_value = numpy.random.random()
            if success_value <= bf_success_rate: desired_weight.data[0] = flipped_weight
            else: print(f'fail to flip bit in simulation: {success_value:.2f} > {bf_success_rate}')

        # if verbose:
        #     value = current_model.state_dict()[bitflip['layer']].view(-1, )[bitflip['offset']].item()
        print(f"After: {desired_weight.item():.2f}")

        if record:
            return attacked_weight, flipped_weight

    if isinstance(bitflip_info, list):
        # print("change weights in list")
        for bitflip in bitflip_info:
            change_model_weight(current_model, bitflip, record, fake_flip, bf_success_rate)
    elif isinstance(bitflip_info, dict):
        if record:
            w_before, w_after = change_model_weight(current_model, bitflip_info, record, bf_success_rate=bf_success_rate)
            return w_before, w_after
        else:
            change_model_weight(current_model, bitflip_info, record, fake_flip, bf_success_rate)
    else: raise Exception('Can not identify current instance: {}'.format(bitflip_info))

# float32 conversion
def floatToBinary32(value):
    # Convert float to binary32
    val = struct.unpack('I', struct.pack('f', value))[0]
    s = bin(val)[2:]
    if len(s) != 32:
        s = '0' * (32 - len(s)) + s
    return s

def binary32ToFloat(binary_str):
    # Convert binary32 to float
    val = int(binary_str, 2)
    return struct.unpack('f', struct.pack('I', val))[0]

# fault injection simulation in float32
def change_model_weights(current_model, bitflip_info, record=False, fake_flip=False, bf_success_rate=1.0, verbose=False):
    # {'layer': 'layer2.0.downsample.0.weight', 'layer_max_range': 0.7840863466262817, 'offset': 2279,
    # 'weight': 0.04178149253129959, 'grad': -13.256144523620605, 'score': 9.840100288391113,
    # 'weight_after': 0.6685038805007935, 'bitflip': (9, 0), 'multi': 16}
    # assert hasattr(bitflip_info[0], 'layer') and hasattr(bitflip_info[0], 'offset') and hasattr(bitflip_info[0], 'bitflip')

    total_bits = len(bitflip_info)
    change_model_weights.total_failed_bits = 0
    if fake_flip: print(f'Fake flip')

    def change_model_weight(current_model, bitflip, record=False, fake_flip=False, bf_success_rate=1.0, verbose=verbose):

        current_model.eval()
        cur_tensor = current_model.state_dict()[bitflip['layer']]
        if bitflip['offset'] >= current_model.state_dict()[bitflip['layer']].view(-1, ).size()[0] or bitflip['offset'] < 0:
            print(f'bit flip at {bitflip["offset"]} is out of current layer offset {current_model.state_dict()[bitflip["layer"]].view(-1, ).size()[0]}, skip it')
            return 0.0, 0.0

        desired_weight = torch.Tensor().to(cur_tensor.device).set_(cur_tensor.storage(), storage_offset=bitflip['offset'], size=(1,), stride=(1,))
        attacked_weight = current_model.state_dict()[bitflip['layer']].view(-1, )[bitflip['offset']].detach().item()

        if verbose:
            print("Flipped layer: {} & Offset {}".format(bitflip['layer'], bitflip['offset']))

            print(f"Before: {desired_weight.item():.2f}", end=" ")

        if fake_flip and False:
            print("wrong channel, never enter this function")
            flipped_weight = 2 ** (((-1) ** (bitflip['bitflip'][1])) * 2 ** (8 - bitflip['bitflip'][0])) * \
                             desired_weight.item()
        else:
            binary = floatToBinary32(desired_weight.item())
            binary_new = binary[:bitflip['bitflip'][0]] + str(1 - bitflip['bitflip'][1]) + binary[
                                                                                           bitflip['bitflip'][0] + 1:]
            if binary == binary_new: change_model_weights.total_failed_bits += 1
            flipped_weight = binary32ToFloat(binary_new)
            success_value = numpy.random.random()
            if success_value <= bf_success_rate:
                # current_model.state_dict()[bitflip['layer']].view(-1, )[bitflip['offset']].data = torch.tensor(flipped_weight)
                desired_weight.data[0] = flipped_weight
            else: print(f'fail to flip bit in simulation: {success_value:.2f} > {bf_success_rate}')

        # if verbose:
        #     value = current_model.state_dict()[bitflip['layer']].view(-1, )[bitflip['offset']].item()
        if verbose:
            print(f"After: {desired_weight.item():.2f}")

        if record:
            return attacked_weight, desired_weight.item()

    if isinstance(bitflip_info, list):
        # print("change weights in list")
        for t, bitflip in enumerate(bitflip_info):
            change_model_weight(current_model, bitflip, record, fake_flip, bf_success_rate, verbose=True)
    elif isinstance(bitflip_info, dict):
        if record:
            w_before, w_after = change_model_weight(current_model, bitflip_info, record, bf_success_rate=bf_success_rate)
            return w_before, w_after
        else:
            change_model_weight(current_model, bitflip_info, record, fake_flip, bf_success_rate)
    else: raise Exception('Can not identify current instance: {}'.format(bitflip_info))
    print(f'total bit flips: {total_bits}; fail flip rate: {change_model_weights.total_failed_bits/total_bits}')

def change_model_weights32(current_model, bitflip_info, record=False, fake_flip=False, bf_success_rate=1.0, verbose=False):
    # {'layer': 'layer2.0.downsample.0.weight', 'layer_max_range': 0.7840863466262817, 'offset': 2279,
    # 'weight': 0.04178149253129959, 'grad': -13.256144523620605, 'score': 9.840100288391113,
    # 'weight_after': 0.6685038805007935, 'bitflip': (9, 0), 'multi': 16}
    # assert hasattr(bitflip_info[0], 'layer') and hasattr(bitflip_info[0], 'offset') and hasattr(bitflip_info[0], 'bitflip')

    total_bits = len(bitflip_info)
    change_model_weights.total_failed_bits = 0
    if fake_flip: print(f'Fake flip')

    def change_model_weight(current_model, bitflip, record=False, fake_flip=False, bf_success_rate=1.0, verbose=verbose):

        current_model.eval()
        cur_tensor = current_model.state_dict()[bitflip['layer']]
        if bitflip['offset'] >= current_model.state_dict()[bitflip['layer']].view(-1, ).size()[0] or bitflip['offset'] < 0:
            print(f'bit flip at {bitflip["offset"]} is out of current layer offset {current_model.state_dict()[bitflip["layer"]].view(-1, ).size()[0]}, skip it')
            return 0.0, 0.0

        desired_weight = torch.Tensor().to(cur_tensor.device).set_(cur_tensor.storage(), storage_offset=bitflip['offset'], size=(1,), stride=(1,))
        attacked_weight = current_model.state_dict()[bitflip['layer']].view(-1, )[bitflip['offset']].detach().item()

        if verbose:
            print("Flipped layer: {} & Offset {}".format(bitflip['layer'], bitflip['offset']))

            print(f"Before: {desired_weight.item():.2f}", end=" ")

        if fake_flip and False:
            print("wrong channel, never enter this function")
            flipped_weight = 2 ** (((-1) ** (bitflip['bitflip'][1])) * 2 ** (8 - bitflip['bitflip'][0])) * \
                             desired_weight.item()
        else:
            binary = floatToBinary32(desired_weight.item())
            binary_new = binary[:bitflip['bitflip'][0]] + str(1 - bitflip['bitflip'][1]) + binary[
                                                                                           bitflip['bitflip'][0] + 1:]
            if binary == binary_new: change_model_weights.total_failed_bits += 1
            flipped_weight = binary32ToFloat(binary_new)
            success_value = numpy.random.random()
            if success_value <= bf_success_rate: desired_weight.data[0] = flipped_weight
            else: print(f'fail to flip bit in simulation: {success_value:.2f} > {bf_success_rate}')

        # if verbose:
        #     value = current_model.state_dict()[bitflip['layer']].view(-1, )[bitflip['offset']].item()
        if verbose:
            print(f"After: {desired_weight.item():.2f}")

        if record:
            return attacked_weight, flipped_weight

    if isinstance(bitflip_info, list):
        # print("change weights in list")
        for t, bitflip in enumerate(bitflip_info):
            change_model_weight(current_model, bitflip, record, fake_flip, bf_success_rate, verbose=True if t <=10 else False)
    elif isinstance(bitflip_info, dict):
        if record:
            w_before, w_after = change_model_weight(current_model, bitflip_info, record, bf_success_rate=bf_success_rate)
            return w_before, w_after
        else:
            change_model_weight(current_model, bitflip_info, record, fake_flip, bf_success_rate)
    else: raise Exception('Can not identify current instance: {}'.format(bitflip_info))
    print(f'total bit flips: {total_bits}; fail flip rate: {change_model_weights.total_failed_bits/total_bits}')


################################Basic Functions###################################################

def all_pos_neg(lst):
    """
    Check if all elements in the list are either positive or negative.

    Args:
        lst (list): A list of numerical values.

    Returns:
        bool: True if all elements are positive or all elements are negative, False otherwise.
    """
    return all(val > 0 for val in lst) or all(val < 0 for val in lst)

def intersection(lst1, lst2):
    """
    Find the intersection of two lists.

    Args:
        lst1 (list): The first list.
        lst2 (list): The second list.

    Returns:
        list: A list containing the elements that are present in both lst1 and lst2.
    """
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def set_seed(seed):
    """
    Set the seed for random number generators in Python, numpy, and PyTorch.

    Args:
        seed (int or None): The seed value to set. If None, use a random seed.
    """
    if seed is None:
        random.seed()
        np.random.seed(None)
        torch.manual_seed(int(torch.initial_seed()))
        torch.cuda.manual_seed_all(int(torch.initial_seed()))
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def rank(data, key, reverse=True):
    """
    Sort a list of dictionaries based on a specified key.

    Args:
        data (list): A list of dictionaries.
        key (str): The key to sort by.
        reverse (bool): Whether to sort in descending order. Default is True.

    Returns:
        list: The sorted list of dictionaries.
    """
    return sorted(data, key=lambda x: x[key], reverse=reverse)

def zero_gradients(object):
    """
    Zero the gradients of a tensor or a list of tensors.

    Args:
        object (torch.Tensor or list or model): The object whose gradients need to be zeroed.
    """
    if torch.is_tensor(object):
        if object.grad is not None:
            object.grad.zero_()
    elif isinstance(object, list):
        for ele in object:
            zero_gradients(ele)
    else:
        for param in object.parameters():
            if param.grad is not None:
                param.grad.zero_()

def ensemble_ban_unstable_bit(lst):
    """
    Identify unstable bits in the binary representation of floating-point numbers.

    Args:
        lst (list): A list of floating-point numbers.

    Returns:
        list: A list of bit positions (5 to 8) that are unstable.
    """
    ban_bit = []
    binary_sample_5 = [floatToBinary32(x)[5] for x in lst]
    binary_sample_6 = [floatToBinary32(x)[6] for x in lst]
    binary_sample_7 = [floatToBinary32(x)[7] for x in lst]
    binary_sample_8 = [floatToBinary32(x)[8] for x in lst]
    if len(np.unique(binary_sample_5)) > 1 : ban_bit.append(5)
    if len(np.unique(binary_sample_6)) > 1 : ban_bit.append(6)
    if len(np.unique(binary_sample_7)) > 1: ban_bit.append(7)
    if len(np.unique(binary_sample_8)) > 1: ban_bit.append(8)

    return ban_bit

def deterministic_run(SEED):
    """
    Set the random seed for reproducibility in Python, numpy, and PyTorch.

    Args:
        SEED (int): The seed value to set.
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def verify_biteffect(weight_sign, grad_sign, abs_w_change_dirct):
    """
    Verify the bit effect based on weight sign, gradient sign, and weight change direction.
    """
    if grad_sign == weight_sign and abs_w_change_dirct == 0: return 0
    if grad_sign != weight_sign and abs_w_change_dirct == 1: return 0
    return 1

def get_sign(value):
    """
    Get the sign of a value or a list of values.

    Args:
        value (float or list): A float or a list of floats.

    Returns:
        int or list: The sign of the float (0 for negative, 1 for positive), or a list of signs.
    """
    if isinstance(value, float):
        if value < 0: return 0
        else: return 1
    elif isinstance(value, list):
        return [get_sign(ele) for ele in value]

def safe_update(current_state_dict, filtered_state_dict):
    """
    Safely update the current state dictionary with the filtered state dictionary.

    Args:
        current_state_dict (dict): The current state dictionary.
        filtered_state_dict (dict): The filtered state dictionary to update with.
    """
    for key in filtered_state_dict.keys():
        if key in current_state_dict:
            if current_state_dict[key].size() == filtered_state_dict[key].size():
                current_state_dict[key] = filtered_state_dict[key]
            else:
                print(f"safe update inconsistent size: {current_state_dict[key].size()}, {filtered_state_dict[key].size()}")

def get_gpu_names():
    """
    Get the names of all available GPUs.

    Returns:
        list: A list of GPU names.
    """
    return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

def save_trigger_to_image(tensor, filename_prefix, directory='trigger'):
    """
    Converts a given tensor of shape (3, H, W) to an image and saves it.

    Args:
        tensor (torch.Tensor): The input tensor with shape (3, H, W).
        filename_prefix (str): The prefix for the saved image filename.

    Returns:
        None
    """
    # Ensure the tensor is on CPU and detach from the computational graph
    tensor = tensor.cpu().detach()

    # Normalize tensor to the range [0, 255] for image representation
    tensor_np = tensor.numpy()
    tensor_np = (tensor_np * 255).astype(np.uint8)

    # Convert the tensor to a PIL Image
    img = Image.fromarray(np.transpose(tensor_np, (1, 2, 0)))  # Convert from (C, H, W) to (H, W, C)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, f"{filename_prefix}.png")

    # Save the image with the given filename prefix
    img.save(filepath)
    print(f"Trigger is saved in {filepath}")

# def save_bitflip_info_to_file(bitflip_info, filename_prefix, directory='bitflip'):
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     filepath = os.path.join(directory, f"{filename_prefix}.txt")
#     with open(filepath, 'w') as file:
#         for index, entry in enumerate(bitflip_info):
#             # file.write(f"Entry {index + 1}:\n")
#             for key, value in entry.items():
#                 file.write(f"  {key}: {value},\t")
#             file.write("\n")  # Add a blank line between entries for readability

from collections import defaultdict
def save_bitflip_info_to_file(bitflip_info, filename_prefix, directory='bitflip'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, f"{filename_prefix}.txt")

    # Group entries by 'iter'
    grouped_info = defaultdict(list)
    for entry in bitflip_info:
        grouped_info[entry['iter']].append(entry)

    # Write grouped info to the file
    with open(filepath, 'w') as file:
        for round_num, (iteration, entries) in enumerate(grouped_info.items(), start=1):
            file.write(f"Attack Round {round_num} at Iteration {iteration}:\n")
            for entry in entries:
                file.write(
                    f"  layer: {entry['layer']},\t  offset: {entry['offset']},\t  bitflip: {entry['bitflip']},\n")
            file.write("\n")  # Add a blank line between rounds for readability








