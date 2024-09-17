import torch
import torch.nn as nn

class LossLib:
    def __init__(self, algorithm_type, loss_weight):
        self.algorithm_type = algorithm_type
        self.loss_weight = loss_weight # [clean, label, neuron]
        self.softmax = nn.Softmax(dim=1)
        self.current_resutls = []
        self.cross_entropy = nn.CrossEntropyLoss()
        self.clean_neuron_gama = 0.0
        print(f"loss library algorithm: {algorithm_type}; loss weight: {loss_weight}")
        pass

    def print_loss_results(self, avg_num=1):
        if len(self.current_resutls) >= 500:
            self.current_resutls = self.current_resutls[-500:]
        average_values = [sum(values) / len(values) for values in zip(*self.current_resutls[-1 * avg_num:])]
        print(f"clean loss {average_values[0]:.3f}; label loss {average_values[1]:.3f}; neuron loss {average_values[2]:.3f}; total loss {average_values[3]:.3f}")

    def neuron_loss(self, fm, selected_neurons, neuron_value, device='cuda:0', target_neuron=True):
        dim0 = fm.size(0)
        scale = 1.0
        if neuron_value != 0.0: scale = 1.0 / (neuron_value * neuron_value)
        if neuron_value == 10.0: scale = 1.0
        fm_target = fm.view(dim0, -1)[:, selected_neurons]
        if target_neuron:
            target = neuron_value * torch.ones_like(fm_target).to(device)
        else:
            target = torch.zeros_like(fm_target).to(device)
        loss = torch.nn.MSELoss(reduction='mean')(fm_target, target) * scale
        return loss
    
    def logits_loss(self, y, target_label=2):
        logits = self.softmax(y)
        target_logits = logits[:, target_label]
        loss = torch.sum(1.0 - target_logits) / target_logits.size(0)
        return loss

    def trigger_loss(self, y, fm, target_label, device, gama=1.0, selected_neuron=[], neuron_value=0.0):
        neuron_loss = self.neuron_loss(fm, selected_neuron, neuron_value, device) if self.loss_weight[2] != 0.0 else torch.tensor(0.0)
        clean_loss = torch.tensor(0.0)
        if self.algorithm_type[0] == 1:
            label_loss = (torch.ones(fm.size()[0], dtype=torch.int64) * target_label).to(device) if self.loss_weight[1] != 0.0 else torch.tensor(0.0)
        elif self.algorithm_type[0] == 2:
            label_loss = self.logits_loss(y, target_label).to(device) if self.loss_weight[1] != 0.0 else torch.tensor(0.0)
        else: raise NotImplementedError
        toal_loss = neuron_loss * self.loss_weight[2] + label_loss * self.loss_weight[1]
        self.current_resutls.append([clean_loss.item(), label_loss.item(), neuron_loss.item(), toal_loss.item()])
        return clean_loss, label_loss, neuron_loss, toal_loss

    def bitsearch_loss(self, y, y_clean, labels, fm, fm_clean, target_label, device, gama=0.0, selected_neuron=[], neuron_value=0.0):
        neuron_loss = self.neuron_loss(fm, selected_neuron, neuron_value, device) if self.loss_weight[2] != 0.0 else torch.tensor(0.0)
        if self.algorithm_type[1] == 1:
            clean_loss = self.cross_entropy(y_clean, labels)
            label_loss = (torch.ones(fm.size()[0], dtype=torch.int64) * target_label).to(device) if self.loss_weight[1] != 0.0 else torch.tensor(0.0)
        elif self.algorithm_type[1] == 2:
            clean_loss = self.cross_entropy(y_clean, labels)
            label_loss = self.logits_loss(y, target_label) if self.loss_weight[1] != 0.0 else torch.tensor(0.0)
        elif self.algorithm_type[1] == 3:
            if self.clean_neuron_gama == 0.0: self.clean_neuron_gama = 1.0 / self.neuron_loss(fm_clean, selected_neuron, 0.0, device)
            clean_loss = self.cross_entropy(y_clean, labels) + self.clean_neuron_gama * self.neuron_loss(fm_clean, selected_neuron, 0.0, device)
            label_loss = self.logits_loss(y, target_label) if self.loss_weight[1] != 0.0 else torch.tensor(0.0)

        else: raise NotImplementedError
        toal_loss = neuron_loss * self.loss_weight[2] + label_loss * self.loss_weight[1] + clean_loss * self.loss_weight[0]
        self.current_resutls.append([clean_loss.item(), label_loss.item(), neuron_loss.item(), toal_loss.item()])
        return clean_loss, label_loss, neuron_loss, toal_loss


