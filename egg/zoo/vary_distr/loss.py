import torch.nn.functional as F

def loss(dataset):

    def loss_(_sender_input, _message, _receiver_input, receiver_output, labels):
        pred = receiver_output.argmax(dim=1)
        acc = (pred == labels).detach().float()
        loss = F.cross_entropy(receiver_output, labels, reduction="none")
        n_distractor = dataset.n_distractors(_sender_input)
        return loss, {'acc': acc, 'n_distractor': n_distractor}

    return loss_


