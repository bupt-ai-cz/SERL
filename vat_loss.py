import torch.nn as nn
import torch
import torch.nn.functional as F
import contextlib


class VATLoss(nn.Module):

    def __init__(self, xi=10.0, eps=1, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, netF,netC, x):
        # print(torch.mean(x))
        with torch.no_grad():
            pred = F.softmax(netC(netF(x)), dim=1)
            # softmax_out = nn.Softmax(dim=1)(pred)
            # entropy = Entropy(softmax_out)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)
        # print("d",torch.mean(d))
        with _disable_tracking_bn_stats(netF):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = netC(netF(x + self.xi * d))
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                netF.zero_grad()
                netC.zero_grad()

            # calc LDS
            # r_adv = d * self.eps
            # print("d, entropy", d.shape, torch.mean(entropy))
            # entropy=entropy.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 3, 224, 224)
            # entropy=torch.sigmoid(entropy)

            # r_adv = d * self.eps*(10/torch.exp(entropy))
            r_adv = d * self.eps
            # r_adv = d * self.eps*entropy
            pred_hat = netC(netF((x + r_adv)))
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')

        return lds
    

@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d