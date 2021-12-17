import torch
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module
import torch.nn.init as init

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #init.constant_(m.weight.data, 0)
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
            #init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)


class DNAnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Linear(512, 512),
        )
    
        self.decoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Linear(512, 512),
        )
        
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                weights_init(m)

    def forward(self, m, f, alpha=None):
        m_gene = self.encoder(m)
        f_gene = self.encoder(f)

        if alpha is not None:
            s_gene = alpha * m_gene + (1 - alpha) * f_gene
        else:
            s_gene = torch.max(torch.cat((m_gene.unsqueeze(0), f_gene.unsqueeze(0)), 0), dim=0)[0]
            if m.size(0) != 1:
                s_gene = s_gene.squeeze(0)
        s = self.decoder(s_gene)
        return s


if __name__ == '__main__':
    net = DNAnet().to('cuda')
    m = torch.randn(3, 512).to('cuda')
    f = torch.randn(3, 512).to('cuda')
    s = net(m, f)
    print(s.size())