import torch.nn as nn
from cadex_encoder import ResnetPointnet
import torch as torch

import torch.nn.functional as F


    
class Autoencoder(nn.Module):

    def __init__(self, k, num_points):
        """
        Arguments:
            k: an integer, dimension of the representation vector.
            num_points: an integer.
        """
        super(Autoencoder, self).__init__()

        # ENCODER

        pointwise_layers = []
        num_units = [3, 64, 128, 128, 256, k]

        for n, m in zip(num_units[:-1], num_units[1:]):
            pointwise_layers.extend([
                nn.Conv1d(n, m, kernel_size=1, bias=False),
                nn.BatchNorm1d(m),
                nn.ReLU(inplace=True)
            ])

        self.pointwise_layers = nn.Sequential(*pointwise_layers)
        self.pooling = nn.AdaptiveMaxPool1d(1)


        c_dim =k
        hidden_dim = 128
        self.homeomorphism_encoder = ResnetPointnet(dim=3, c_dim=c_dim, hidden_dim=hidden_dim).to('cuda')

        #self.encoder_nmf= NMF_Encoder(c_dim)

        # DECODER
        #self.norm0 = InstanceNorm(k)
        self.decoder = nn.Sequential(
            nn.Conv1d(k, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, num_points * 3, kernel_size=1)
        )

     

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, 3, num_points].
        Returns:
            encoding: a float tensor with shape [b, k].
            restoration: a float tensor with shape [b, 3, num_points].
        """
        
        encode = 'auto2018'
        #decode='decode' #'decodeDeformNVP'


        if(encode == 'cadex_encoder'):
            b, _, num_points = x.size()
            #print('shape of x: ', x.shape)
            x = x.permute( (0, 2, 1))
            #print('shape of x after permutation: ', x.shape) #[5,3000,3]
            #print('num_points: ', num_points)
            encoding = self.homeomorphism_encoder(x)
            b, k = encoding.shape
            encoding = encoding.reshape(b,k,1)
        elif(encode == 'auto2018' ):
            #print('size of x: ', x.size())
            b, _, num_points = x.size()
            x = self.pointwise_layers(x)  # shape [b, k, num_points]
            encoding = self.pooling(x)  # shape [b, k, 1]


        #print('encoding shape: ', encoding.shape) # 16, 256, 1
        x = self.decoder(encoding)  # shape [b, num_points * 3, 1]
        #print('x shape before viewing: ', x.shape)
        restoration = x.view(b, 3, num_points)

        return encoding.squeeze(2), restoration
