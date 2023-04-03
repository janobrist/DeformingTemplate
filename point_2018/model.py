import torch.nn as nn
from cadex_encoder import ResnetPointnet
import torch as torch
from nvp_cadex import NVP_v2_5_frame
import torch.nn.functional as F

class InstanceNorm(nn.Module):
    '''
    Instance Normalization. Refer Section 3 in paper (Instance Normalization) for more details and Fig 5 for visual intuition.
    This is crucial for cross-category training.
    '''
    def __init__(self, zdim):
        super(InstanceNorm, self).__init__()
        '''
        Initialization.
        '''
        self.norm = nn.Sequential(nn.Linear(zdim, 256), nn.ReLU(), nn.Linear(256, 3), nn.Sigmoid()).float()
        
    def forward(self, input, code):
        '''
        input: point cloud of shape BxNx3
        code: shape embedding of shape Bx1xzdim
        '''
        centered_input = input - torch.mean(input, axis=1, keepdim=True)   # Center the point cloud
        centered_input = centered_input*(1-self.norm(code))    # anisotropic scaling the point cloud w.r.t. the shape embedding
        
        return centered_input


class NMF_Encoder(nn.Module):
    '''
    PointNet Encoder by Qi. et.al
    '''
    def __init__(self, zdim, input_dim=3):
        super(NMF_Encoder, self).__init__()
        
        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, zdim, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(zdim)

        self.fc1 = nn.Linear(zdim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_bn1 = nn.BatchNorm1d(256)
        self.fc_bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, zdim)
        

    def forward(self, x):
        '''
        Input: Nx#ptsx3
        Output: Nxzdim
        '''
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.zdim)

        
        ms = F.relu(self.fc_bn1(self.fc1(x)))
        ms = F.relu(self.fc_bn2(self.fc2(ms)))
        ms = self.fc3(ms)

        return ms
    

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


        #homeomorphism_decoder:
        n_layers = 6
        #dimension of the code
        feature_dims = 128
        hidden_size = [128, 64, 32, 32, 32]
        #the dimension of the coordinates to be projected onto
        proj_dims = 128
        code_proj_hidden_size = [128, 128, 128]
        proj_type = 'simple'
        block_normalize = True
        normalization = False

        self.deform=NVP_v2_5_frame(n_layers=n_layers, feature_dims=feature_dims, hidden_size=hidden_size, proj_dims=proj_dims,\
        code_proj_hidden_size=code_proj_hidden_size, proj_type=proj_type, block_normalize=block_normalize, normalization=normalization).to('cuda')


            

                

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, 3, num_points].
        Returns:
            encoding: a float tensor with shape [b, k].
            restoration: a float tensor with shape [b, 3, num_points].
        """
        
        encode = 'auto'
        decode='decode' #'decodeDeformNVP'
        instanceNor=False

        if(encode == 'pointnet'):
            b, _, num_points = x.size()
            #print('shape of x: ', x.shape)
            x = x.permute( (0, 2, 1))
            print('shape of x after permutation: ', x.shape) #[5,3000,3]
            #print('num_points: ', num_points)
            encoding = self.homeomorphism_encoder(x)
            b, k = encoding.shape
            encoding = encoding.reshape(b,k,1)
        elif(encode == 'auto' ):
            b, _, num_points = x.size()
            x = self.pointwise_layers(x)  # shape [b, k, num_points]
            encoding = self.pooling(x)  # shape [b, k, 1]
        elif(encode == 'nmf'):
            x = x.permute( (0, 2, 1))
            encoding=self.encoder_nmf(x)
            print('encoding shape: ', encoding.shape)


        
        #print('encoding shape: ', encoding.shape)
        if(decode=='decodeDeformNVP'):
            encoding = encoding.reshape(b,k)
            #print('######## encoding shape: ', encoding.shape)
            #print('######## x shape: ', x.shape)
            x = self.deform.forward(encoding, x)
            
            restoration = x.view(b, 3, num_points)
        else:
            if(instanceNor):
                encoding=self.norm0(encoding)
            print('encoding shape: ', encoding.shape) # 16, 256, 1
            x = self.decoder(encoding)  # shape [b, num_points * 3, 1]
            print('x shape before viewing: ', x.shape)
            restoration = x.view(b, 3, num_points)

        return encoding, restoration
