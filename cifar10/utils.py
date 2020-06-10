import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.models.inception import inception_v3
from torch.nn.functional import adaptive_avg_pool2d
import pathlib
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from inception import *
from tqdm import tqdm
import math
from inception_score import IS
import pickle, os, time


import numpy as np

class BenchMark:

    def __init__(self, dset, CONF, dims, device, nz):
        self.Dset=CONF['Dset']
        self.IS = IS(os.path.join(CONF['DataRoot'], 'imagenet_inceptionNet'))
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.device = device
        self.model = InceptionV3([block_idx]).to(device)
        while(os.path.exists(os.path.join(CONF['DataRoot'], CONF['Dset']+"_FID_tmp.pkl.occ"))):
            time.sleep(60)
        if not os.path.exists(os.path.join(CONF['DataRoot'], CONF['Dset']+"_FID_tmp.pkl")):
            os.makedirs(os.path.join(CONF['DataRoot'], CONF['Dset']+"_FID_tmp.pkl.occ"))
            self.m1, self.s1 = self.calculate_activation_statistics(dset, self.model, device, dims=dims)
            pickle.dump({'mean': self.m1, 'std': self.s1}, open(os.path.join(CONF['DataRoot'], CONF['Dset']+"_FID_tmp.pkl"), "wb"))
            os.rmdir(os.path.join(CONF['DataRoot'], CONF['Dset']+"_FID_tmp.pkl.occ"))
        else:
            tmp = pickle.load(open(os.path.join(CONF['DataRoot'], CONF['Dset']+"_FID_tmp.pkl"), "rb"))
            self.m1, self.s1 = (tmp['mean'],tmp['std'])
        self.nz = nz

    def gen_dataset(self, model, num_samples=50000, batch_size=50):
        D=[]
        with torch.no_grad():
            for i in range(num_samples//batch_size):
                noise = torch.randn(batch_size, self.nz, device=self.device)
                fake = model(noise)
                D.append(fake.data.cpu())
            D=torch.cat(D,0)
        return D

    ###############################################
    #                      FID                    #
    ###############################################
    
    def get_activations(self, dset, model, dims=2048, Dset = True):

        model.eval()

        pred_arr = []

        for D in dset:
            batch = D[0] if Dset else D
            batch = batch.to(self.device)
            if self.Dset=='cifar10':
                batch = ((batch*0.5)+0.5)
            batch_size = batch.shape[0]
            
            pred = model(batch)

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.shape[2] != 1 or pred.shape[3] != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred_arr.append(pred.cpu().numpy().reshape(batch_size, -1))

        return np.concatenate(pred_arr)
    
    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

        Stable version by Dougal J. Sutherland.

        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.

        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)

    def calculate_activation_statistics(self, dset, model, batch_size=50,
                                        dims=2048, cuda=False, verbose=False, Dset = True):
        act = self.get_activations(dset, model, dims, Dset = Dset)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    def compute_statistics_of_data(self, data, dim_feature=2048, num_samples=50000, batch_size=50):
        with torch.no_grad():
            dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=2)
            m, s = self.calculate_activation_statistics(dataloader, self.model, batch_size, dim_feature, Dset=False)
        return m, s

    def calculate_fid(self, m, dim_feature=2048, num_samples=50000, batch_size=50):
        D = self.gen_dataset(model, num_samples, batch_size)
        dataloader = torch.utils.data.DataLoader(D, batch_size=batch_size, shuffle=False, num_workers=2)
        m2, s2 = self.calculate_activation_statistics(dataloader, self.model, batch_size, dim_feature, Dset=False)
        return self.calculate_frechet_distance(self.m1,self.s1,m2,s2)

    ###############################################
    #                      IS                     #
    ###############################################
    
    def calculate_IS(self, model, num_samples=50000, batch_size=50):
        D = self.gen_dataset(model, num_samples, batch_size)
        if self.Dset=='cifar10':
            D = (((D.cpu().numpy()*0.5)+0.5)*255).astype('int32')
        else:
            raise Exception("Not Implemented")
        return self.IS.get_inception_score(D.transpose(0,2,3,1), 10)


    ###############################################
    #                     BOTH                    #
    ###############################################

    def calculate_fid_IS(self, model, dim_feature=2048, num_samples=50000, batch_size=128):
        D = self.gen_dataset(model, num_samples, batch_size)
        dataloader = torch.utils.data.DataLoader(D, batch_size=batch_size, shuffle=False, num_workers=2)
        m2, s2 = self.calculate_activation_statistics(dataloader, self.model, batch_size, dim_feature, Dset=False)
        fid = self.calculate_frechet_distance(self.m1,self.s1,m2,s2)
        if self.Dset=='cifar10':
            D = ((D.cpu().numpy()+1.)*255/2).astype('int32')
        else:
            raise Exception("Not Implemented")
        return self.IS.get_inception_score(D.transpose(0,2,3,1), 10), fid 
        

class CheckPoint(object):

    def __init__(self, checkpoints, epoch_length):
        L=[]
        for r,v in checkpoints:
            L.append((math.ceil(epoch_length*r), v))
        self.checkpoints=L
        self.epoch = 0
    
    def step(self, value):
        self.epoch+=1
        epoch = self.epoch
        if len(self.checkpoints)==0:
            return True
        e, v = self.checkpoints[0]
        if epoch>=e:
            self.checkpoints=self.checkpoints[1:]
            return v<=value
        else:
            return True

    def update(self, epoch, value):
        self.epoch = epoch
        if len(self.checkpoints)==0:
            return True
        e, v = self.checkpoints[0]
        if epoch>=e:
            self.checkpoints=self.checkpoints[1:]
            return v<=value
        else:
            return True
    
    def next(self):
        if len(self.checkpoints)==0:
            return 0, 0
        return self.checkpoints[0]