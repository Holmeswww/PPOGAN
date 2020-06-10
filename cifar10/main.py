from printlib import print_normal as print
import argparse
import os, sys
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default="./",
                    help='GPUs to use.')
parser.add_argument('--snapshot_dir', type=str, default="./",
                    help='snapshot')
parser.add_argument('--tb_dir', type=str, default="./",
                    help='tensorboard')
parser.add_argument('--id', type=str, help='unique identifier')
opt = parser.parse_args()

sys.stderr = open(os.path.join(opt.output_dir,'err.txt'), 'w')

import yaml
CONF = yaml.load(open(os.path.join(opt.output_dir,'conf.yml')), Loader=yaml.FullLoader)
print(caption = "STARTED")

print(CONF)

LAMBDA = CONF['LAMBDA']
nz = CONF['nz']

import random
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from utils import BenchMark as BM
from utils import CheckPoint
from torch import autograd

writer = SummaryWriter(log_dir=os.path.join(opt.tb_dir, opt.id))

random.seed(CONF['SEED'])
torch.manual_seed(CONF['SEED'])

cudnn.benchmark = True

Dset=CONF['Dset']
DataRoot = CONF['DataRoot']
ImageSize=CONF['ImageSize']
batchSize=CONF['BS']

if Dset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=DataRoot,
                               transform=transforms.Compose([
                                   transforms.Resize(ImageSize),
                                   transforms.CenterCrop(ImageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    nc=3
elif Dset == 'lsun':
    dataset = dset.LSUN(root=DataRoot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Resize(ImageSize),
                            transforms.CenterCrop(ImageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
    nc=3
elif Dset == 'cifar10':
    dataset = dset.CIFAR10(root=DataRoot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(ImageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    nc=3

elif Dset == 'mnist':
        dataset = dset.MNIST(root=DataRoot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(ImageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ]))
        nc=1

elif Dset == 'fake':
    dataset = dset.FakeData(image_size=(3, ImageSize, ImageSize),
                            transform=transforms.ToTensor())
    nc=3

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize,
                                         shuffle=True, num_workers=2)

device = torch.device("cuda:0")
print(caption = "Preparing FID")
bm=BM(dataloader, CONF, 2048, device, nz)

channels = 3

class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.kaiming_uniform_(self.conv1.weight.data, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight.data, mode='fan_in', nonlinearity='relu')

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2
            )
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Upsample(scale_factor=2)

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, shape, stride=1, bn=False):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.kaiming_uniform_(self.conv1.weight.data, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight.data, mode='fan_out', nonlinearity='relu')
        norm = lambda c : (nn.Sequential() if not bn else nn.BatchNorm2d(c))
        if stride == 1:
            self.model = nn.Sequential(
                norm(in_channels),
                nn.ReLU(),
                self.conv1,
                norm(out_channels),
                nn.ReLU(),
                self.conv2,
                )
        else:
            self.model = nn.Sequential(
                norm(in_channels),
                nn.ReLU(),
                self.conv1,
                norm(out_channels),
                nn.ReLU(),
                self.conv2,
                nn.AvgPool2d(2, stride=stride, padding=0)
                )
        self.bypass = nn.Sequential()
        if stride != 1:

            self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform(self.bypass_conv.weight.data, nn.init.calculate_gain('linear'))

            self.bypass = nn.Sequential(
                self.bypass_conv,
                nn.AvgPool2d(2, stride=stride, padding=0)
            )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.kaiming_uniform_(self.conv1.weight.data, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight.data, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_uniform(self.bypass_conv.weight.data, nn.init.calculate_gain('linear'))

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            self.bypass_conv,
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

GEN_SIZE=CONF['G_SIZE']
DISC_SIZE=int(CONF['G_SIZE']*CONF['D_SCALE'])

class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        self.dense = nn.Linear(self.z_dim, 4 * 4 * GEN_SIZE)
        self.final = nn.Conv2d(GEN_SIZE, channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform(self.dense.weight.data, nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform(self.final.weight.data, nn.init.calculate_gain('tanh'))

        self.model = nn.Sequential(
            ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
            ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
            ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
            nn.BatchNorm2d(GEN_SIZE),
            nn.ReLU(),
            self.final,
            nn.Tanh())

    def forward(self, z):
        return self.model(self.dense(z).view(-1, GEN_SIZE, 4, 4))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # self.model = nn.Sequential(
        self.r1 = FirstResBlockDiscriminator(channels, DISC_SIZE, stride=2)
        self.r2 = ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, 16, stride=2)
        self.r3 = ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, 8)
        self.r4 = ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, 8)
            # nn.ReLU(),
        self.pool = nn.AvgPool2d(8)
            # )
        self.fc = nn.Linear(DISC_SIZE, 1)
        nn.init.xavier_uniform(self.fc.weight.data, nn.init.calculate_gain('linear'))

    def forward(self, x, d1=0.2, d2=0.5, d3=0.5):
        x = F.dropout(self.r2(self.r1(x)), p=d1)
        x = F.dropout(self.r3(x), p=d2)
        o2 = self.pool(F.relu(F.dropout(self.r4(x), p=d3), inplace=True))
        
        return self.fc(o2.view(-1,DISC_SIZE)), o2.squeeze()

class Discriminator_D(nn.Module):
    def __init__(self):
        super(Discriminator_D, self).__init__()

        self.model = nn.Sequential(
                FirstResBlockDiscriminator(channels, DISC_SIZE, stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, 16, stride=2, bn = True),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, 8, bn = True),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, 8, bn = True),
                nn.ReLU(),
                nn.AvgPool2d(8),
            )
        self.fc = nn.Linear(DISC_SIZE, 1)
        nn.init.xavier_uniform(self.fc.weight.data, nn.init.calculate_gain('linear'))

    def forward(self, x):
        return F.sigmoid(self.fc(self.model(x).view(-1,DISC_SIZE)))

netG = Generator(nz).cuda()
netD = Discriminator().cuda()
D = Discriminator_D().cuda()
print(netG)
print(netD)
print(D)

criterion = nn.BCELoss()

def calc_gradient_penalty(netD, real_data, fake_data, BATCH_SIZE):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, real_data.nelement()//BATCH_SIZE).contiguous().view(BATCH_SIZE, 3, ImageSize, ImageSize)
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)[0]

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

fixed_noise = torch.randn(batchSize, nz, device=device)
real_label = 1
fake_label = 0

optimizerD = optim.Adam(netD.parameters(), lr=CONF['LR'], betas=(0.0,0.9))
schedulerD = optim.lr_scheduler.LambdaLR(optimizerD, lr_lambda=[lambda epoch: max(0., 1.-epoch/(CONF['niter']*len(dataloader)))])
optimizerD_D = optim.Adam(D.parameters(), lr=CONF['LR']/CONF['PPO_iters'], betas=(0.0,0.9))
schedulerD_D = optim.lr_scheduler.LambdaLR(optimizerD_D, lr_lambda=[lambda epoch: max(0., 1.-epoch/(CONF['niter']*len(dataloader)))])
optimizerG  = optim.Adam(netG.parameters(), lr=CONF['LR']/CONF['PPO_iters'], betas=(0.0,0.9))
schedulerG = optim.lr_scheduler.LambdaLR(optimizerG, lr_lambda=[lambda epoch: max(0., 1.-epoch/(CONF['niter']*len(dataloader)))])

niter = CONF['niter']
# c=0

one = torch.FloatTensor([1])
mone = one * -1
one, mone = one.cuda(), mone.cuda()

from tqdm import tqdm_notebook

bFID = 1000.0
bIS = 0.0
mean = 0.0


G_progress = 0

diff = 0

while G_progress < niter*len(dataloader):
    data_iter = iter(dataloader)
    i=0
    while i<len(dataloader):
        
        j = 0
        while j < CONF['Diters'] and i < len(dataloader):
            data = data_iter.next()
            i += 1
            j += 1
            ############################
            # (1) Update D network: maximize D(x) - D(G(z))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)

            D_real1_1, D_real1_2 = netD(real_cpu)
            errD_real = D_real1_1.mean()
            errD_real.backward(mone,retain_graph=True)
            D_x = D_real1_1.mean().item()

            D_real2_1, D_real2_2 = netD(real_cpu)

            ct_penalty = CONF["LAMBDA2"]*((D_real1_1-D_real2_1)**2)
            ct_penalty = ct_penalty + CONF["LAMBDA2"]*0.1*((D_real1_2-D_real2_2)**2).mean(dim=1)
            ct_penalty = torch.max(0. * (ct_penalty-CONF["Factor_M"]),ct_penalty-CONF["Factor_M"])
            ct_penalty = ct_penalty.mean()
            ct_penalty.backward()

            # train with fake
            noise = torch.randn(batch_size, nz, device=device)
            fake = netG(noise)
            D_fake1_1, _ = netD(fake.detach())
            with torch.no_grad():
                W = batch_size * F.softmax(D_fake1_1.data, dim = 0)
            errD_fake = (W*D_fake1_1).mean()
            errD_fake.backward(one)
            gradient_penalty = calc_gradient_penalty(netD, real_cpu.data, fake.data, batch_size)
            gradient_penalty.backward()

            D_G_z1 = D_fake1_1.mean().item()
            errD = errD_real - errD_fake
            errD_GP = -(errD_real - errD_fake) + gradient_penalty
            optimizerD.step()

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################
        j = 0
        with torch.no_grad():
            D0 = (D(fake.data)).data
            P0 = (1.-D0)/torch.clamp(D0, min = 1e-7)

        while j < CONF['PPO_iters'] and i < len(dataloader):
            data = data_iter.next()
            i += 1
            j += 1
            D.zero_grad()
            real_tmp = data[0].to(device)
            batch_size = real_tmp.size(0)
            label = torch.full((batch_size,), real_label, device=device)
            output = D(real_tmp)
            errDD_real = criterion(output, label)
            errDD_real.backward()
            label.fill_(fake_label)
            Noise = torch.randn(batch_size, nz, device=device)
            fake = netG(Noise)
            output = D(fake.detach())
            errDD_fake = criterion(output, label)
            errDD_fake.backward()
            nn.utils.clip_grad_norm_(D.parameters(), CONF['max_grad_norm'])
            optimizerD_D.step()
            netG.zero_grad()
            fake = netG(noise)
            D1 = D(netG(noise))
            P1 = (1.-D1)
            ratio = (P1/torch.clamp(D1*P0, min = 1e-7))
            adv_targ, _ = netD(fake)
            surr1 = ratio * adv_targ
            ratio_clipped = torch.clamp(ratio, 1.0 - CONF['clip_param'], 1.0 + CONF['clip_param'])
            surr2 = ratio_clipped * adv_targ
            target = torch.where(adv_targ>0, torch.min(surr1, surr2), torch.max(surr1, surr2))
            errG = target.mean()
            errG.backward(mone)
            D_G_z2 = errG.item()
            optimizerG.step()
            writer.add_histogram('R_{}'.format(j), ratio.data.cpu().numpy(), global_step=G_progress)
            writer.add_histogram('R_clip_{}'.format(j), ratio_clipped.data.cpu().numpy(), global_step=G_progress)
            writer.add_histogram('Adv_{}'.format(j), adv_targ.data.cpu().numpy(), global_step=G_progress)
            writer.add_histogram('L_{}'.format(j), target.data.cpu().numpy(), global_step=G_progress)

        writer.add_scalar("Loss_D", errD.item(), global_step=G_progress)
        writer.add_scalar("Loss_G", errG.item(), global_step=G_progress)
        writer.add_scalar("Loss_D_GP", errD_GP.item(), global_step=G_progress)
        writer.add_scalars("stats", {'D(G)_1':D_G_z1, 'D(G)_2':D_G_z2}, global_step=G_progress)

        if G_progress % 1000 == 0:
            netG.eval()
            (mean, std), FID = bm.calculate_fid_IS(netG)
            if bIS < mean:
                bIS = mean
                torch.save(netG, os.path.join(opt.output_dir,"G.cpt"))
                torch.save(netD, os.path.join(opt.output_dir,"D.cpt"))
                torch.save(D, os.path.join(opt.output_dir,"D_D.cpt"))
                print(value = bIS, caption = "{} : {}".format(opt.id, bIS))
            if FID < bFID:
                bFID = FID
            writer.add_scalar("Inception_mean", mean, global_step=G_progress)
            writer.add_scalar("FID", FID, global_step=G_progress)
            netG.train()

        G_progress+=1
        schedulerD.step()
        schedulerG.step()
        schedulerD_D.step()
    writer.add_image("real_sample", torchvision.utils.make_grid(real_cpu, nrow=8, normalize = True), global_step=G_progress)
    writer.add_image("fake_sample", torchvision.utils.make_grid(fake.data, nrow=8, normalize = True), global_step=G_progress)
print(progress = "{}/{}".format(niter*len(dataloader),niter*len(dataloader)), caption = "Done... Best IS:{}".format(bIS))