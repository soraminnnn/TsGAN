import argparse
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import numpy as np
from model.generator.netG import Generator
from model.discriminator.netD import Discriminator

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="data/planetLab/train.npy",help="path to load the training data")

parser.add_argument("--use_GPU", type=bool, default=True, help="whether use GPU")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--models_dir", type=str, default="model_saved",help='path to save the trained models.')
parser.add_argument("--models_iter",type=int, default=50, help="number of epochs of saving model")
parser.add_argument("--samples_dir", type=str, default="samples",help='path to save the output samples.')
parser.add_argument("--samples_iter",type=int, default=50, help="number of epochs of generating validate samples")

###training parameters###
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="batch size of training")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--lr_decay", type=bool, default=False, help="whether use learning rate decay(linear decay)")
parser.add_argument("--start_decay", type=int, default=100, help="epochs of starting learning rate decay")
parser.add_argument("--n_generator", type=int, default=5, help="iterations for generator of training")

###generator parameters###
parser.add_argument("--g_num_layers", type=int, default=6, help="number of layers of the generator")
parser.add_argument("--g_num_filters", type=int, default=32, help="number of filters of the generator")
parser.add_argument("--g_kernel_size", type=int, default=7, help="kernel size of convolution for the generator")
###discriminator parameters###
parser.add_argument("--d_num_layers", type=int, default=3, help="number of layers of the discriminator")
parser.add_argument("--d_downsample", type=bool, nargs='+', default=[True]*3, help="whether use average pool in each layer of the discriminator")

args = parser.parse_args()

if args.use_GPU:
    device = 'cuda:0'
else:
    device = 'cpu'

fix_z = torch.randn(5,args.latent_dim).to(device)

def decay_lr(opt, max_iter, start_iter, initial_lr):
    '''Decay learning rate linearly till 0.'''
    coeff = -initial_lr / (max_iter - start_iter)
    for pg in opt.param_groups:
        pg['lr'] += coeff

def train():
    data = np.load(args.data)
    print('dataset shape: {}'.format(data.shape))

    #data preprocess
    max=[]
    min=[]
    for i in range(data.shape[1]):
        max.append(np.max(data[:, i, :]))
        min.append(np.min(data[:, i, :]))
        data[:, i, :] = (data[:, i, :] - min[i]) / (max[i] - min[i])
    data = (data-0.5)/0.5

    #dataset info
    samples_size = data.shape[0]
    series_len = data.shape[2]*data.shape[3]
    series_dim = data.shape[1]

    net_g = Generator(latent_dim=args.latent_dim, seq_len=series_len, input_dim=1, output_dim=series_dim, num_channels=[args.g_num_filters]*args.g_num_layers,kernel_size=args.g_kernel_size).to(device)
    net_d = Discriminator(in_channels=series_dim,num_layers=args.d_num_layers,downsample=args.d_downsample).to(device)
    latent_shape = [1, args.latent_dim]
    output_shape = [series_dim,data.shape[2],data.shape[3]]

    optimizer_d = torch.optim.Adam(net_d.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_g = torch.optim.Adam(net_g.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    data = torch.Tensor(data)
    dataset = torch.utils.data.TensorDataset(data,torch.ones(samples_size))
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)
    print('##network structure discription##')
    print(net_g)
    print(net_d)

    d_losses = []
    g_losses = []
    D_x = []
    D_gz = []
    iters = 0

    # loss function
    criterion = nn.BCELoss().to(device)

    # training iterations
    total_step = len(data_loader)
    start_epoch = 0
    checkpoint_path = '%s/check.pth'%args.models_dir

    '''
    reload training from checkpoint
    '''
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        net_g.load_state_dict(checkpoint['net_g'])
        net_g.train()
        optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        net_d.load_state_dict(checkpoint['net_d'])
        net_d.train()
        optimizer_d.load_state_dict(checkpoint['optimizer_d'])
        start_epoch = checkpoint['epoch'] + 1
        print('####restart from epoch %s####' % start_epoch)
    start = time.perf_counter()  # time.clock()
    for epoch in range(start_epoch, args.n_epochs):
        net_g.train()
        if(args.lr_decay and epoch+1 >= args.start_decay):
            decay_lr(optimizer_g,args.n_epochs,args.start_decay,args.lr)
            decay_lr(optimizer_d,args.n_epochs,args.start_decay,args.lr)
        for i, (real_data, _) in enumerate(data_loader):
            true_batch_size = real_data.shape[0]
            latent_shape[0] = true_batch_size

            true_lable = torch.ones(true_batch_size, 1).to(device)
            fake_lable = torch.zeros(true_batch_size, 1).to(device)

            # ================================================================== #
            #                      Train the discriminator                       #
            # ================================================================== #
            optimizer_d.zero_grad()
            # real_loss for the discriminator
            real_data = real_data.to(device)
            outputs = net_d(real_data)
            d_x = outputs
            d_loss_real = criterion(outputs, true_lable)
            # fake_loss for the discriminator
            z = torch.randn(latent_shape).to(device)
            fake_data = net_g(z).view(true_batch_size,*output_shape)
            outputs = net_d(fake_data)
            d_loss_fake = criterion(outputs, fake_lable)
            # total_loss
            d_loss = d_loss_fake + d_loss_real
            d_loss.backward()
            optimizer_d.step()

            # ================================================================== #
            #                      Train the generator                           #
            # ================================================================== #
            if (i+1) % args.n_generator == 0:
                z = torch.randn(latent_shape).to(device)
                optimizer_g.zero_grad()
                fake_data = net_g(z).view(true_batch_size,*output_shape)
                outputs = net_d(fake_data)
                d_gz = outputs
                # loss
                g_loss = criterion(outputs, true_lable)
                g_loss.backward()
                optimizer_g.step()

                print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                      .format(epoch, args.n_epochs, i + 1, total_step, d_loss.item(), g_loss.item(),
                              d_x.mean().item(), d_gz.mean().item()))

                D_x.append(d_x.mean().item())
                D_gz.append(d_gz.mean().item())
                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())

            iters += 1

        if (epoch+1) % args.models_iter == 0:
            torch.save(net_g.state_dict(), '%s/%sepochs.pkl' % (args.models_dir,epoch+1))
            #save checkpoint
            checkpoint = {
                'net_g': net_g.state_dict(),
                'optimizer_g': optimizer_g.state_dict(),
                'net_d': net_d.state_dict(),
                'optimizer_d': optimizer_d.state_dict(),
                'epoch': epoch
            }
            torch.save(checkpoint,checkpoint_path)
            print('####Check point saved####')

        if (epoch+1) % args.samples_iter == 0:
            net_g.eval()
            samples = net_g(fix_z).cpu().detach().numpy()
            samples = (samples * 0.5) + 0.5
            for n in range(samples.shape[1]):
                samples[:,n,:] = samples[:,n,:] * (max[n]-min[n]) + min[n]
            for n in range(samples.shape[0]):
                plt.clf()
                for N in range(samples.shape[1]):
                    plt.plot(samples[n,N,:])
                plt.savefig('%s/%s_%s.png'%(args.samples_dir,epoch+1,n))

    print("####training finished####")
    torch.save(net_g.state_dict(), '%s/%sepochs.pkl' %(args.models_dir,args.n_epochs))
    print('model saved successfully')
    end = time.clock()
    print('Running time: %s Seconds' % (end - start))

    #plt loss history
    plt.figure(figsize=(10, 5))
    plt.title("Training Loss")
    plt.plot(d_losses, label="G_loss" , color='black')
    plt.plot(g_losses, label="D_loss", color='red')
    plt.plot(D_x, label="D(x)", color='green')
    plt.plot(D_gz, label="D(g(z))", color='yellow')
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('loss.png')

if __name__ == '__main__':
    train()
