import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from model.generator.netG import Generator
parser = argparse.ArgumentParser()

parser.add_argument("--data", type=str, default="data/planetLab/train.npy",help="path to load the training data")
parser.add_argument("--n_classes", type=int, default=0, help="if generating condition samples, it represents the number of classes of training data")
parser.add_argument("--model_path", type=str, default='models_saved/alibaba.pkl', help="path to load the generator parameters")
parser.add_argument("--samples_size", type=int, default=10, help="number of samples to be generated")
parser.add_argument("--samples_save", type=str, default='samples', help="path to save the generated samples")
parser.add_argument("--samples_plg", type=bool, default=True, help="whether plog the generated samples")

parser.add_argument("--use_GPU", type=bool, default=True, help="whether use GPU")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--g_num_layers", type=int, default=6, help="number of layers of the generator")
parser.add_argument("--g_num_filters", type=int, default=32, help="number of filters of the generator")
parser.add_argument("--g_kernel_size", type=int, default=7, help="kernel size of convolution for the generator")

args = parser.parse_args()

if args.use_GPU:
    device = 'cuda:0'
else:
    device = 'cpu'

def samples_generate():
    train_data = np.load(args.data)
    series_len = train_data.shape[2]*train_data.shape[3]
    series_dim = train_data.shape[1]

    net_g = Generator(latent_dim=args.latent_dim, seq_len=series_len, input_dim=1, output_dim=series_dim,
                      num_channels=[args.g_num_filters] * args.g_num_layers, kernel_size=args.g_kernel_size,n_classes=args.n_classes).to(device)
    net_g.load_state_dict(torch.load(args.model_path))
    net_g.eval()

    z = torch.randn(args.samples_size, args.latent_dim).to(device)
    batch_size = 200
    samples = torch.Tensor([])
    samples_label = torch.LongTensor([])
    if args.samples_size <= batch_size:
        if args.n_classes > 0:
            z_label = torch.LongTensor(np.random.randint(0, args.n_classes, args.samples_size)).to(device)
            samples_label = torch.cat([samples_label, z_label.cpu().detach()], 0)
            samples = net_g(z,z_label).cpu().detach()
        else:
            samples = net_g(z).cpu().detach()
    else:
        iters = args.samples_size // batch_size
        for i in range(iters):
            z = torch.randn(batch_size, args.latent_dim).to(device)
            if args.n_classes > 0:
                z_label = torch.LongTensor(np.random.randint(0, args.n_classes, args.samples_size)).to(device)
                samples_label = torch.cat([samples_label, z_label.cpu().detach()], 0)
                out = net_g(z,z_label).cpu().detach()
            else:
                out = net_g(z).cpu().detach()
            samples = torch.cat([samples,out],0)
        samples_left = args.samples_size - iters*batch_size
        if samples_left != 0:
            z = torch.randn(samples_left,args.latent_dim).to(device)
            if args.n_classes > 0:
                z_label = torch.LongTensor(np.random.randint(0, args.n_classes, args.samples_size)).to(device)
                samples_label = torch.cat([samples_label, z_label.cpu().detach()], 0)
                out = net_g(z,z_label).cpu().detach()
            else:
                out = net_g(z).cpu().detach()
            samples = torch.cat([samples,out],0)

    samples = samples * 0.5 + 0.5
    for n in range(train_data.shape[1]):
        max = np.max(train_data[:, n, :])
        min = np.min(train_data[:, n, :])
        samples[:, n, :] = samples[:, n, :] * (max - min) + min
    np.save('%s/samples.npy'%args.samples_save,samples.numpy())
    if args.n_classes > 0:
        np.save('%s/samples_label.npy' % args.samples_save, samples_label.numpy())

    if args.samples_plg:
        for I in range(args.samples_size):
            plt.clf()
            plt.axis([0, series_len, 0, 100])
            for j in range(samples.shape[1]):
                plt.plot(samples[I, j, :])
            if args.n_classes > 0:
                plt.title('class %d' % (samples_label[I]), fontsize=15)
            else:
                plt.title('%d' % (I + 1), fontsize=15)
            plt.savefig('%s/%s.png' % (args.samples_save,I))

    print("generating samples finished")

if __name__ == '__main__':
    samples_generate()