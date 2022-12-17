# A GAN-based method for time-dependent cloud workload generation
PyTorch implementation of A GAN-based method for time-dependent cloud workload generation.

# Getting started
## PlanetLab
train:  
`python gan.py --data="data/planetLab/train.npy" --use_GPU=True --g_num_layers=6 --g_num_filters=32 --g_kernel_size=7 --d_num_layers=3 --d_downsample True True True`

generate samples:  
`python samples_generate.py --data="data/planetLab/train.npy" --model_path="model_saved/planetLab.pkl" --use_GPU=True --g_num_layers=6 --g_num_filters=32 --g_kernel_size=7`

## Alibaba machine
train:  
`python gan.py --data="data/alibaba_machine/train.npy" --use_GPU=True --batch_size=64 --g_num_layers=8 --g_num_filters=25 --g_kernel_size=7 --d_num_layers=4 --d_downsample True True True True`

generate samples:  
`python samples_generate.py --data="data/alibaba_machine/train.npy" --model_path="model_saved/alibaba_machine.pkl" --use_GPU=True --g_num_layers=8 --g_num_filters=25 --g_kernel_size=7`

## Alibaba container
train:  
`python cgan.py --data="data/alibaba_container/train.npy" --label="data/alibaba_container/label.npy" --use_GPU=True --g_num_layers=8 --g_num_filters=25 --g_kernel_size=7 --d_num_layers=4 --d_downsample True True True True`

generate samples:  
`python samples_generate.py --data="data/alibaba_container/train.npy" --n_classes=7 --model_path="model_saved/1000epochs.pkl" --use_GPU=True --g_num_layers=8 --g_num_filters=25 --g_kernel_size=7`