# CS744-Project

### 1. Install nvidia drivers
```
bash nvidia_setup.sh
```

reboot the VMs `sudo reboot`

### 2. Install miniconda
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init
source ~/.bashrc
```

### 3. Install python modules
```
pip install -r requirments.txt 
```

### Train vgg model
```
cd vgg
bash run.sh <MASTER_IP> <RANK>

The deepspeed that works is CS744-Project/vgg/traindeepspeed.py and can be run by vgg/deepspeed.sh
conda create -n cs744 python=3.11
conda activate cs744
pip install -r vgg/requirements.txt
./deepspeed.sh

```

For deepspeed update g++ libs run `conda install -c conda-forge libstdcxx-ng`