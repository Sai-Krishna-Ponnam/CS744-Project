import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.utils.tensorboard import SummaryWriter
import wandb
import numpy as np
import random
import argparse

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser(description="Distributed Training with FSDP")
parser.add_argument("--local_rank", type=int, default=0, help="Local rank of the process")
parser.add_argument("--world_size", type=int, default=1, help="Total number of processes")
parser.add_argument("--master_addr", type=str, default="localhost", help="Master address")
parser.add_argument("--master_port", type=int, default=12345, help="Master port")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--sharding_strategy", type=int, default=0, help="Sharding strategy (0: FULL_SHARD, 1: SHARD_GRAD_OP, 2: NO_SHARD)")
parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
args = parser.parse_args()

def setup():
    """Initialize distributed training environment."""
    dist.init_process_group(backend="nccl")  # NCCL for GPUs


def cleanup():
    """Destroy the process group after training."""
    dist.destroy_process_group()


def compute_accuracy(model, dataloader, device, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loss += criterion(outputs, labels)

    test_loss /= total
    accuracy = 100 * correct / total
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy}%")


def train():
    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    if args.sharding_strategy == 0:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif args.sharding_strategy == 1:
        sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
    elif args.sharding_strategy == 2:
        sharding_strategy = ShardingStrategy.NO_SHARD

    lr = 0.001

    """Distributed training function."""
    setup()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])  # Rank within the node
    print(f"Rank: {rank}, World size: {world_size}, Local rank: {local_rank}")

    torch.cuda.set_device(local_rank)

    # Model setup
    model = models.vit_b_16(weights=None).cuda(local_rank)
    model = FSDP(model, sharding_strategy=sharding_strategy)

   

    # Dataset & Dataloader
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    if local_rank == 0:
        datasets.CIFAR10(root="./data", train=True, download=True)
        datasets.CIFAR10(root="./data", train=False, download=True)
    dist.barrier()

    train_dataset = datasets.FakeData(12810, (3, 224, 224), 1000, transforms.ToTensor())
    test_dataset = datasets.FakeData(500, (3, 224, 224), 1000, transforms.ToTensor())

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss().cuda(local_rank)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    wandb_run_id = "FDSPVITmodel" + "-"+ str(args.batch_size) + "-" + str(args.sharding_strategy) + "-" + str(args.num_epochs) 
    settings = wandb.Settings(
        mode="shared",
        x_stats_sampling_interval=1,
        # GPU index to capture metrics from.
        # In DDP, each process has a single GPU, but all GPUs on the node may be visible.
        x_stats_gpu_device_ids=[local_rank],
        x_label=f"rank-{global_rank}",
    )
    if global_rank != 0:
        # Do not upload wandb files except console logs.
        settings.x_primary = False
        # Do not change the state of the run on run.finish().
        settings.x_update_finish_state = False


    run = wandb.init(
        project="your_project_name",
        id=wandb_run_id,
        config={
            "epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "sharding_strategy": args.sharding_strategy,
            "world_size": world_size,
        },
        settings=settings,
    )


    # Update the run metadata with the number of CPUs and GPUs in the cluster.
    run._metadata.gpu_count = world_size
    
    dist.barrier()

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_sampler.set_epoch(epoch)
        total = 0
        correct = 0
        step = 0
        for images, labels in train_dataloader:
            images, labels = images.cuda(local_rank), labels.cuda(local_rank)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
            total += labels.size(0)  # Number of samples in the batch
            correct += (predicted == labels).sum().item()  # Count correct predictions

            step += 1
            if step % 100 == 0 and rank == 0:  # Log only from rank 0
                print(f"Rank {rank}, Step {step}, Epoch [{epoch+1}/5], Loss: {loss.item():.4f}")

        accuracy = 100 * correct / total
        print(f"Rank {rank}, Epoch [{epoch+1}/5], Loss: {loss.item():.4f}, Train Accuracy: {accuracy:.2f}%")
        compute_accuracy(model, test_dataloader, local_rank, criterion)

        if rank == 0:
            run.log({
                "epoch": epoch + 1,
                "train_loss": loss.item(),
                "train_accuracy": accuracy,
            })


        # Optionally, add model parameters, gradients, histograms, etc.
#        writer.add_histogram("model_weights", model.parameters(), epoch)

    if rank == 0:
        torch.save(model.module.state_dict(), "vgg16_fsdp.pth")
        
        print("Model saved successfully!")

    cleanup()
    if rank == 0:
        run.finish()


if __name__ == "__main__":
    
    
    
    train()
