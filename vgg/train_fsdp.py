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

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)


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
    NUM_EPOCHS = 5
    BATCH_SIZE = 32
    lr = 0.001

    """Distributed training function."""
    setup()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])  # Rank within the node
    print(f"Rank: {rank}, World size: {world_size}, Local rank: {local_rank}")

    torch.cuda.set_device(local_rank)

    # Model setup
    model = models.vgg16(weights=None).cuda(local_rank)
    model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD)

    if dist.get_rank() == 0:
        wandb.init(
            project="cs744",
            name=f"fsdp-vgg16-rank-{dist.get_rank()}",
            config={
                "epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": lr,
                "model": "vgg16",
                "sharding_strategy": "FULL_SHARD",
            }
        )
        wandb.watch(model, log="all")

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

    train_dataset = datasets.CIFAR10(root="./data", train=True, download=False, transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=False, transform=transform)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss().cuda(local_rank)
    optimizer = optim.Adam(model.parameters(), lr=lr)

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
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": loss.item(),
                "train_accuracy": accuracy,
            })

    if rank == 0:
        torch.save(model.module.state_dict(), "vgg16_fsdp.pth")
        wandb.save("vgg16_fsdp.pth")
        print("Model saved successfully!")

    cleanup()
    if rank == 0:
        wandb.finish()


if __name__ == "__main__":
    train()