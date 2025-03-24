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

    # Dataset & Dataloader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False, sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss().cuda(local_rank)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(2):
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
        torch.save(model.module.state_dict(), "vgg16_fsdp.pth")
        print("Model saved successfully!")

    cleanup()

if __name__ == "__main__":
    train()
