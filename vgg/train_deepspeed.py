import os
import torch
import torch.nn as nn
import torch.optim as optim
import deepspeed
import torch.distributed as dist
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
from deepspeed.ops.adam import DeepSpeedCPUAdam
from torch.utils.tensorboard import SummaryWriter
import argparse


# Define the VGG model
class VGGModel(nn.Module):
    def __init__(self):
        super(VGGModel, self).__init__()
        self.model = models.vgg16(weights=None)  # Use VGG16 model
        self.model.classifier[6] = nn.Linear(4096, 10)  # Modify for 10 classes (CIFAR-10)

    def forward(self, x):
        return self.model(x)


# DeepSpeed Config
deepspeed_config = {
    "train_micro_batch_size_per_gpu": 32,
    "optimizer": {"type": "AdamW", "params": {"lr": 0.001, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 1e-2}},
    "fp16": {"enabled": False},
    "zero_optimization": {"stage": 2},
}


# Initialize DeepSpeed
def setup_deepspeed(args, model):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    model, optimizer, _, _ = deepspeed.initialize(
        args=args, model=model, model_parameters=parameters, config=deepspeed_config
    )
    return model, optimizer


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def train(args):
    torch.manual_seed(42)  # Set seed for reproducibility

    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)

    # Distributed initialization
    deepspeed.init_distributed()

    # Setup TensorBoard
    writer = SummaryWriter(log_dir="./runs/deepspeed_experiment")

    # Define model and wrap it with DeepSpeed
    model = VGGModel()
    model, optimizer = setup_deepspeed(args, model)

    # Data transforms
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    # test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    # Distributed Sampler
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
    # test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss().cuda()

    # Training loop
    epochs = 5
    for epoch in range(epochs):
        model.train()
        train_sampler.set_epoch(epoch)  # Ensure different sampling each epoch
        running_loss = 0.0
        correct = 0
        total = 0
        step = 0

        for inputs, labels in train_dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()  # Reset gradients
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            model.backward(loss)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            step += 1
            if step % 100 == 0:  # Log only from rank 0
                print(f"Rank {model.global_rank}, Step {step}, Epoch [{epoch+1}/5], Loss: {loss.item():.4f}")

        train_loss = running_loss / len(train_dataloader)
        train_accuracy = 100.0 * correct / total

        # Log metrics to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

        # Save checkpoint using DeepSpeed (only rank 0)
        if model.global_rank == 0:
            model.save_checkpoint("deepspeed_checkpoints")

    writer.close()
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSpeed Training Script")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    args = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    train(args)
