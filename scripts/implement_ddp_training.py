#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "torch>=2.0.0",
# ]
# ///
"""
Example implementation of DistributedDataParallel for multi-GPU training.
This addresses the DataParallel deadlock issue by using DDP with spawn method.

Based on research findings:
- DDP is faster and more stable than DataParallel
- Use spawn method to avoid deadlocks
- find_unused_parameters=True when needed
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path

# Set spawn method early (before any torch imports in subprocesses)
mp.set_start_method('spawn', force=True)


def setup_ddp(rank, world_size, master_port='12355'):
    """Initialize DDP process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl',  # Use NCCL for GPU, 'gloo' for CPU
        init_method='env://',
        rank=rank,
        world_size=world_size
    )
    
    # Set device for this process
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Cleanup DDP process group."""
    dist.destroy_process_group()


def train_ddp_worker(
    rank,
    world_size,
    model_class,
    train_dataset,
    val_dataset,
    config,
    base_dir,
    master_port='12355'
):
    """Training function for a single DDP worker."""
    # Setup DDP
    setup_ddp(rank, world_size, master_port)
    
    # Set device
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    # Create model
    model = model_class()
    model = model.to(device)
    
    # Wrap in DDP
    model = DDP(
        model,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=True,  # Important if not all params used in loss
    )
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Create data loaders (no shuffle in DataLoader when using DistributedSampler)
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,  # Use sampler instead of shuffle
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    
    # Only rank 0 prints/logs
    is_main_process = rank == 0
    
    if is_main_process:
        print(f"✅ DDP Training initialized on {world_size} GPUs")
        print(f"   Batch size per GPU: {config['batch_size']}")
        print(f"   Effective batch size: {config['batch_size'] * world_size}")
    
    # Training loop (simplified example)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('lr', 1e-3))
    
    for epoch in range(config.get('epochs', 100)):
        # Set epoch for sampler (important for proper shuffling)
        train_sampler.set_epoch(epoch)
        
        # Training
        model.train()
        for batch in train_loader:
            # Training step...
            optimizer.zero_grad()
            # loss = compute_loss(model, batch)
            # loss.backward()
            # optimizer.step()
            pass
        
        # Validation (only on rank 0 to avoid duplicate metrics)
        if is_main_process and epoch % 10 == 0:
            model.eval()
            # Validation step...
            pass
        
        # Save checkpoint (only rank 0)
        if is_main_process and epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),  # Note: .module for DDP
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, base_dir / f"checkpoint_epoch_{epoch}.pt")
    
    # Cleanup
    cleanup_ddp()


def launch_ddp_training(
    model_class,
    train_dataset,
    val_dataset,
    config,
    base_dir=Path("models")
):
    """Launch DDP training across all available GPUs."""
    world_size = torch.cuda.device_count()
    
    if world_size <= 1:
        print("⚠️  Only 1 GPU available, DDP not needed")
        return
    
    print(f"🚀 Launching DDP training on {world_size} GPUs")
    
    # Spawn processes
    mp.spawn(
        train_ddp_worker,
        args=(
            world_size,
            model_class,
            train_dataset,
            val_dataset,
            config,
            base_dir,
        ),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    # Example usage
    print("DDP Training Example")
    print("This is a template - integrate into train_flexible_opportunistic.py")
    print("\nKey changes needed:")
    print("1. Replace DataParallel with DDP")
    print("2. Use DistributedSampler for data loading")
    print("3. Use mp.spawn() to launch training")
    print("4. Only rank 0 saves checkpoints/logs")
    print("5. Access model via model.module when saving state_dict")

