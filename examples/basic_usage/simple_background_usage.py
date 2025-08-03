#!/usr/bin/env python3
"""
Simple Background Processing Usage Example

This example shows the minimal code needed to use background batch processing
for eliminating training bottlenecks.
"""
import torch
import torch.nn as nn

from represent.dataloader import MarketDepthDataset, HighPerformanceDataLoader
from represent.constants import SAMPLES
from tests.unit.fixtures.sample_data import generate_realistic_market_data


def main():
    """Simple usage example."""
    print("🚀 Simple Background Processing Usage")
    print("=" * 50)
    
    # 1. Create dataset and fill with data
    dataset = MarketDepthDataset(buffer_size=SAMPLES)
    market_data = generate_realistic_market_data(SAMPLES)
    dataset.add_streaming_data(market_data)
    
    # 2. Create high performance dataloader
    dataloader = HighPerformanceDataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=0,      # Single-threaded for stability
        pin_memory=False    # Safe for all devices
    )
    
    # 4. Create your model (example CNN)
    model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((10, 10)),
        nn.Flatten(),
        nn.Linear(3200, 1)
    )
    
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    print("Training with HighPerformanceDataLoader...")
    
    # 5. Training loop - reliable and stable
    for epoch in range(10):
        for features, targets in dataloader:
            # Features already have correct dimensions from dataloader
            batch = features
            target = targets
            
            # Standard PyTorch training step
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch + 1}: Loss = {loss.item():.6f}")
            break  # Only process one batch per epoch for demo
    
    print("\\n✅ Done! HighPerformanceDataLoader provides reliable training.")


if __name__ == "__main__":
    main()