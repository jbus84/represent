#!/usr/bin/env python3
"""
Simple Background Processing Usage Example

This example shows the minimal code needed to use background batch processing
for eliminating training bottlenecks.
"""
import torch
import torch.nn as nn

from represent.dataloader import MarketDepthDataset, AsyncDataLoader
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
    
    # 2. Create async dataloader with background processing
    async_loader = AsyncDataLoader(
        dataset=dataset,
        background_queue_size=4,  # Keep 4 batches ready
        prefetch_batches=2        # Pre-generate 2 batches
    )
    
    # 3. Start background batch production
    async_loader.start_background_production()
    
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
    
    print("Training with background batch processing...")
    
    # 5. Training loop - batches are instant!
    for epoch in range(10):
        # Get batch (sub-millisecond when queue is full)
        batch = async_loader.get_batch()
        
        # Prepare for training
        batch = batch.unsqueeze(0).unsqueeze(0)  # Add batch & channel dims
        target = torch.randn(1, 1)  # Random target for demo
        
        # Standard PyTorch training step
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch + 1}: Loss = {loss.item():.6f}")
    
    # 6. Check background processing status
    status = async_loader.queue_status
    print("\\n📊 Background Processing Stats:")
    print(f"   Batches produced: {status['batches_produced']}")
    print(f"   Average generation: {status['avg_generation_time_ms']:.2f}ms")
    print(f"   Average retrieval: {status['avg_retrieval_time_ms']:.3f}ms")
    print(f"   Queue utilization: {status['queue_size']}/{status['max_queue_size']}")
    
    # 7. Cleanup
    async_loader.stop()
    
    print("\\n✅ Done! Background processing eliminated the bottleneck.")


if __name__ == "__main__":
    main()