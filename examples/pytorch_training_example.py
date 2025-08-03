#!/usr/bin/env python3
"""
Complete PyTorch Training Example with Background Batch Processing

This example demonstrates a realistic ML training scenario using the represent
package's background batch processing for market depth prediction.

Key Features Demonstrated:
- Market depth data preprocessing
- Custom PyTorch model architecture
- Background batch processing integration
- Training loop with validation
- Performance monitoring
- Model checkpointing
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from represent.dataloader import MarketDepthDataset, HighPerformanceDataLoader
from represent.constants import SAMPLES
from tests.unit.fixtures.sample_data import generate_realistic_market_data

class MarketDepthCNN(nn.Module):
    """
    Convolutional Neural Network for market depth prediction.
    
    Architecture:
    - Input: (batch_size, 1, 402, 500) - market depth representation
    - Conv layers: Extract spatial patterns in price/time dimensions
    - Dense layers: Final prediction
    - Output: (batch_size, 1) - price movement prediction
    """
    
    def __init__(self, dropout_rate=0.3):
        super(MarketDepthCNN, self).__init__()
        
        # Convolutional layers for spatial feature extraction
        self.conv_layers = nn.Sequential(
            # First conv block - detect local patterns
            nn.Conv2d(1, 32, kernel_size=(5, 5), padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(dropout_rate),
            
            # Second conv block - detect larger patterns
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(dropout_rate),
            
            # Third conv block - high-level features
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((10, 10)),  # Reduce to fixed size
        )
        
        # Dense layers for prediction
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 10 * 10, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),  # Single output: price movement prediction
            nn.Tanh()  # Output in [-1, 1] range
        )
    
    def forward(self, x):
        """Forward pass through the network."""
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x

def create_synthetic_targets(batch_size: int) -> torch.Tensor:
    """
    Create synthetic targets for demonstration.
    In practice, these would be derived from future price movements.
    """
    # Simulate price movement predictions in [-1, 1] range
    # Positive = price increase, negative = price decrease
    return torch.randn(batch_size, 1) * 0.5  # Scale to reasonable range

def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch with background batch processing."""
    model.train()
    total_loss = 0.0
    batch_count = 0
    batch_times = []
    training_times = []
    
    print(f"\n🚀 Training Epoch {epoch + 1}")
    print("-" * 50)
    
    for batch_idx in range(50):  # Train on 50 batches per epoch
        # Time batch loading (should be instant with background processing!)
        batch_start = time.perf_counter()
        
        # Get batch from background processor
        market_depth = next(iter(dataloader))[0]  # Shape: (402, 500)
        
        # Create synthetic target (in practice, use real targets)
        target = create_synthetic_targets(1).to(device)
        
        batch_end = time.perf_counter()
        batch_time = (batch_end - batch_start) * 1000
        batch_times.append(batch_time)
        
        # Prepare data for model
        # Add batch and channel dimensions: (1, 1, 402, 500)
        market_depth = market_depth.unsqueeze(0).unsqueeze(0).to(device)
        
        # Training step
        train_start = time.perf_counter()
        
        optimizer.zero_grad()
        output = model(market_depth)
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        train_end = time.perf_counter()
        training_time = (train_end - train_start) * 1000
        training_times.append(training_time)
        
        total_loss += loss.item()
        batch_count += 1
        
        # Progress reporting
        if batch_idx % 10 == 0:
            avg_batch_time = sum(batch_times[-10:]) / min(10, len(batch_times))
            avg_train_time = sum(training_times[-10:]) / min(10, len(training_times))
            
            print(f"  Batch {batch_idx:2d}: Loss={loss.item():.6f}, "
                  f"Batch={avg_batch_time:.3f}ms, Train={avg_train_time:.2f}ms")
    
    # Epoch statistics
    avg_loss = total_loss / batch_count
    avg_batch_time = sum(batch_times) / len(batch_times)
    avg_training_time = sum(training_times) / len(training_times)
    
    print(f"\n📊 Epoch {epoch + 1} Results:")
    print(f"   Average Loss: {avg_loss:.6f}")
    print(f"   Average Batch Time: {avg_batch_time:.3f}ms")
    print(f"   Average Training Time: {avg_training_time:.2f}ms")
    print(f"   Batch Loading Overhead: {(avg_batch_time/(avg_batch_time+avg_training_time))*100:.1f}%")
    
    return avg_loss, avg_batch_time, avg_training_time

def validate_model(model, dataloader, criterion, device):
    """Run validation with background batch processing."""
    model.eval()
    total_loss = 0.0
    batch_count = 0
    
    print("\n🔍 Running Validation")
    print("-" * 30)
    
    with torch.no_grad():
        for batch_idx in range(10):  # Validate on 10 batches
            # Get validation batch
            market_depth = next(iter(dataloader))[0]
            target = create_synthetic_targets(1).to(device)
            
            # Prepare data
            market_depth = market_depth.unsqueeze(0).unsqueeze(0).to(device)
            
            # Forward pass
            output = model(market_depth)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            batch_count += 1
    
    avg_val_loss = total_loss / batch_count
    print(f"   Validation Loss: {avg_val_loss:.6f}")
    
    return avg_val_loss

def main():
    """Main training function demonstrating background batch processing."""
    print("🎯 PYTORCH TRAINING WITH BACKGROUND BATCH PROCESSING")
    print("=" * 70)
    print("This example demonstrates realistic ML training using background")
    print("batch processing to eliminate data loading bottlenecks.")
    print("=" * 70)
    
    # Training configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    # Create output directory
    output_dir = Path("training_output")
    output_dir.mkdir(exist_ok=True)
    
    # Setup TensorBoard logging
    writer = SummaryWriter(output_dir / "tensorboard_logs")
    
    # 1. Create dataset with market data
    print("\n1️⃣  Setting up market depth dataset...")
    dataset = MarketDepthDataset(buffer_size=SAMPLES)
    
    # Generate realistic market data for training
    print(f"   Generating {SAMPLES:,} samples of market data...")
    market_data = generate_realistic_market_data(SAMPLES)
    dataset.add_streaming_data(market_data)
    print(f"   ✅ Dataset ready with {dataset.ring_buffer_size:,} samples")
    
    # 2. Create background batch processor
    print("\n2️⃣  Creating background batch processor...")
    dataloader = HighPerformanceDataLoader(
        dataset=dataset,
        background_queue_size=6,  # Keep 6 batches ready
        prefetch_batches=3        # Start with 3 pre-generated batches
    )
    
    # Start background processing

    # Check initial status
    status = {"status": "HighPerformanceDataLoader"}
    print("   ✅ Background processing started")
    print(f"   📊 Queue: {status['queue_size']}/{status['max_queue_size']} batches ready")
    print(f"   ⚡ Background healthy: {status['background_healthy']}")
    
    # 3. Create model
    print("\n3️⃣  Creating MarketDepthCNN model...")
    model = MarketDepthCNN(dropout_rate=0.3).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   📋 Total parameters: {total_params:,}")
    print(f"   🎯 Trainable parameters: {trainable_params:,}")
    
    # 4. Setup training components
    print("\n4️⃣  Setting up training components...")
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    print("   ✅ Loss function: MSE Loss")
    print("   ✅ Optimizer: AdamW (lr=0.001, weight_decay=0.01)")
    print("   ✅ Scheduler: ReduceLROnPlateau")
    
    # 5. Training loop
    print("\n5️⃣  Starting training loop...")
    num_epochs = 10
    best_val_loss = float('inf')
    
    training_stats = {
        'epochs': [],
        'train_losses': [],
        'val_losses': [],
        'batch_times': [],
        'training_times': []
    }
    
    try:
        for epoch in range(num_epochs):
            # Train epoch
            train_loss, batch_time, training_time = train_epoch(
                model, dataloader, optimizer, criterion, device, epoch
            )
            
            # Validate
            val_loss = validate_model(model, dataloader, criterion, device)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Save statistics
            training_stats['epochs'].append(epoch + 1)
            training_stats['train_losses'].append(train_loss)
            training_stats['val_losses'].append(val_loss)
            training_stats['batch_times'].append(batch_time)
            training_stats['training_times'].append(training_time)
            
            # Log to TensorBoard
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('Performance/BatchTime_ms', batch_time, epoch)
            writer.add_scalar('Performance/TrainingTime_ms', training_time, epoch)
            writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, output_dir / 'best_model.pth')
                print(f"   💾 Saved new best model (val_loss: {val_loss:.6f})")
            
            print(f"   📈 Learning Rate: {current_lr:.6f}")
            
            # Background processing status
            status = {"status": "HighPerformanceDataLoader"}
            print(f"   🔄 Background: {status['batches_produced']} batches produced, "
                  f"{status['avg_generation_time_ms']:.2f}ms avg generation")
    
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    
    finally:
        # 6. Cleanup and final statistics
        print("\n6️⃣  Training completed! Generating final report...")
        
        # Stop background processing

        # Final background processing statistics
        final_status = {"status": "HighPerformanceDataLoader"}
        
        print("\n" + "=" * 70)
        print("🏆 FINAL TRAINING REPORT")
        print("=" * 70)
        
        print("📊 Training Performance:")
        if training_stats['batch_times']:
            avg_batch_time = sum(training_stats['batch_times']) / len(training_stats['batch_times'])
            avg_training_time = sum(training_stats['training_times']) / len(training_stats['training_times'])
            
            print(f"   Average batch loading time: {avg_batch_time:.3f}ms")
            print(f"   Average training time per batch: {avg_training_time:.2f}ms")
            print(f"   Data loading overhead: {(avg_batch_time/(avg_batch_time+avg_training_time))*100:.1f}%")
            
            if avg_batch_time < 1.0:
                print("   🚀 EXCELLENT: Sub-millisecond batch loading achieved!")
            elif avg_batch_time < 5.0:
                print("   ✅ VERY GOOD: Fast batch loading maintained")
            else:
                print("   ⚠️  WARNING: Batch loading may be bottlenecking training")
        
        print("\n📈 Background Processing:")
        print(f"   Total batches produced: {final_status['batches_produced']}")
        print(f"   Average generation time: {final_status['avg_generation_time_ms']:.2f}ms")
        print(f"   Background processing rate: {final_status['background_rate_bps']:.1f} batches/sec")
        
        print("\n🎯 Model Performance:")
        if training_stats['val_losses']:
            final_val_loss = training_stats['val_losses'][-1]
            best_val_loss = min(training_stats['val_losses'])
            print(f"   Final validation loss: {final_val_loss:.6f}")
            print(f"   Best validation loss: {best_val_loss:.6f}")
        
        print("\n💾 Output Files:")
        print(f"   Model checkpoint: {output_dir / 'best_model.pth'}")
        print(f"   TensorBoard logs: {output_dir / 'tensorboard_logs'}")
        print(f"   Run: tensorboard --logdir {output_dir / 'tensorboard_logs'}")
        
        # Close TensorBoard writer
        writer.close()
        
        print("\n✅ Training completed successfully!")
        print("   Use the saved model for inference or continue training.")
        print("=" * 70)

if __name__ == "__main__":
    main()