"""
PyTorch training integration example.

Demonstrates how to use the represent library with PyTorch for model training
using the new RepresentConfig system and parquet-based data pipeline.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from represent import RepresentConfig, create_parquet_dataloader


class SimpleMarketDepthCNN(nn.Module):
    """Simple CNN for market depth classification."""
    
    def __init__(self, num_features=1, num_classes=13):
        super().__init__()
        
        input_channels = num_features
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_model_example():
    """Demonstrate PyTorch training with represent library."""
    
    print("🧠 PyTorch Training Integration Example")
    print("="*45)
    
    # Configuration for training
    config = RepresentConfig(
        currency="AUDUSD",
        lookback_rows=3000,      # Configurable
        lookforward_input=2000,  # Configurable
        batch_size=32,           # Training batch size
        features=["volume"],     # Single feature for this example
        nbins=13                 # Number of classification classes
    )
    
    print("📊 Training Configuration:")
    print(f"   Currency: {config.currency}")
    print(f"   Features: {config.features}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Classes: {config.nbins}")
    print(f"   Lookback: {config.lookback_rows}")
    
    # Check for training data
    data_path = Path("data/classified/sample_classified.parquet")
    if not data_path.exists():
        print(f"\n❌ Training data not found: {data_path}")
        print("💡 Please run the classification examples first to generate training data")
        print("💡 Or use: python examples/new_architecture/dbn_to_parquet_example.py")
        return
    
    try:
        # Create dataloader
        print("\n📦 Creating training dataloader...")
        dataloader = create_parquet_dataloader(
            parquet_path=data_path,
            batch_size=config.batch_size,
            shuffle=True,
            sample_fraction=0.1,  # Use 10% of data for quick demo
            num_workers=2
        )
        
        print(f"✅ Dataloader created with {len(dataloader)} batches")
        
        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {device}")
        
        num_features = len(config.features)
        model = SimpleMarketDepthCNN(num_features=num_features, num_classes=config.nbins)
        model.to(device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        
        print("🏗️ Model initialized:")
        print(f"   Input features: {num_features}")
        print(f"   Output classes: {config.nbins}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Training loop
        print("\n🚀 Starting training...")
        model.train()
        
        epoch_losses = []
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (features, labels) in enumerate(dataloader):
            # Move to device
            features = features.to(device)
            labels = labels.to(device)
            
            # Handle single feature case (add channel dimension)
            if num_features == 1 and features.dim() == 3:
                features = features.unsqueeze(1)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            epoch_losses.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            # Progress update
            if batch_idx % 10 == 0:
                accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0
                print(f"   Batch {batch_idx:3d}: Loss = {loss.item():.4f}, "
                      f"Accuracy = {accuracy:.2f}% ({total_correct}/{total_samples})")
        
        # Final statistics
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        final_accuracy = 100.0 * total_correct / total_samples
        
        print("\n✅ Training Complete!")
        print(f"   Average Loss: {avg_loss:.4f}")
        print(f"   Final Accuracy: {final_accuracy:.2f}%")
        print(f"   Total Samples: {total_samples:,}")
        print(f"   Batches Processed: {len(dataloader)}")
        
        # Save model
        model_path = Path("examples/pytorch_integration/trained_model.pth")
        model_path.parent.mkdir(exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'accuracy': final_accuracy,
            'loss': avg_loss
        }, model_path)
        
        print(f"💾 Model saved: {model_path}")
        
        # Demonstrate inference
        demonstrate_inference(model, dataloader, device, config)
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        print("💡 Check that classified parquet data is available")


def demonstrate_inference(model, dataloader, device, config):
    """Demonstrate model inference."""
    
    print("\n🔍 Inference Demonstration:")
    
    model.eval()
    with torch.no_grad():
        # Get first batch for inference demo
        features, labels = next(iter(dataloader))
        features = features.to(device)
        labels = labels.to(device)
        
        # Handle single feature case
        if len(config.features) == 1 and features.dim() == 3:
            features = features.unsqueeze(1)
        
        # Run inference
        outputs = model(features)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        
        # Show results for first few samples
        print("   Sample predictions (first 5):")
        for i in range(min(5, len(predicted))):
            actual = labels[i].item()
            pred = predicted[i].item()
            confidence = probabilities[i][pred].item()
            
            status = "✅" if actual == pred else "❌"
            print(f"     {status} Actual: {actual}, Predicted: {pred}, "
                  f"Confidence: {confidence:.3f}")
        
        # Overall batch accuracy
        batch_accuracy = (predicted == labels).float().mean().item()
        print(f"   Batch accuracy: {batch_accuracy:.3f}")


if __name__ == "__main__":
    train_model_example()