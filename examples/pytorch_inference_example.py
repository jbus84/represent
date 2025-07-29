#!/usr/bin/env python3
"""
PyTorch Inference Example with Background Batch Processing

This example demonstrates how to use a trained model for real-time inference
with the represent package's background batch processing.

Key Features:
- Model loading and inference setup
- Real-time market depth processing
- Background batch processing for low-latency inference
- Batch inference for multiple predictions
- Performance monitoring
"""
import time
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np

from represent.dataloader import MarketDepthDataset, AsyncDataLoader
from represent.constants import SAMPLES
from tests.unit.fixtures.sample_data import generate_realistic_market_data


class MarketDepthCNN(nn.Module):
    """
    Simple CNN model for market depth inference.
    (Same architecture as training example)
    """
    
    def __init__(self, dropout_rate=0.3):
        super(MarketDepthCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 5), padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((10, 10)),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 10 * 10, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x


def create_model_for_inference(device):
    """Create and initialize model for inference."""
    model = MarketDepthCNN(dropout_rate=0.0)  # No dropout for inference
    model.to(device)
    model.eval()  # Set to evaluation mode
    return model


def load_trained_model(model, checkpoint_path, device):
    """Load a trained model from checkpoint."""
    if Path(checkpoint_path).exists():
        print(f"📂 Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        epoch = checkpoint.get('epoch', 'unknown')
        val_loss = checkpoint.get('val_loss', 'unknown')
        print(f"   ✅ Loaded model from epoch {epoch}, validation loss: {val_loss}")
        return True
    else:
        print(f"⚠️  No checkpoint found at {checkpoint_path}")
        print("   Using randomly initialized model for demonstration")
        return False


def single_prediction_demo(model, async_loader, device):
    """Demonstrate single prediction with timing."""
    print("\n🎯 Single Prediction Demo")
    print("-" * 40)
    
    # Get a single batch
    start_time = time.perf_counter()
    
    market_depth = async_loader.get_batch()  # Shape: (402, 500)
    batch_time = time.perf_counter() - start_time
    
    # Prepare for inference
    input_tensor = market_depth.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 402, 500)
    
    # Run inference
    inference_start = time.perf_counter()
    
    with torch.no_grad():
        prediction = model(input_tensor)
    
    inference_time = time.perf_counter() - inference_start
    total_time = time.perf_counter() - start_time
    
    # Display results
    pred_value = prediction.cpu().item()
    print(f"   📊 Prediction: {pred_value:.6f}")
    print(f"   ⚡ Batch loading: {batch_time*1000:.3f}ms")
    print(f"   🧠 Inference: {inference_time*1000:.3f}ms")
    print(f"   🕐 Total time: {total_time*1000:.3f}ms")
    
    # Interpret prediction
    if pred_value > 0.1:
        direction = "📈 STRONG UP"
    elif pred_value > 0.02:
        direction = "↗️ UP"
    elif pred_value > -0.02:
        direction = "➡️ FLAT"
    elif pred_value > -0.1:
        direction = "↘️ DOWN"
    else:
        direction = "📉 STRONG DOWN"
    
    print(f"   🎯 Signal: {direction}")
    
    return pred_value, total_time


def batch_prediction_demo(model, async_loader, device, batch_size=10):
    """Demonstrate batch prediction for multiple samples."""
    print(f"\n📦 Batch Prediction Demo (batch_size={batch_size})")
    print("-" * 50)
    
    predictions = []
    batch_times = []
    inference_times = []
    
    print("   Processing batches...")
    
    for i in range(batch_size):
        # Get batch
        batch_start = time.perf_counter()
        market_depth = async_loader.get_batch()
        batch_time = time.perf_counter() - batch_start
        batch_times.append(batch_time * 1000)
        
        # Inference
        input_tensor = market_depth.unsqueeze(0).unsqueeze(0).to(device)
        
        inference_start = time.perf_counter()
        with torch.no_grad():
            prediction = model(input_tensor)
        inference_time = time.perf_counter() - inference_start
        inference_times.append(inference_time * 1000)
        
        pred_value = prediction.cpu().item()
        predictions.append(pred_value)
        
        if i % 3 == 0:  # Progress update
            print(f"      Batch {i+1:2d}: pred={pred_value:+.4f}, "
                  f"load={batch_time*1000:.2f}ms, infer={inference_time*1000:.2f}ms")
    
    # Statistics
    avg_batch_time = np.mean(batch_times)
    avg_inference_time = np.mean(inference_times)
    total_throughput = batch_size / (sum(batch_times) + sum(inference_times)) * 1000
    
    print("\n   📊 Batch Statistics:")
    print(f"      Average batch loading: {avg_batch_time:.3f}ms")
    print(f"      Average inference: {avg_inference_time:.3f}ms")
    print(f"      Total throughput: {total_throughput:.1f} predictions/second")
    
    # Prediction statistics
    pred_mean = np.mean(predictions)
    pred_std = np.std(predictions)
    pred_min = np.min(predictions)
    pred_max = np.max(predictions)
    
    print("   🎯 Prediction Statistics:")
    print(f"      Mean: {pred_mean:+.6f}")
    print(f"      Std:  {pred_std:.6f}")
    print(f"      Range: [{pred_min:+.6f}, {pred_max:+.6f}]")
    
    return predictions, avg_batch_time, avg_inference_time


def streaming_inference_demo(model, async_loader, device, duration_seconds=10):
    """Demonstrate continuous streaming inference."""
    print(f"\n🌊 Streaming Inference Demo ({duration_seconds}s)")
    print("-" * 45)
    
    predictions = []
    timestamps = []
    processing_times = []
    
    start_time = time.perf_counter()
    next_report = start_time + 2  # Report every 2 seconds
    
    print("   Starting continuous inference...")
    
    while time.perf_counter() - start_time < duration_seconds:
        iteration_start = time.perf_counter()
        
        # Get batch and run inference
        market_depth = async_loader.get_batch()
        input_tensor = market_depth.unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = model(input_tensor)
        
        # Record results
        pred_value = prediction.cpu().item()
        current_time = time.perf_counter()
        processing_time = (current_time - iteration_start) * 1000
        
        predictions.append(pred_value)
        timestamps.append(current_time - start_time)
        processing_times.append(processing_time)
        
        # Periodic reporting
        if current_time >= next_report:
            elapsed = current_time - start_time
            recent_times = processing_times[-20:]
            
            print(f"      {elapsed:.1f}s: {len(predictions)} predictions, "
                  f"avg_time={np.mean(recent_times):.2f}ms, "
                  f"latest_pred={pred_value:+.4f}")
            
            next_report += 2
        
        # Small delay to prevent overwhelming
        time.sleep(0.01)
    
    # Final statistics
    total_time = time.perf_counter() - start_time
    avg_processing_time = np.mean(processing_times)
    throughput = len(predictions) / total_time
    
    print("\n   📊 Streaming Results:")
    print(f"      Total predictions: {len(predictions)}")
    print(f"      Total time: {total_time:.2f}s")
    print(f"      Average processing time: {avg_processing_time:.3f}ms")
    print(f"      Throughput: {throughput:.1f} predictions/second")
    
    # Performance assessment
    if avg_processing_time < 5.0:
        print("      🚀 EXCELLENT: Ultra-low latency achieved!")
    elif avg_processing_time < 10.0:
        print("      ✅ VERY GOOD: Low latency maintained")
    else:
        print("      ⚠️  WARNING: Higher latency detected")
    
    return predictions, processing_times, throughput


def main():
    """Main inference demonstration."""
    print("🔮 PYTORCH INFERENCE WITH BACKGROUND BATCH PROCESSING")
    print("=" * 70)
    print("This example demonstrates real-time inference using background")
    print("batch processing for ultra-low latency market predictions.")
    print("=" * 70)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    # 1. Create and load model
    print("\n1️⃣  Setting up inference model...")
    model = create_model_for_inference(device)
    
    # Try to load trained model (from training example)
    checkpoint_path = Path("training_output/best_model.pth")
    load_trained_model(model, checkpoint_path, device)
    
    # 2. Setup data and background processing
    print("\n2️⃣  Setting up market data and background processing...")
    dataset = MarketDepthDataset(buffer_size=SAMPLES)
    
    # Generate market data for inference
    market_data = generate_realistic_market_data(SAMPLES)
    dataset.add_streaming_data(market_data)
    
    # Create background processor optimized for inference
    async_loader = AsyncDataLoader(
        dataset=dataset,
        background_queue_size=8,  # Larger queue for inference
        prefetch_batches=4        # Keep queue well-stocked
    )
    
    # Start background processing
    async_loader.start_background_production()
    
    # Wait for queue to fill
    print("   ⏳ Warming up background processing...")
    time.sleep(0.5)
    
    status = async_loader.queue_status
    print("   ✅ Background processing ready")
    print(f"   📊 Queue: {status['queue_size']}/{status['max_queue_size']} batches ready")
    
    try:
        # 3. Run inference demonstrations
        print("\n3️⃣  Running inference demonstrations...")
        
        # Single prediction
        pred_value, total_time = single_prediction_demo(model, async_loader, device)
        
        # Batch predictions
        predictions, avg_batch_time, avg_inference_time = batch_prediction_demo(
            model, async_loader, device, batch_size=15
        )
        
        # Streaming inference
        stream_predictions, stream_times, throughput = streaming_inference_demo(
            model, async_loader, device, duration_seconds=8
        )
        
        # 4. Performance summary
        print("\n4️⃣  Performance Summary")
        print("=" * 40)
        
        print("📊 Inference Performance:")
        print(f"   Single prediction: {total_time*1000:.3f}ms end-to-end")
        print(f"   Batch loading: {avg_batch_time:.3f}ms average")
        print(f"   Model inference: {avg_inference_time:.3f}ms average")
        print(f"   Streaming throughput: {throughput:.1f} predictions/second")
        
        if avg_batch_time < 1.0:
            print("   🎯 RESULT: Background processing eliminates data bottlenecks!")
            print(f"   ⚡ GPU utilization: ~{(avg_inference_time/(avg_batch_time+avg_inference_time))*100:.0f}%")
        
        print("\n🔄 Background Processing:")
        final_status = async_loader.queue_status
        print(f"   Batches produced: {final_status['batches_produced']}")
        print(f"   Background generation: {final_status['avg_generation_time_ms']:.2f}ms average")
        print(f"   Queue efficiency: {(final_status['queue_size']/final_status['max_queue_size']*100):.1f}%")
        
        print("\n💡 Key Benefits Demonstrated:")
        print("   ✅ Sub-millisecond batch loading with background processing")
        print("   ✅ Consistent low-latency inference performance")
        print("   ✅ High throughput for real-time trading applications")
        print("   ✅ No data loading bottlenecks during inference")
        
    except KeyboardInterrupt:
        print("\n⚠️  Inference demo interrupted by user")
    
    finally:
        # Cleanup
        print("\n5️⃣  Cleaning up...")
        async_loader.stop()
        print("   ✅ Background processing stopped")
        
        print("\n🏁 Inference demonstration completed!")
        print("   This shows how background processing enables real-time,")
        print("   low-latency inference for trading applications.")
        print("=" * 70)


if __name__ == "__main__":
    main()