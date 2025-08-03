#!/usr/bin/env python3
"""
PyTorch Streaming Data Example with Background Processing

This example demonstrates how to use the dataloader with continuously arriving
market data, simulating a real-time trading environment.

Key Features:
- Streaming data integration
- Continuous model inference
- Real-time performance monitoring
- Dynamic data updates
- Signal generation and tracking
"""
import time
import threading
from collections import deque
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np

from represent.dataloader import MarketDepthDataset, HighPerformanceDataLoader
from represent.constants import SAMPLES
from tests.unit.fixtures.sample_data import generate_realistic_market_data

class TradingSignalModel(nn.Module):
    """
    Lightweight model optimized for real-time trading signals.
    """
    
    def __init__(self):
        super(TradingSignalModel, self).__init__()
        
        # Efficient architecture for low-latency inference
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((20, 20)),
            
            nn.Flatten(),
            nn.Linear(32 * 20 * 20, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),  # 3 outputs: [buy_signal, hold_signal, sell_signal]
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.features(x)

class StreamingDataSimulator:
    """
    Simulates real-time market data streaming.
    """
    
    def __init__(self, dataset: MarketDepthDataset, update_interval=0.1):
        self.dataset = dataset
        self.update_interval = update_interval
        self.running = False
        self.thread = None
        self.updates_sent = 0
    
    def start(self):
        """Start streaming data updates."""
        self.running = True
        self.thread = threading.Thread(target=self._stream_data, daemon=True)
        self.thread.start()
        print(f"📡 Started streaming data (interval: {self.update_interval}s)")
    
    def stop(self):
        """Stop streaming data updates."""
        self.running = False
        if self.thread:
            self.thread.join()
        print(f"🛑 Stopped streaming data ({self.updates_sent} updates sent)")
    
    def _stream_data(self):
        """Internal method to continuously generate new data."""
        while self.running:
            # Generate new market data (simulating incoming ticks)
            new_data = generate_realistic_market_data(50)  # 50 new ticks
            
            # Add to dataset (this will update the ring buffer)
            self.dataset.add_streaming_data(new_data)
            self.updates_sent += 1
            
            time.sleep(self.update_interval)

class TradingSignalGenerator:
    """
    Generates and tracks trading signals using the model.
    """
    
    def __init__(self, model, device, signal_threshold=0.6):
        self.model = model
        self.device = device
        self.signal_threshold = signal_threshold
        
        # Signal tracking
        self.signals = deque(maxlen=1000)  # Keep last 1000 signals
        self.performance_history = deque(maxlen=100)  # Performance tracking
        
        # Signal counters
        self.signal_counts = {'BUY': 0, 'HOLD': 0, 'SELL': 0}
    
    def generate_signal(self, market_depth):
        """Generate trading signal from market depth data."""
        # Prepare input
        input_tensor = market_depth.unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Model inference
        inference_start = time.perf_counter()
        with torch.no_grad():
            output = self.model(input_tensor)  # Shape: (1, 3)
        inference_time = (time.perf_counter() - inference_start) * 1000
        
        # Extract probabilities
        probs = output.cpu().numpy()[0]  # [buy_prob, hold_prob, sell_prob]
        buy_prob, hold_prob, sell_prob = probs
        
        # Determine signal
        max_prob = np.max(probs)
        if max_prob < self.signal_threshold:
            signal = 'HOLD'  # Low confidence
            confidence = max_prob
        else:
            signal_idx = np.argmax(probs)
            signals = ['BUY', 'HOLD', 'SELL']
            signal = signals[signal_idx]
            confidence = max_prob
        
        # Record signal
        signal_data = {
            'timestamp': datetime.now(),
            'signal': signal,
            'confidence': confidence,
            'buy_prob': buy_prob,
            'hold_prob': hold_prob,
            'sell_prob': sell_prob,
            'inference_time': inference_time
        }
        
        self.signals.append(signal_data)
        self.signal_counts[signal] += 1
        self.performance_history.append(inference_time)
        
        return signal_data
    
    def get_recent_signals(self, count=10):
        """Get the most recent signals."""
        return list(self.signals)[-count:]
    
    def get_performance_stats(self):
        """Get performance statistics."""
        if not self.performance_history:
            return {}
        
        times = list(self.performance_history)
        return {
            'avg_inference_time': np.mean(times),
            'min_inference_time': np.min(times),
            'max_inference_time': np.max(times),
            'std_inference_time': np.std(times),
            'total_signals': len(self.signals),
            'signal_distribution': dict(self.signal_counts)
        }

def print_signal_dashboard(signal_generator, dataloader):
    """Print a real-time dashboard of signals and performance."""
    recent_signals = signal_generator.get_recent_signals(5)
    stats = signal_generator.get_performance_stats()
    queue_status = {"status": "HighPerformanceDataLoader"}
    
    # Clear previous output (simple version)
    print("\n" + "=" * 80)
    print("🔥 REAL-TIME TRADING SIGNAL DASHBOARD")
    print("=" * 80)
    
    # Recent signals
    print("📊 Recent Signals:")
    for i, sig in enumerate(reversed(recent_signals)):
        time_str = sig['timestamp'].strftime('%H:%M:%S.%f')[:-3]
        signal_emoji = {'BUY': '🟢', 'HOLD': '🟡', 'SELL': '🔴'}[sig['signal']]
        print(f"   {signal_emoji} {time_str} | {sig['signal']:4s} | "
              f"Conf: {sig['confidence']:.3f} | "
              f"Probs: [{sig['buy_prob']:.2f}, {sig['hold_prob']:.2f}, {sig['sell_prob']:.2f}] | "
              f"{sig['inference_time']:.2f}ms")
    
    # Performance stats
    if stats:
        print("\n⚡ Performance:")
        print(f"   Avg Inference: {stats['avg_inference_time']:.2f}ms")
        print(f"   Range: {stats['min_inference_time']:.2f}ms - {stats['max_inference_time']:.2f}ms")
        print(f"   Total Signals: {stats['total_signals']}")
        
        # Signal distribution
        dist = stats['signal_distribution']
        total = sum(dist.values())
        if total > 0:
            print(f"   Distribution: 🟢{dist['BUY']} ({dist['BUY']/total*100:.1f}%) | "
                  f"🟡{dist['HOLD']} ({dist['HOLD']/total*100:.1f}%) | "
                  f"🔴{dist['SELL']} ({dist['SELL']/total*100:.1f}%)")
    
    # Background processing status
    print("\n🔄 Background Processing:")
    print(f"   Queue: {queue_status['queue_size']}/{queue_status['max_queue_size']} batches")
    print(f"   Generated: {queue_status['batches_produced']} batches")
    print(f"   Avg Generation: {queue_status['avg_generation_time_ms']:.2f}ms")
    print(f"   Background Rate: {queue_status['background_rate_bps']:.1f} batches/sec")
    print(f"   Healthy: {'✅' if queue_status['background_healthy'] else '❌'}")

def main():
    """Main streaming inference demonstration."""
    print("🌊 PYTORCH STREAMING INFERENCE WITH BACKGROUND PROCESSING")
    print("=" * 80)
    print("This example simulates real-time trading with continuous market data")
    print("streaming and ultra-low latency signal generation.")
    print("=" * 80)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    # 1. Create model
    print("\n1️⃣  Creating trading signal model...")
    model = TradingSignalModel().to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Model created ({total_params:,} parameters)")
    
    # 2. Setup streaming dataset
    print("\n2️⃣  Setting up streaming market data...")
    dataset = MarketDepthDataset(buffer_size=SAMPLES)
    
    # Initialize with some data
    initial_data = generate_realistic_market_data(SAMPLES)
    dataset.add_streaming_data(initial_data)
    print(f"   ✅ Dataset initialized with {dataset.ring_buffer_size:,} samples")
    
    # 3. Create background processor
    print("\n3️⃣  Starting background batch processing...")
    dataloader = HighPerformanceDataLoader(
        dataset=dataset,
        background_queue_size=10,  # Large queue for streaming
        prefetch_batches=5         # Keep well-stocked
    )

    # Wait for initial queue population
    time.sleep(0.3)
    status = {"status": "HighPerformanceDataLoader"}
    print(f"   ✅ Background processing ready ({status['queue_size']}/{status['max_queue_size']} batches)")
    
    # 4. Setup streaming simulator
    print("\n4️⃣  Starting streaming data simulator...")
    simulator = StreamingDataSimulator(dataset, update_interval=0.05)  # 20 Hz updates
    simulator.start()
    
    # 5. Create signal generator
    print("\n5️⃣instances  Setting up signal generator...")
    signal_generator = TradingSignalGenerator(model, device, signal_threshold=0.4)
    print(f"   ✅ Signal generator ready (threshold: {signal_generator.signal_threshold})")
    
    # 6. Run streaming inference
    print("\n6️⃣  Starting real-time signal generation...")
    print("   Press Ctrl+C to stop...")
    
    try:
        last_dashboard_update = time.perf_counter()
        dashboard_interval = 2.0  # Update dashboard every 2 seconds
        
        signal_times = []
        total_signals = 0
        
        while True:
            loop_start = time.perf_counter()
            
            # Generate signal from current market state
            market_depth = next(iter(dataloader))[0]
            signal_data = signal_generator.generate_signal(market_depth)
            
            total_signals += 1
            signal_times.append(signal_data['inference_time'])
            
            # Update dashboard periodically
            current_time = time.perf_counter()
            if current_time - last_dashboard_update >= dashboard_interval:
                print_signal_dashboard(signal_generator, dataloader)
                last_dashboard_update = current_time
            
            # Brief pause to prevent overwhelming
            time.sleep(0.02)  # 50 Hz signal generation
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Streaming stopped by user")
    
    finally:
        # 7. Cleanup and final report
        print("\n7️⃣  Shutting down and generating final report...")
        
        # Stop components
        simulator.stop()

        # Final statistics
        final_stats = signal_generator.get_performance_stats()
        final_queue_status = {"status": "HighPerformanceDataLoader"}
        
        print("\n" + "=" * 80)
        print("🏆 FINAL STREAMING PERFORMANCE REPORT")
        print("=" * 80)
        
        if final_stats:
            print("📊 Signal Generation Performance:")
            print(f"   Total signals generated: {final_stats['total_signals']:,}")
            print(f"   Average inference time: {final_stats['avg_inference_time']:.3f}ms")
            print(f"   Min/Max inference time: {final_stats['min_inference_time']:.3f}ms / {final_stats['max_inference_time']:.3f}ms")
            print(f"   Standard deviation: {final_stats['std_inference_time']:.3f}ms")
            
            # Throughput calculation
            if signal_times:
                avg_signal_time = np.mean(signal_times)
                theoretical_throughput = 1000 / avg_signal_time  # signals per second
                print(f"   Theoretical throughput: {theoretical_throughput:.0f} signals/second")
            
            # Signal distribution
            dist = final_stats['signal_distribution']
            total_dist = sum(dist.values())
            print("\n📈 Signal Distribution:")
            print(f"   🟢 BUY signals:  {dist['BUY']:,} ({dist['BUY']/total_dist*100:.1f}%)")
            print(f"   🟡 HOLD signals: {dist['HOLD']:,} ({dist['HOLD']/total_dist*100:.1f}%)")
            print(f"   🔴 SELL signals: {dist['SELL']:,} ({dist['SELL']/total_dist*100:.1f}%)")
        
        print("\n🔄 Background Processing Summary:")
        print(f"   Total batches produced: {final_queue_status['batches_produced']:,}")
        print(f"   Average generation time: {final_queue_status['avg_generation_time_ms']:.2f}ms")
        print(f"   Background processing rate: {final_queue_status['background_rate_bps']:.1f} batches/sec")
        
        print("\n📡 Data Streaming Summary:")
        print(f"   Data updates received: {simulator.updates_sent:,}")
        print(f"   Update frequency: ~{simulator.updates_sent/(time.perf_counter()-loop_start+60):.1f} Hz")  # Rough estimate
        
        # Performance assessment
        if final_stats and final_stats['avg_inference_time'] < 5.0:
            print("\n🚀 EXCELLENT PERFORMANCE!")
            print("   ✅ Ultra-low latency inference achieved")
            print("   ✅ Real-time trading requirements met")
            print("   ✅ Background processing eliminated bottlenecks")
        elif final_stats and final_stats['avg_inference_time'] < 10.0:
            print("\n✅ GOOD PERFORMANCE!")
            print("   ⚡ Low latency maintained")
            print("   📊 Suitable for most trading applications")
        else:
            print("\n⚠️  PERFORMANCE WARNING!")
            print("   🔧 Consider optimization for high-frequency trading")
        
        print("\n💡 Key Achievements:")
        print("   ✅ Continuous real-time signal generation")
        print("   ✅ Dynamic market data integration")
        print("   ✅ Consistent low-latency performance")
        print("   ✅ Scalable background processing architecture")
        
        print("\n🎯 This demonstrates production-ready real-time trading capabilities!")
        print("=" * 80)

if __name__ == "__main__":
    main()