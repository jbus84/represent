#!/usr/bin/env python3
"""
Quick test of visualization capabilities for the lazy dataloader benchmark.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add represent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_basic_visualization():
    """Test basic visualization functionality."""
    print("🎨 Testing Visualization Capabilities")
    print("=" * 40)

    # Set up output directory
    viz_dir = Path(__file__).parent / "test_visualizations"
    viz_dir.mkdir(exist_ok=True)

    # Set up matplotlib style
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    # Test 1: Simple performance timeline
    print("📊 Creating performance timeline...")

    # Mock performance data
    sample_indices = list(range(100))
    access_times = np.random.normal(1.0, 0.3, 100)  # 1ms average with variation
    access_times = np.maximum(access_times, 0.1)  # Ensure positive

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    # Timeline plot
    ax1.plot(sample_indices, access_times, alpha=0.7, color="blue", linewidth=1)
    ax1.axhline(
        y=np.mean(access_times),
        color="red",
        linestyle="--",
        label=f"Average: {np.mean(access_times):.2f}ms",
    )
    ax1.axhline(y=1.0, color="orange", linestyle="--", label="Target: 1.0ms", alpha=0.8)
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("Access Time (ms)")
    ax1.set_title("Random Access Performance Timeline")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Distribution histogram
    ax2.hist(access_times, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
    ax2.axvline(x=np.mean(access_times), color="red", linestyle="--", linewidth=2, label="Mean")
    ax2.axvline(
        x=np.median(access_times), color="green", linestyle="--", linewidth=2, label="Median"
    )
    ax2.axvline(x=1.0, color="orange", linestyle="--", linewidth=2, label="Target")
    ax2.set_xlabel("Access Time (ms)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Access Time Distribution")
    ax2.legend()

    plt.tight_layout()
    timeline_file = viz_dir / "test_timeline.png"
    plt.savefig(timeline_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"   ✅ Timeline saved: {timeline_file}")

    # Test 2: Cache performance heatmap
    print("📊 Creating cache performance heatmap...")

    cache_sizes = ["50", "100", "500", "1000"]
    metrics = ["Sequential", "Repeated", "Random", "Speedup"]

    # Mock performance data
    data = np.array(
        [
            [2.5, 1.8, 3.2, 0.8],  # Sequential times (ms)
            [0.5, 0.3, 0.7, 0.2],  # Repeated times (ms) - much faster due to cache
            [2.0, 1.5, 2.8, 0.7],  # Random times (ms)
            [5.0, 6.0, 4.5, 4.0],  # Speedup ratios
        ]
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create heatmap
    im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto")

    # Set labels
    ax.set_xticks(range(len(cache_sizes)))
    ax.set_xticklabels(cache_sizes)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics)

    ax.set_xlabel("Cache Size")
    ax.set_ylabel("Access Pattern")
    ax.set_title("Cache Performance Heatmap")

    # Add value annotations
    for i in range(len(metrics)):
        for j in range(len(cache_sizes)):
            value = data[i][j]
            if i == 3:  # Speedup ratio
                text = f"{value:.1f}x"
            else:
                text = f"{value:.1f}ms"
            ax.text(j, i, text, ha="center", va="center", color="white", fontsize=10, weight="bold")

    plt.colorbar(im, ax=ax, label="Performance (lower is better for time metrics)")
    plt.tight_layout()

    heatmap_file = viz_dir / "test_heatmap.png"
    plt.savefig(heatmap_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"   ✅ Heatmap saved: {heatmap_file}")

    # Test 3: Throughput comparison
    print("📊 Creating throughput comparison...")

    batch_sizes = ["16", "32", "64"]
    throughputs = [8500, 12000, 15000]  # samples/sec
    batch_times = [45, 38, 42]  # ms

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Throughput chart
    bars1 = ax1.bar(batch_sizes, throughputs, color="lightblue", edgecolor="navy", alpha=0.7)
    ax1.axhline(y=10000, color="red", linestyle="--", linewidth=2, label="Target: 10K samples/sec")
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Throughput (samples/sec)")
    ax1.set_title("Batch Loading Throughput")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add value labels
    for bar, value in zip(bars1, throughputs):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 300,
            f"{value}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Batch time chart
    bars2 = ax2.bar(batch_sizes, batch_times, color="lightcoral", edgecolor="darkred", alpha=0.7)
    ax2.axhline(y=50, color="red", linestyle="--", linewidth=2, label="Target: 50ms")
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Average Batch Time (ms)")
    ax2.set_title("Batch Loading Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add value labels
    for bar, value in zip(bars2, batch_times):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{value}ms",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()

    throughput_file = viz_dir / "test_throughput.png"
    plt.savefig(throughput_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"   ✅ Throughput chart saved: {throughput_file}")

    # Test 4: Performance dashboard summary
    print("📊 Creating mini dashboard...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Performance score gauge (simplified)
    ax1.pie(
        [85, 15],
        labels=["Met", "Not Met"],
        colors=["lightgreen", "lightcoral"],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax1.set_title("Performance Targets\n85% Met")

    # Access time comparison
    cache_sizes = ["50", "100", "500"]
    access_times = [1.2, 0.8, 0.9]
    ax2.bar(cache_sizes, access_times, color="skyblue")
    ax2.axhline(y=1.0, color="red", linestyle="--", label="Target")
    ax2.set_xlabel("Cache Size")
    ax2.set_ylabel("Access Time (ms)")
    ax2.set_title("Single Sample Access")
    ax2.legend()

    # Memory usage over time
    time_points = list(range(10))
    memory_usage = [450 + i * 5 + np.random.normal(0, 10) for i in range(10)]
    ax3.plot(time_points, memory_usage, "g-o", linewidth=2)
    ax3.set_xlabel("Time (minutes)")
    ax3.set_ylabel("Memory Usage (MB)")
    ax3.set_title("Memory Usage Over Time")
    ax3.grid(True, alpha=0.3)

    # Summary text
    summary = """🎯 PERFORMANCE SUMMARY
    
• Best access time: 0.8ms ✅
• Best throughput: 15K samples/sec ✅  
• Memory efficient: <500MB ✅
• Cache speedup: 6x ✅

Status: EXCELLENT
Ready for production!"""

    ax4.text(
        0.05,
        0.95,
        summary,
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
    )
    ax4.axis("off")

    plt.suptitle("Lazy DataLoader Performance Dashboard (Test)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    dashboard_file = viz_dir / "test_dashboard.png"
    plt.savefig(dashboard_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"   ✅ Dashboard saved: {dashboard_file}")

    # Summary
    print("\n🎉 Visualization Test Complete!")
    print(f"📁 Test visualizations saved to: {viz_dir}")

    file_sizes = []
    for viz_file in [timeline_file, heatmap_file, throughput_file, dashboard_file]:
        size_kb = viz_file.stat().st_size / 1024
        file_sizes.append(size_kb)
        print(f"   📈 {viz_file.name}: {size_kb:.1f}KB")

    total_size = sum(file_sizes)
    print(f"📊 Total size: {total_size:.1f}KB")

    print("\n✅ Visualization system ready for full benchmark!")

    return viz_dir


def main():
    """Run visualization test."""
    try:
        test_basic_visualization()
    except ImportError as e:
        print(f"❌ Missing visualization dependencies: {e}")
        print("Install with: uv add matplotlib seaborn")
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
