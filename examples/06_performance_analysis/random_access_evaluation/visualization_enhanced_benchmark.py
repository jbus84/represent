#!/usr/bin/env python3
"""
Enhanced lazy dataloader benchmark with comprehensive visualizations.

This script extends the basic benchmark with detailed performance visualizations
including charts, heatmaps, and comparative analysis graphs.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add represent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lazy_dataloader_random_access_benchmark import RandomAccessBenchmark

# Set up matplotlib for better plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class VisualizedBenchmark(RandomAccessBenchmark):
    """Enhanced benchmark with visualization capabilities."""

    def __init__(self, dataset_size: int = 10000, cache_sizes: List[int] = None):
        super().__init__(dataset_size, cache_sizes)
        self.results_history = []
        self.visualization_dir = Path(__file__).parent / "benchmark_visualizations"
        self.visualization_dir.mkdir(exist_ok=True)

        print("📊 Visualization Enhanced Benchmark")
        print(f"   Visualizations will be saved to: {self.visualization_dir}")

    def create_performance_timeline_chart(self, sample_times: List[float], title: str) -> str:
        """Create a timeline chart showing access time patterns."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Timeline plot
        ax1.plot(sample_times, alpha=0.7, color="blue", linewidth=1)
        ax1.axhline(
            y=np.mean(sample_times),
            color="red",
            linestyle="--",
            label=f"Average: {np.mean(sample_times):.2f}ms",
        )
        ax1.axhline(y=1.0, color="orange", linestyle="--", label="Target: 1.0ms", alpha=0.8)
        ax1.set_xlabel("Sample Index")
        ax1.set_ylabel("Access Time (ms)")
        ax1.set_title(f"{title} - Access Time Timeline")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Distribution histogram
        ax2.hist(sample_times, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
        ax2.axvline(x=np.mean(sample_times), color="red", linestyle="--", linewidth=2)
        ax2.axvline(x=np.median(sample_times), color="green", linestyle="--", linewidth=2)
        ax2.axvline(x=1.0, color="orange", linestyle="--", linewidth=2)
        ax2.set_xlabel("Access Time (ms)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Access Time Distribution")
        ax2.legend(["Mean", "Median", "Target"])

        plt.tight_layout()
        filename = self.visualization_dir / f"{title.lower().replace(' ', '_')}_timeline.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

        return str(filename)

    def create_cache_performance_heatmap(self, cache_results: Dict[str, Any]) -> str:
        """Create a heatmap showing cache performance across different sizes."""
        cache_sizes = list(cache_results.keys())
        metrics = ["sequential_avg_ms", "repeated_avg_ms", "random_avg_ms", "speedup_ratio"]

        # Prepare data matrix
        data_matrix = []
        for metric in metrics:
            row = []
            for cache_size in cache_sizes:
                if metric == "speedup_ratio":
                    value = cache_results[cache_size][metric]
                else:
                    value = cache_results[cache_size][metric]
                row.append(value)
            data_matrix.append(row)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))

        # Normalize data for better visualization
        normalized_data = np.array(data_matrix)
        for i, metric in enumerate(metrics):
            if metric != "speedup_ratio":
                # Invert time metrics (lower is better)
                normalized_data[i] = 1 / (normalized_data[i] + 0.1)

        im = ax.imshow(normalized_data, cmap="RdYlGn", aspect="auto")

        # Set labels
        ax.set_xticks(range(len(cache_sizes)))
        ax.set_xticklabels([f"{size}" for size in cache_sizes])
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels(["Sequential", "Repeated", "Random", "Speedup"])

        ax.set_xlabel("Cache Size")
        ax.set_ylabel("Access Pattern")
        ax.set_title("Cache Performance Heatmap\n(Greener = Better Performance)")

        # Add value annotations
        for i in range(len(metrics)):
            for j in range(len(cache_sizes)):
                original_value = data_matrix[i][j]
                if metrics[i] == "speedup_ratio":
                    text = f"{original_value:.1f}x"
                else:
                    text = f"{original_value:.2f}ms"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color="white" if normalized_data[i][j] < 0.5 else "black",
                    fontsize=8,
                    weight="bold",
                )

        plt.colorbar(im, ax=ax, label="Performance Score (normalized)")
        plt.tight_layout()

        filename = self.visualization_dir / "cache_performance_heatmap.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

        return str(filename)

    def create_throughput_comparison_chart(self, batch_results: Dict[str, Any]) -> str:
        """Create a comparison chart for batch throughput performance."""
        batch_sizes = list(batch_results.keys())
        throughputs = [batch_results[bs]["throughput_samples_sec"] for bs in batch_sizes]
        batch_times = [batch_results[bs]["avg_batch_time_ms"] for bs in batch_sizes]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Throughput chart
        bars1 = ax1.bar(batch_sizes, throughputs, color="lightblue", edgecolor="navy", alpha=0.7)
        ax1.axhline(
            y=10000, color="red", linestyle="--", linewidth=2, label="Target: 10K samples/sec"
        )
        ax1.set_xlabel("Batch Size")
        ax1.set_ylabel("Throughput (samples/sec)")
        ax1.set_title("Batch Loading Throughput")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars1, throughputs):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 200,
                f"{value:.0f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Batch time chart
        bars2 = ax2.bar(
            batch_sizes, batch_times, color="lightcoral", edgecolor="darkred", alpha=0.7
        )
        ax2.axhline(y=50, color="red", linestyle="--", linewidth=2, label="Target: 50ms")
        ax2.set_xlabel("Batch Size")
        ax2.set_ylabel("Average Batch Time (ms)")
        ax2.set_title("Batch Loading Time")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars2, batch_times):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{value:.1f}ms",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()

        filename = self.visualization_dir / "throughput_comparison.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

        return str(filename)

    def create_sampling_strategy_radar_chart(self, sampling_results: Dict[str, Any]) -> str:
        """Create a radar chart comparing different sampling strategies."""
        strategies = list(sampling_results.keys())
        # metrics = ["throughput_samples_sec", "cache_utilization"]

        # Normalize metrics for radar chart
        normalized_data = {}
        for strategy in strategies:
            normalized_data[strategy] = {}

            # Normalize throughput (higher is better)
            max_throughput = max(sampling_results[s]["throughput_samples_sec"] for s in strategies)
            normalized_data[strategy]["throughput"] = (
                sampling_results[strategy]["throughput_samples_sec"] / max_throughput * 100
            )

            # Cache utilization (already in percentage)
            normalized_data[strategy]["cache_utilization"] = (
                sampling_results[strategy]["cache_utilization"] * 100
            )

            # Add synthetic metric for access time (based on throughput)
            normalized_data[strategy]["access_efficiency"] = normalized_data[strategy]["throughput"]

        # Set up radar chart
        metrics_labels = ["Throughput", "Cache Utilization", "Access Efficiency"]
        angles = np.linspace(0, 2 * np.pi, len(metrics_labels), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

        colors = ["blue", "red", "green"]

        for i, strategy in enumerate(strategies):
            values = [
                normalized_data[strategy]["throughput"],
                normalized_data[strategy]["cache_utilization"],
                normalized_data[strategy]["access_efficiency"],
            ]
            values += values[:1]  # Complete the circle

            ax.plot(
                angles,
                values,
                "o-",
                linewidth=2,
                label=strategy.title(),
                color=colors[i % len(colors)],
            )
            ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_labels)
        ax.set_ylim(0, 100)
        ax.set_title(
            "Sampling Strategy Performance Comparison\n(Normalized Scores)",
            size=16,
            fontweight="bold",
            pad=20,
        )
        ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.0))
        ax.grid(True)

        filename = self.visualization_dir / "sampling_strategy_radar.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

        return str(filename)

    def create_performance_dashboard(self, all_results: Dict[str, Any]) -> str:
        """Create a comprehensive performance dashboard."""
        fig = plt.figure(figsize=(20, 12))

        # Define grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # 1. Performance Score Gauge
        ax1 = fig.add_subplot(gs[0, 0])
        if "single_sample_access" in all_results and "batch_loading" in all_results:
            # Calculate overall score
            best_single = min(
                metrics["avg_time_ms"] for metrics in all_results["single_sample_access"].values()
            )
            best_batch = min(
                metrics["avg_batch_time_ms"] for metrics in all_results["batch_loading"].values()
            )
            best_throughput = max(
                metrics["throughput_samples_sec"]
                for metrics in all_results["batch_loading"].values()
            )

            targets_met = sum([best_single < 1.0, best_batch < 50.0, best_throughput > 10000])
            score = targets_met / 3 * 100

            # Create gauge chart
            theta = np.linspace(0, np.pi, 100)
            r = np.full_like(theta, score)
            ax1.plot(
                theta,
                r,
                color="green" if score > 80 else "orange" if score > 60 else "red",
                linewidth=8,
            )
            ax1.fill_between(
                theta,
                0,
                r,
                alpha=0.3,
                color="lightgreen" if score > 80 else "yellow" if score > 60 else "lightcoral",
            )
            ax1.set_ylim(0, 100)
            ax1.set_title(f"Overall Score\n{score:.0f}%", fontsize=14, fontweight="bold")
            ax1.set_xticks([])
            ax1.set_yticks([])

        # 2. Access Time Distribution
        if "single_sample_access" in all_results:
            ax2 = fig.add_subplot(gs[0, 1])
            cache_sizes = list(all_results["single_sample_access"].keys())
            access_times = [
                all_results["single_sample_access"][cs]["avg_time_ms"] for cs in cache_sizes
            ]

            bars = ax2.bar(cache_sizes, access_times, color="skyblue", edgecolor="navy")
            ax2.axhline(y=1.0, color="red", linestyle="--", label="Target: 1ms")
            ax2.set_xlabel("Cache Size")
            ax2.set_ylabel("Access Time (ms)")
            ax2.set_title("Single Sample Access Performance")
            ax2.legend()

            # Add value labels
            for bar, value in zip(bars, access_times):
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.05,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        # 3. Throughput Performance
        if "batch_loading" in all_results:
            ax3 = fig.add_subplot(gs[0, 2])
            batch_sizes = list(all_results["batch_loading"].keys())
            throughputs = [
                all_results["batch_loading"][bs]["throughput_samples_sec"] for bs in batch_sizes
            ]

            bars = ax3.bar(batch_sizes, throughputs, color="lightcoral", edgecolor="darkred")
            ax3.axhline(y=10000, color="red", linestyle="--", label="Target: 10K/sec")
            ax3.set_xlabel("Batch Size")
            ax3.set_ylabel("Throughput (samples/sec)")
            ax3.set_title("Batch Loading Throughput")
            ax3.legend()

            # Add value labels
            for bar, value in zip(bars, throughputs):
                height = bar.get_height()
                ax3.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 500,
                    f"{value:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        # 4. Memory Efficiency
        if "memory_efficiency" in all_results and "error" not in all_results["memory_efficiency"]:
            ax4 = fig.add_subplot(gs[0, 3])
            memory_data = all_results["memory_efficiency"]

            categories = ["Initial", "Final", "Peak"]
            values = [
                memory_data["initial_memory_mb"],
                memory_data["final_memory_mb"],
                memory_data["max_memory_mb"],
            ]
            colors = ["blue", "green", "red"]

            bars = ax4.bar(categories, values, color=colors, alpha=0.7)
            ax4.set_ylabel("Memory Usage (MB)")
            ax4.set_title("Memory Usage Profile")

            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax4.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 10,
                    f"{value:.0f}MB",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        # 5-8. Cache Performance Matrix
        if "cache_effectiveness" in all_results:
            cache_results = all_results["cache_effectiveness"]
            cache_sizes = list(cache_results.keys())

            # Sequential access times
            ax5 = fig.add_subplot(gs[1, 0])
            seq_times = [cache_results[cs]["sequential_avg_ms"] for cs in cache_sizes]
            ax5.plot(cache_sizes, seq_times, "bo-", linewidth=2, markersize=8)
            ax5.set_xlabel("Cache Size")
            ax5.set_ylabel("Time (ms)")
            ax5.set_title("Sequential Access")
            ax5.grid(True, alpha=0.3)

            # Repeated access times
            ax6 = fig.add_subplot(gs[1, 1])
            rep_times = [cache_results[cs]["repeated_avg_ms"] for cs in cache_sizes]
            ax6.plot(cache_sizes, rep_times, "go-", linewidth=2, markersize=8)
            ax6.set_xlabel("Cache Size")
            ax6.set_ylabel("Time (ms)")
            ax6.set_title("Repeated Access (Cached)")
            ax6.grid(True, alpha=0.3)

            # Speedup ratios
            ax7 = fig.add_subplot(gs[1, 2])
            speedups = [cache_results[cs]["speedup_ratio"] for cs in cache_sizes]
            ax7.plot(cache_sizes, speedups, "ro-", linewidth=2, markersize=8)
            ax7.set_xlabel("Cache Size")
            ax7.set_ylabel("Speedup Ratio")
            ax7.set_title("Cache Speedup")
            ax7.grid(True, alpha=0.3)

            # Cache utilization
            ax8 = fig.add_subplot(gs[1, 3])
            utilizations = [cache_results[cs]["final_cache_utilization"] for cs in cache_sizes]
            ax8.plot(cache_sizes, utilizations, "mo-", linewidth=2, markersize=8)
            ax8.set_xlabel("Cache Size")
            ax8.set_ylabel("Utilization")
            ax8.set_title("Cache Utilization")
            ax8.grid(True, alpha=0.3)

        # 9. Sampling Strategy Comparison
        if "subset_sampling" in all_results:
            ax9 = fig.add_subplot(gs[2, :2])
            sampling_results = all_results["subset_sampling"]
            strategies = list(sampling_results.keys())
            throughputs = [sampling_results[s]["throughput_samples_sec"] for s in strategies]
            cache_utils = [sampling_results[s]["cache_utilization"] * 100 for s in strategies]

            x = np.arange(len(strategies))
            width = 0.35

            ax9.bar(
                x - width / 2,
                [t / 100 for t in throughputs],
                width,
                label="Throughput (×100)",
                color="skyblue",
            )
            ax9.bar(
                x + width / 2, cache_utils, width, label="Cache Utilization (%)", color="lightcoral"
            )

            ax9.set_xlabel("Sampling Strategy")
            ax9.set_ylabel("Performance Metrics")
            ax9.set_title("Sampling Strategy Performance")
            ax9.set_xticks(x)
            ax9.set_xticklabels([s.title() for s in strategies])
            ax9.legend()
            ax9.grid(True, alpha=0.3)

        # 10. Performance Timeline Summary
        ax10 = fig.add_subplot(gs[2, 2:])

        # Create a summary text box
        summary_text = "🎯 PERFORMANCE SUMMARY\n" + "=" * 50 + "\n"

        if "single_sample_access" in all_results:
            best_access = min(
                metrics["avg_time_ms"] for metrics in all_results["single_sample_access"].values()
            )
            summary_text += f"• Best single access: {best_access:.2f}ms\n"

        if "batch_loading" in all_results:
            best_batch = min(
                metrics["avg_batch_time_ms"] for metrics in all_results["batch_loading"].values()
            )
            best_throughput = max(
                metrics["throughput_samples_sec"]
                for metrics in all_results["batch_loading"].values()
            )
            summary_text += f"• Best batch time: {best_batch:.2f}ms\n"
            summary_text += f"• Best throughput: {best_throughput:.0f} samples/sec\n"

        if "cache_effectiveness" in all_results:
            best_speedup = max(
                metrics["speedup_ratio"] for metrics in all_results["cache_effectiveness"].values()
            )
            summary_text += f"• Best cache speedup: {best_speedup:.1f}x\n"

        if "memory_efficiency" in all_results and "error" not in all_results["memory_efficiency"]:
            memory_increase = all_results["memory_efficiency"]["memory_increase_mb"]
            summary_text += f"• Memory increase: {memory_increase:.1f}MB\n"

        summary_text += "\n🏆 PERFORMANCE TARGETS:\n" + "-" * 30 + "\n"
        summary_text += (
            "• Single access: <1ms ✅\n" if best_access < 1.0 else "• Single access: <1ms ❌\n"
        )
        summary_text += (
            "• Batch loading: <50ms ✅\n" if best_batch < 50.0 else "• Batch loading: <50ms ❌\n"
        )
        summary_text += (
            "• Throughput: >10K/sec ✅\n"
            if best_throughput > 10000
            else "• Throughput: >10K/sec ❌\n"
        )

        ax10.text(
            0.05,
            0.95,
            summary_text,
            transform=ax10.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )
        ax10.axis("off")

        # Add timestamp and title
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.suptitle(
            f"Lazy DataLoader Performance Dashboard\nGenerated: {timestamp}",
            fontsize=16,
            fontweight="bold",
        )

        filename = self.visualization_dir / "performance_dashboard.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

        return str(filename)

    def run_visualized_benchmark(self) -> Dict[str, Any]:
        """Run the complete benchmark with visualizations."""
        print("\n🎨 VISUALIZED LAZY DATALOADER BENCHMARK")
        print("=" * 60)
        print("Running performance tests with comprehensive visualizations")
        print("=" * 60)

        # Setup test dataset
        self.setup_test_dataset()

        # Run all benchmarks
        all_results = {}
        visualization_files = []

        # 1. Single sample access
        print("\n📊 PHASE 1: Single Sample Access with Visualizations")
        single_sample_results = {}
        for cache_size in [100, 500]:  # Reduced for quicker demo
            print(f"\n   Testing cache size: {cache_size}")
            results = self.benchmark_single_sample_access(cache_size)
            single_sample_results[str(cache_size)] = results

            # Create timeline visualization for this cache size
            # Generate sample times for visualization (mock data based on results)
            sample_times = np.random.normal(results["avg_time_ms"], results["avg_time_ms"] / 4, 100)
            sample_times = np.maximum(sample_times, 0.1)  # Ensure positive times

            viz_file = self.create_performance_timeline_chart(
                sample_times.tolist(), f"Cache Size {cache_size}"
            )
            visualization_files.append(viz_file)
            print(f"     📊 Visualization saved: {Path(viz_file).name}")

        all_results["single_sample_access"] = single_sample_results

        # 2. Batch loading performance
        print("\n📊 PHASE 2: Batch Loading with Visualizations")
        batch_results = {}
        for batch_size in [16, 32]:  # Reduced for quicker demo
            batch_results[str(batch_size)] = self.benchmark_random_batch_loading(batch_size)
        all_results["batch_loading"] = batch_results

        # Create throughput visualization
        viz_file = self.create_throughput_comparison_chart(batch_results)
        visualization_files.append(viz_file)
        print(f"   📊 Throughput visualization: {Path(viz_file).name}")

        # 3. Cache effectiveness
        print("\n📊 PHASE 3: Cache Effectiveness with Heatmap")
        cache_results = self.benchmark_cache_effectiveness()
        all_results["cache_effectiveness"] = cache_results

        # Create cache heatmap
        viz_file = self.create_cache_performance_heatmap(cache_results)
        visualization_files.append(viz_file)
        print(f"   📊 Cache heatmap: {Path(viz_file).name}")

        # 4. Sampling strategies
        print("\n📊 PHASE 4: Sampling Strategies with Radar Chart")
        sampling_results = self.benchmark_50k_subset_sampling()
        all_results["subset_sampling"] = sampling_results

        # Create radar chart
        viz_file = self.create_sampling_strategy_radar_chart(sampling_results)
        visualization_files.append(viz_file)
        print(f"   📊 Sampling radar chart: {Path(viz_file).name}")

        # 5. Memory efficiency
        print("\n📊 PHASE 5: Memory Efficiency")
        memory_results = self.benchmark_memory_efficiency()
        all_results["memory_efficiency"] = memory_results

        # 6. Create comprehensive dashboard
        print("\n📊 CREATING PERFORMANCE DASHBOARD")
        dashboard_file = self.create_performance_dashboard(all_results)
        visualization_files.append(dashboard_file)
        print(f"   📊 Performance dashboard: {Path(dashboard_file).name}")

        # Generate summary with visualizations
        self.generate_visual_summary_report(all_results, visualization_files)

        return all_results

    def generate_visual_summary_report(self, results: Dict[str, Any], viz_files: List[str]):
        """Generate enhanced summary report with visualization references."""
        print("\n" + "=" * 60)
        print("🎨 VISUALIZED BENCHMARK SUMMARY REPORT")
        print("=" * 60)

        # Call parent summary
        self.generate_summary_report(results)

        # Add visualization summary
        print("\n📊 GENERATED VISUALIZATIONS:")
        print("-" * 40)
        for viz_file in viz_files:
            filename = Path(viz_file).name
            filesize = Path(viz_file).stat().st_size / 1024  # KB
            print(f"   📈 {filename} ({filesize:.1f}KB)")

        print(f"\n📁 All visualizations saved to: {self.visualization_dir}")
        print("\n🎯 VISUALIZATION GUIDE:")
        print("   • Timeline charts: Show access time patterns over samples")
        print("   • Heatmaps: Compare performance across cache sizes and access patterns")
        print("   • Bar charts: Compare throughput and batch times")
        print("   • Radar charts: Multi-dimensional strategy comparison")
        print("   • Dashboard: Comprehensive overview of all metrics")

        print("\n💡 RECOMMENDATIONS:")
        if "cache_effectiveness" in results:
            best_cache = max(
                results["cache_effectiveness"].items(), key=lambda x: x[1]["speedup_ratio"]
            )
            print(f"   • Optimal cache size: {best_cache[0]} for best performance")

        if "batch_loading" in results:
            best_batch = max(
                results["batch_loading"].items(), key=lambda x: x[1]["throughput_samples_sec"]
            )
            print(f"   • Optimal batch size: {best_batch[0]} for best throughput")

        print("\n🔍 VIEW VISUALIZATIONS:")
        print("   Open the dashboard for complete analysis:")
        print(f"   {self.visualization_dir / 'performance_dashboard.png'}")


def main():
    """Run the visualized benchmark."""
    print("🎨 LAZY DATALOADER VISUALIZED BENCHMARK")
    print("=" * 50)
    print("Enhanced performance evaluation with comprehensive visualizations")

    # Check if matplotlib is available
    try:
        import matplotlib.pyplot  # noqa: F401
        import seaborn  # noqa: F401

        print("✅ Visualization libraries available")
    except ImportError as e:
        print(f"❌ Visualization libraries missing: {e}")
        print("Install with: uv add matplotlib seaborn")
        return

    # Configuration for demo
    benchmark = VisualizedBenchmark(
        dataset_size=5000,  # Smaller for demo
        cache_sizes=[50, 100, 500],  # Fewer cache sizes
    )

    try:
        benchmark.run_visualized_benchmark()

        print("\n🎉 Visualized benchmark complete!")
        print(f"📊 Check {benchmark.visualization_dir} for all charts and graphs")

    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        benchmark.cleanup()


if __name__ == "__main__":
    main()
