#!/usr/bin/env python
"""
Quick Demo Runner

Simplified runner for the complete workflow demo.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run the complete workflow demo and open the results."""
    script_path = Path(__file__).parent / "complete_workflow_demo.py"

    print("🚀 Running Complete Workflow Demo...")
    print("=" * 50)

    try:
        # Run the demo
        result = subprocess.run([sys.executable, str(script_path)],
                              capture_output=False, text=True)

        if result.returncode == 0:
            # Try to open the HTML report
            report_path = Path(__file__).parent / "complete_workflow_output" / "complete_workflow_report.html"

            if report_path.exists():
                print("\n✅ Demo completed successfully!")
                print(f"📊 HTML Report: {report_path}")
                print(f"📁 Output Directory: {report_path.parent}")

                # Try to open in browser (optional)
                try:
                    import webbrowser
                    webbrowser.open(f"file://{report_path.absolute()}")
                    print("🌐 Report opened in browser")
                except Exception:
                    print("💡 Manually open the HTML report to view results")
            else:
                print("⚠️  Demo completed but report not found")
        else:
            print(f"❌ Demo failed with exit code: {result.returncode}")
            return 1

    except Exception as e:
        print(f"❌ Error running demo: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
