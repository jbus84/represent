#!/usr/bin/env python3
"""
Simple API Demo - Quick test for the examples runner

This is a minimal example that demonstrates basic represent functionality
and generates some sample output for testing the HTML report system.
"""

import sys
from pathlib import Path
import time
import json

# Add represent package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def main():
    """Simple demonstration of represent configuration."""
    
    print("🚀 Simple API Demo - Represent Package")
    print("=" * 50)
    
    # Simulate some processing
    print("📊 Creating sample configuration...")
    time.sleep(0.5)
    
    try:
        from represent import create_represent_config
        
        # Create configuration
        config = create_represent_config(
            currency="AUDUSD",
            samples=1000,
            features=["volume"]
        )
        
        print("✅ Configuration created successfully!")
        print(f"   Currency: {config.currency}")
        print(f"   Features: {config.features}")
        print(f"   Samples: {config.samples}")
        print(f"   NBins: {config.nbins}")
        print(f"   Lookback Rows: {config.lookback_rows}")
        print(f"   Lookforward Input: {config.lookforward_input}")
        
        # Create output directory
        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        
        # Generate sample output file
        config_data = {
            "currency": config.currency,
            "features": config.features,
            "samples": config.samples,
            "nbins": config.nbins,
            "lookback_rows": config.lookback_rows,
            "lookforward_input": config.lookforward_input,
            "timestamp": time.time(),
            "demo_type": "simple_api"
        }
        
        output_file = output_dir / "simple_api_config.json"
        with open(output_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"📁 Sample output saved to: {output_file}")
        
        print("\n🎯 Key Features Demonstrated:")
        print("   ✅ Configuration creation with currency-specific optimizations")
        print("   ✅ Automatic parameter calculation and validation")
        print("   ✅ JSON export of configuration data")
        print("   ✅ Output file generation for report system")
        
        print("\n✅ Simple API Demo completed successfully!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure the represent package is installed and accessible")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()