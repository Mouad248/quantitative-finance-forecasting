import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from portopt.data import download_prices
    print("✓ portopt.data import successful")
except ImportError as e:
    print(f"✗ portopt.data import failed: {e}")

try:
    from portopt.opt import max_sharpe
    print("✓ portopt.opt import successful")
except ImportError as e:
    print(f"✗ portopt.opt import failed: {e}")

try:
    from portopt.plot import plot_efficient_frontier
    print("✓ portopt.plot import successful")
except ImportError as e:
    print(f"✗ portopt.plot import failed: {e}")

