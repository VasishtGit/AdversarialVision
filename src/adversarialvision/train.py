"""Train module proxy for adversarialvision."""

import sys
from pathlib import Path

# Add project root to path if needed
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from train import main

if __name__ == "__main__":
    main()
