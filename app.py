import sys
import os

# Add root so 'accessguard' is a proper package (fixes relative imports)
root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)
# Add accessguard dir so bare imports like 'mfa_verification' resolve
sys.path.insert(0, os.path.join(root, "accessguard"))

from accessguard.main import main
main()
