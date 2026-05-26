"""Entry point so the example runs as `python -m company_research "question"`."""

import sys

from company_research.cli import main

if __name__ == "__main__":
    sys.exit(main())
