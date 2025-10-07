"""Legacy symbol optimization runner (deprecated)."""

import sys


def main() -> None:
    print(
        "Symbol optimization workflows have been removed from represent. "
        "The previous implementation depended on the tstrends library, which is no longer bundled."
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
