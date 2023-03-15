#!/usr/bin/env python3
import os
import argparse
import sys

try:
    from emlib import doctools
except ImportError:
    import textwrap
    msg = ("**WARNING**: Trying to update documentation, but the python present in the current environment"
           " does not have the needed packages (emlib). Documentation will not be"
           " updated")
    print()
    print("\n".join(textwrap.wrap(msg, width=72)))
    print()
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--destfolder", default="docs", help="Destination forlder for documentation")
    args = parser.parse_args()
    assert os.path.exists(args.destfolder)
