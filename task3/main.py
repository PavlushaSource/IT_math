import sys

import util_lib as util

if __name__ == '__main__':
    try:
        util.main()
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
