import argparse


def str2bool(v):
    """ Parse a boolean argument from command line.
    Copied from: https://stackoverflow.com/a/43357954
    
    Example:
        parser.add_argument("--nice", type=str2bool, nargs='?',
                            const=True, default=False,
                            help="Activate nice mode.")
        
        cmdline 1: "script --nice"
        cmdline 2: "script --nice <bool>"
    """
    if v is None:  # allowing not specified
        return None
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
