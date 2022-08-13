import argparse

import re


def ip_validator(arg_value, pat=re.compile(r"^((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)(\.(?!$)|$)){4}$")):
    if not pat.match(arg_value):
        raise argparse.ArgumentTypeError("Invalid IP address")
    return arg_value
