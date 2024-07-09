import pd


def pdprint(value):
    """Print a Pd Data type to the Pd console"""

    prefix = pd.get_object_args()
    prefix = " ".join(prefix)
    if len(prefix) > 0:
        pd.print(str(prefix) + " " + str(value), show_prefix=False)
    else:
        pd.print(str(value))
