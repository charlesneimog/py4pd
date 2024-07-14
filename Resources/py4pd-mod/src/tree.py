import pd


def extract_numbers(lst):
    """ Extract numbers from a nested list of lists."""
    result = []
    for item in lst:
        if isinstance(item, int):
            result.append(item)
        elif isinstance(item, list):
            result.append(item[0])
    return result




