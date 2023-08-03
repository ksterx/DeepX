def check_duplicate_keys(*args: dict):
    """Check for duplicate keys in the given dictionaries.

    Args:
        *args (dict): Dictionaries to check for duplicate keys.

    Raises:
        ValueError: If duplicate keys are found.
    """
    keys = set()
    for d in args:
        if isinstance(d, list):
            for k in d.keys():
                if k in keys:
                    raise ValueError(f"Duplicate key found: {k}")
                keys.add(k)
