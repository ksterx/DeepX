def clean():
    """Cleans the MLFlow directory. Even if a run is 'deleted' in the UI, it is not actually deleted
    from the filesystem. This function deletes the 'deleted' runs from the filesystem.
    """
