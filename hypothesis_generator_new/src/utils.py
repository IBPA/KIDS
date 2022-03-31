import pickle


def save_pkl(obj, save_to: str) -> None:
    """
    Pickle the given object.

    Args:
        obj: Object to pickle.
        save_to: Filepath to pickle the object to.
    """
    with open(save_to, 'wb') as fid:
        pickle.dump(obj, fid)


def load_pkl(load_from: str):
    """
    Load the pickled object.

    Args:
        save_to: Filepath to pickle the object to.

    Returns:
        Loaded object.
    """
    with open(load_from, 'rb') as fid:
        obj = pickle.load(fid)

    return obj
