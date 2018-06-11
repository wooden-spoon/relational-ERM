import argparse
import tempfile


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class MaybeTempDirectory:
    """ A utility class which combines the possibility of allocating
    a temporary directory or using a given directory.
    """
    def __init__(self, directory_name=None):
        """ Creates a context manager for a temporary or named directory.

        Parameters
        ----------
        directory_name: The name of the directory. If None, this class
            automatically manages a temporary directory. If not None,
            this is simply passed through as the `name` attribute of
            the returned object when entering the context.
        """
        self.name = directory_name
        self._tempdir = tempfile.TemporaryDirectory()

    def __enter__(self):
        if self.name is not None:
            return self.name
        else:
            return self._tempdir.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.name is None:
            return self._tempdir.__exit__(exc_type, exc_val, exc_tb)
