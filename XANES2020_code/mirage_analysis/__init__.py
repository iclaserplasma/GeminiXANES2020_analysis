def base_path():
    """Returns the configured base path for data storage. Raises RuntimeError
    if the path has not been configured yet.
    """

    if BASE_PATH is None:
        raise RuntimeError('data storage base path has not been configured')

    return BASE_PATH

from .pipeline import DataPipeline
from .loader import register_data_loader, ImageDataLoader, LundatronLoader

BASE_PATH = None

def configure(base_path):
    """Select the base path for data storage. This path should contain one
    folder per diagnostic.
    """

    global BASE_PATH
    BASE_PATH = base_path

# add an item for each diagnostic below...
register_data_loader('Espec_high', ImageDataLoader)
register_data_loader('Lundatron', LundatronLoader)
register_data_loader('Gematron', LundatronLoader)
