from . import config
from .logging_setup import setup_logging

setup_logging()
config.ensure_dirs()
