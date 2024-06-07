import logging
import sys


def get_logger(verbose:str = True) -> None:
    """ GETS LOGGER """

    logformat = "%(asctime)s|%(levelname)s|%(message)s" # class and func name are inside message in decorator
    datefmt = "%d/%m %H:%M:%S"

    logging.basicConfig(filename="code.log", level=logging.INFO, filemode="w",
                        format=logformat, datefmt=datefmt)
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(logging.Formatter(fmt=logformat, datefmt=datefmt))
    logger = logging.getLogger("helical")
    if verbose == True:
        logger.addHandler(stream_handler)
    else:
        logger.addHandler(logging.NullHandler())

    return logger


def set_logging_level(level: str) -> None:
    """ SETS LOGGING LEVEL """

    levels = {
        'CRITICAL': logging.CRITICAL,
        'ERROR': logging.ERROR,
        'WARNING': logging.WARNING,
        'INFO': logging.INFO,
        'DEBUG': logging.DEBUG
    }

    if level.upper() in levels:
        logging.getLogger().setLevel(levels[level.upper()])
    else:
        raise ValueError(f"Invalid logging level: {level}")
    
    return


set_logging_level(level='DEBUG')