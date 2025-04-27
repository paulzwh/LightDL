import os
import sys
import logging

try:
    from termcolor import colored
    color_print = True
except:
    color_print = False


def create_logger(
    name: str = None,
    print_to_console: bool = True,
    output_path: os.PathLike | str = None,
    console_level: int = logging.DEBUG,
    file_level: int = logging.INFO
):
    """
    Args:
        name: for Logger
        print_to_console: whether to print to console
        output_path: directory to save log file. not save when None. set to the same path if no need to separate when distributed
        console_level: set level to print to console
        file_level: set level to output to file 
    """
    # create logger
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    file_fmt = "[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s"
    if color_print:
        console_fmt = colored("[%(asctime)s %(name)s]", "green") + colored("(%(filename)s %(lineno)d)", "yellow") + ": %(levelname)s %(message)s"
    else:
        console_fmt = file_fmt
    datefmt = "%Y-%m-%d %H:%M:%S"

    # create console handler
    if print_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(
            logging.Formatter(fmt=console_fmt, datefmt=datefmt)
        )
        logger.addHandler(console_handler)

    # create file handler
    if output_path is not None:
        file_handler = logging.FileHandler(output_path, mode='a')
        file_handler.setLevel(file_level)
        file_handler.setFormatter(
            logging.Formatter(fmt=file_fmt, datefmt=datefmt)
        )
        logger.addHandler(file_handler)
    
    return logger
