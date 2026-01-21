import logging

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Creates and returns a logger object with the specified name and level.
    The logger outputs to the console with a standard format.

    Args:
        name (str): Name of the logger.
        level (int): Logging level (default: logging.INFO).

    Returns:
        logging.Logger: Configured logger object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    return logger