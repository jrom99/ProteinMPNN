import logging


def main():
    """Runs data processing scripts to turn raw data from into
    cleaned data ready to be analyzed.
    """
    # logger = logging.getLogger(__name__)


if __name__ == "__main__":
    log_fmt = "{asctime} - {name} - {levelname} - {message}"
    logging.basicConfig(level=logging.INFO, format=log_fmt, style="{")

    main()
