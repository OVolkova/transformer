import logging

logger = logging.getLogger("transformers")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(pathname)s - %(funcName)s: %(lineno)d - %(message)s"
)
handler.setFormatter(formatter)

logger.addHandler(handler)
