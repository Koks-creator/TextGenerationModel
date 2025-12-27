import sys
from pathlib import Path
import os
from logging import Logger
sys.path.append(str(Path(__file__).resolve().parent.parent))
from fastapi import FastAPI

from config import Config
from custom_logger import CustomLogger
from text_generation import TextGenerating


def setup_logging() -> Logger:
    """Configure logging for the api"""
    log_dir = os.path.dirname(Config.API_LOG_FILE)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = CustomLogger(
        logger_name="middleware_logger",
        logger_log_level=Config.CLI_LOG_LEVEL,
        file_handler_log_level=Config.FILE_LOG_LEVEL,
        log_file_name=Config.API_LOG_FILE
    ).create_logger()

    return logger

logger = setup_logging()

text_gen = TextGenerating(
        models_folder=Config.MODELS_FOLDER
    )
models = {}
for model_name in text_gen.available_models:
    try:
        models[model_name] = text_gen.load_model(model_name=model_name)
    except Exception as e:
        models[model_name] = None
        logger.error(f"Error in initing model: {model_name=}", exc_info=True)

logger.info("Starting api...")

app = FastAPI(title="TextGenerationApi")

from api import routes