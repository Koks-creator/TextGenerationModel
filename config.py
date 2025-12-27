from pathlib import Path
import json
from typing import Union
import os
import logging


class Config:
    # Overall
    ROOT_PATH: Path = Path(__file__).resolve().parent

    MODELS_FOLDER: Union[str, Path] = fr"{ROOT_PATH}/models"

    # MODEL
    MODEL_MIN_GEN_LEN: int = 10
    MODEL_MAX_GEN_LEN: int = 1000
    MODEL_MIN_TEMPERATURE: float = 0.1
    MODEL_MAX_TEMPERATURE: float = 2.0

    # API
    API_PORT: int = 5000
    API_HOST: str = "127.0.0.1"
    API_LOG_FILE: str = f"{ROOT_PATH}/logs/api_logs.log"
    API_MAX_BULK_REQ: int = 5

    # WEB APP
    WEB_APP_PORT: int = 8000
    WEB_APP_HOST: str = "127.0.0.1"
    WEB_APP_DEBUG: bool = True
    WEB_APP_LOG_FILE: str = f"{ROOT_PATH}/logs/web_app.logs"
    WEB_APP_TEMP_UPLOADS_FOLDER = f"{ROOT_PATH}/ocr_webapp/static/temp_uploads"
    WEB_APP_FILES_LIFE_TIME: int = 300
    WEB_APP_USE_SSL: bool = False
    WEB_APP_SSL_FOLDER: str = f"{ROOT_PATH}/ocr_webapp/ssl_cert"
    WEB_APP_TESTING: bool = False

    # LOGGER
    UVICORN_LOG_CONFIG_PATH: Union[str, os.PathLike, Path] = f"{ROOT_PATH}/api/uvicorn_log_config.json"
    CLI_LOG_LEVEL: int = logging.DEBUG
    FILE_LOG_LEVEL: int = logging.DEBUG

    def get_uvicorn_logger(self) -> dict:
        with open(self.UVICORN_LOG_CONFIG_PATH) as f:
            log_config = json.load(f)
            log_config["handlers"]["file_handler"]["filename"] = f"{Config.ROOT_PATH}/logs/api_logs.log"
            return log_config