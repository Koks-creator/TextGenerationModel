import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from typing import List, Dict
from pydantic import BaseModel, Field
import asyncio
from fastapi import HTTPException, status

from api import app, logger, Config, models


class BaseTextGenBody(BaseModel):
    model_name: str
    start_phrase: str
    generation_length: int = Field(None, ge=Config.MODEL_MIN_GEN_LEN, le=Config.MODEL_MAX_GEN_LEN)
    temperature: float = Field(None, ge=Config.MODEL_MIN_TEMPERATURE, le=Config.MODEL_MAX_TEMPERATURE)

class BaseTextGenRes(BaseModel):
    model_name: str
    res_text: str
    generation_length: int = Field(None, ge=Config.MODEL_MIN_GEN_LEN, le=Config.MODEL_MAX_GEN_LEN)
    temperature: float = Field(None, ge=Config.MODEL_MIN_TEMPERATURE, le=Config.MODEL_MAX_TEMPERATURE)

class BaseTextGenAdvBody(BaseModel):
    inputs: List[BaseTextGenBody]
    ignore_errors: bool = False

class BaseTextGenAdvRes(BaseModel):
    results: List[BaseTextGenRes]
    ignore_errors: bool = False

class HealthResponse(BaseModel):
    status: Dict[str, Dict[str, str]]


@app.get("/")
async def alive():
    return "Hello, I'm alive :) https://www.youtube.com/watch?v=9DeG5WQClUI"

@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK, tags=["Health"])
async def health_check():
    try:
        models_status = {}
        for model_name, instance in models.items():
            if instance:
                models_status[model_name] = {"status": "Loaded"}
            else:
                models_status[model_name] = {"status": "Failed to load"}
        return HealthResponse(status=models_status)
    except Exception as e:
        logger.error(f"health_check unhandled error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"health_check unhandled error: {e}"
        )
    
@app.post("/gen_text", response_model=BaseTextGenRes)
async def gen_text(body: BaseTextGenBody):
    logger.debug(f"gen_text requests with body: {body}")
    try:
        model = models.get(body.model_name)
        if not model:
            raise HTTPException(status_code=500, detail=f"Model: {body.model_name} not found or not initialized - check /health")
        res_text = await asyncio.to_thread(
            model.generate,
            body.start_phrase,
            body.generation_length,
            body.temperature
        )

        return BaseTextGenRes(
            model_name=body.model_name,
            res_text=res_text,
            generation_length=body.generation_length,
            temperature=body.temperature
        )
    except Exception as e:
        logger.error(f"gen_text unhandled error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"gen_text unhandled error: {e}"
        )

@app.post("/gen_text_bulk", response_model=BaseTextGenAdvRes)
async def gen_text_bulk(bodies: BaseTextGenAdvBody):
    logger.debug(f"gen_text_bulk requests with bodies: {bodies}")
    try:
        if len(bodies.inputs) > Config.API_MAX_BULK_REQ:
            raise HTTPException(
                status_code=500,
                detail=f"Too much subrequests - limit is: {Config.API_MAX_BULK_REQ}"
            )
        list_of_results = []

        for body in bodies.inputs:
            model = models.get(body.model_name)
            if model:
                res_text = await asyncio.to_thread(
                    model.generate,
                    body.start_phrase,
                    body.generation_length,
                    body.temperature
                )

                list_of_results.append(
                    BaseTextGenRes(
                        model_name=body.model_name,
                        res_text=res_text,
                        generation_length=body.generation_length,
                        temperature=body.temperature
                    )
                )
            elif not bodies.ignore_errors:
                raise HTTPException(
                    status_code=500,
                    detail=f"Model: {body.model_name} not found or not initialized - check /health"
                )
        return BaseTextGenAdvRes(
            results=list_of_results,
            ignore_errors=bodies.ignore_errors
        )
    except Exception as e:
        logger.error(f"gen_text_bulk unhandled error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"gen_text_bulk unhandled error: {e}"
        )