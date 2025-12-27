from dataclasses import dataclass
from pathlib import Path
from typing import Union
import pickle
import os
import numpy as np
import tensorflow as tf

from model_definition import CharRNN
from config import Config


@dataclass
class TextGenerating:
    models_folder: Union[Path, str]

    def __post_init__(self) -> None:
        self.available_models = os.listdir(self.models_folder)
    
    def load_model(self, model_name: str) -> CharRNN:
        model_folder = fr"{self.models_folder}/{model_name}"
        model_files = os.listdir(model_folder)

        artifacts = {
            "config": None,
            "weights": None
        }
        # da sie szybciej?
        for phrase in ["config", "weights"]:
            for file in model_files:
                if phrase in file.lower():
                    if phrase != "weights":
                        with open(fr"{model_folder}/{file}", "rb") as f:
                            artifacts[phrase] = pickle.load(f)
                    else:
                        artifacts[phrase] = np.load(fr"{model_folder}/{file}")
                    break
        
        config = artifacts["config"]
        model_new = CharRNN(**config)

        dummy_input = tf.zeros((1, 10), dtype=tf.int32)
        _ = model_new(dummy_input)
        loaded = artifacts["weights"]
        weights = [loaded[f'arr_{i}'] for i in range(len(loaded.files))]

        model_new.set_weights(weights)
        # print(artifacts)
        return model_new



if __name__ == "__main__":
    text_gen = TextGenerating(
        models_folder=Config.MODELS_FOLDER
    )
    model = text_gen.load_model("pride_and_prejudice")
    for temp in [.5]:
        print(f"\nTemperature = {temp}:")
        generated = model.generate(
            start_string="Are you ready",
            generation_length=100,
            temperature=temp
        )
        print(generated)
