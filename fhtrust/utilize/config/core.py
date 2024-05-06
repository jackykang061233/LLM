from typing import List
from pydantic import BaseModel
from strictyaml import load

class NonQuantizedConfig(BaseModel):
    name: str

class QuantizedConfig(BaseModel):
    name: str
    modelfile: str
    modeltype: str
    tokenizer: str
    
class ModelConfig(BaseModel):
    non_quantized_model: NonQuantizedConfig
    quantized_model: QuantizedConfig
    quantized: bool
    
class Config(BaseModel):
    model: ModelConfig
    final_prompt: str
    
def get_config() -> Config:
    with open('config/config.yaml', "r") as conf_file:
        parsed_config = load(conf_file.read())

    config = Config(
        **parsed_config.data,
    )

    return config

config = get_config()