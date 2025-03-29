import bentoml
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from bentoml.io import Text


@bentoml.service(
    resources={"gpu": 1} if torch.cuda.is_available() else {"cpu": 1},
    traffic={"timeout": 60},
)
class GPT2Service:
    def __init__(self):
        self.model_name = "gpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(
            self.device
        )

    @bentoml.api()
    def generate_response(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs, max_length=50, do_sample=True, temperature=0.7
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
