# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import BitsAndBytesConfig


def load_model():

    model_name = "gpt2"

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)


    prompt = "Say hello like a pirate!"

    # Prepare the input and generate a response
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=50, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("Response:", response)
    print("Model imported successfully!")


if __name__ == "__main__":
    print("Importing model...")
    load_model()
