import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor


class LlamaModel:

    def __init__(self,
                 model_path: str,
                 instruct: list,
                 ):
        self.model_path = model_path
        self.instruct = instruct

        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.input_text = self.processor.apply_chat_template(self.instruct, add_generation_prompt=True)

        return

    def __call__(self, image) -> str:
        inputs = self.processor(
            image,
            self.input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)

        description = self.model.generate(**inputs, max_new_tokens=200)
        description = self.processor.decode(description[0])
        return description
