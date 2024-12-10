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
            device_map="cuda:0",
        )
        self.model.tie_weights()

        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.input_text = self.processor.apply_chat_template(self.instruct, add_generation_prompt=True)

        return

    def __call__(self, batch) -> str:

        inputs = self.processor(
            batch,
            [self.input_text] * len(batch),
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)

        descriptions = self.model.generate(**inputs,
                                          max_new_tokens=200,
                                          do_sample=False,
                                          num_beams=1,
                                          num_return_sequences=1,
                                          top_p=None)

        descriptions = [self.processor.decode(d) for d in descriptions]
        return descriptions
