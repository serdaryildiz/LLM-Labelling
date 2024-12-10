from multiprocessing import Process, Queue, Event

from utils.llama.model import LlamaModel


class LlamaProcess(Process):

    def __init__(self,
                 model: LlamaModel,
                 input_queue: Queue,
                 output_queue: Queue,
                 stop_event: Event,
                 batch_size: int = 1
                 ):
        super().__init__()
        self.model = model
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.batch_size = batch_size
        return

    def run(self):

        while not self.stop_event.is_set():

            # get input queue
            batch_image = []
            batch_sample_id = []
            for _ in range(self.batch_size):
                try:
                    sample = self.input_queue.get(timeout=60 * 60)

                    sample_id = sample["id"]
                    image = sample["image"]

                    batch_image.append(image)
                    batch_sample_id.append(sample_id)

                except Exception as e:
                    break

            # inference
            descriptions = self.model(batch_image)

            # add output queue
            for description, sample_id in zip(descriptions, batch_sample_id):
                self.output_queue.put(
                    {
                        "id": sample_id,
                        "description": description
                    }
                )
