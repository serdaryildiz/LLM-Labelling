from multiprocessing import Process, Queue, Event

from utils.llama.model import LlamaModel


class LlamaProcess(Process):

    def __init__(self,
                 model: LlamaModel,
                 input_queue: Queue,
                 output_queue: Queue,
                 stop_event: Event
                 ):
        super().__init__()
        self.model = model
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        return

    def run(self):

        while not self.stop_event.is_set():

            sample = self.input_queue.get(timeout=60 * 60)

            sample_id = sample["id"]
            image = sample["image"]

            description = self.model(image)

            self.output_queue.put(
                {
                    "id": sample_id,
                    "description": description
                }
            )
