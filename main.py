import argparse
from datetime import datetime

import glob
import io
import json
import multiprocessing
import os.path
from threading import Thread

import lmdb
import tqdm
from PIL import Image

from utils.llama.model import LlamaModel
from utils.llama.process import LlamaProcess


def get_processed_id_list(output_path):
    json_files = glob.glob(os.path.join(output_path, "*.json"))

    id_list = []
    for json_path in json_files:
        processed = json.load(open(json_path, "r"))
        id_list += processed.keys()

    return id_list


def main(opt):

    instruct = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text",
             "text": "There is one person in the image focus. If it is not a person body image crop, you can just say it is not."
                     " list all attributes this person appearance by gender, age, head, upper body, lower body, shoes and carrying."
                     " If you do not have and information, pass it."
                     "For example :  "
                     "*   Gender:  "
                     "*   Age:  "
                     "*   Head:  "
                     "*   Upper body:  "
                     "*   Lower body:  "
                     "*   Shoes:  "
                     "*   Carrying: "
                     ":"}
        ]
         }
    ]

    # create output dir if not exists
    os.makedirs(opt.output_path, exist_ok=True)

    # get processed ids
    processed_id_list = get_processed_id_list(opt.output_path)

    # model
    model = LlamaModel(
        model_path=opt.model_dir,
        instruct=instruct
    )

    # queues
    input_queue = multiprocessing.Manager().Queue(maxsize=opt.max_queue_size)
    output_queue = multiprocessing.Manager().Queue(maxsize=opt.max_queue_size)

    # event
    stop_event = multiprocessing.Event()

    # process list
    process_list = [LlamaProcess(
        model=model,
        input_queue=input_queue,
        output_queue=output_queue,
        stop_event=stop_event,
        batch_size=opt.batch_size
    ) for _ in range(opt.num_process)]

    # start process
    for p in process_list:
        p.start()

    def _producer():
        # Open the LMDB file
        env = lmdb.open(opt.dataset_lmdb, readonly=True, lock=False, readahead=False, meminit=False)

        with env.begin(write=False) as txn:
            cursor = txn.cursor()

            stats = txn.stat()
            total_keys = stats['entries']
            print(f"Total number of keys: {total_keys}")

            found = 0
            for key, value in tqdm.tqdm(cursor, total=total_keys):
                key = key.decode(encoding="utf-8")

                if found < len(processed_id_list) and key in processed_id_list:
                    found += 1
                    continue

                image = Image.open(io.BytesIO(value))

                input_queue.put(
                    {
                        "id": key,
                        "image": image
                    }
                )

    def _consumer():
        output = {}
        cnt = 0
        while not stop_event.is_set():

            sample = output_queue.get()

            key = sample["id"]
            description = sample["description"]

            output[key] = description

            cnt += 1
            if cnt % opt.part_size == 0:
                current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                json.dump(output,
                          open(os.path.join(opt.output_path, f"part_{current_time}.json"), "w"),
                          indent=1,
                          )

                output = {}

    # threads
    producer_thread = Thread(target=_producer)
    consumer_thread = Thread(target=_consumer)

    producer_thread.start()
    consumer_thread.start()

    producer_thread.join()
    consumer_thread.join()

    return


if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser("Llama dataset labelling script.")
    parser.add_argument("--model-dir", type=str,
                        default="./models--meta-llama--Llama-3.2-11B-Vision-Instruct/snapshots/cee5b78e6faed15d5f2e6d8a654fd5b247c0d5ca",
                        help="model's weight dir"
                        )

    parser.add_argument("--output-path", type=str, default="./output", help="output dir path")
    parser.add_argument("--dataset-lmdb", type=str, default="./data/LUPws/lmdb", help="output dir path")
    parser.add_argument("--part-size", type=int, default=1000, help="label part size")
    parser.add_argument("--max-queue-size", type=int, default=1, help="label part size")
    parser.add_argument("--num-process", type=int, default=1, help="number of llama process")
    parser.add_argument("--batch-size", type=int, default=1, help="llama model batch size")

    parser.add_argument("--show", action="store_true", help="show image")

    args = parser.parse_args()

    main(args)
