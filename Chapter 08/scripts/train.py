from transformers import Trainer, TrainingArguments, AutoProcessor, BlipForConditionalGeneration
from datasets import load_from_disk
import logging
import sys
import argparse
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=str, default=5e-5)

    # Data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])

    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # load datasets
    train_dataset = load_from_disk(args.training_dir)

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")

    checkpoint = args.model_name
    processor = AutoProcessor.from_pretrained(checkpoint)

    def transforms(example_batch):
        images = [x for x in example_batch["image"]]
        captions = [x for x in example_batch["text"]]
        inputs = processor(images=images, text=captions, padding="max_length")
        #inputs = {k:v.squeeze() for k,v in inputs.items()}
        inputs.update({"labels": inputs["input_ids"]})
        return inputs


    train_dataset.set_transform(transforms)

    # download model from model hub
    model = BlipForConditionalGeneration.from_pretrained(args.model_name)
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        learning_rate=float(args.learning_rate),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=2,
        logging_steps=50,
        remove_unused_columns=False,
        push_to_hub=False,
        label_names=["labels"],
        logging_dir=f"{args.output_data_dir}/logs",
    )

    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )

    # train model
    trainer.train()

    # Saves the model to s3
    trainer.save_model(args.model_dir)
