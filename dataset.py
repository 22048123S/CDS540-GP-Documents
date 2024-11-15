import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import DatasetDict, load_dataset
import cv2
import numpy as np


def load_concatenated_datasets():
    # Load the dataset
    train_set = load_dataset("Multimodal-Fatima/StanfordCars_train")
    test_set = load_dataset("Multimodal-Fatima/StanfordCars_test")

    # def process_image(example):
    #     # Convert PIL image to a NumPy array in RGB format
    #     pil_image = example["image"]
    #     image = np.array(pil_image)

    #     # Ensure the image is in RGB format for cv2
    #     if image.ndim == 2:  # If grayscale, convert to RGB
    #         image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    #     elif image.shape[2] == 4:  # If RGBA, convert to RGB
    #         image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    #     # Center crop to a square
    #     h, w, _ = image.shape
    #     crop_size = min(h, w)
    #     start_x = (w - crop_size) // 2
    #     start_y = (h - crop_size) // 2
    #     image = image[start_y : start_y + crop_size, start_x : start_x + crop_size]

    #     # Resize to 224x224 using cv2
    #     image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    #     # Update the image in the example
    #     example["image"] = image
    #     return example

    # # Apply the function to process images
    # train_set = train_set.map(process_image)
    # test_set = test_set.map(process_image)

    # Organize into a DatasetDict
    dataset = DatasetDict({"train": train_set["train"], "test": test_set["test"]})

    return dataset


if __name__ == "__main__":
    dataset = load_concatenated_datasets()
    dataset.save_to_disk("./processed_dataset")
