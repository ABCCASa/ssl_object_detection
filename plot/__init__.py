import os
import torch
from torchvision.utils import draw_bounding_boxes
import torchvision


def plot_model(model, dataset, device, classes, save_folder, threshold=None):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    model.eval()
    for index in range(len(dataset)):
        image, _ = dataset[index]
        with torch.no_grad():
            predictions = model([image.to(device)])
            pred = predictions[0]
        image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
        scores = pred["scores"]
        if threshold is None:
            threshold = torch.mean(scores)
        mask = scores > threshold
        selected_scores = scores[mask]
        selected_labels = pred["labels"][mask]
        selected_boxes = pred["boxes"][mask].long()
        selected_labels = [f"{classes[label]}: {score:.3f}" for label, score in zip(selected_labels, selected_scores)]
        output_image = draw_bounding_boxes(image, selected_boxes, selected_labels, colors="red")
        torchvision.transforms.ToPILImage()(output_image).save(f"{save_folder}/{index}.png")


def plot_dataset(dataset, save_folder, classes):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for index in range(len(dataset)):
        image, target = dataset[index]
        plot_data(image, target, classes, save_folder, f"{index}.png")


def plot_dataloader(dataloader, save_folder, classes):
    index = 0
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for images, targets in dataloader:
        for image, target in zip(images, targets):
            plot_data(image, target, classes, save_folder, f"{index}.png")
            index += 1

def plot_data(image, target, classes, save_folder, data_name):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    labels = target["labels"]
    boxes = target["boxes"].long()
    if "scores" in target.keys():
        labels = [f"{classes[label]}: {score:.3f}" for label, score in zip( labels, target["scores"])]
    else:
        labels = [classes[label.item()] for label in labels]

    output_image = draw_bounding_boxes(image, boxes, labels, colors="red")
    torchvision.transforms.ToPILImage()(output_image).save(f"{save_folder}/{data_name}")