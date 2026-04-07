## Purpose
This project aims to reproduce the core ideas from UAP_VLP GitHub repository on a simplified use case and reduced scale. The goal is to make the implementation easier to understand, run, and experiment with, while preserving the key concepts behind universal adversarial perturbations for vision-language models.

## System Requirements
- **Operating System:** macOS  
- **Hardware:** Apple Silicon (M4 Pro, 2024)

## Dataset Setup
Download the Flickr30k dataset and place the images in the following directory: `./flickr30k/Images`
Ensure the dataset is properly extracted before running the project.

## Environment Setup
` conda create -n clipuap python=3.11 -y `
` conda activate clipuap ` 
` python -m pip install -U pip `
` pip inistall -r requirements.txt ` 

## Run
`python train_uap.py`
