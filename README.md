# Pokémon Classifier
[Methodology and Project Detailed Explanations.pdf](https://github.com/user-attachments/files/18239978/Team.Project.Report.pdf)

A deep learning project that compares different CNN architectures (AlexNet, Inception, ResNet50, VGG) for Pokémon image classification.

## Project Overview

This project implements and evaluates multiple CNN architectures for classifying Pokémon images. The models tested include:
- AlexNet
- Inception (with transfer learning)
- ResNet50 
- VGG

### Key Results
- Inception with transfer learning: ~100% validation accuracy
- ResNet50: 98% validation accuracy
- AlexNet: 70% validation accuracy
- VGG: 60% validation accuracy

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd pokemon-classifier
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Dataset

The project uses the [Pokémon Classification Dataset](https://www.kaggle.com/datasets/lantian773030/pokemonclassification) from Kaggle. You'll need to:
1. Download the dataset
2. Place it in the `data` directory

## Project Structure

### Core Files
- `alexnet_ex.py`: Implements the AlexNet architecture for Pokémon classification. Features Conv2D layers with ReLU activation and max-pooling layers. Achieves ~70% validation accuracy.

- `inception.py`: Contains the Inception model implementation with transfer learning. Uses pre-trained weights and multi-scale feature recognition. Achieves highest accuracy (~100%) among all models.

- `resnet50.py`: Implements ResNet50 architecture with residual connections. Includes both training and evaluation code. Achieves 98% validation accuracy.

- 'test.py`: Separate testing script with random pokemon image. Handles custom image inputs and provides prediction outputs. Use this for testing the model with new Pokémon images.

- `vgg.py`: VGG model implementation focusing on deep convolutional layers. Includes data preprocessing and training pipeline. Achieves ~60% validation accuracy.

### Supporting Directories
- `data/`: Store the Pokémon Classification Dataset here
- `test_images/`: Directory for custom test images
- `models/`: Saved model checkpoints (created during training)
- `logs/`: Training logs and metrics (created during training)

## Usage

### Testing Different Models

1. For AlexNet:
```bash
python alexnet_ex.py
```

2. For Inception:
```bash
python inception.py
```

3. For ResNet50:
```bash
python resnet50.py
```

4. For VGG:
```bash
python vgg.py
```

### Testing with Custom Images

To test the ResNet50 model with random Pokémon images:
1. Place your test images in the `test_images` directory
2. Run:
```bash
python resnet50_test.py
```

## Model Performance Analysis

### Inception Model
- Best performing model
- Excels at multi-scale feature recognition
- Most effective for animated character classification
- Achieved nearly perfect validation accuracy

### ResNet50
- Strong second-best performer
- Effective at handling complex features
- Well-suited for detailed feature extraction
- Achieved 98% validation accuracy

### AlexNet & VGG
- Moderate to lower performance
- Less effective for animated character recognition
- Better suited for real-world image classification

## References

1. [Few-Shot Classification Repository](https://github.com/bochendong/few_shot_classification)
2. [Pokémon Classification Dataset](https://www.kaggle.com/datasets/lantian773030/pokemonclassification)

## Contributors
- 오성식 (2019315300)
- 박호진 (2020310574)
- 손채리 (2019311148)
- Azamfirei andrei (2024319807)
