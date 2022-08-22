
---

<div align="center">    
 
# Temporally Adjustable Longitudinal FAIR MRI Estimation / Synthesis for Multiple Sclerosis

<!-- [![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539) -->
<!-- [![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018) -->
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
</div>

## Abstract
Multiple Sclerosis (MS) is a chronic progressive neurological disease characterized by the development of lesions in the white matter of the brain. T2-fluid-attenuated inversion recovery (FLAIR) brain magnetic resonance imaging (MRI) provides superior visualization and characterization of MS lesions, relative to other MRI modalities. Longitudinal brain FLAIR MRI in MS, involving repetitively imaging a patient over time, provides helpful information for clinicians towards monitoring disease progression. Predicting future whole brain MRI examinations with variable time lag has only been attempted in limited applications, such as healthy aging and structural degeneration in Alzheimer’s Disease. In this article, we present novel modifications to deep learning architectures for MS FLAIR image synthesis / estimation, in order to support prediction of longitudinal images in a flexible continuous way. This is achieved with learned transposed convolutions, which supports modeling time as a spatially distributed array with variable temporal properties at different spatial locations. Thus, this approach can theoretically model spatially-specific time-dependent brain development, supporting the modeling of more rapid growth at appropriate physical locations, such as the site of an MS brain lesion. This approach also supports the clinician user to define how far into the future a predicted examination should target. Four distinct deep learning architectures have been developed. The ISBI2015 longitudinal MS dataset was used to validate and compare our proposed approaches. Results demonstrate that a modified ACGAN achieves the best performance and reduces variability in model accuracy.

## Model Architecture
<img src="figs\architecture.png" width = "1000" alt="quanlitative_result" align=center/>

## Requirements
First, install following dependencies
```bash
torch==1.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
scikit-image>=0.18.2
scikit-learn>=1.0.1
nibabel>=3.2.1
scipy>=1.7.0
monai>=0.8.0
pytorch-lightning==1.4.0
```

## Train
To run the experiments from our paper the following bash scripts specifies the commands to run: 
|              Network               | Job Script              |
| :--------------------------------: | ----------------------- |
|           Modified ACGAN           | ./Job_scripts/ACGAN.sh  |
| Discriminator-Induced Time 3D cGAN | ./Job_scripts/dt-GAN.sh |
|   Generator-Induced Time 3D cGAN   | ./Job_scripts/gt-GAN.sh |
|              3D rUNet              | ./Job_scripts/UNet.sh   |

You can also find the hparams for each models here. It would take around 4 ~ 6 hours to train each model parallelled on 4 A100 GPUs (40GB).

## Folder Structure
```bash
Temporally-Adjustable-Longitudinal-MRI-Synthesis/
├── data
│   └── MS - ISBI 2015 Longitudinal MS Lesion Segmentation Challenge dataset, each folder contains subjects with the same number of time-points
│       ├── exams-4
│       ├── exams-5
│       └── exams-6
├── Job_scripts - Job scripts to run different models
│   ├── ACGAN.sh
│   ├── dt-GAN.sh
│   ├── UNet.sh
│   └── gt-GAN.sh
├── LICENSE
├── project
│   ├── __init__.py
│   ├── lig_module - Pytorch Lightning modules
│   │   ├── data_model
│   │   │   └── data_model_synthesis.py
│   │   └── lig_model
│   │       ├── lig_model_ACGAN.py
│   │       ├── lig_model_GAN.py
│   │       └── lig_model_synthesis.py
│   ├── main.py - main file to start/resume training
│   ├── model - The architecture of the model
│   │   ├── create_structure_data.py
│   │   ├── GAN
│   │   │   ├── ACGAN_discriminator.py
│   │   │   └── discriminator.py
│   │   └── unet
│   │       ├── conv.py
│   │       ├── decoding.py
│   │       ├── encoding.py
│   │       ├── __init__.py
│   │       └── unet.py
│   └── utils - utils and visualization
│       ├── compute_std.py
│       ├── const.py
│       ├── transforms.py
│       ├── visualize_6_timepoint_sample.py
│       ├── visualize.py
│       └── visualize_training.py
└── README.md
```

## Results
<img src="figs\predicted_images.jpg" width = "1300" alt="quanlitative_result" align=center />
