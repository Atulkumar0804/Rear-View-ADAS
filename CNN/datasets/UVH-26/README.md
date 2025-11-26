---
pretty_name: UVH-26 (Urban Vision Hackathon Dataset)
license: cc-by-4.0
tags:
  - computer-vision
  - object-detection
  - traffic
  - vehicles
  - india
  - cctv
task_categories:
  - object-detection
task_ids:
  - vehicle-detection
language:
  - und
annotations_creators:
  - crowd-sourced
source_datasets: []
size_categories:
  - 10K<n<100K

dataset_info:
  features:
    image: 
      type: image
    objects:
      type: sequence
      feature:
        type: object
        properties:
          bbox:
            type: sequence
            feature:
              type: float32
          category:
            type: class_label
            names:
              - Hatchback
              - Sedan
              - SUV
              - MUV
              - Bus
              - Truck
              - Three-wheeler
              - Two-wheeler
              - LCV
              - Mini-bus
              - Tempo-traveller
              - Bicycle
              - Van
              - Other

configs:
- config_name: mv
  data_files:
  - split: train
    path: "UVH-26-Train/UVH-26-MV-Train.json"
  - split: val
    path: "UVH-26-Val/UVH-26-MV-Val.json"
    
---

<div align="center">

  <!-- Banner Image -->
  <img width="50%" src="https://cdn-uploads.huggingface.co/production/uploads/653a21df882e390615659caf/aoWWXP4iJjEWSFevCt5cl.png" alt="UVH-26 Banner">

  <div align="center">
    <a href="https://arxiv.org/abs/2511.02563" ><img src="https://cdn-uploads.huggingface.co/production/uploads/653a21df882e390615659caf/kkT7Yr_s2wmWNbJsx5-Aa.png" height="16" width="11.96" style="display: inline-block; vertical-align: middle; margin: 2px;"> <b style="display: inline-block;"> ArXiv </b></a>  |  
    <a href="https://huggingface.co/iisc-aim/UVH-26"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" height="16" width="16" style="display: inline-block; vertical-align: middle; margin: 2px;"><b style="display: inline-block;"> Models </b></a>
  </div>

  <!-- Performance Graphic -->
  <img width="95%" src="https://cdn-uploads.huggingface.co/production/uploads/653a21df882e390615659caf/gGuP07sJGejnJfqKywp4a.png" alt="Performance on UVH-26">

  <p><em>
    Models trained on UVH-26 deliver up to <b>31.5% higher mAP</b> than COCO-pretrained baselines,
    demonstrating significant gains in real-world performance for Indian traffic scenarios.
  </em></p>

</div>


# Dataset Card for UVH-26 (Urban Vision Hackathon Dataset)

## Dataset Summary
**UVH-26** is a large-scale, India-specific traffic-camera image dataset released by **AIM @ IISc** for research in intelligent transportation systems and vehicle detection.  
It contains **26,646** high-resolution (1080p) frames sampled from ≈ 2,800 Bengaluru *Safe City* CCTV cameras over a 4-week period.  
Images were annotated through a nationwide crowdsourced hackathon involving **565 college students**, producing **≈ 1.8 million bounding boxes** across **14 fine-grained vehicle classes** representative of Indian traffic conditions.

To capture different levels of annotation consensus, UVH-26 includes **two separate annotation sets**:
1. **`UVH-26-MV`** — final labels computed via *majority voting* across multiple annotators per image.  
2. **`UVH-26-ST`** — labels generated using the *STAPLE* algorithm (an Expectation–Maximization–based probabilistic consensus method) for higher reliability.

These versions share identical image data but differ in bounding box consensus logic.

## Attribution
More technical details about the dataset and models are available in our [Technical Report available on arXiv](https://arxiv.org/abs/2511.02563). 
If you use these datasets or models, kindly cite the following: 
**The Urban Vision Hackathon Dataset and Models: Towards Image Annotations and Accurate Vision Models for Indian Traffic, Preliminary Dataset Release, UVH-26-v1.0**, 
Akash Sharma, Chinmay Mhatre, Sankalp Gawali, Ruthvik Bokkasam, Brij Kishore, Vishwajeet Pattanaik, Tarun Rambha, Abdul R. Pinjari, Vijay Kovvali, Anirban Chakraborty, Punit Rathore, Raghu Krishnapuram and Yogesh Simmhan, 
*Technical Report, Indian Institute of Science*, [arXiv.2511.02563](https://arxiv.org/abs/2511.02563), Nov, 2025.

```bibtex
@techreport{sharma2025uvh26,
  title        = {Towards Image Annotations and Accurate Vision Models for Indian Traffic, Preliminary Dataset Release, UVH-26-v1.0},
  author       = {Akash Sharma and Chinmay Mhatre and Sankalp Gawali and Ruthvik Bokkasam and Brij Kishore and Vishwajeet Pattanaik and Tarun Rambha and Abdul R. Pinjari and Vijay Kovvali and Anirban Chakraborty and Punit Rathore and Raghu Krishnapuram and Yogesh Simmhan},
  institution  = {Indian Institute of Science},
  type         = {Technical Report},
  number       = {arXiv:2511.02563},
  year         = {2025},
  month        = {November},
  doi          = {10.48550/arXiv.2511.02563}
}
```

## Dataset Structure

The datasets released follow the folder structure described below.

### **1. UVH-26-Train/**
Contains **80% of the UVH-26 dataset** used for training.
* **`images/`** – Training images organized into subfolders (`000/`, `001/`, …) for convenience.
  * `images/000/*` – Actual training images (`1.png`, `2.png`, …). Each image filename is unique across the entire dataset.
  * `images/001/*`, etc. – Additional subfolders following the same structure.
* **`UVH-26-MV-Train.json`** – Majority Voting consensus annotations for training images in **COCO JSON format**.
* **`UVH-26-ST-Train.json`** – STAPLE consensus annotations for training images in **COCO JSON format**.

### **2. UVH-26-Val/**
Contains **20% of the UVH-26 dataset** used for validation.
* **`images/`** – Validation images organized into subfolders (`000/`, `001/`, …).
  * `images/000/*` – Actual validation images. All filenames are globally unique across both training and validation sets.
  * `images/001/*`, etc. – Additional subfolders following the same structure.
* **`UVH-26-MV-Val.json`** – Majority Voting consensus annotations for validation images in **COCO JSON format**.
* **`UVH-26-ST-Val.json`** – STAPLE consensus annotations for validation images in **COCO JSON format**.

## Annotation JSON Schema
Each annotation file follows the standard COCO structure:
- **`images`** — list of image metadata  
  `id`, `file_name`, `width`, `height`
- **`annotations`** — object instances  
  `id`, `image_id`, `category_id`, `bbox [x, y, width, height]`, `area`  
- **`categories`** — class taxonomy (IDs and names below)

### Annotation Pipeline
- **Source:** frames captured between 06:00 – 18:00 IST during February 2025  
- **Pre-annotation:** generated using a fine-tuned **RT-DETR v2-X** model trained on ≈ 3 k expert-labeled images  
- **Crowdsourcing:** > 550 student volunteers corrected or validated predictions through a gamified web interface with leaderboards  
- **Consensus:** both *majority voting* and *STAPLE* algorithms applied to derive final annotations

## Vehicle Classes
| ID | Class Name        | Description                                                                                                                                            |
| -- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1  | Hatchback         | Small passenger cars without a protruding rear boot (“dickey”).                                                                                        |
| 2  | Sedan             | Passenger cars with a low-slung design and a separate protruding rear boot (“dickey”).                                                                 |
| 3  | SUV               | Car-like vehicles with high ground clearance, a sturdy body, and no protruding boot.                                                                   |
| 4  | MUV               | Large vehicles with three seating rows, combining passenger and cargo functionality.                                                                   |
| 5  | Bus               | Large passenger vehicles used for public or private transport, including office shuttles and intercity buses.                                          |
| 6  | Truck             | Heavy goods carriers with a front cabin and a rear cargo compartment.                                                                                  |
| 7  | Three-wheeler     | Compact vehicles with one front wheel and two rear wheels, featuring a covered passenger cabin.                                                        |
| 8  | Two-wheeler       | Motorbikes and scooters for single or double riders. Bounding boxes include both vehicle and rider.                                                    |
| 9  | LCV               | Lightweight goods carriers used for short- to medium-distance transport.                                                                               |
| 10 | Mini-bus          | Shorter, compact buses with fewer seats; larger than a Tempo Traveller, often featuring a flat front.                                                  |
| 11 | Tempo-traveller   | Medium-sized passenger vans with tall roofs and side windows; larger than vans but smaller than minibuses, with a protruding front.                    |
| 12 | Bicycle           | Non-motorized, manually pedalled vehicles including geared, non-geared, women’s, and children’s cycles. Bounding boxes include both vehicle and rider. |
| 13 | Van               | Medium-sized vehicles for transporting goods or people, typically with a flat front and sliding side doors; smaller than Tempo Travellers.             |
| 14 | Other             | Vehicles not covered in other classes, including agricultural, specialized, or unconventional designs.                                                  |

## Collection and Processing
- **Source:** ≈ 2,800 *Safe City* surveillance cameras operated by Bengaluru Police  
- **Coverage:** both junction and mid-block perspectives across multiple city zones  
- **Selection:** images with high vehicle density, occlusion, and diverse viewpoints prioritized  

## Intended Uses
- Building accurate, lightweight, edge-deployed perception systems for **Intelligent Transportation Systems (ITS)**  
- Training and benchmarking vehicle detection models

## License
- **Dataset:** [CC BY 4.0 International](https://creativecommons.org/licenses/by/4.0/)  
- **Pre-trained Models:** [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

## Acknowledgements
We thank the **Bengaluru Traffic Police (BTP)** and the **Bengaluru Police** for providing access to the *Safe City* camera data from which the image datasets used for this release were derived.  
We thank **Capital One** for sponsoring the prizes for the **Urban Vision Hackathon** competition.  
We thank **IISc’s AI and Robotics Technology Park (ARTPARK)** and the **Centre for Infrastructure, Sustainable Transportation and Urban Planning (CiSTUP)** for funding the annotation and model-training efforts, and the **Kotak IISc AI-ML Centre (KIAC)** for providing the GPU resources required to train the models.  
We acknowledge the outreach support provided by the **ACM India Council** and the **IEEE India Council** to encourage chapter volunteers to participate in the hackathon.  
Lastly, we thank the **AI Centers of Excellence (AI COE)** initiative of the **Ministry of Education**, their **Apex Committee members**, and the **AIRAWAT Research Foundation**, whose support helped catalyze these efforts.  

Created by the **AI for Integrated Mobility (AIM)** group at the **Indian Institute of Science (IISc)**, Bengaluru.
