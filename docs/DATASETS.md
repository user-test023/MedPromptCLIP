# Dataset Preparation for ModalCLIP
In this markdown file, you can download the needed datasets and prepare each dataset.
To ensure reproducibility and fair comparison for future work, we provide json files for all datasets(you can prepare youlr dataset following the josn files). The fixed splits are either from the original datasets (if available) or created by us.

**NOTE:** Modalclip does not use image samples during training. The following steps to configure datasets are mainly for evaluation purposes and fewshot experiment. 

We recommend putting all datasets under the same folder (say `DATA`) to ease management and following the instructions below to organize datasets to avoid modifying the source code. 
The file structure should look like:

```
DATA/
|–– f1000images/
|–– ODIR/
|–– ODIR_3x200/
|–– OCTDL/
|–– OCT_C8/
|–– SLID_E/
|–– SLO/
|–– FFA/
|--...
```
| Dataset Name            | Volume and Type of Data                                | Disease Categories/Labels                          | Source and Link                                                                                                                                                                      | Application Tasks                           |
|-------------------------| ----------------------------------------------------- | ------------------------------------------------- |--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| ------------------------------------------ |
| **f1000images**         | 1000 colored fundus images                             | 39 disease categories                             | [Kaggle - FundusImage1000](https://www.kaggle.com/datasets/linchundan/fundusimage1000 )                                                                                              | Ophthalmic disease classification and grading |
| **ODIR**                | Fundus images and diagnostic labels for 5000 patients  | 8 types of ophthalmic diseases                    | [Kaggle - ODIR](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k )                                                                                        | Multi-disease classification               |
| **ODIR_3x200**          | 600 fundus images (3 classes, 200 per class)          | Pathological myopia, cataract, normal             | [Kaggle - ODIR](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k )                                                                                        | Three - class classification task          |
| **OCTDL**               | 2000+ OCT images                                      | Multiple retinal conditions (grouped and labeled) | [Kaggle - OCTDL](https://www.kaggle.com/datasets/shakilrana/octdl-retinal-oct-images-dataset )                                                                                       | OCT image classification and early diagnosis |
| **OCT_C8**              | 24000 high-quality retinal OCT images                  | 8 types of retinal diseases                       | [Kaggle - OCT_C8](https://www.kaggle.com/datasets/obulisainaren/retinal-oct-c8 )                                                                                                     | Retinal disease classification             |
| **SLID_E**              | 2999 slit lamp images from 1563 patients              | 4 epiphora grades (normal, mild, moderate, severe)| [Figshare - SLID-E](https://figshare.com/articles/dataset/_i_SLID-E_Slit_Lamp_Image_Dataset_for_Epiphora_-_A_Benchmark_Resource_for_Automated_Tear_Overflow_Analysis_i_/26172919/1 ) | Epiphora grading and classification        |
| **SLO (Retina-SLO)**    | 7943 SLO images (4102 eyes, 2440 participants)       | ME, DR, glaucoma (including suspects)            | [SpringerLink - Retina-SLO](https://link.springer.com/chapter/10.1007/978-3-031-72086-4_5 )                                                                                          | Multi-disease detection and classification (excluding unclear images) |
| **FFA (FFA-Synthesis)** | 600 pairs of CFP-FFA image pairs                      | 4 ophthalmic diseases (using only FFA modality)  | [GitHub - FFA-Synthesis](https://github.com/whq-xxh/FFA-Synthesis )                                                                                                                  | Image synthesis, classification, medical analysis |
| **SkinCancer** | 29,364 2D pathology .jpg images                                                         | 16 categories related to skin conditions                                                                                 | [Heidelberg University - SkinCancer](https://heidata.uni-heidelberg.de/dataset.xhtml?persistentId=doi:10.11588/data/7QCR8S)                                                               | Skin pathology classification                      |
| **NCT-CRC-HE-100K** | 100,000 2D pathology H&E stained .tif images from 86 patients                             | 9 types of human colorectal cancer and healthy tissues                          | [NCT-CRC-HE-100K](https://www.kaggle.com/datasets/imrankhan77/nct-crc-he-100k)                                                                                    | Colorectal cancer tissue classification            |
| **LC25000** | 25,000 2D pathology .jpeg images (768x768 pixels each)                                   | 5 categories: Colon Adenocarcinoma, Benign Colon Tissue, Lung Adenocarcinoma, Lung Squamous Cell Carcinoma, Benign Lung Tissue (5000 images per class) | [LC2500 Lung and Colon Cancer Histopathological Images](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images?select=lung_colon_image_set)                                                  | Lung and Colon cancer histopathological image classification |
| **LungHist700** | 691 2D pathology JPG images                                                             | 7 categories: Adenocarcinoma, Squamous Cell Carcinoma, Normal Tissue (Lung)                                              | [LungHist700](https://figshare.com/articles/dataset/LungHist700_A_Dataset_of_Histological_Images_for_Deep_Learning_in_Pulmonary_Pathology/25459174?file=45206104)                     | Pulmonary pathology classification (Lung)          |
| **Breast Ultrasound Images Dataset** | 780 2D ultrasound PNG images (average size 500x500 pixels) from 600 female patients (25-75 years old, collected in 2018) | 3 categories: Normal, Benign, Malignant                                                                                  | [PaddlePaddle AI Studio - Breast Ultrasound Images Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)                              | Breast ultrasound image classification             |
| **MIAS Mammography** | 322 2D X-ray PNG images                                                                 | 7 categories related to breast conditions                                                                                | [MIAS Mammography](https://www.kaggle.com/datasets/kmader/mias-mammography)                                                                              | Breast X-ray image classification (Chest/Breast)   |
| **Chest_Xray PD Dataset** | 4575 2D X-ray jpg, png images                                                          | 3 categories related to lung conditions                                                                                  | [Chest X-ray PD Dataset](https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/ChestX-rayPDDataset.md)                                                                                 | Lung X-ray image classification (Chest)            |
| **Chest_CT** | 1000 2D CT JPG, PNG images                                                              | 4 categories related to lung conditions                                                                                  | [Chest CT-Scan images ](https://tianchi.aliyun.com/dataset/93929)                                                | Lung CT image classification (Chest)               |
| **The Nerthus Dataset** | 5525 2D Endoscopy jpg images                                                            | 4 categories related to intestinal conditions                                                                            | [GitHub - Awesome-Medical-Dataset/Nerthus](https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/Nerthus.md)                                                         | Intestine endoscopy image evaluation               |
| **Br35H** | 3000 2D MRI JPG images                                                                  | 2 categories: Brain Tumor presence (Yes/No)                                                                              | [Heywhale - Br35H](https://www.heywhale.com/mw/dataset/61d3e5682d30dc001701f728/file)                                  | Brain tumor detection (Brain)                      |
| **Head CT-hemorrhage** | 200 2D CT .png images                                                                   | 2 categories related to brain/head hemorrhage                                                                            | [kaggle- Head CT-hemorrhage](https://www.kaggle.com/datasets/felipekitamura/head-ct-hemorrhage)                                                                                  | Brain/Head CT image classification for hemorrhage  |
| **Knee Osteoarthritis Severity Grading Dataset** | 9786 2D X-Ray PNG images                                                          | 5 categories for Knee Osteoarthritis Severity                                                                            | [Mendeley Data - Knee Osteoarthritis Severity Grading](https://data.mendeley.com/datasets/56rmx5bjcr/1)                                                                                 | Knee X-Ray classification for Osteoarthritis Severity |


If you want to add new dataset, please prepare it as follows:

1.  Download the dataset and add it to `/DATA`.
2.  Split the dataset as 70% train, 15% test, 15% val and keep it as a json file. You can regard [datapre.py](../DATA/datapre.py) as a reference.
3.  Add a new `dataset.py` (rename according to your dataset) to `/datasets`.
4.  Add relative `.yaml` to `/configs` (one for datasets, others for trainers).




