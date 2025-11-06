<div align="center">
<h1> Explainable Fake Image Detection Using Large Multimodal Models </h1>
<h2> Capstone Project — CS56-1 | The University of Sydney, Semester 2, 2025 </h2>
</div>

---

## 👥 Team Members
Yuyang Chen  
Harry Cao  
Jiaze Li  
Yuxuan Ke  
Zhihcheng Gao  
Haodi Qi  

---

## 🧭 Project Overview

This project — *Explainable Fake Image Detection Using Large Multimodal Models* — aims to build an interpretable and reliable system for detecting and localizing fake or manipulated images.  

It leverages the capabilities of **large multimodal models (LMMs)** to perform detection, localization, and explanation generation, making the decision process more transparent and understandable to humans.

Our implementation is built upon the open-source **FakeShield** framework and extended for educational and research purposes as part of the Capstone program.  

The system integrates multiple components for **forgery detection, multimodal reasoning, and visual explanation** generation.

---

## 🏗️ System Components

- **Domain Tag-guided Explainable Forgery Detection Module (DTE-FDM):**  
  Detects whether an image is authentic or manipulated while generating preliminary textual reasoning.

- **Multimodal Forgery Localization Module (MFLM):**  
  Identifies and highlights tampered regions at the pixel level.

- **Explanation Generator:**  
  Produces clear, human-readable explanations aligning visual and textual evidence.

---



## 🛠️ Requirements and Installation

### Installation via Pip

1. Ensure your environment meets the following requirements:
    - Python == 3.9
    - Pytorch == 1.13.0
    - CUDA Version == 11.6

2. Clone the repository:
    ```bash
    git clone https://github.com/YuyangChen-622/Capstone5703-Group56-1
    cd FakeShield
    ```
3. Install dependencies:
    ```bash
    apt update && apt install git
    pip install -r requirements.txt

    ## Install MMCV
    git clone https://github.com/open-mmlab/mmcv
    cd mmcv
    git checkout v1.4.7
    MMCV_WITH_OPS=1 pip install -e .
    ```
4. Install DTE-FDM:
    ```bash
    cd ../DTE-FDM
    pip install -e .
    pip install -e ".[train]"
    pip install flash-attn --no-build-isolation
    ```


## 🤖 Prepare Model

1. **Download FakeShield weights from Hugging Face**
   
   The model weights consist of three parts: `DTE-FDM`, `MFLM`, and `DTG`. For convenience, we have packaged them together and uploaded them to the [Hugging Face repository](https://huggingface.co/zhipeixu/fakeshield-v1-22b/tree/main).

   We recommend using `huggingface_hub` to download the weights:
   ```bash
   pip install huggingface_hub
   huggingface-cli download --resume-download zhipeixu/fakeshield-v1-22b --local-dir weight/
   ```

2. **Download pretrained SAM weight**
   
   In MFLM, we will use the SAM pre-training weights. You can use `wget` to download the `sam_vit_h_4b8939.pth` model:
   ```bash
   wget https://huggingface.co/ybelkada/segment-anything/resolve/main/checkpoints/sam_vit_h_4b8939.pth -P weight/
   ```

3. **Ensure the weights are placed correctly**
   
   Organize your `weight/` folder as follows:
   ```
    FakeShield/
    ├── weight/
    │   ├── fakeshield-v1-22b/
    │   │   ├── DTE-FDM/
    │   │   ├── MFLM/
    │   │   ├── DTG.pth
    │   ├── sam_vit_h_4b8939.pth
   ```

## 🚀 Quick Start

### CLI Demo

You can quickly run the demo script by executing:

```bash
bash scripts/cli_demo.sh
```

The `cli_demo.sh` script allows customization through the following environment variables:
- `WEIGHT_PATH`: Path to the FakeShield weight directory (default: `./weight/fakeshield-v1-22b`)
- `IMAGE_PATH`: Path to the input image (default: `./playground/image/Sp_D_CRN_A_ani0043_ani0041_0373.jpg`)
- `DTE_FDM_OUTPUT`: Path for saving the DTE-FDM output (default: `./playground/DTE-FDM_output.jsonl`)
- `MFLM_OUTPUT`: Path for saving the MFLM output (default: `./playground/DTE-FDM_output.jsonl`)

Modify these variables to suit different use cases.

## 🏋️‍♂️ Train

### Data Preparation



Download them from the above links and organize them as follows:

```bash
dataset/
├── photoshop/                # PhotoShop Manipulation Dataset
│   ├── CASIAv2_Tp/           # CASIAv2 Tampered Images
│   │   ├── image/
│   │   └── mask/
│   ├── CASIAv2_Au/           # CASIAv2 Authentic Images
│   │   └── image/
│   ├── FR_Tp/                # Fantastic Reality Tampered Images
│   │   ├── image/
│   │   └── mask/
│   ├── FR_Au/                # Fantastic Reality Authentic Images
│   │   └── image/
│   ├── CASIAv1+_Tp/          # CASIAv1+ Tampered Images
│   │   ├── image/
│   │   └── mask/
│   ├── CASIAv1+_Au/          # CASIAv1+ Authentic Images
│   │   └── image/
│   ├── IMD2020_Tp/           # IMD2020 Tampered Images
│   │   ├── image/
│   │   └── mask/
│   ├── IMD2020_Au/           # IMD2020 Authentic Images
│   │   └── image/
│   ├── Columbia/             # Columbia Dataset
│   │   ├── image/
│   │   └── mask/
│   ├── coverage/             # Coverage Dataset
│   │   ├── image/
│   │   └── mask/
│   ├── NIST16/               # NIST16 Dataset
│   │   ├── image/
│   │   └── mask/
│   ├── DSO/                  # DSO Dataset
│   │   ├── image/
│   │   └── mask/
│   └── Korus/                # Korus Dataset
│       ├── image/
│       └── mask/
│
├── deepfake/                 # DeepFake Manipulation Dataset
│   ├── FaceAPP_Train/        # FaceAPP Training Data
│   │   ├── image/
│   │   └── mask/
│   ├── FaceAPP_Val/          # FaceAPP Validation Data
│   │   ├── image/
│   │   └── mask/
│   ├── FFHQ_Train/           # FFHQ Training Data
│   │   └── image/
│   └── FFHQ_Val/             # FFHQ Validation Data
│       └── image/
│
├── aigc/                     # AIGC Editing Manipulation Dataset
│   ├── SD_inpaint_Train/     # Stable Diffusion Inpainting Training Data
│   │   ├── image/
│   │   └── mask/
│   ├── SD_inpaint_Val/       # Stable Diffusion Inpainting Validation Data
│   │   ├── image/
│   │   └── mask/
│   ├── COCO2017_Train/       # COCO2017 Training Data
│   │   └── image/
│   └── COCO2017_Val/         # COCO2017 Validation Data
│       └── image/
│
└── MMTD_Set/                 # Multi-Modal Tamper Description Dataset
    └── MMTD-Set-34k.json     # JSON Training File
```





### LoRA Finetune DTE-FDM

You can fine-tune DTE-FDM using LoRA with the following script:

```bash
bash ./scripts/DTE-FDM/finetune_lora.sh
```

The script allows customization through the following environment variables:
- `OUTPUT_DIR`: Directory for saving training output
- `DATA_PATH`: Path to the training dataset (JSON format)
- `WEIGHT_PATH`: Path to the pre-trained weights

Modify these variables as needed to adapt the training process to different datasets and setups.

### LoRA Finetune MFLM

You can fine-tune MFLM using LoRA with the following script:

```bash
bash ./scripts/MFLM/finetune_lora.sh
```

The script allows customization through the following environment variables:
- `OUTPUT_DIR`: Directory for saving training output
- `DATA_PATH`: Path to the training dataset
- `WEIGHT_PATH`: Path to the pre-trained weights
- `TRAIN_DATA_CHOICE`: Selecting the training dataset
- `VAL_DATA_CHOICE`: Selecting the validation dataset

Modify these variables as needed to adapt the training process to different datasets and setups.


## 🎯 Test

You can test FakeShield using the following script:

```bash
bash ./scripts/test.sh
```

The script allows customization through the following environment variables:

- `WEIGHT_PATH`: Path to the directory containing the FakeShield model weights.
- `QUESTION_PATH`: Path to the test dataset in JSONL format. This file can be generated using [`./playground/eval_jsonl.py`](https://github.com/zhipeixu/FakeShield/blob/main/playground/eval_jsonl.py).
- `DTE_FDM_OUTPUT`: Path for saving the output of the DTE-FDM model.
- `MFLM_OUTPUT`: Path for saving the output of the MFLM model.

Modify these variables as needed to adapt the evaluation process to different datasets and setups.

## 🙏 Acknowledgement

We are thankful to LLaVA, groundingLMM, and LISA for releasing their models and code as open-source contributions.
