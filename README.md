# BMF-Med: Boundary-Aware Multiscale Fusion Network for Medical Image Segmentation

## Abstract

Precise medical image segmentation is crucial for clinical diagnosis and treatment. Blurred boundaries and multi-scale anatomical structures remain challenges for current methods. To address this, we propose the Boundary-Aware Multi-Scale Fusion Network for Medical Image Segmentation.We design a multi-scale edge enhancement feature extraction module that integrates learnable edge detection with contextual information to improve boundary perception.We introduce an edge-enhanced parallel hybrid Transformer module that models spatial and channel features concurrently.We include a feature reconstruction upsampling module to strengthen detail recovery.Experiments across five cross-modal datasets demonstrate that BMF-Med achieves an average Dice score of 84.56% (HD95 12.68 mm) on Synapse, 92.21% on ACDC, 79.53% on BUSI, and 92.39% on CVC-ClinicDB. These results significantly outperform state-of-the-art methods while maintaining a compact parameter count of 11.39 million, achieving a favorable balance between accuracy and memory efficiency.

---

## Requirements

We trained on NVIDIA RTX 4090 with Python 3.9.10 and PyTorch 1.12.1.

**Recommended library versions:**

- Python 3.9.10
- Torch 1.12.1+cu113
- torchvision 0.13.1+cu113
- numpy 1.21.5

You can install the same experimental environment using:

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

**Synapse Dataset**: download from [here](https://www.synapse.org/#!Synapse:syn3193805/wiki/).

The expected directory structure is as follows:

```
└── data
    └── Synapse
        ├── test_vol
        │   ├── case0001.npy.h5
        │   └── *.npy.h5
        └── train
            ├── case0005_slice000.npz
            └── *.npz
```

---

## Training & Testing

### Train on Synapse dataset

```bash
python train.py \
  --root_path ./data/Synapse/train_npz \
  --test_path ./data/Synapse/test_vol_h5 \
  --batch_size 20 \
  --eval_interval 20 \
  --max_epochs 400
```

| Argument           | Description                           |
| ------------------ | ------------------------------------- |
| `--root_path`      | Path to training data (npz format)    |
| `--test_path`      | Path to testing data (h5 format)      |
| `--eval_interval`  | Evaluate every N epochs               |

### Test on Synapse dataset

```bash
python test.py \
  --volume_path ./data/Synapse/ \
  --output_dir './model_out'
```

| Argument         | Description                         |
| ---------------- | ----------------------------------- |
| `--volume_path`  | Root directory of the test data     |
| `--output_dir`   | Directory where your model weights are stored |

---

## Results

**Performance comparison on Synapse Multi-Organ Segmentation dataset**
<img width="906" height="445" alt="image" src="https://github.com/user-attachments/assets/d61c0b08-41ca-45a3-ab9c-37f50e22e937" />



---

