# Image classification using the Cats and Dogs Classification Dataset
A simple convolutional‐neural‑network (CNN) built with PyTorch that tells cats from dogs with a ~89% test accuracy


## Dataset
Dataset
- Name: Dog and Cat Classification Dataset
- Source: [kaggle dataset by Bhavik Jikadara](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset/data) 

Dataset should be downloaded seperatly and placed as follows:
   ```
   data/raw/Cat/*.jpg
   data/raw/Dog/*.jpg
   ```

## Model
A simple convolutional network (CNN) with small hyperparameter tuning and regularization

```text
Conv‑BN‑ReLU( 3 → 32, k=3) → MaxPool(2)
Conv‑BN‑ReLU(32 → 64, k=3) → MaxPool(2)
Conv‑BN‑ReLU(64 → 128, k=3) → MaxPool(2)
Conv‑BN‑ReLU(128 → 64, k=3) → MaxPool(2)
Conv‑BN‑ReLU(64 → 32, k=3) → MaxPool(2)
Flatten → FC(128) → Dropout(0.5) → FC(2)
```


## Requirements
- Python ≥ 3.10
- PyTorch ≥ 2.0 & torchvision
- numpy, scikit‑learn, matplotlib

Also see requirements.txt


## Results

| Metric                | Best value                    |
| --------------------- | ----------------------------- |
| **Test accuracy**     | **≈ 89 %**                    |
| **Peak val accuracy** | 90 %                          |
| **Train time**        | \~15 mins on NVIDIA RTX 3060TI|


## TODO
- Update README with images of performance
- More indepth hyperparameter tuning and regularization
- Transfer learning
- Pretrained weights