# Improving methane plume detections in images with missing pixels using Fair ML

[Project page](https://ada-research.github.io/maclean-missing-pixels/)

This repository contains the code and supplementary material [coming soon] for the paper "Improving methane plume detections in images with missing pixels using Fair ML" by Julia Wąsala, Bram Maasakkers, Ilse Aben, Rochelle Schneider, Holger Hoos, and Mitra Baratchi. The paper was accepted at the [MACLEAN'25 workshop at ECML-PKDD](https://sites.google.com/view/maclean25).The paper proposes two approaches to deal with missing not at random pixels in satellite data: an imputation approach, and a resampling scheme that resamples the training data at training time to remove the association between the number of missing pixels in an image and the label (see paper for more details). 

# How to use the code
We provide our imputation and resampling strategies as plug-in modules, and removed most code specific to the task presented in the paper (multimodal methane plume classification). 

> [!TIP]  
> Imputation and resampling can be combined, but we show individual usage examples. 

## Imputation
The imputation module is implemented as a `torch` module which can be passed as a `transform` to `torch.utils.data.Dataset` in the same way you would apply normalisation or data augmentation.

Usage example:
```python
from imputation import ImputeNaN
from data import YourDatasetClass
from torchvision.transforms import v2 as transforms
from torch.data.utils import DataLoader

n_channels=7 # number of channels in your image, defaults to 7
imputation_strategy="median" # imput
transforms = transforms.Compose([
    ImputeNaN(strategy=imputation_strategy,n_channels=n_channels)
])

dataset=YourDatasetClass(transform=transforms)
loader=DataLoader(dataset,batch_size=64)
```

## Resampling
The resampling function `calc_sampling_weights` takes a `pandas` DataFrame with class labels and attribute to balance as input. The function returns a `pd.Series` with sampling weights, which can be used with  `torch.utils.data.WeightedrandomSampler` to resample the data during training. 

The resampling strategy is meant to address problems where sampling bias causes a spurious association between a metadata attribute (in our case, "valid_pixel" or the coverage of an image, inversely related to the number of missing pixels) and the label. It does so by binning the data based on the confounding attribute, and assigning weights to each image/training instance to ensure class balance in each bin. 

> [!IMPORTANT]  
> It is important that the training instances in the `DataFrame` used for calculating sample weights is in the same order as the training instances in your dataset. We excluded code that checks for this, because ours was highly specific to our dataset implementation. We suggest to implement a `annotation` member variable in your dataset, and pass that to the `calc_sampling_weights` function. 

Usage example:
```python
from resampling import calc_sampling_weights
from data import YourDatasetClass
from torchvision.transforms import v2 as transforms
from torch.data.utils import DataLoader

dataset=YourDatasetClass()
# the annotations file should have labels and a column with the values of the 
# attribute you want to bin the data on (default is 'valid_pixels')
sampling_weights=calc_sampling_weights(dataset.annotations)
sampler = torch.utils.data.WeightedRandomSampler(sampling_weights, len(sampling_weights))
loader=DataLoader(dataset,batch_size=64,sampler=sampler, shuffle=False)
```

# Supplementary material
We provide additional details on the composition and processing of our dataset in the `supplementary` folder. More supplementary information may be added later. 