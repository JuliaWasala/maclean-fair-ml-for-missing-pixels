This document provides additional detail on the data processing that did not fit in the paper.  

## Dataset description. 

We compiled a methane plume detection dataset from TROPOMI methane observations using images validated by methane analysis experts over the course of approximately 3 years. The dataset consists of 90456 images of 32x32 pixels, with 7 channels: methane enhancement (concentration corrected for background), albedo, aerosol optical thickness, chi squared of the retrieval, quality assurance values, cloud product [1], and a land cover classification. 

The dataset is an extended version of the dataset compiled by Wąsala et al. [5] based on the work by Schuit et al. [1,2].  

The expert-validated data has rich annotations, which we have simplified to: 

-‘plume’: images containing methane plumes. 
- ‘artefact’: images that have plume-like features that are not real plumes. 
- ‘empty’: images that are clearly empty/show no plume-like features. 

We perform binary classification, grouping ‘artefact’ and ‘empty’ in the single class ‘not plume’ (resulting in a class balance of 56% plume). 

In general, we followed the data processing and filtering procedure from Schuit et al. [1], with a few updated steps: 

- We use a newer data version (v19 [4]). 
- We apply stricter filtering, discarding all images that have missing pixels where the methane channel does not.  

The following describes the steps in our data processing pipeline, from the operational Sentinel-5P TROPOMI methane level-2 data product to ML-ready training data. Additionally, we refer to results from preliminary data processing fine-tuning experiments. Outline: 

- Label sources 
- Data processing 
- Data partitioning 
- Normalisation 
- Imputation 
- Data augmentation 

##  Label sources 

The starting point of our dataset are the methane plume detection datasets created by Schuit et al. (and available on Zenodo). This dataset, described in Schuit et al., consists of about 4000 images, with approximately 32% having ‘plume’ labels. Since the creation of this dataset, the model proposed by Schuit et al. has been used to detect more plumes in the TROPOMI methane data product. These analyses resulted in additional labelled images. We combine multiple sets of labels (from independent analyses) to create a larger dataset with a better class balance. 

| Description | Year | Citation | Note |
| ----------- | ---- | -------- | ---- |
| SVC train set | 2020 | [1,2] |The dataset used to train the SVC in Schuit et al. Contains mostly plumes and artefacts. |
| CNN train set | 2020 | [1,2] | The dataset used to train the CNN in Schuit et al. Contains mostly clear plumes and clear empty images. |
| SVC detections | 2021 |[1,2] | Results from the application analysis in Schuit et al.  |
| Detections close to landfills | 2021 | [3] |ML-based detections of plumes close to landfills, part of the Dogniaux et al.'s analysis on landfill emissions. Overlaps with ‘SVC detections’ have been removed. We excluded detections from 2022 to keep as a separate validation set.  |
| CNN detections | 2023 | http://earth.sron.nl/methane-emissions | These are all detections from the first step of the pipeline from Schuit et al (so before filtering with CNN). These detections also contain images determined to be something else than a plume (and are therefore not accessible on the website) |

We combine the label sets as follows. Each image is assigned a unique identifier based on the orbit number and its index/location within the orbit. Each image is a 32x32 crop from a larger orbit file. We discard all duplicate images. Then, we filter based on the label and only keep the images classified as ‘plume’, ‘empty’, or ‘artefact’, and remove the rest (‘area source’ and ‘doubt’).  

## Data processing 

Since the creation of the dataset by Schuit et al, a new version of the TROPOMI level-2 methane data has been released, which has been processed with an improved retrieval algorithm that addresses previous issues with albedo. Therefore, we download images from the newer data version. Some older orbits from the start of the mission have not been reprocessed. Furthermore, we discard images that contain data processed using the backup cloud product, which is used when VIIRS-based cloud data is unavailable.  

We follow Schuit et al. and apply additional filtering rules to discard images with less than 20% valid pixels.. The filtering can have slightly different results due to the different data version.  

## Data partitioning 

As mentioned in the paper, we use a spatial blocking scheme to partition the data into training, validation, and testing partitions, avoiding spatial autocorrelation's confounding effect. We wanted our data partitioning scheme to have the following properties: 

- Addresses spatial autocorrelation 
- Ensures important regions with many known methane sources are represented in each partition 
- Ensures approximate class balance across partitions. 

Since some regions almost exclusively have scenes with ‘plume’ labels, while some have none at all, we chose to partition the data into smaller cells, as creating bigger blocks would make it more challenging to achieve class balance.  

![Label distribution](https://github.com/JuliaWasala/maclean-fair-ml-for-missing-pixels/blob/main/img0.png?raw=true)

We create a 3x3 degree grid over the map, assigning images to grid cells. Then, we randomly assign each grid cell to a partition. Before training any models, we have fine-tuned the random seeds for this random assignment only to achieve approximate class balance, without having access to any model performance to avoid data leakage.  

The resulting splits look like this: 

![Data partitioning scheme](https://github.com/JuliaWasala/maclean-fair-ml-for-missing-pixels/blob/main/img0.png?raw=true)

## Normalisation 

Satellite bands often follow gamma distributions; therefore, [some care may be needed in normalisation to avoid squashing values in a very narrow range](https://medium.com/sentinel-hub/how-to-normalize-satellite-images-for-deep-learning-d5b668c885af). 

We conducted preliminary experiments to find the optimal normalisation procedure (based on performance on the validation set). We always normalised the methane data layer with the normalisation scheme proposed by Schuit et al., and evaluated all combinations of the following strategies: 

- **Standardisation** (z-scaling) vs **Min-max** Normalisation (scaling between 0 and 1) 
- **Sample-wise vs feature-wise** normalisation 
- **Robust vs standard** normalisation:   
    - Using median and the interquartile range instead of mean and standard deviation for standardisation (calculated based on the training set only) 
    - Scaling between the 1st and 99th percentiles for min-max normalisation instead of using the actual min and max. 

We found that, in general, robust standardisation performed worse than standard normalisation, feature-wise performed better than sample-wise, and there was no large difference between normalisation and standardisation.  

We proceeded with standard feature-wise standardisation.  

## Imputation 

In addition to the imputation strategies proposed in the paper, we evaluated two additional, more aggressive strategies: 
- **Random imputation**: at each epoch, for each image, select a random imputation from zero, median, pixel-sample, or noise-augmented imputation. 
- **Channel-wise randomised imputation**: at each epoch, for each channel within each image, select a random imputation for each channel.

Both methods lead to highly unstable performance: from good/comparable with the other networks, to extremely bad. Likely, the non-determinism in these imputation strategies was too strong and harmed convergence during training.  

## Data augmentation 

We opted for simple data augmentation: flipping vertically and horizontally, and rotating with increments of 90 degrees. Because plumes can be relatively small, any augmentations that involve cropping or masking may accidentally mask out any plumes and lead to label noise (images that do not have any visible plume anymore will still have the plume label).  

Colour transformations such as saturation/solarise are not directly applicable as the channels are physically unrelated quantities (they do not form a single colour image together).  

# References 

[1] Schuit, B.J., Maasakkers, J.D., Bijl, P., Mahapatra, G., van den Berg, A.W., Pandey, S., Lorente, A., Borsdorff, T., Houweling, S., Varon, D.J., McKeever, J., Jervis, D., Girard, M., Irakulis-Loitxate, I., Gorroño, J., Guanter, L., Cusworth, D.H., Aben, I.: Automated detection and monitoring of methane super-emitters using satellite data. Atmospheric Chemistry and Physics 23(16), 9071–9098 (Sep 2023). https://doi.org/10.5194/acp-23-9071-2023 

[2] Schuit, B.J., Maasakkers, J.D., Bijl, P., Mahapatra, G., Van den Berg, A.W., Pandey, S., Lorente, A., Borsdorff, T., Houweling, S., Varon, D.J., McKeever, J., Jervis, D., Girard, M., Irakulis-Loitxate, I., Gorroño, J., Guanter, L., Cusworth, D.H., Aben, I.: Interactive map with TROPOMI and high-resolution scenes [Schuit et al. 2023: Automated detection and monitoring of methane super-emitters using satellite data] (Jun 2023). https://doi.org/10.5281/zenodo.8089889 

[3] Dogniaux, M., Maasakkers, J.D., Girard, M., Jervis, D., McKeever, J., Schuit, B.J., Sharma, S., Lopez-Noreña, A., Varon, D.J., Aben, I.: Satellite survey sheds new light on global solid waste methane emissions (Jul 2024). 

[4] Lorente, A., Borsdorff, T., Martinez-Velarte, M.C., Butz, A., Hasekamp, O.P., Wu, L., Landgraf, J.: Evaluation of the methane full-physics retrieval applied to TROPOMI ocean sun glint measurements. Atmospheric Measurement Techniques 15(22), 6585–6603 (2022). https://doi.org/10.5194/amt-15-6585-2022 

[5] Wąsala, J., Marselis, S., Arp, L., Hoos, H., Longépé, N., Baratchi, M.: AutoMergeNet: Reducing False Positives in Satellite Data Using Automated Image Data Fusion. Manuscript currently under review. (2025) 

 