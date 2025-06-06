import torch

class ImputeNaN(torch.nn.Module):
    """Class to Impute NaNs in tensor. Inputs:
    - strategy: str     the imputation strategy. Choice of:
        - zero: impute with 0
        - median: impute with per-channel median (sample-wise)
        - noise: impute with gaussian noise (mean = median, std = std)
        - sample: impute with pixel values sampled from the image
    - n_channels: number of channels in the tensor"""

    def __init__(self, strategy:str="zero",n_channels:int=7):
        super().__init__()
        STRATEGY_OPTIONS = ["zero", "median",
                        "sample", "noise"]
        assert strategy in STRATEGY_OPTIONS, "Strategy must be one of zero,median,sample,noise"
        self.strategy = strategy

    def forward(self,strategy:str, tensor: Tensor) -> Tensor:
        
        # assumes NaNs are in same places in each channel, bases NaN
        # masks off where the NaNs are in the first channel.
        # change this line if this is not the case in your data. 
        nan_mask = torch.isnan(tensor[0, :])

        filled_tensor=tensor.clone()
        if strategy=="zero":
            # apply mask to each channel in tensor
            filled_tensor[:,nan_mask] = 0
        elif strategy=="median":
            # yeah nanmedian doesn't exist in pytorch, need to do it ith numpy
            medians=torch.tensor(np.nanmedian(tensor, axis=(1,2)))
            filled_tensor[:,nan_mask] = medians.view(*medians.size(),1).expand_as(filled_tensor[:,nan_mask])
        elif strategy=="noise":
            stds = torch.tensor(np.nanstd(tensor, axis=(1, 2)))
            medians = torch.tensor(np.nanmedian(tensor , axis=(1, 2)))

            # define a normal distribution for each channel, and draw N=N_nans samples from each
            samples = [torch.normal(
                medians[i], stds[i], (nan_mask.sum(),), dtype=float) for i in range(tensor.shape[0])]
            
            # concatenate list of tensors into tensor, swap so it's channels first, convert to same
            # dtype as tensor. 
            filled_tensor[:, nan_mask] = torch.swapaxes(
                torch.stack(samples, dim=1), 0, 1).to(torch.float64)
        elif strategy=="sample":
            samples = [torch.Tensor(np.random.choice(tensor[i,~nan_mask],size=nan_mask.sum().item())) for i in range(tensor.shape[0])]

            filled_tensor[:, nan_mask] = torch.swapaxes(
                torch.stack(samples, dim=1), 0, 1).to(torch.float64)
        else:
            raise ValueError(
                "Strategy must be one of zero, median,sample,noise")
        return filled_tensor