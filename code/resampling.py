

def calc_sampling_weights(df,attribute_col="valid_pixels"):
    """Calculate sampling weights to balance the class balance in each valid pixel/NaN bracket. The goal is to break the correlation between the label and the valid pixel fraction.

    inputs:
    - df: pd.DataFrame -- dataframe with a label and the column with the attribute you want to balance (valid/missing pixels in the paper). WARNING: This needs to be the same data as in your torch.utils.data.Dataset, in the same order. Only applied to train data
    in our paper. We didn't include our code for checking this, as it was specific to our Dataset implementation. 
    - attribute_col: string, the col name of the attribute you want to balance. needs to be numeric, as data will be binned on these values.
    
    returns: 
    - pd.Series with sampling weights.
    """

    bins = pd.cut(df[attribute_col],
                    bins=20, include_lowest=True)
    df["bin"] = bins
    df["bin_count"] = df.groupby("bin")[
        "label"].transform("count")
    df["num_pos_in_bin"] = df.groupby("bin")[
        "label"].transform("sum")
    df["class_weight"] = df.apply(
        lambda x: 1.0/x.num_pos_in_bin if x.label == 1 else 1.0/(x.bin_count-x.num_pos_in_bin), axis=1)

    return df["class_weight"] * \
        (df["bin_count"]/len(df))