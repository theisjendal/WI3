# Matrix factorization
## Pre-processing
- We tried to implement pre-processing, but we found that some movies had no ratings in the test set
- This makes it difficult to find the actual RMSE since we can't re-create the structure

## Learning rate
- We experimented with different learning rates and found the proposed learning rate of `0.001` to be a little too slow
in convergence. On the other hand, `0.01` produced better results
- We tried different learning rates because we found that especially with increasing latent dimension sizes, SGD would
often get stuck in local minima

## Latent dimension size
- We found that larger latent dimension sizes yielded better RMSE during training, but not necessarily during testing
- This is indicative that the larger latent dimension sizes causes the model to overfit to the training data
- For this relatively small dataset (compared to Netflix), smaller latent dimension sizes seem to be needed