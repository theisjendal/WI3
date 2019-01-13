# Matrix factorization
## Pre-processing
- We tried to implement pre-processing, but we found that some movies had no ratings in the test set
- This makes it difficult to find the actual RMSE since we can't re-create the structure

## Learning rate and momentum
- We experimented with different learning rates and found the proposed learning rate of `0.001` to be a little too slow
in convergence. We tried higher learning rates, but while a local minimum was found faster, it had trouble converging
- We tried to implement momentum, where we consider the previous weight update when updating a weight
(`w=momentum*m-lr*g`), where `m` is the previous weight update
   - This turned out to work really well!

## Latent dimension size
- We found that larger latent dimension sizes yielded better RMSE during training, but not necessarily during testing
- This is indicative that the larger latent dimension sizes causes the model to overfit to the training data
- For this relatively small dataset (compared to Netflix), smaller latent dimension sizes seem to be needed