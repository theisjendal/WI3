# Content-based recommendations
## Key ideas
- Each product is represented by all of its reviews merged into one review

## Choice of approach
- We chose to do item-based recommendations because we found that the number of unique users was much higher than that
of products. It is therefore more memory and time efficient to use item-based
- Item-based approaches offer great transparency. This means that a user can see why an item was recommended

## Drawbacks
- Using item based it means, that it only chooses new items similar to already seen items and therefore
relatively safe choices and not novel.
- If a user has only given items the same rating, then our system would give any unseen product that rating
- Being a memory-based model, it can easily adapt to new products and users, unlike model-basel approaches. On the other
hand, a model-based approach is much more efficient
- The model cannot recommend before ratings of a product occur
- As we use the most frequent used terms, making the matrix more dense we also risk that these terms,
are not representative 

## Extending to hybrid content-based system
- Combining the ratings of CF and CB might introduce some novel recommendations, that are not produced by
our current implementation
- The use of CF might also be good for newer items

## Conclusions
- Stemming and dimensionality reduction is important, we started with out reducing, and had an algorithm that ran
for a very long time. Reducing this decreased runtime significantly. 

## Examples from data
- We ran the algorithm on a user who had made some positive reviews on a microphone windscreen and guitar strings. These were the top-5 recommended items:
1. Bluecell Black 5 Pack Microphone Windscreen Foam Cover
2. Bluecell 5 Pack Blue/Green/Yellow/Hot Pink/Orange Handheld Stage Microphone Windscreen Foam Cover
3. Ernie Ball Earthwood Extra Light Phosphor Bronze Acoustic String Set
4. D'Addario EXL115W Nickel Wound Electric Guitar Strings, Medium/Blues-Jazz Rock, Wound 3rd, 11-49
5. Planet Waves Ergonomic Guitar Peg Winder

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
