# Increase 0.001  --> lower INCREASE the EPOCH and 
# lower the LEARNING-RATE + BATCH NORMALIZATION should be as small as you can!

# patience (int) – Number of epochs with no improvement after which learning 
# rate will be reduced. For example, if patience = 2, then we will ignore the 
# first 2 epochs with no improvement, and will only decrease the LR after the 
# 3rd epoch if the loss still hasn’t improved then. Default: 10.

# Plot LR changes

# Larger Batches better generalize!
# Smaller batchsize should help (64/32)

# The lack of generalization ability is due to the fact that large-batch methods 
# tend to converge to sharp minimizers of the training function. These minimizers 
# are characterized by large positive eigenvalues in ∇2f(x)
# and tend to generalize less well. In contrast, small-batch methods converge to 
# flat minimizers characterized by small positive eigenvalues of ∇2f(x).
# We have observed that the loss function landscape of deep neural networks 
# is such that large-batch methods are almost invariably at

# https://pytorch.org/docs/stable/tensorboard.html
