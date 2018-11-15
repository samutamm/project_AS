
import torch

class SNIP:

    def __init__(self, params):
        self.W = params
        self.C = torch.zeros(params.shape)

    def prune(self, loss_function, training_X, training_y, K):
        """
        Algorithm 1 SNIP: Single-shot Network Pruning based on Connection Sensitivity
        from the paper with corresponding name. https://openreview.net/pdf/3b4408062d47079caf01147df0e4321eb792f507.pdf

        This function implements only lines 1-5 of the algorithm.

        :param loss_function: Loss function that is used later in learning to optimize given task.
        :param training_X: learning examples for calculating Connection Sensivity (line 3)
        :param training_y: labels for examples
        :param K: number of non-zero weights to be returned in output C
        :return: binary vector C where 1's are weights to be retained and 0's weights to drop
        """

        # TODO VarianceScalingInitialization
        # TODO sampling examples
        # TODO count Connection Sensivity
        # TODO sort s and take top-k
        return self.C
