from enum import Enum


class ClassifierType(Enum):
    Single_CosineLinearFeature = 1
    Separate_CosineLinearLayers = 2
    Single_Linear = 3
    Separate_LinearLayers = 4
    Single_Stochastic_Classifier = 5
    Separate_Stochastic_Classifiers = 6


class GenerationStrategy(Enum):
    GMM = 1
    GMM_with_generator_loss = 2
