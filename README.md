# ClassifierChainEnsembler
Ensembling class for classifier chains

skmultilearn implemented classifier chains but not classifier chain ensembles. scikitlearn has ensembles of classifier 
chains, but only in the dev version. So, I wrote my own ensembling class for classifier chains. It's built on top of skmultilearn.problem_transform ClassifierChain class that creates one chain in the same order as the order of the labels
presented to it.

Basic algorithm is to:
- randomly reorder the labels and build a ClassifierChain for that ordering.
- Repeat multiple times
- Use a bagged approach with equal weighting for each ClassifierChain to predict each label and to average the
probabilities.
