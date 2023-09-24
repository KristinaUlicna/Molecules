# Molecules 🧪 🥼 🖥️

This repository trains a whole-graph classifier to predict a chemical property of a molecule. The molecules are encoded as [SMILES representations](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system) and get converted into a `torch_geometric` dataset using the `smiles.from_smiles` [method](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/smiles.html). 

In this repository, I'm predicting active compounds which have their pIC50 > 8.0.


## Brief repo overview 📙

1. This pipeline first reads the `csv` dataset with ChemBL molecules. 
1. Their smiles strings are then decoded into a `torch_geometric` graphical structure.
1. The molecules are then shuffled & split into training, validation and inference datasets.
1. The train & valid datasets are loaded into the `Dataloader` & used for mode training, whilst the `infer` dataset is kept separate & unseen during training.
1. The graph neural network, concretely a Graph Attention Network (**GAT**, by [Velickovic et al., 2018](https://arxiv.org/abs/1710.10903)), is used. A 3-layer GAT network is constructed with optional batch normalisation & dropout. 
1. As the task is to identify active compounds, the problem is framed as a classification problem & measures positive class accuracy.
1. The `y` (true) label of each compound is `1` if higher than the pre-defined cut off threshold, and `0` otherwise.
1. The training optimises `CrossEntropyLoss` criterion and measures accuracy at train and valid checkpoints after every epoch. Binary Cross Entropy loss function could also have been used, but this choice generalises to other, more complex problems, too. 
1. The dataset poses a class imbalance problem, which makes it difficult to train the model smoothly. To avoid skewing the performance to the inactive compound classification performance, loss weighting is implemented. 
    + The weight of the positive class is the inverse of the representation of the positive class in the training dataset.
    + This means, as there are ~600 active compounds compared to ~3000 inactive compounds, the weighting could be specified as `[1.0, 5.0]`. Empirically, it can be seen that the `[1.0, 2.5]` weighting works just fine. 
1. For training, `Adam` optimiser is implemented, together with `ExponentialLR` scheduler.
1. To evaluate the performance, a variety of metrics is calculated:
    + Accuracy, precision, recall & f1-score
    + Confusion matrix 
    + Receiver-operator curve
    + Precision-recall curve

*Note:* All configuration parameters can be tweaked in the [config](#Configuration), which can be directly modified in file `./results/config.json`

*Note:* All inference metrics and figures, including the trained classifier, save into `./results` subfolder. 


## Run & installation 🏃‍♀️

```sh
conda 
```

To run the pipeline, navigate into the `pipeline.ipynb` and run the [notebook](`./pipeline.ipynb`).


## Configuration 🛠️

Here is an example [config](`./results/config.json`) I used for my training:

```sh
{
    "problem_type": "classification",
    "batch_size": 64,
    "dropout_prob": 0.2,
    "num_features": 9,
    "num_attn_heads": 8,
    "embedding_size": 64,
    "use_batch_norm": False,
    "learning_rate": 0.001,
    "weight_decay": 1e-05,
    "scheduler_gamma": 0.99,
    "num_epochs": 500,
    "pos_class_weight": 2.5,
    "logging_frequency": 20
}
```

## Happy coding 💻

Cite this repository in your work if you found it helpful:

```
@misc{Ulicna2023,
  author = {Ulicna, Kristina},
  title = {Molecules},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/KristinaUlicna/Molecules}},
}
```
