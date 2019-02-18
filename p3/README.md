
### Required packages
This projects includes some packages not found in the standard bundle:
- Numpy
- Torch
- Torchvision
- Tdqm
- PIL
- Matplotlib

Other packages imported are generally included in the standard python interpreter 

### Folder Structure
```
p3
└── .git/
└── evaluation
│   └── all_models.py
│   └── all_weights.py
│   └── eval.py
│   └── eval_vote.py
│   └── models.py
│   └── probabilities.py
└── experiments
│   └── base_model
│       └── params.json
└── model
│   └── architecture.py
│   └── metrics.py
└── processing
│   └── loader.py
│   └── transforms.py
└── utils
│   └── classes.py
│   └── functions.py
└── README.md
└── run.pbs
└── train.py
```

### Instructions to train
Run the train.py with the full path to:
* `--data_dir` path to the folder containing the Cat and Dog folders.

* `--model` '1' or '2' for ResNet and VGG, respectively.

* `--model__dir` path to the params.json containing the model hyperparameters. To run a new experiment
copy the model in the base_model folder on experiments.

### Instructions to evaluate
To evaluate on a single model you can use the eval.py on the evaluation folder. The parameters are:
* `--image_dir` path to the folder containing test images.
* `--model_path` path to the folder containing the pth model saved during training. 
* `--image_dir` path to the folder where predictions are going to be saved.
To evaluate on an ensemble of models you can use the eval_vote.py on the evaluation folder. The parameters slightly
 change:
* `--image_dir` path to the folder containing test images.
* `--model_path` path to the .txt file containing the full path of the models to be used (one per line). 
* `--image_dir` path to the folder where predictions are going to be saved.
* `--weights_dir` path to the .txt file containing vote weight for each of the models. If not provided voting will be
uniform.

### Additional information
Contact ramon.emiliani@umontreal.ca