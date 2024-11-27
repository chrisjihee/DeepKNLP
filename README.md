# DeepKNLP
Transformer-based Korean Natural Language Processing

## Main Reference
![overview](images/overview.png?raw=true)
* ratsgo ratsnlp: https://github.com/ratsgo/ratsnlp
* ratsgo nlpbook: https://ratsgo.github.io/nlpbook/
* HF Transformers: https://github.com/huggingface/transformers
* Lightning Fabric: https://lightning.ai/docs/fabric/stable/

## Installation

1. Install Miniforge
    ```bash
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh
    ```
2. Clone the repository
    ```bash
    rm -rf DeepKNLP*
    git clone git@github.com:chrisjihee/DeepKNLP.git
    cd DeepKNLP*
    ```
3. Create a new environment
    ```bash
    mamba create -n DeepKNLP python=3.11 -y
    mamba activate DeepKNLP
    ```
4. Install the required packages
    ```bash
    pip install -U -r requirements.txt
    pip list | grep -E "torch|lightning|transformers|chris"
    ```
5. Link huggingface cache
    ```bash
    ln -s ~/.cache/huggingface/hub ./.cache_hf_hub
    ```

## Core Implentation

### `train()` definition
```python
import lightning as L

fabric = L.Fabric(...)

# Instantiate the LightningModule
model = LitModel()

# Get the optimizer(s) from the LightningModule
optimizer = model.configure_optimizers()

# Get the training data loader from the LightningModule
train_dataloader = model.train_dataloader()

# Set up objects
model, optimizer = fabric.setup(model, optimizer)
train_dataloader = fabric.setup_dataloaders(train_dataloader)

# Call the hooks at the right time
model.on_train_start()

model.train()
for epoch in range(num_epochs):
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        loss = model.training_step(batch, i)
        fabric.backward(loss)
        optimizer.step()

        # Control when hooks are called
        if condition:
            model.any_hook_you_like()
```

### `LitModel` definition
```python
import lightning as L


class LitModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ...

    def training_step(self, batch, batch_idx):
        # Main forward, loss computation, and metrics goes here
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y, y_hat)
        acc = self.accuracy(y, y_hat)
        ...
        return loss

    def configure_optimizers(self):
        # Return one or several optimizers
        return torch.optim.Adam(self.parameters(), ...)

    def train_dataloader(self):
        # Return your dataloader for training
        return DataLoader(...)

    def on_train_start(self):
        # Do something at the beginning of training
        ...

    def any_hook_you_like(self, *args, **kwargs):
        ...
```

## Target Task
* Document Classification: https://ratsgo.github.io/nlpbook/docs/doc_cls/overview/
  - `python task1-cls.py --help`
  - `python task1-cls.py train --help`
  - `python task1-cls.py test --help`
  - `python task1-cls.py serve --help`
* Word Sequence Labelling: https://ratsgo.github.io/nlpbook/docs/ner/overview/
  - `python task2-ner.py --help`
  - `python task2-ner.py train --help`
  - `python task2-ner.py test --help`
  - `python task2-ner.py serve --help`
* [Not Yet Provided] Sentence Pair Classification: https://ratsgo.github.io/nlpbook/docs/pair_cls/overview/
* [Not Yet Provided] Extractive Question Answering: https://ratsgo.github.io/nlpbook/docs/qa/overview/
