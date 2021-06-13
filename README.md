# structural-alternations

Examining whether pre-trained language models have understanding of structural alternations

## Usage

There are two main types of experiments which can be run using the 
`structural-alternations` framework: **non-tuning** and **tuning** experiments.
**Non-tuning** experiments involve taking an off-the-shelf pre-trained model
and examining it's logit distributions on masked language modeling (MLM) tasks
for a variety of pre-determined tokens or token groups, allowing you to examine
things like entropy or token-group confidence in particular positions in testing
data for pre-trained BERT models. **Tuning** experiments allow you to take a 
small set of tuning data and fine-tune a pre-trained model by introducing nonce 
words into the model's vocabulary. You can then test how the model performs on
MLM tasks vis-a-vis its predictions on how these nonce tokens are used.

### Directory Structure

`structural-alternations` uses the `hydra` framework to encapsulate experiment
parameters and hyperparameters in modular YAML files for repoducibility. All
configuration files are stored within the `conf/` directory:

  - `conf/tune.yaml` controls the hyperparameters for a tuning experiment. You must provide a `model` type (default: `distilbert`) and a source of tuning data (default `active_DO`), and you may optionally override hyperparameters like the number of tuning `epochs` (default: `20`), whether the tuning inputs are `masked` or not (default: `false`), and the learning rate `lr` of the model during tuning (default: `0.001`).
  - `conf/eval.yaml` controls the `data` used for model evaluation (default: `ptb_active_dbl_obj`) and the relative directory path `checkpoint_dir` of the model used for evaluation (no default provided; these will appear in the `outputs/` directory once a model has been downloaded and optionally tuned).
  - `conf/tuning/` is a directory containing configuration files for tuning a model. Each file must specify three things: a `name`, a dictionary of tokens `to_mask` of the form `"short_name" : "model_vocab_idx"` which will be masked out during tuning, and a list of tuning `data` sentences.
  - `conf/model/` is a directory containing configuration files for the types of pre-trained models. Each file must specify the `base_class`, the `tokenizer` class, and the `string_id` of a pre-trained model family from HuggingFace.
  - `conf/data/` is a directory containing configuration files for the various evaluation datasets. Each file must specify five things: a unique `name`, pointing to a CSV file in the `data/` directory (note, the project-root-level subdirectory, not the configuration directory) containing the actual evaluation data, a short `description` of the dataset, whether or not the entries in the data file represent `entail`ment relations (i.e., whether during evaluation it must be the case that success on the second sentence in a loaded line is predicated by success on the first sentence), a dictionary of `eval_groups` of the form `"group_name" : ["list", "of", "tokens"]` specifying groups of tokens used by the evaluation scripts to compute model performance on the dataset, and a dictionary of tokens `to_mask` from the evaluation data.

### Framework Interface

In order to tune a new model, run `python tune.py`. To override the default choices
for `model` type and `tuning` data, specify them as `key=value` arguments to the
tuning script, such as:
```bash
python tune.py model=bert tuning=untuned
```
This script will pull a vanilla BERT model from HuggingFace, do nothing with it 
(i.e., tune it on an empty dataset), and save it to the outputs directory. Note 
that the valid `key`s for a loaded script represent folders within the `conf/` directory
and the `value`s are the names of the YAML configuration files within those 
directories (without the `.yaml` file extension).

## Installation

We use the `poetry` tool to manage dependencies and ensure that the correct versions of all requisite software are installed in a virtual environment for portability and reproducibility. To install, run `poetry shell` and `poetry install` to activate a new virtual environment and install the dependencies. Then, from within the virtual environment, run the script (either `tune.py` or `eval.py`).
