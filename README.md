# structural-alternations

Examining whether pre-trained language models have understanding of structural alternations


## Installation

Dependencies are managed using `conda`. To set up the conda environment for the framework, issue the following command from within the `structural-alternations` directory.
```bash
conda env create -f environment.yaml
```
Once the environment has been created, activate with `conda activate salts`.

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

You can set `masked_tuning_style` to either `bert` (default) or `always`. `bert` style uses the masked tuning method from the original BERT specification (masked tokens are 80% masked, 10% identical, 10% random word). If tuning a RoBERTa model, the `bert` option does not work, so it is automatically set to `always`. NOTE THAT IN THIS CASE, THE INCORRECT TUNING SETTING IS REFLECTED IN THE DIRECTORY NAME AND CONFIG FILES, SO BE CAREFUL!

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

You can run multiple models without changing model parameters by using hydra's `--multirun`/`-m` option and setting `n=range(0,x)`, where `x` is the number of models to tune for each other combination of parameters. 

(Note that currently this is producing strange behavior where the first model differs from all others, but every model besides the first is identical. Tuning multiple models manually does not result in identical models, even if tuned using the same hyperparameters. An alternative suggestion is to use the terminal/command prompt's for loop command to do the same thing, which doesn't have the same issues. On Windows, the syntax for this is
```
FOR /L %i in (1,1,end) DO python tune.py ...
```
On a Mac, the syntax for this is
```bash
for i in {1..end}; do python tune.py ...; done
````
Where `end` is the number of models you want to tune.)

In order to evaluate a single model's performance, run `python eval.py`. To override the default choices for `checkpoint_dir` and evaluation `data`, specify them as `key=value` arguments to the tuning script, as above. This outputs a CSV and a pickle file containing a dataframe that reports the evaluation results in `checkpoint_dir/eval`, as well as plots for each pairwise combination of sentence types in the data file.

In order to evaluate multiple models' performance, run `python multieval.py`. In this case, all models in every subdir of the set `checkpoint_dir` will be evaluated on the specified evaluation `data`, and results will be saved in the appropriate subdirectory (if there is no evaluation data already present for that model). Once all models in every subdir of `checkpoint_dir` have been evaluated, the results are combined and reduced to a single point (plus or minus the standard error) for each model. As before, this outputs a CSV and a pickle file containing the summary, as well as plots, that report this information, in `checkpoint_dir/multieval`.

### Directory Structure

`structural-alternations` uses the `hydra` framework to encapsulate experiment
parameters and hyperparameters in modular YAML files for repoducibility. All
configuration files are stored within the `conf/` directory:

  - `conf/tune.yaml` controls the hyperparameters for a tuning experiment. You must provide a `model` type (default: `distilbert`) and a source of tuning data (default `dative_DO`), and you may optionally override hyperparameters like the number of tuning `epochs` (default: `70`), whether the tuning inputs are `masked` or not (default: `false`), and the learning rate `lr` of the model during tuning (default: `0.001`). You can also set the `masked_tuning_style` (default: `bert`).
  - `conf/eval.yaml` controls the `data` used for model evaluation (default: `ptb_active_dbl_obj`) and the relative directory path `checkpoint_dir` of the model used for evaluation (no default provided; these will appear in the `outputs/` directory once a model has been downloaded and optionally tuned).
  - `conf/tuning/` is a directory containing configuration files for tuning a model. Each file must specify three things: a `name`, a dictionary of tokens `to_mask` of the form `"short_name" : "model_vocab_idx"` which will be masked out during tuning, and a list of tuning `data` sentences.
  - `conf/model/` is a directory containing configuration files for the types of pre-trained models. Each file must specify the `base_class`, the `tokenizer` class, and the `string_id` of a pre-trained model family from HuggingFace.
  - `conf/data/` is a directory containing configuration files for the various evaluation datasets. Each file must specify five things: a unique `name`, pointing to a CSV file in the `data/` directory (note, the project-root-level subdirectory, not the configuration directory) containing the actual evaluation data, a short `description` of the dataset, whether or not the entries in the data file represent `entail`ment relations (i.e., whether during evaluation it must be the case that success on the second sentence in a loaded line is predicated by success on the first sentence), a list of `sentence_types` in the order they occur in the data file, a dictionary of `eval_groups` of the form `"group_name" : ["list", "of", "tokens"]` specifying groups of tokens used by the evaluation scripts to compute model performance on the dataset, and a dictionary of tokens `to_mask` from the evaluation data.