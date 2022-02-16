# structural-alternations

Examining whether pre-trained language models have understanding of structural alternations


## Installation

Dependencies are managed using `conda`. To set up the conda environment for the framework, issue the following command from within the `structural-alternations` directory.
```bash
conda env create -f environment.yaml
```
Once the environment has been created, activate with `conda activate salts`.

## Usage

There are three main types of experiments which can be run using the `structural-alternations` framework: **non-tuning**, **tuning**, and **new verb** experiments. 

**Non-tuning** experiments involve taking an off-the-shelf pre-trained model and examining its logit distributions on masked language modeling (MLM) tasks for a variety of pre-determined tokens or token groups, allowing you to examine things like entropy or token-group confidence in particular positions in testing data for pre-trained BERT models. 

**Tuning** experiments allow you to take a small set of tuning data and fine-tune a pre-trained model by introducing nonce words into the model's vocabulary. You can then test how the model performs on MLM tasks vis-a-vis its predictions on how these nonce tokens are used.

**New verb** experiments are a combination of the non-tuning and tuning experiments described above. In this case, the model is tuned on a nonce word that is a verb that occurs with various kinds of arguments that are assigned to specific grammatical functions (e.g., tuning data might encode that the word "laughter" is a possible subject but not a possible object for the nonce verb, while "tree" has the opposite distribution). However, it is not directly tuned on the arguments of that verb themselves (the embeddings are frozen). Then, you can examine what the model has learned about the new verb by examining its predictions regarding those non-tuned arguments in various structures pre- and post-tuning.

### Configuration and hyperparameters

There are a number of configuration options you can set when running experiments.

`dev` is used to specify a set of data to use as dev sets during fine-tuning. Metrics for each dev set will be calculated, and mean loss across all dev sets will be used to determine early halting. In addition to any manually specified dev sets, a version of the fine-tuning set without masking and dropout is included among the dev sets. Data for dev sets is specified in files that are just like the fine-tuning sets. There is also a special option, `best_matches`. This automatically includes as dev sets all fine-tuning sets that differ from the current fine-tuning set in only one parameter (e.g., active vs. passive, DO vs. PD, etc.).

`dev_exclude` is a list of strings. Any fine-tuning sets containing any of these strings will be excluded from the set of dev sets. This is most useful when you want to use `dev=best_matches` but not include everything that setting would include by default. Note that `dev_exclude` does *NOT* support wildcard matching.

`n` specifies how many models to fit with the current configuration. It is only used when using hydra's `-m/--multirun` option.

#### Hyperparameters

`lr` is the learning rate. Default is `0.001`.

`max_epochs` specifies the maximum number of epochs to allow for fine-tuning. Default is `70`.

`min_epochs` specifies the minimum number of epochs to require for fine-tuning. Default is `max_epochs`.

`patience` specifies how many epochs to continue training with no improvement on mean dev loss. The default is `max_epochs` (ensuring that unless you change this, fine-tuning will always continue for the full number of epochs).

`delta` specifies an amount of change to count as improvement for calculating remaining patience. For instance, with `delta=0.5`, mean dev loss must improve by $0.5$ within `patience` epochs for fine-tuning to continue. Default is `0` (meaning any improvement within `patience` epochs will ensure continued fine-tuning).

`masked_tuning_style` specifies the type of masking to use during tuning. Possible options are `always` (default), `bert`, `roberta`, or `none`. `bert` style uses the masked tuning method from the original BERT specification: prior to fine-tuning, the input data are processed such that masked tokens stay masked tokens 80% of the time, are replaced with the correct target 10% of the time, and are replaced with a random token in the model's vocabulary 10% of the time. `roberta`-style tuning performs the BERT tuning method on the input data every epoch instead of only once prior to fine-tuning. `none` means that no masking is used.

`strip_punct` specifies whether to remove punctuation from sentences during fine-tuning. If set during fine-tuning, punctuation will also be removed during evaluation. Default is `false` (meaning punctuation is retained). Note that any punctuation used to signal masked tokens will not be stripped from anywhere in the sentence. Punctuation that will not be stripped is `[]<>,`. In addition, commas cannot be used in evaluation data, as they are used to separate sentences of different types. Note that currently, models can only be evaluated on data that matches the data they were trained on for this setting (e.g., if a model was trained with no punctuation, it will be evaluated on data with no punctuation, and vice versa). Keep this in mind when using `multieval.py`, as the individual models being summarized across may be evaluated on slightly different eval data if they were trained differently in this way. In case a reported summary statistic comes from models with multiple different parameters, the value for that parameter will be reported as `multiple`, with no direct information about which underlying parameters generated the summary statistic retained in the full summary.

### Framework Interface

#### Fine-tuning

In order to tune a new model, run `python tune.py`. To override the default choices any option, specify them as `key=value` arguments to the tuning script, such as:
```bash
python tune.py model=bert tuning=untuned
```
This script will pull a vanilla BERT model from HuggingFace, do nothing with it (i.e., tune it on an empty dataset), and save the randomly initialized weights of any tokens marked as to be masked to the outputs directory. Note that the valid `key`s for a loaded script represent folders within the `conf/` directory and the `value`s are the names of the YAML configuration files within those directories (without the `.yaml` file extension).

Note that if you want to tune or evaluate models with multiple sets of parameters, it is highly recommended you install the `hydra-joblib-launcher` module using `pip install hydra-joblib-launcher --upgrade`. Then, when running models using hydra's `--multirun/-m` flag, you can set `hydra/launcher=joblib` and `hydra.launcher.n_jobs=#`, where `#` is replaced with the number of concurrent jobs you want to run. In addition, when using `-m`, you can specify parameters to sweep over as comma separated lists (e.g., `hyperparameters.lr=0.001,0.01`). This will greatly speed things up if you want to fine-tune models with multiple configuration/hyperparameter settings.

When you actually tune a model, a file containing the weights of each token in the tuning configuration's `to_mask` list for each epoch (including the randomly initialized weights) will be saved in the outputs directory. In addition, a CSV and pickle file containing various training metrics for each epoch with be saved in the outputs directory, along with a PDF containing plots of these metrics over time. These metrics include the loss and the log probability for each masked token. For new verb experiments, these metrics also report a difference measure, which is the difference for each argument position between the mean surprisal of the unexpected tokens in that position and the mean surprisal of the expected tokens in that position. This is expected to increase as the model learns about which position each set of arguments goes in.

#### Evaluation

In order to evaluate a single model's performance, run `python eval.py`. To override the default choices for `checkpoint_dir` and evaluation `data`, specify them as `key=value` arguments to the tuning script, as above. This outputs CSVs and a pickle file containing a dataframe that reports the evaluation results in `checkpoint_dir/eval`, as well as various plots. 

##### Configuration

For evaluation, you can set the following options.

`checkpoint_dir` specifies where the weights for the model you want to evaluate are saved.

`epoch` specifies which epoch you want to evaluate the performance of the tuned model on. This can be a number, or one of {`best_mean`, `best_sumsq`}. If a number, the script will pull the weights from the corresponding epoch in the saved weights file. If it is `best_mean`, the evaluation script will pull data from the saved `metrics.csv` for that model, determine which epoch had the lowest mean dev loss, and load weights from that epoch. If it is `best_sumsq`, the script will pull data from the `metrics.csv` for that model, and load weights from the epoch that (a) has the lowest loss for at least one of the dev sets, and (b) minimizes the sum of the squared differences between the average dev loss and the loss for each individual dev set among epochs meeting criterion (a). The idea is to pick an epoch where the model's performance across all dev sets is maximally similar.

`k` is used to get the `k` most similar subword tokens to the novel tokens at the evaluation epoch (using cosine similarity). Default is `50`.

`num_tsne_words` specifies how many words to include in t-SNE plots. Words included are the first `num_tsne_words` in the model's vocabulary that are either nouns or verbs (depending on the experiment's type). Default is `500`.

`checkpoint_dir` specifies which model to evaluate.

#### Multievaluation

In order to evaluate multiple models' performance, run `python multieval.py`.

The same options as for `eval.py` can be set. In addition, `multieval.py` also allows you to set the following options.

`dir` specifies a directory containing subdirectories with model information in them.

`criteria` is string that specifies a comma separated list of strings (make sure to include the initial and trailing quotes `'` when setting this, or hydra will try to parse them as separate parameters). To be included in the evaluation, the directory name containing the model must be in a subdirectory of `dir` and must contain all criteria strings. Note that to facilitate not outputting the evaluation results within several subdirectories, the character `^` can be used instead of the system path separator to specify that a given string should occur at the beginning of a subdirectory. This is replaced with the system path separator when the script is run. On Windows, you need to escape this character in the criteria string with an additional `^` (e.g., `^^bert`) so that Windows doesn't parse it out before passing it to hydra. If no criteria are set, all models in every subdir of the set `dir` will be evaluated on the specified evaluation `data`, and results will be saved in the appropriate subdirectory (if there is no evaluation data already present for that model).

`summarize` is a boolean specifying whether to summarize information across the evaluated models. If `true`, once all models in every subdir of `dir` have been evaluated, the results will be combined and summarized as a single point (plus or minus the standard error) for each model. This outputs CSVs and a pickle file containing the summary, as well as plots that report this information, in `dir/multieval-[criteria]-[data]`. Default is `true`.

`entail` and `new_verb` specify which kind of experiment is being performed.

### Directory Structure

`structural-alternations` uses the `hydra` framework to encapsulate experiment
parameters and hyperparameters in modular YAML files for repoducibility. All
configuration files are stored within the `conf/` directory:

  - `conf/tune.yaml` controls the hyperparameters for a tuning experiment. You must provide a `model` type (default: `distilbert`) and a source of tuning data (default `dative_DO_give_active`).
  - `conf/eval.yaml` controls the `data` used for model evaluation (default: `ptb_active_dbl_obj`) and the relative directory path `checkpoint_dir` of the model used for evaluation (no default provided; these will appear in the `outputs/` directory once a model has been downloaded and optionally tuned).
  - `conf/tuning/` is a directory containing configuration files for tuning a model. Each file must specify four things: a `name`, the `reference_sentence_type` (i.e., what kind of sentence(s) do the tuning data represent), whether the tuning data is for a `new_verb` experiment, a dictionary of tokens `to_mask` of the form `short_name : model_vocab_idx` which will be masked out during tuning, and a list of tuning `data` sentences. If you are running a new verb experiment, you must also provide `args`, a dictionary of the form `str : List[str]`, where the final dataset for tuning will be constructed from the `data` by generating replacing `str` in every sentence in `data` with every possible value in the list.
  - `conf/model/` is a directory containing configuration files for the types of pre-trained models. Each file must specify the `base_class`, the `tokenizer` class, and the `string_id` of a pre-trained model family from HuggingFace.
  - `conf/data/` is a directory containing configuration files for the various evaluation datasets. Each file must specify a unique `name`, pointing to a `.data` file in the `data/` directory (note, the project-root-level subdirectory, not the configuration directory) containing the actual evaluation data, a `friendly_name` for the dataset, a short `description` of the dataset, whether or not the entries in the data file represent `entail`ment relations (i.e., whether during evaluation it must be the case that success on the second sentence in a loaded line is predicated by success on the first sentence), whether or not the data is for a `new_verb` experiment, a list of `sentence_types` in the order they occur in the data file, a dictionary of `eval_groups` of the form `group_name : [list, of, tokens]` specifying groups of tokens used by the evaluation scripts to compute model performance on the dataset, a list of tokens `to_mask` from the evaluation data, a `masked_token_targets` dictionary of the form `added_token: [comparison, tokens]` to compare to the learned tokens (using cosine similarity and t-SNE), and a dictionary of `masked_token_target_labels` used to specify what the `masked_token_target` groups correspond to.
  
## `gen_args.py`

`gen_args.py` is a utility to help in running new verb experiments. Part of the key to these experiments is that the arguments associated with each position should be unbiased toward those positions prior to tuning, since that means that whatever knowledge the model ends up with about the positions of the arguments with the nonce verb must have come from the tuning and not the pre-training. For this reason, it is useful to generate random nouns as candidates, and check how well they adhere to this ideal. To do this, you will need the SUBTLEX corpus file `SUBTLEX-US frequency list with PoS information.xlsx`. The first time the `gen_args.py` is run, it will process this file and save a reformatted version as a CSV that can be loaded faster later. The script will filter out nouns based on a `min_freq`, and also based on whether they exist within all model tokenizers as a single token. Then, a random set of `tuning.num_words` * `tuning.args` nouns is generated. `n_sets` determines how many distinct sets of candidates will be checked. The predictions each model makes for the combinations of every noun with every other noun are collected and converted to surprisal values. The raw results are stored in `predictions.csv`. Summary statistics, including a difference of differences measure is calculated. In an ideal case, this would be 0, which would indicate that the expected argument is not considered by the pre-trained models to be more likely in the expected position than the unexpected argument. The difference of these measures is also reported, which gives an idea of how much more biased the models are toward putting the expected noun in the expected position vs. in the unexpected position. Positive means that it expects the first type of noun in the expected position more than the other type; negative means the opposite. The lower this difference of differences gets post-fine-tuning, the more equally the model has learned about which nouns go where.

You can set the following configuration options.

`dataset_loc` specifies where the CSV containing information about nouns and frequencies is located.

`target_freq` specifies the center of a range; candidate nouns must be within `range` of `target_freq` to be included as candidates. Default is `4000`.

`range` specifies a range around the `target_freq`. Nouns must be within `range` of `target_freq` to be included as candidates. Default is `1000`.

`strip_punct` specifies whether to consider sentences with or withoun punctuation when getting predictions from the models.

`n_sets` specifies how many sets of nouns to consider.

`n_jobs` specifies how many processors to use in parallel.