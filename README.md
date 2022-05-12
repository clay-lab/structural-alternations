# structural-alternations

Examining whether pre-trained language models have understanding of structural alternations


## Installation

Dependencies are managed using `conda`. To set up the conda environment for the framework, issue the following command from within the `structural-alternations` directory.
```bash
conda env create -f environment.yaml
```
Once the environment has been created, activate with `conda activate salts`.

## Usage

There are two main types of experiments which can be run using the `structural-alternations` framework: **non-tuning** and **tuning** experiments. Within tuning experiments, there are two sub-types, **new argument** and **new verb** experiments. (There are also some additional scripts to aid in the setup of new verb experiments, **cls_emb.py** and **check_args.py**, to be discussed later.)

**Non-tuning** experiments involve taking an off-the-shelf pre-trained model and examining its logit distributions on masked language modeling (MLM) tasks for a variety of pre-determined tokens or token groups, allowing you to examine things like entropy or token-group confidence in particular positions in testing data for pre-trained BERT models. (Note that due to recent updates, non-tuning experiments may no longer work out of the box. This will be worked on later.)

**Tuning** experiments allow you to take a small set of tuning data and fine-tune a pre-trained model by introducing nonce words into the model's vocabulary. You can then test how the model performs on MLM tasks vis-a-vis its predictions on how these nonce tokens are used.

**New argument** experiments are a sub-type of tuning experiments that introduce new argument nouns into a models vocabulary.

**New verb** experiments introduce a novel verb into the model's vocabulary. Unlike new argument experiments, predictions are not collected on the novel word; instead, predictions are collected regarding the arguments of the novel verb. In other words, you can examine what the model has learned about the new verb by examining its predictions regarding its possible arguments in various structures pre- and post-tuning.

### Configuration

Configuration is handled using [Hydra](https://github.com/facebookresearch/hydra). Default values are specified in `.yaml` files located in the `conf` directory (and subdirectories). when running from the command line, default values are overridden using `key=value` syntax. Additional explanation of how to flexibly specify parameters using Hydra can be found at [hydra.cc](https://hydra.cc/).

#### Options and defaults

The name of an option and its default value are listed here as `name (default)` with explanations.

##### `tune.py` (defaults in `conf/tune.yaml`)

* `model (distilbert)`: which pretrained model to use. This should correspond to the name of a `.yaml` file in `conf/model`. This `.yaml` file should contain:
    * `string_id`: the huggingface string id of the model
    * `friendly_name`: whatever you'd like to use as a non-string id name for the model
    * `model_kwargs`: a dict of kwargs to pass to huggingface transformers' `AutoModelForMaskedLM.from_pretrained`.
    * `tokenizer_kwargs`: a dict of kwargs to pass to huggingface transformers' `AutoTokenizer.from_pretrained`.
* `tuning (dative_DO_give_active)`: which tuning data to use. This should correspond to the name of a `.yaml` file in `conf/tuning`. The contents of this `.yaml` file will differ depending on whether you're running a new argument or a new verb experiment, and will be detailed below.
* `override hydra/job_logging`: this points to a custom logger that allows the use of utf-8 in log files (instead of just ASCII). You shouldn't need to touch this.
* `dev ([])`: a list of names of `.yaml` files in `conf/tuning` to use as dev sets during fine-tuning experiments. Average loss across all dev sets is used to determine early stopping. In addition to any dev sets provided, the training set with dropout disabled and novel token masking enabled is always used as a dev set. A special option for `dev`, `best_matches`, uses the filename of the current tuning file as a base, and finds all tuning files that differ from it in one string when split by underscores (e.g. `tuning=dative_DO_give_active dev=best_matches` would use `dative_DO_send_active`, `dative_DO_give_passive`, `dative_PD_give_active`, and `dative_DO_mail_active` as dev sets).
* `dev_exclude ([])`: a list of strings to use to exclude dev sets when using `dev=best_matches`. A dev set containing any string in `dev_exclude` will not be included in the dev sets, even if it would be selected by the `best_matches` criterion.
* `n (0)`: how many models to fine-tune using the specified options with different seeds for the randomly initialized novel token embeddings. Only used when using Hydra's `-m/--multirun` option.
* `hyperparameters`: a dict containing hyperparameters, which are the following.
    * `lr (0.001)`: the learning rate. If using gradual or complete unfreezing, it is recommended to set this to `0.0001` instead of the default; if unfreezing only the embeddings of the novel tokens, the default works better.
    * `max_epochs (70)`: the maximum number of epochs to fine-tune for.
    * `min_epochs (=max_epochs)`: the minimum number of epochs to fine-tune for.
    * `patience (=max_epochs)`: how many epochs to continue fine-tuning for with no improvement on average loss across the dev sets.
    * `delta (0)`: how much of an improvement on average dev loss is sufficient to allow training to continue. `0` means any improvement, no matter how small, resets the patience counter.
    * `masked_tuning_style (always)`: how to mask the novel tokens. Possible options are `always`, `bert`, `roberta` or `none`.
        * `always`: always mask the novel tokens.
        * `bert`: before fine-tuning, decide to mask 80% of the novel tokens in the fine-tuning data, leave 10% intact, and replace 10% with a random token from the model's vocabulary.
        * `roberta`: like `bert`, but rerun the decision about what to do with each novel token every epoch.
        * `none`: do not mask the novel tokens.
    * `strip_punct (false)`: whether to remove most punctuation from fine-tuning data. Punctuation that is not stripped is `[]<>,`.
    * `unfreezing (none)`: one of `gradual{int}`, `mixout{float}`, `{int}`, `complete`, or `none`. When using `unfreezing=none`, only the weights of the novel tokens are updated, and only those are saved. Because this requires much less space than saving the full model, weights are saved for every epoch. When using any other option, only the full model checkpoint with the lowest average dev loss will be saved.
        * `gradual{int}`: gradual unfreezing unfreezes one layer of the model at a time, starting from the highest numbered layer and proceeding backward until all layers are unfrozen. `{int}` should be replaced with an integer specifying how many epochs to wait between unfreezing one layer and the previous layer. If no integer is provided, the default is 1 epoch (i.e., `hyperparameters.unfreezing=gradual` is equivalent to `hyperparameters.unfreezing=gradual1`).
        * `mixout{float}`: mixout unfreezing completely unfreezes the model, but replaces dropout layers with mixout layers. Mixout layers randomly replace a parameter with the original model's parameter with a probability of `{float}`.
        * `{int}`: an integer specifying the highest layer of the model that should remain unfrozen. E.g., `unfreezing=6` means that layers 0--6 are frozen, and layers 7+ are unfrozen.
        * `complete`: unfreeze all model parameters, including word embeddings. Note that the previous options do not unfreeze word embeddings, but only hidden layers.
        * `none`: do not unfreeze any model parameters except for the embeddings of the novel tokens (which must be unfrozen for any learning to take place).
    * `mask_args (false)`: whether to mask argument positions in new verb experiments, separately from whether to mask the novel verb (which is set by `hyperparameters.masked_tuning_style`). Only used for new verb experiments.
    * `use_kl_baseline_loss (false)`: whether to use a loss term that combines the default cross entropy loss with a loss based on the KL divergence of the predictions of the model being fine-tuned and the predictions of the pre-fine-tuning version of that model. Only used if setting `hyperparameters.unfreezing` to anything other than `none`.
* `kl_loss_params`: if `hyperparameters.use_kl_baseline_loss` is set to `true`, these control how that loss is calculated.
    * `dataset (datamaker/datasets/miniboki-2022-04-01_22-58-30/miniboki)`: a directory containing a dataset in huggingface's datasets format with sentences to use to compute the KL divergence term. The term compares the distribution of the model being fine-tuned to a baseline, non-fine-tuned version of the model to minimize the divergence between the predictions of the two. The default is a dataset of 10,000 sentences constructed to mimic BERT's pretraining dataset, using current data from Wikipedia and Bookcorpus in the same ratio as the occurred in BERT's pretrained dataset.
    * `n_examples_per_step (100)`: how many randomly chosen examples from the dataset to use when calculating the KL loss term every epoch.
    * `scaleby (0.5)`: a multiplier for the KL divergence loss term to control how much to weight it relative to the default cross entropy loss.
    * `masking (none)`: how to mask inputs when calculating the KL loss divergence. When using `kl_loss_params.masking=none`, KL divergence is calculated based on predictions for the entire sentence (i.e., input token sequence); with other values, it is calculated based only on the mask tokens, to mimic BERT's pretraining objective.
        * `always`: randomly choose 15% of tokens in the input sentences and mask them.
        * `bert`: randomly choose 15% of tokens in the input sentences. Of those, mask 80%, do nothing to 10%, and replace 10% with a random token from the model's vocabulary (not including the novel tokens).
        * `none`: do not mask any tokens.
* `debug (false)`: whether to log predictions for sample sentences every epoch during fine-tuning.
* `use_gpu (false)`: whether to use GPU support. if a GPU is not available, this will automatically be set to false. If you save a model fine-tuned using a GPU, you will still be able to load it for evaluation on a CPU.

##### `eval.py` (defaults in `conf/eval.yaml`)

* `data (syn_give_give_ext)`: which data to use for evaluation. This should correspond to the name of a file in `conf/data`, which should include the following information.
    * `name`: the name of the dataset (including the file extension). This should correspond to the name of a file in `./data`. A dataset consists of rows with lists of sentences separated by ` , ` (a space, followed by a comma, followed by a space). If we think of this like a CSV, rows correspond to different examples of sentence types, which each column corresponding to a single sentence type.
    * `description`: a description of the dataset.
    * `sentence_types`: a list of the sentence types in the dataset, with one for each column.
    * `eval_groups`: a dict mapping a label to the novel tokens.
    * `to_mask`: a list containing the novel tokens.
  
    For new argument experiments only:
    * `masked_token_targets`: a dict mapping each novel token to a list of existing tokens to compare it to. Used to get *t*SNEs and cosine similarities between each token and its targets to compare the learned embeddings to the embeddings of the existing tokens.
    * `masked_token_target_labels`: a dict mapping each novel token to a label for the target group it is being compared to.
  
    For new verb experiments only:
    * `added_args`: a list of dicts specifying additional in-group but out-of-training arguments to include during evaluation. The dict key should correspond to an arg group from the tuning file, and its value should be a dict mapping argument types to a list of strings of additional arguments to include in that group during evaluation.
  
    * `prediction_sentences`: sentences to log and save full model predictions for during evaluation.
* `override hydra/job_logging`: this points to a custom logger that allows the use of utf-8 in log files (instead of just ASCII). You shouldn't need to touch this.
* `criteria (all)`: a comma separate list of strings passed as a single string. To be included in evaluation, a model's checkpoint directory must contain all these strings. `all` is a special value meaning no exclusions.
* `create_plots (true)`: whether to create plots of *t*SNEs, cosine similarities, and odds ratios. You can skip plot creation to save time and just get the CSVs.
* `epoch (best_mean)`: which epoch to evaluate the model at. If using any `hyperparameters.unfreezing` other than `none`, this can only be `0` or `best_mean`. If using `hyperparameters.unfreezing=none`, other options are available. Pass an integer to evaluate the model at that epoch. Pass `best_mean` to evaluate the model at the state with the lowest average dev loss. Pass `best_sumsq` to evaluate the model at an epoch where at least one dev set is at its lowest loss, and the difference between performance on the dev sets is minimized.
* `topk_mask_token_predictions ()`: how many of the top predictions to get for masked tokens in `data.prediction_sentences`.
* `k (50)`: find and save the *k* subword tokens with the most similar embeddings to the novel embeddings (using cosine similarity).
* `num_tsne_words (500)`: plot the first two *t*SNE components of the first *n* tokens in the model vocubulary and the novel tokens (to compare the learned representations of the novel tokens to the learned representations of existing tokens).
* `comparison_dataset (datamaker/datasets/miniboki-2022-04-01_22-58-30/miniboki)`: if provided, compare the fine-tuned model's predictions to the same model pre-fine-tuning on this dataset. The default is as described above in the options for `tune.yaml`, `kl_loss_params.dataset`.
* `comparison_n_exs (100)`: how many sentences to draw from the dataset to calculate KL divergence on.
* `comparison_masking (none)`: how mask tokens in sentences during comparison to the model baselines. Options are the same as those described in `tune.yaml`, `kl_loss_params.masking`.
* `dir ()`: a directory containing subdirectories (arbitrarily nested) with model checkpoints to evaluate. All subdirectories of `dir` containing valid model checkpoints will be evaluated.
* `summarize (false)`: whether to summarize when evaluating multiple models in the same run. Summarization involves average predictions for the most similar tokens, cosine similarities for target tokens, and odds ratios. For odds ratios comparisons, the models predictions for each kind of evaluation token group is reduced to a single point which represents the mean of that group.
* `debug (false)`: whether to log model predictions for sample sentences in addition to the `data.prediction_sentences`.
* `use_gpu (false)`: whether to use GPU support.
* `rerun (false)`: whether to rerun evaluations on directories already containing the expected number of results files.

##### Tuning options

For both new argument and new verb experiments, the following should be specified in a tuning config file.

* `name`: the name of the tuning dataset.
* `reference_sentence_type`: the name of the reference sentence type, which is the kind of sentence included in the fine-tuning dataset. This should match whatever you call the same sentence type in the evaluation data file; plots are drawn to compare each other sentence type to the reference sentence type.
* `exp_type`: for new argument experiments, `newarg`; for new verb experiments, `newverb`.
* `to_mask`: a list of the novel tokens to mask in the input sentences.
* `data`: a list of sentences to use as fine-tuning data.

In addition, new verb experiments should specify more options.

* `num_words`: used only by `check_args.py`, not during fine-tuning. Specifies how many of the best generated predictions to display for each argument type.
* `which_args`: which set of arguments to use during fine-tuning. This should corresponding to the name of an option specified in the file, or `model`, which uses the arguments corresponding to the model's friendly name.
* `check_args_data`: a list of sentences to use when generating sets of arguments using `check_args.py`. Not used during fine-tuning.

Argument sets are specified as a dictionary mapping an argument label to a list of arguments to substitute for that label during fine-tuning. You can also specify a specific random seed to use when initializing the new verb token for the model; this is useful when using the arguments generated using `check_args.py`, as it ensures that the random state used to generate the unbiased arguments matches the one used during fine-tuning. The random seed is used when `tuning.which_args` is set to `model`, the actual model's name (i.e., `model=bert tuning.which_args=bert`), or when using `best_average` or `most_similar`.

#### `check_args.py`

`check_args.py` provides an interface for generating sets of argument nouns that start off as unbiased toward a particular structural position. Simply choosing nouns like `woman` and `lawyer` to be subjects and `file` and `computer` as objects would make results of new verb experiments harder to evaluate: does the model do well because it generically predicts the former two nouns to be more likely in subject positions compared to object positions and vice versa for the latter two, or was it learned on the basis of the fine-tuning data? By starting off with nouns that are unbiased toward particular argument positions, it is easier to be sure that any improvement could not be due to pre-existing knowledge of specific tokens, but instead would have to represent generalizations across structural contexts. All candidate arguments must be tokenized as a single token in all models.

The following are the configuration options for `check_args.py`, specified in `check_args.yaml`.

* `dataset_loc (conf/subtlex_freqs_formatted.csv.gz)`: the location of a dataset containing words along with frequency information to pull candidate words from. Currently only SUBTLEX is supported, though it would be easy to add support for other datasets.
* `tunings ([])`: a list of strings corresponding to tuning files in `conf/tuning`. Data (including the `check_args_data`) from these tuning files is used to determine which arguments are unbiased toward specific structural positions.
* `target_freq (any)`: the target frequency for candidate nouns. `any` means all nouns, regardless of frequency of occurrence, are possible candidates.
* `range (1000)`: a noun's frequency must be within +/- this number of the target freq to be a possible candidate. Only used if `target_freq` is not set to `any`.
* `min_length (4)`: the minimum acceptable length in characters for a candidate noun to be a candidate. Changing this can help avoid unwanted things like acronyms and abbreviations.
* `strip_punct (false)`: whether to remove punctuation from sentences when checking argument bias.
* `patterns`: a dict mapping argument types to a list of indices. When outputting an `args.yaml` file containing argument sets for each model, the top least biased `tuning.num_words` arguments $\times$ the number of argument types will be separated into each argument list according to the indices here.

The output of `check_args.py` includes model predictions for each argument which are the log odds ratio of a noun in one argument position compared to every other argument position, a correlation heatmap comparing predictions across each pair of models, and a file `args.yaml` that has argument sets formatted in such a way that they can be copy-pasted into a tuning configuration file and used for fine-tuning.

Arguments are considered less biased if the average log odds ratio of them occurring in the position of one argument type versus another is closer to 0.

#### `cls_emb.py`

`cls_emb.py` fits a Support Vector Machine to embeddings of a model tokens, to determine whether they are linearly separable. Originally, this was included because we wanted to ensure that it was possible linearly separate the subject and object nouns generated by `check_args.py`. However, this is not very useful in practice, because as it turns out, it is quite trivial to linearly separate two groups of six embeddings in a 768-dimensional embedding space.

* `model (distilbert)`: which model's embeddings to classify.
* `tuning (newverb_transitive_ext)`: which tuning file to use to find sets of argument tokens to classify. You can set `tuning.which_args` to change which set of arguments' embeddings to fit the SVM to.

### Framework Interface

#### Fine-tuning

In order to tune a new model, run `python tune.py`. To override the default choices any option, specify them as `key=value` arguments to the tuning script, such as:
```bash
python tune.py model=bert tuning=untuned
```
This script will pull a vanilla BERT model from HuggingFace, do nothing with it (i.e., tune it on an empty dataset), and save the randomly initialized weights of any tokens marked as to be masked to the outputs directory.

When you actually tune a model, a file containing the weights of each token in the tuning configuration's `to_mask` list for each epoch (including the randomly initialized weights) will be saved in the outputs directory. In addition, a CSV file containing various training metrics for each epoch with be saved in the outputs directory, along with a PDF containing plots of these metrics during fine-tuning.

#### Evaluation

In order to evaluate a single model's performance, run `python eval.py`. To override the default choices for `dir` and evaluation `data`, specify them as `key=value` arguments to the tuning script, as above. This outputs CSVs and a pickle file containing a dataframe that reports the evaluation results in a subdirectory of each models' checkpoint dir, as well as various plots.

Outputs include files with the following:

* the model's predictions regarding the log odds ratio of each token in its expected position compared to unexpected positions. Plots comparing these for the reference sentence type against every other sentence type are output as well.
* accuracy information for each sentence type compared to the reference sentence type. Accuracy is defined as having a  log odds ratio >0 for a novel token in the expected position compared to unexpected positions.
* the most similar tokens to the novel tokens after fine-tuning, using cosine similarity as a measure. The cosine similarities of the tokens in the target groups are included as well. If there are at least two novel tokens, plots of these are included as well.
* the first two *t*SNE components of the novel tokens, the first `num_tsne_tokens` in the model vocabulary, and the target tokens. Plots of these are included as well.
* If a comparison dataset is provided, information about the KL divergences for the `comparison_n_exs` examples are output, along with a histogram.
* Saved model outputs for the prediction sentencess, as well as a CSV with information about these.

If `summarize=true`, files summarizing this information across multiple model will be output in a subdirectory of `dir`. The cosine similarities summary gets the average cosine similarity between novel tokens and the target tokens across models. The log odds ratios summary condenses the models' predictions to a single point for each novel token type, with an accuracy file generated based on these averages showing the proportion of models with means for each token type that are accurate. Plots of these are included as well.