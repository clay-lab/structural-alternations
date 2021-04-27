# tuner.py
# 
# Tunes a model on training data.

import os
import hydra
from typing import Dict, List
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.distributions import Categorical
from transformers import  DistilBertForMaskedLM, DistilBertTokenizer
from omegaconf import DictConfig
from tqdm import trange
import logging

log = logging.getLogger(__name__)

class Tuner:

  # START Computed Properties

  @property
  def model_class(self):
    if self.cfg.model.base_class == "DistilBertForMaskedLM":
      return DistilBertForMaskedLM
  
  @property
  def tokenizer_class(self):
    if self.cfg.model.tokenizer == "DistilBertTokenizer":
      return DistilBertTokenizer
  
  @property
  def model_bert_name(self) -> str:
    if self.cfg.model.base_class == "DistilBertForMaskedLM":
      return 'distilbert'
  
  @property
  def mask_tok(self):
    return self.tokenizer.mask_token
  
  @property
  def mask_tok_id(self):
    return self.tokenizer(self.mask_tok, return_tensors="pt")["input_ids"][:,1]
  
  @property
  def string_id(self) -> str:
    return self.cfg.model.string_id
  
  @property
  def tuning_data(self) -> List[str]:
    return [s.lower() for s in self.cfg.tuning.data]
  
  @property
  def masked_tuning_data(self) -> List[str]:
    
    data = []
    for s in self.tuning_data:
      for key in list(self.tokens_to_mask.keys()):
        s = s.replace(key, self.mask_tok)
      data.append(s)
    
    return data
  
  @property
  def tokens_to_mask(self) -> Dict[str,str]:
    return self.cfg.tuning.to_mask
  
  # END Computed Properties

  def __init__(self, cfg: DictConfig) -> None:
    
    self.cfg = cfg

    # Construct Model & Tokenizer

    log.info(f"Initializing Tokenizer: {self.cfg.model.tokenizer}")

    self.tokenizer = self.tokenizer_class.from_pretrained(
      self.string_id, 
      do_basic_tokenize=False
    )

    log.info(f"Initializing Model: {self.cfg.model.base_class}")
    self.model = self.model_class.from_pretrained(self.string_id)

    self.old_embeddings = getattr(self.model, self.model_bert_name).embeddings.word_embeddings.weight.clone()

    log.info(f"Freezing model parameters")
    # Freeze parameters
    for name, param in self.model.named_parameters():
      if 'word_embeddings' not in name:
        param.requires_grad = False
    
    for name, param in self.model.named_parameters():
      if param.requires_grad:
        assert 'word_embeddings' in name, f"{name} is not frozen!"

  def load_eval_data_file(self, data_path: str, replacing: Dict[str, str]):
    """
    Loads a file from the specified path, returning a tuple of (input, label)
    for model evaluation.
    """

    resolved_path = os.path.join(
      hydra.utils.get_original_cwd(),
      "data",
      data_path
    )

    with open(resolved_path, 'r') as f:
      raw_sentences = [line.strip().lower() for line in f]
      sentences = []
      for s in raw_sentences:
        for key in replacing:
          s = s.replace(key, self.tokens_to_mask[replacing[key]])
        sentences.append(s)
      
      masked_sentences = []
      for s in sentences:
        m = s
        for val in list(self.tokens_to_mask.values()):
          m = m.replace(val, self.mask_tok)
        masked_sentences.append(m)
      
      inputs = self.tokenizer(masked_sentences, return_tensors="pt", padding=True)
      labels = self.tokenizer(sentences, return_tensors="pt", padding=True)["input_ids"]

      return inputs, labels
  
  def collect_results(self, inputs, eval_groups, outputs) -> Dict:

    results = {}
    
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=2)
    log_probabilities = torch.nn.functional.log_softmax(logits, dim=2)
    predicted_ids = torch.argmax(log_probabilities, dim=2)

    for i, _ in enumerate(predicted_ids):

      sentence_results = {}

      # Foci = indices where input sentences have a [mask] token
      foci = torch.nonzero(inputs["input_ids"][i]==self.mask_tok_id, as_tuple=True)[0]
      
      for idx in foci:
        idx_results = {}
        for group in eval_groups:
          tokens = eval_groups[group]
          group_mean = 0.0
          for token in tokens:
            token_id = self.tokenizer(token, return_tensors="pt")["input_ids"][:,1]
            group_mean += log_probabilities[:,idx,:][i,token_id].item()
          idx_results[group] = group_mean
        
        sentence_results[idx] = {
          'mean grouped log_probability' : idx_results,
          'log_probabilities' : log_probabilities[:,idx,:][i,:],
          'probabilities' : probabilities[:,idx,:][i,:],
          'logits': logits[:,idx,:][i,:]
        }
      results[i] = sentence_results
    
    return results
          
  def summarize_results(self, results: Dict, labels) -> Dict:
    
    summary = {}

    # Define theme and recipient ids
    ricket = self.tokenizer(self.tokens_to_mask["RICKET"], return_tensors="pt")["input_ids"][:,1]
    thax = self.tokenizer(self.tokens_to_mask["THAX"], return_tensors="pt")["input_ids"][:,1]

    # Cumulative log probabilities for <token> in <position>
    theme_in_theme = []
    theme_in_recipient = []
    recipient_in_theme = []
    recipient_in_recipeint = []

    # Confidence in predicting <token> over the alternative
    ricket_confidence = []
    thax_confidence = []

    # Confidence that position is an <animacy> noun
    animate_confidence = []
    inanimate_confidence = []

    # Entropies in various positions
    theme_entropy = []
    recipient_entropy = []

    for i in results:
      label = labels[i]
      result = results[i]

      for idx in result:

        target = label[idx.item()]
        scores = result[idx]['mean grouped log_probability']
        probabilities = result[idx]['probabilities']

        categorical_distribution = Categorical(probs=probabilities)
        entropy = categorical_distribution.entropy()

        if target == ricket:
          theme_in_recipient.append(scores['theme'])
          recipient_in_recipeint.append(scores['recipient'])
          recipient_entropy.append(entropy)
          ricket_confidence.append(scores['recipient'] - scores['theme'])
          animate_confidence.append(scores['animate'] - scores['inanimate'])
        elif target == thax:
          theme_in_theme.append(scores['theme'])
          recipient_in_theme.append(scores['recipient'])
          theme_entropy.append(entropy)
          thax_confidence.append(scores['theme'] - scores['recipient'])
          inanimate_confidence.append(scores['inanimate'] - scores['animate'])

    summary['theme'] = {
      'entropy' : theme_entropy,
      'animacy_conf' : inanimate_confidence,
      'token_conf' : thax_confidence
    }

    summary['recipient'] = {
      'entropy' : recipient_entropy,
      'animacy_conf' : animate_confidence,
      'token_conf' : ricket_confidence
    }

    return summary
  
  def graph_results(self, results: Dict, summary: Dict, eval_cfg: DictConfig):

    dataset = str(eval_cfg.data.name).split('.')[0]

    fig, axs = plt.subplots(2, 2, sharey='row', sharex='row', tight_layout=True)

    theme_entr = [x.item() for x in summary['theme']['entropy']]
    recip_entr = [x.item() for x in summary['recipient']['entropy']]

    inan = summary['theme']['animacy_conf']
    anim = summary['recipient']['animacy_conf']

    # Entropy Plots
    axs[0][0].hist(theme_entr, density=True)
    axs[0][0].axvline(np.mean(theme_entr), color='r')
    axs[0][0].set_title('entropy [theme]')

    axs[0][1].hist(recip_entr, density=True)
    axs[0][1].axvline(np.mean(recip_entr), color='r')
    axs[0][1].set_title('entropy [recipient]')

    # Animacy Plots

    axs[1][0].hist(inan, density=True)
    axs[1][0].axvline(np.mean(inan), color='r')
    axs[1][0].set_title('animacy confidence [theme]')

    axs[1][1].hist(anim, density=True)
    axs[1][1].axvline(np.mean(anim), color='r')
    axs[1][1].set_title('animacy confidence [recipient]')

    fig.suptitle(f"{eval_cfg.data.description}")

    plt.savefig(f"{dataset}.png")
  
  def eval(self, eval_cfg: DictConfig, checkpoint_dir: str):
    
    # Load model from disk
    log.info("Loading model from disk.")
    model_path = os.path.join(checkpoint_dir, "model.pt")
    self.model.load_state_dict(torch.load(model_path))
    self.model.eval()

    # Load data
    inputs, labels = self.load_eval_data_file(eval_cfg.data.name, eval_cfg.data.to_mask)

    # Calculate results on given data
    with torch.no_grad():

      log.info("Evaluating model on testing data")
      outputs = self.model(**inputs)

      results = self.collect_results(inputs, eval_cfg.data.eval_groups, outputs)
      summary = self.summarize_results(results, labels)

      log.info("Creating graphs")
      self.graph_results(results, summary, eval_cfg)

  def tune(self):
    """
    Fine-tunes the model on the provided tuning data. Saves model state to disk.
    """

    # Collect Hyperparameters
    lr = self.cfg.hyperparameters.lr
    epochs = self.cfg.hyperparameters.epochs
    optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    # Construct inputs, labels
    if self.cfg.hyperparameters.masked:
      inputs = self.tokenizer(self.masked_tuning_data, return_tensors="pt", padding=True)
    else:
      inputs = self.tokenizer(self.tuning_data, return_tensors="pt", padding=True)
    labels = self.tokenizer(self.tuning_data, return_tensors="pt", padding=True)["input_ids"]

    self.model.train()

    log.info("Fine-tuning model")
    with trange(epochs) as t:

      for _ in t:

        outputs = self.model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()

        # Zero-out gradient of embeddings asside from the rows we care about
        nz_grad = {}
        for key in self.tokens_to_mask:
          tok = self.tokens_to_mask[key]
          tok_id = self.tokenizer(tok, return_tensors="pt")["input_ids"][:,1]
          grad = getattr(self.model, self.model_bert_name).embeddings.word_embeddings.weight.grad[tok_id, :].clone()
          nz_grad[tok_id] = grad
        
        getattr(self.model, self.model_bert_name).embeddings.word_embeddings.weight.grad.data.fill_(0)

        for key in nz_grad:
          getattr(self.model, self.model_bert_name).embeddings.word_embeddings.weight.grad[key, :] = nz_grad[key]
        
        optimizer.step()
        t.set_postfix(loss=loss.item())
        
        # Check that we changed the correct number of parameters
        new_embeddings = getattr(self.model, self.model_bert_name).embeddings.word_embeddings.weight.clone()
        sim = torch.eq(self.old_embeddings, new_embeddings)
        changed_params = int(list(sim.all(dim=1).size())[0]) - sim.all(dim=1).sum().item()

        exp_ch = len(list(self.tokens_to_mask.keys()))
        assert changed_params == exp_ch, f"Exactly {exp_ch} embeddings should have been updated, but {changed_params} were!"
        
    log.info("Saving model state dictionary.")
    torch.save(self.model.state_dict(), "model.pt")

