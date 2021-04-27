# tuner.py
# 
# Tunes a model on training data.

import os
import hydra
from typing import Dict, List
import torch
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
      raw_sentences = [next(f).strip().lower()]
      sentences = []
      for s in raw_sentences:
        for key in replacing:
          s = s.replace(key, replacing[key])
        sentences.append(s)
      
      masked_sentences = []
      for s in sentences:
        m = s
        for key in list(self.tokens_to_mask.keys()):
          m = m.replace(key, self.mask_tok)
        masked_sentences.append(m)
      
      inputs = self.tokenizer(masked_sentences, return_tensors="pt", padding=True)
      labels = self.tokenizer(sentences, return_tensors="pt", padding=True)["input_ids"]

      return inputs, labels
  
  def eval(self, eval_cfg: DictConfig, checkpoint_dir: str):
    
    # Load model from disk
    log.info("Loading model from disk.")
    model_path = os.path.join(checkpoint_dir, 'model.pt')
    self.model.load_state_dict(torch.load(model_path))
    self.model.eval()

    # Load data
    inputs, labels = self.load_eval_data_file(eval_cfg.data.name, eval_cfg.data.to_mask)

    # Calculate results on given data
    results = {}
    with torch.no_grad():

      outputs = self.model(**inputs)

      logits = outputs.logits
      probabilities = torch.nn.functional.softmax(logits, dim=2)
      log_probabilities = torch.nn.functional.log_softmax(logits, dim=2)
      predicted_ids = torch.argmax(log_probabilities, dim=2)

      for i, _ in enumerate(predicted_ids):

        foci = torch.nonzero(inputs["input_ids"][i]==103, as_tuple=True)[0]
        sentence_results = {}

        for pos in foci:

          pos_results = {}
          for tok_group in eval_cfg.data.eval_groups:
            tokens = eval_cfg.data.eval_groups[tok_group]
            mean = 0.0
            for token in tokens:
              tok_id = self.tokenizer(token, return_tensors="pt")["input_ids"][:,1]
              mean += log_probabilities[:,pos,:][i,tok_id].item()
            pos_results[tok_group] = mean
          
          sentence_results[pos] = {
            'mean token group scores' : pos_results,
            'log_probabilities' : log_probabilities[:,pos,:][i,:],
            'probabilities' : probabilities[:,pos,:][i,:],
            'logits': logits[:,pos,:][i,:]
          }
        results[i] = sentence_results
    
    print(results)


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

