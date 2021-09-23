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
from transformers import  DistilBertForMaskedLM, DistilBertTokenizer, RobertaForMaskedLM, RobertaTokenizer, BertForMaskedLM, BertTokenizer
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig
from tqdm import trange
import logging

import random

log = logging.getLogger(__name__)

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

class Tuner:

  # START Computed Properties

  @property
  def model_class(self):
    if self.cfg.model.base_class == "DistilBertForMaskedLM":
      return DistilBertForMaskedLM
    elif self.cfg.model.base_class == "RobertaForMaskedLM":
      return RobertaForMaskedLM
    elif self.cfg.model.base_class == "BertForMaskedLM":
      return BertForMaskedLM
  
  @property
  def tokenizer_class(self):
    if self.cfg.model.tokenizer == "DistilBertTokenizer":
      return DistilBertTokenizer
    elif self.cfg.model.tokenizer == "RobertaTokenizer":
      return RobertaTokenizer
    elif self.cfg.model.tokenizer == "BertTokenizer":
      return BertTokenizer
  
  @property
  def model_bert_name(self) -> str:
    if self.cfg.model.base_class == "DistilBertForMaskedLM":
      return 'distilbert'
    elif self.cfg.model.base_class == "RobertaForMaskedLM":
      return 'roberta'
    elif self.cfg.model.base_class == "BertForMaskedLM":
      return 'bert'
  
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
    data = []
    for s in self.cfg.tuning.data:
      for key in self.tokens_to_mask:
        s = s.replace(key, self.tokens_to_mask[key])
      data.append(s)
    return [d.lower() for d in data]
  
  @property
  def masked_tuning_data(self) -> List[str]:
    
    data = []
    for s in self.tuning_data:
      for val in list(self.tokens_to_mask.values()):
        s = s.replace(val.lower(), self.mask_tok)
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
      do_basic_tokenize=False,
      local_files_only=True
    )

    log.info(f"Initializing Model: {self.cfg.model.base_class}")

    self.model = self.model_class.from_pretrained(
      self.string_id,
      local_files_only=True
    )

    # randomly initialize the embeddings of the novel tokens we care about
    # to provide some variablity in model tuning
    model_e_dim = getattr(
      self.model, 
      self.model_bert_name
    ).embeddings.word_embeddings.embedding_dim
    num_new_tokens = len(list(self.tokens_to_mask.keys()))
    new_embeds = torch.nn.Embedding(
      num_new_tokens, 
      model_e_dim
    )
    
    with torch.no_grad():

      unused_embedding_weights = getattr(
          self.model, 
          self.model_bert_name
        ).embeddings.word_embeddings.weight[range(0,999), :]

      std, mean = torch.std_mean(unused_embedding_weights)
      log.info(f"Initializing unused tokens with random data drawn from N({mean:.2f}, {std:.2f})")

      # These are experimentally determined values to match the
      # default embedding weights of BERT's unused vocab items
      torch.nn.init.normal_(new_embeds.weight, mean=mean, std=std)

      for i, key in enumerate(self.tokens_to_mask):
        tok = self.tokens_to_mask[key]
        tok_id = self.tokenizer(tok, return_tensors="pt")["input_ids"][:,1]

        getattr(
          self.model, 
          self.model_bert_name
        ).embeddings.word_embeddings.weight[tok_id, :] = new_embeds.weight[i,:]
   

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

    with open(resolved_path, "r") as f:
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
  
  def load_eval_multi_file(self, data_path: str, replacing: Dict[str, str]):

    resolved_path = os.path.join(
      hydra.utils.get_original_cwd(),
      "data",
      data_path
    )

    with open(resolved_path, "r") as f:
      
      raw_input = [line.strip().lower() for line in f]
      sentences = []

      for r in raw_input:
        line = []
        s_splits = r.split(',')
        for s in s_splits:
          for key in replacing:
            s = s.replace(key, self.tokens_to_mask[replacing[key]])
          line.append(s.strip())
        sentences.append(line)
      
      masked_sentences = []
      for s_group in sentences:
        m_group = []
        for s in s_group:
          m = s
          for val in list(self.tokens_to_mask.values()):
            m = m.replace(val, self.mask_tok)
          m_group.append(m)
        masked_sentences.append(m_group)

    sentences_transposed = list(map(list, zip(*sentences)))
    masked_transposed = list(map(list, zip(*masked_sentences)))

    inputs = []
    labels = []

    for s in sentences_transposed:
      label = self.tokenizer(s, return_tensors="pt", padding=True)["input_ids"]
      labels.append(label)
    
    for m in masked_transposed:
      input = self.tokenizer(m, return_tensors="pt", padding=True)
      inputs.append(input)

    return {
      "inputs" : inputs,
      "labels" : labels
    }
  
  def collect_results(self, inputs, eval_groups, outputs) -> Dict:

    results = {}
    
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=2)
    log_probabilities = torch.nn.functional.log_softmax(logits, dim=2)
    predicted_ids = torch.argmax(log_probabilities, dim=2)

    # print(f"Mask token id: {self.mask_tok_id}")
    # print("Inputs:")
    # print(inputs["input_ids"])

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
          inanimate_confidence.append(scores['animate'] - scores['inanimate'])

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
  
  def collect_entailed_results(self, inputs, eval_groups, outputs):

    results_arr = []
    
    for j in range(len(outputs)):

      results = {}

      logits = outputs[j].logits
      probabilities = torch.nn.functional.softmax(logits, dim=2)
      log_probabilities = torch.nn.functional.log_softmax(logits, dim=2)
      predicted_ids = torch.argmax(log_probabilities, dim=2)

      for i, _ in enumerate(predicted_ids):

        sentence_results = {}
        foci = torch.nonzero(inputs[j]["input_ids"][i]==self.mask_tok_id, as_tuple=True)[0]

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
      
      results_arr.append(results)
    
    return results_arr

  def summarize_entailed_results(self, results_arr, labels_arr):

    # Define theme and recipient ids
    ricket = self.tokenizer(self.tokens_to_mask["RICKET"], return_tensors="pt")["input_ids"][:,1]
    thax = self.tokenizer(self.tokens_to_mask["THAX"], return_tensors="pt")["input_ids"][:,1]
    
    active_results = results_arr[0]
    active_labels = labels_arr[0]

    passive_results = results_arr[1]
    passive_labels = labels_arr[1]

    confidences = []

    for r in active_results:
      
      active_result = active_results[r]
      active_label = active_labels[r]

      passive_result = passive_results[r]
      passive_label = passive_labels[r]

      active_token_confidence = {}
      passive_token_confidence = {}

      for idx in active_result:

        target = active_label[idx.item()]
        scores = active_result[idx]['mean grouped log_probability']

        token_conf = scores['theme'] - scores['recipient']

        if target == ricket:
          # print("I'm in a recipient position")
          active_token_confidence["recipient"] = -token_conf
        else:
          # print("I'm in a theme position")
          active_token_confidence["theme"] = token_conf
      
      for idx in passive_result:

        target = passive_label[idx.item()]
        scores = passive_result[idx]['mean grouped log_probability']

        # print(scores)
        # raise SystemExit

        token_conf = scores['theme'] - scores['recipient']

        if target == ricket:
          # print("I'm in a recipient position")
          passive_token_confidence["recipient"] = -token_conf
        else:
          # print("I'm in a theme position")
          passive_token_confidence["theme"] = token_conf

      confidences.append({
        "active" : active_token_confidence,
        "passive" : passive_token_confidence
      })
    
    return confidences

  def graph_results(self, results: Dict, summary: Dict, eval_cfg: DictConfig):

    dataset = str(eval_cfg.data.name).split('.')[0]

    fig, axs = plt.subplots(2, 2, sharey='row', sharex='row', tight_layout=True)

    theme_entr = [x.item() for x in summary['theme']['entropy']]
    recip_entr = [x.item() for x in summary['recipient']['entropy']]

    inan = summary['theme']['animacy_conf']
    anim = summary['recipient']['animacy_conf']

    # Entropy Plots
    axs[0][0].hist(theme_entr)
    axs[0][0].axvline(np.mean(theme_entr), color='r')
    axs[0][0].set_title('entropy [theme]')

    axs[0][1].hist(recip_entr)
    axs[0][1].axvline(np.mean(recip_entr), color='r')
    axs[0][1].set_title('entropy [recipient]')

    # Animacy Plots

    axs[1][0].hist(inan)
    axs[1][0].axvline(np.mean(inan), color='r')
    axs[1][0].set_title('animacy confidence [theme]')

    axs[1][1].hist(anim)
    axs[1][1].axvline(np.mean(anim), color='r')
    axs[1][1].set_title('animacy confidence [recipient]')

    fig.suptitle(f"{eval_cfg.data.description}")

    plt.savefig(f"{dataset}.png")

    with open(f"{dataset}-scores.npy", "wb") as f:
      np.save(f, np.array(theme_entr))
      np.save(f, np.array(recip_entr))
      np.save(f, np.array(inan))
      np.save(f, np.array(anim))
  
  def get_entailed_summary(self, outputs, labels):
    """
    Returns a list of dictionaries of the form
     n : {"active" : {"theme" : __, "recipient" : __}, "passive" : ...}
    where n is the index of the active-passive pair in the testing data,
    and each dictionary contains two sub-dictionaries (for active and passive 
    sentence, respectively). In the active dict, e.g., 
      "theme" == confidence in THAX versus RICKET (diff. of log probs)
    while
      "recipeint" == confidence in RICKET vs THAX
    """
    summary = []

    active_outputs = outputs[0]
    active_logprobs = torch.nn.functional.log_softmax(active_outputs.logits, dim=2)
    active_labels = labels[0]

    passive_outputs = outputs[1]
    passive_logprobs = torch.nn.functional.log_softmax(passive_outputs.logits, dim=2)
    passive_labels = labels[1]

    tok_indices = self.tokenizer.convert_tokens_to_ids(list(self.tokens_to_mask.values()))
    rick_id = tok_indices[0]
    thax_id = tok_indices[1]

    # print("\n\n########################")
    # print("active_labels:", active_labels.size())
    # print(active_labels)
    # print("passive_labels:", passive_labels.size())
    # print(passive_labels)
    # print("########################\n\n")

    thax_foci = [
      ((active_labels == thax_id).nonzero(as_tuple=True)[1]),
      ((passive_labels == thax_id).nonzero(as_tuple=True)[1])
    ]

    ricket_foci = [
      ((active_labels == rick_id).nonzero(as_tuple=True)[1]),
      ((passive_labels == rick_id).nonzero(as_tuple=True)[1])
    ]
    
    # Active Theme confidence

    # print("\n\n######################")
    # print("passive_logprobs", active_logprobs.size())
    # print("thax_id", thax_id)
    # print("passive_logprobs[:,:,thax_id]", active_logprobs[:,:,thax_id].size())
    # print("range(passive_logprobs.shape[0])", range(active_logprobs.shape[0]))
    # print("thax_foci[0]", thax_foci[0].size())
    # print("######################\n\n")

    ac_thax_in_theme = active_logprobs[:,:,thax_id][range(active_logprobs.shape[0]), thax_foci[0]]
    ac_rick_in_theme = active_logprobs[:,:,rick_id][range(active_logprobs.shape[0]), thax_foci[0]]
    active_theme = ac_thax_in_theme - ac_rick_in_theme
    
    # Passive Theme confidence

    # print("\n\n######################")
    # print("passive_logprobs", passive_logprobs.size())
    # print("thax_id", thax_id)
    # print("passive_logprobs[:,:,thax_id]", passive_logprobs[:,:,thax_id].size())
    # print("range(passive_logprobs.shape[0])", range(passive_logprobs.shape[0]))
    # print("thax_foci[1]", thax_foci[1].size())
    # print("######################\n\n")

    pa_thax_in_theme = passive_logprobs[:,:,thax_id][range(passive_logprobs.shape[0]), thax_foci[1]]
    pa_rick_in_theme = passive_logprobs[:,:,rick_id][range(passive_logprobs.shape[0]), thax_foci[1]]
    passive_theme = pa_thax_in_theme - pa_rick_in_theme

    if not(ricket_foci[0].shape[0]==0 and ricket_foci[1].shape[0]==0):
      # Active Recipient Confidence
      ac_thax_in_recip = active_logprobs[:,:,thax_id][range(active_logprobs.shape[0]), ricket_foci[0]]
      ac_rick_in_recip = active_logprobs[:,:,rick_id][range(active_logprobs.shape[0]), ricket_foci[0]]
      active_recip = ac_thax_in_recip - ac_rick_in_recip 
    
      # Active Recipient Confidence
      pa_thax_in_recip = passive_logprobs[:,:,thax_id][range(passive_logprobs.shape[0]), ricket_foci[1]]
      pa_rick_in_recip = passive_logprobs[:,:,rick_id][range(passive_logprobs.shape[0]), ricket_foci[1]]
      passive_recip = pa_thax_in_recip - pa_rick_in_recip 

    else:
      active_recip = torch.empty_like(active_theme)
      passive_recip = torch.empty_like(passive_theme)


    for i in range(active_theme.shape[0]):
      summary.append({
        "active" : {
          "theme" : active_theme[i],
          "recipient" : active_recip[i]
        }, "passive" : {
          "theme" : passive_theme[i],
          "recipient" : passive_recip[i]
        }
      })
    
    return summary
  
  def graph_entailed_results(self, summary, eval_cfg: DictConfig):

    dataset_name = eval_cfg.data.name.split('.')[0]

    theme_points = [(l['active']['theme'], l['passive']['theme']) for l in summary]
    recipient_points = [(l['active']['recipient'], l['passive']['recipient']) for l in summary]

    x_points = []
    y_points = []

    for i, _ in enumerate(theme_points):
      x_points.append(theme_points[i][0])
      x_points.append(recipient_points[i][0])

      y_points.append(theme_points[i][1])
      y_points.append(recipient_points[i][1])
    
    x_lim = np.max([np.abs(p) for p in x_points]) + 0.5
    y_lim = np.max([np.abs(p) for p in y_points]) + 0.5

    lim = np.max([x_lim, y_lim])

    fig, ax = plt.subplots()

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    ax.scatter(
      x=[t[0] for t in theme_points], 
      y=[t[1] for t in theme_points], 
      c='teal',
      label='theme position'
    )
    ax.scatter(
      x=[t[0] for t in recipient_points], 
      y=[t[1] for t in recipient_points], 
      c='r',
      label="recipient position"
    )

    # Line
    xpoints = ypoints = ax.get_xlim()
    ax.plot(xpoints, ypoints, linestyle='--', color='k', scalex=False, scaley=False)

    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    ax.set_xlabel("[thax] confidence in active sentences")
    ax.set_ylabel("[thax] confidence in passive sentences")
    ax.legend()

    fig.suptitle(f"{eval_cfg.data.description}")
    fig.tight_layout()
    plt.savefig(f"{dataset_name}-paired.png")

    with open(f"{dataset_name}-scores.npy", "wb") as f:
      np.save(f, np.array(theme_points))
      np.save(f, np.array(recipient_points))
  
  def eval_entailments(self, eval_cfg: DictConfig, checkpoint_dir: str):
    """
    Computes model performance on data consisting of 
      sentence 1 , sentence 2
    where credit for a correct prediction on sentence 2 is contingent on
    also correctly predicting sentence 1.
    """

    print(f"SAVING TO: {os.getcwd()}")
    
    # Load model
    log.info("Loading model from disk")
    model_path = os.path.join(checkpoint_dir, "model.pt")
    self.model.load_state_dict(torch.load(model_path))
    self.model.eval()

    # Load model
    data = self.load_eval_multi_file(eval_cfg.data.name, eval_cfg.data.to_mask)
    inputs = data["inputs"]
    labels = data["labels"]

    assert len(inputs) == len(labels), f"Inputs (size {len(inputs)}) must match labels (size {len(labels)}) in length"

    # Calculate performance on data
    with torch.no_grad():

      log.info("Evaluating model on testing data")

      outputs = []

      for i in range(len(inputs)):
        output = self.model(**inputs[i])
        outputs.append(output)

      summary = self.get_entailed_summary(outputs, labels)
      self.graph_entailed_results(summary, eval_cfg)

      for line in summary:
        print(line)

  def eval(self, eval_cfg: DictConfig, checkpoint_dir: str):
    
    # Load model from disk
    log.info("Loading model from disk")
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

    log.info(f"Training model @ '{os.getcwd()}'")

    if not self.tuning_data:
      log.info("Saving model state dictionary.")
      torch.save(self.model.state_dict(), "model.pt")
      return

    # Collect Hyperparameters
    lr = self.cfg.hyperparameters.lr
    epochs = self.cfg.hyperparameters.epochs
    optimizer = torch.optim.AdamW(
      self.model.parameters(), 
      lr=lr,
      weight_decay=0
    )

    writer = SummaryWriter()

    # Construct inputs, labels
    if self.cfg.hyperparameters.masked:
      inputs = self.tokenizer(self.masked_tuning_data, return_tensors="pt", padding=True)
    else:
      inputs = self.tokenizer(self.tuning_data, return_tensors="pt", padding=True)

    labels = self.tokenizer(self.tuning_data, return_tensors="pt", padding=True)["input_ids"]

    self.model.train()

    set_seed(42)

    log.info("Fine-tuning model")
    with trange(epochs) as t:
      for epoch in t:

        optimizer.zero_grad()

        # Compute loss
        outputs = self.model(**inputs, labels=labels)
        loss = outputs.loss
        t.set_postfix(loss=loss.item())
        loss.backward()
        
        # Log results
        writer.add_scalar(f"loss/{self.model_bert_name}", loss, epoch)
        masked_input = self.tokenizer(
          self.masked_tuning_data, 
          return_tensors="pt", 
          padding=True
        )
        results = self.collect_results(masked_input, self.tokens_to_mask, outputs)

        sent_key = list(results.keys())[0]
        pos_key = list(results[sent_key].keys())[0]
        spec_results = results[sent_key][pos_key]["mean grouped log_probability"]

        for key in spec_results:
          writer.add_scalar(f"{key} LogProb/{self.model_bert_name}", spec_results[key], epoch)

        # GRADIENT ADJUSTMENT
        # 
        # The word_embedding remains unfrozen, but we only want to update
        # the embeddings of the novel tokens. To do this, we zero-out
        # all gradients except for those at these token indices.

        # Copy gradients of the relevant tokens
        nz_grad = {}
        for key in self.tokens_to_mask:
          tok = self.tokens_to_mask[key]
          tok_id = self.tokenizer(tok, return_tensors="pt")["input_ids"][:,1]
          grad = getattr(self.model, self.model_bert_name).embeddings.word_embeddings.weight.grad[tok_id, :].clone()
          nz_grad[tok_id] = grad
        
        # Zero out all gradients of word_embeddings in-place
        getattr(self.model, self.model_bert_name).embeddings.word_embeddings.weight.grad.data.fill_(0)

        # print(optimizer)
        # raise SystemExit

        # Replace the original gradients at the relevant token indices
        for key in nz_grad:
          getattr(self.model, self.model_bert_name).embeddings.word_embeddings.weight.grad[key, :] = nz_grad[key]
        
        optimizer.step()
        
        # Check that we changed the correct number of parameters
        new_embeddings = getattr(self.model, self.model_bert_name).embeddings.word_embeddings.weight.clone()
        sim = torch.eq(self.old_embeddings, new_embeddings)
        changed_params = int(list(sim.all(dim=1).size())[0]) - sim.all(dim=1).sum().item()

        exp_ch = len(list(self.tokens_to_mask.keys()))
        assert changed_params == exp_ch, f"Exactly {exp_ch} embeddings should have been updated, but {changed_params} were!"
        
    log.info("Saving model state dictionary.")
    torch.save(self.model.state_dict(), "model.pt")

    writer.flush()
    writer.close()

