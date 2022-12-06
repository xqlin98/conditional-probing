import pickle
import yaml as yaml
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import sys
import click
import os

import model
import dataset
import task
import cache
import probe
import trainer
import reporter
from utils import TRAIN_STR, DEV_STR, TEST_STR

ontonotes_fields = ["one_offset_word_index", "token", "None", "ptb_pos", "ptb_pos2", "None2", "dep_rel", "None3", "None4", "source_file", "part_number", "zero_offset_word_index", "token2", "ptb_pos3", "parse_bit", "predicate_lemma", "predicate_frameset_id", "word_sense", "speaker_author", "named_entities"]

from model    import * 
from dataset  import * 
from task     import * 
from cache    import * 
from probe    import * 
from trainer  import * 
from reporter import * 
from utils import *
from dvutils.Data_Shapley import *

@click.command()
@click.argument('yaml_path')
@click.option('--just-cache-data', default=0, help='If 1, just writes data to cache; does not run experiment')
@click.option('--do_test', default=0, help='If 1, evaluates on the test set; hopefully just run this once!')
def run_yaml_experiment(yaml_path, just_cache_data, do_test):
  """
  Runs an experiment as configured by a yaml config file
  """

  # Take constructed classes from yaml
  yaml_args = yaml.load(open(yaml_path), Loader=yaml.Loader)
  list_dataset = yaml_args['dataset']
  
  regimen_model = yaml_args['regimen']
  reporter_model = yaml_args['reporter']
  cache_model = yaml_args['cache']

  dshap_com = yaml_args['dshap_com']
  dshap_sing = yaml_args['dshap_sing']
  
  # Make results directory
  os.makedirs(regimen_model.reporting_root, exist_ok=True)

  # Make dataloaders and load data
  train_dataloader = list_dataset.get_train_dataloader(shuffle=False)
  dev_dataloader = list_dataset.get_dev_dataloader(shuffle=False)
  if do_test:
    test_dataloader = list_dataset.get_test_dataloader(shuffle=False)
  cache_model.release_locks()

  if just_cache_data:
    print("Data caching done. Exiting...")
    return

  # sample a set of data points to conduct data valuation
  np.random.seed(10)
  sample_num = 2
  val_sample_num = 300
  tmc_iterations = 1
  
  train_num = list_dataset.train_num
  val_num = list_dataset.dev_num
  sampled_idx = np.random.choice(np.arange(train_num), sample_num, replace=False).tolist()
  sampled_val_idx = np.random.choice(np.arange(val_num), val_sample_num, replace=False).tolist()
  
  # data Shapley with pretrained embedding + layer 0 embedding
  dshap_value_com = dshap_com.run(data_idx=sampled_idx, val_data_idx=sampled_val_idx, iteration=tmc_iterations)

  # data Shapley with layer 0 embedding only
  dshap_value_sing = dshap_sing.run(data_idx=sampled_idx, val_data_idx=sampled_val_idx, iteration=tmc_iterations)
  
  cond_dshap_value = dshap_value_com - dshap_value_sing
  # record the result
  result_dict = {'dshap_value_com': dshap_value_com,
                 'dshap_value_sing': dshap_value_sing,
                 'cond_dshap_value': cond_dshap_value,
                 'sampled_idx': sampled_idx}
  raw_selected_data = list_dataset.data_loader.sentence_raw_idx_extractor(sampled_idx)
  
  # record the raw data input for inspection
  raw_selected_data.to_csv(os.path.join(regimen_model.reporting_root, 'raw_selected_data.csv'),index=False)
  with open(os.path.join(regimen_model.reporting_root, 'result_dict.pickle'), 'wb') as handle:
      pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
  run_yaml_experiment()
