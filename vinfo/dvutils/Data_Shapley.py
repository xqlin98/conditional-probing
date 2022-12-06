from collections import defaultdict
from math import comb
from tqdm import tqdm
import numpy as np

from dvutils.Adpt_Shapley import Adpt_Shapley
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from copy import deepcopy

from utils import InitYAMLObject

class Data_Shapley(Adpt_Shapley, InitYAMLObject):
    yaml_tag = '!Data_Shapley'
    
    def __init__(self, dataset, train_process, probe_model, list_model, reporter):
        Adpt_Shapley.__init__(self, probe_model, None, None, None)
        self.dataset = dataset
        self.train_process = train_process
        self.probe_model = probe_model
        self.list_model = list_model
        self.reporter = reporter
        self.sv_result = []
        self.init_model_weight = deepcopy(self.probe_model.state_dict())

        self.train_loader = None
        self.val_loader = None
        self.data_idx = None
    def tmc_sv_from_mem(self):
        """To compute sv with TMC from memory"""
        sv_list = defaultdict(list)
        for coalition in self.memory:
            for i, idx in enumerate(self.memory[coalition]):
                sv_list[idx].extend(self.memory[coalition][idx])
        sv_result = []
        for i in range(self.n_participants):
            sv_result.append(np.mean(sv_list[i]))
        self.sv_result = sv_result
        return sv_result

    def exact_sv_from_mem(self):
        """To compute the sv with exact algo from memory"""
        gamma_vec = [1/comb(self.n_participants-1, k) for k in range(self.n_participants)]

        sv_list = defaultdict(list)
        for coalition in self.memory:
            if len(coalition) != self.n_participants:
                for i, idx in enumerate(self.memory[coalition]):
                    set_size = len(coalition)
                    if set_size < self.n_participants:
                        weighted_marginal = self.memory[coalition][idx] * gamma_vec[set_size] * (1/self.n_participants)
                        sv_list[idx].append(weighted_marginal)
        sv_result = []
        for i in range(self.n_participants):
            sv_result.append(np.sum(sv_list[i]))
        self.sv_result = sv_result
        return sv_result
    
    def tmc_one_iteration(self, tolerance, metric, early_stopping=True):
        """Runs one iteration of TMC-Shapley algorithm."""
        idxs = np.random.permutation(self.n_participants)
        marginal_contribs = np.zeros(self.n_participants)
        truncation_counter = 0
        new_score = self.get_null_score()
        selected_idx = []
        for n, idx in tqdm(enumerate(idxs), leave=False):
        # for n, idx in enumerate(idxs):
            old_score = new_score
            selected_idx.append(self.data_idx[idx])

            tmp_loader = self.dataset.get_idx_dataloader(selected_idx)
            self.train_process.train_until_convergence(self.probe_model, self.list_model, None, tmp_loader, self.val_loader, gradient_steps_between_eval=2*len(tmp_loader))
            dev_predictions = self.train_process.predict(self.probe_model, self.list_model, self.val_loader)
            new_score = -self.reporter(dev_predictions, self.val_loader, None)
            
            marginal_contribs[idx] = (new_score - old_score)
            distance_to_full_score = np.abs(new_score - self.get_full_score())
            if distance_to_full_score <= tolerance * np.abs(self.get_full_score()-self.get_null_score()):
                truncation_counter += 1
                if truncation_counter > 1:
                    break
            else:
                truncation_counter = 0
            self.restart_model()
            
        self.tmc_record(idxs=idxs,
                        marginal_contribs=marginal_contribs)

    def get_null_score(self):
        """To compute the performance with initial weight"""
        try:
            self.null_score
        except:
            dev_predictions = self.train_process.predict(self.probe_model, self.list_model, self.val_loader)
            self.null_score = -self.reporter(dev_predictions, self.val_loader, "123")
        return self.null_score
    
    def get_full_score(self):
        """To compute the performance on grand coalition"""
        try:
            self.full_score
        except:
            self.train_process.train_until_convergence(self.probe_model, self.list_model, None, self.train_loader, self.val_loader, gradient_steps_between_eval=min(1000,len(self.train_loader)))
            dev_predictions = self.train_process.predict(self.probe_model, self.list_model, self.val_loader)
            self.full_score = -self.reporter(dev_predictions, self.val_loader, None)
            self.restart_model()
        return self.full_score

    def restart_model(self,seed=10):
        self.probe_model.load_state_dict(deepcopy(self.init_model_weight))
        # self.optimizer = self.optimizer_fn(self.model.parameters(), lr=self.lr)

    def run(self,
            data_idx,
            val_data_idx,
            method="tmc",
            iteration=2000,  
            tolerance=0.01,
            metric="accu",
            early_stopping=True):
            """Compute the sv with different method"""
            self.train_loader = self.dataset.get_idx_dataloader(data_idx, split="train")
            self.val_loader = self.dataset.get_idx_dataloader(val_data_idx, split="val")
            self.data_idx = data_idx
            self.n_participants = len(data_idx)
            
            self.metric=metric
            self.memory = defaultdict()
            if method == "tmc":
                for iter in tqdm(range(iteration), desc='[TMC iterations]'):
                    self.tmc_one_iteration(tolerance=tolerance, metric=metric, early_stopping=early_stopping)
                sv_result = self.tmc_sv_from_mem()
            elif method == "exact":
                self.exact_method(metric)
                sv_result = self.exact_sv_from_mem()
            return np.array(sv_result)
