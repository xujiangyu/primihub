import pickle
import json
import pandas as pd
import numpy as np
from sklearn import metrics
from primihub.new_FL.algorithm.utils.net_work import GrpcServers
from primihub.utils.evaluation import evaluate_ks_and_roc_auc, plot_lift_and_gain, eval_acc
from primihub.new_FL.algorithm.utils.base import BaseModel


class HeteroLrHostInfer(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_inputs()
        self.channel = GrpcServers(local_role=self.other_params.party_name,
                                   remote_roles=self.role_params['neighbors'],
                                   party_info=self.node_info,
                                   task_info=self.other_params.task_info)

    def set_inputs(self):
        # set common params
        self.model = self.common_params['model']
        self.task_name = self.common_params['task_name']
        self.metric_path = self.common_params['metric_path']
        self.model_pred = self.common_params['model_pred']

        # set role params
        self.data_set = self.role_params['data_set']
        self.id = self.role_params['id']
        self.selected_column = self.role_params['selected_column']
        self.label = self.role_params['label']
        self.model_path = self.role_params['model_path']

        # read from data path
        value = eval(self.other_params.party_datasets[
            self.other_params.party_name].data['data_set'])

        data_path = value['data_path']
        self.data = pd.read_csv(data_path)

    def load_dict(self):
        with open(self.model_path, "rb") as current_model:
            model_dict = pickle.load(current_model)

        self.weights = model_dict['weights']
        self.bias = model_dict['bias']
        self.col_names = model_dict['columns']
        self.std = model_dict['std']

    def preprocess(self):
        if self.label is not None:
            self.y = self.data.pop(self.label).values

        if len(self.col_names) > 0:
            self.data = self.data[self.col_names].values

        if self.std is not None:
            self.data = self.std.transform(self.data)

    def predict_raw(self, x):
        host_part = np.dot(x, self.weights) + self.bias
        guest_part = self.channel.recv("guest_part")[
            self.role_params['neighbors'][0]]
        h = host_part + guest_part

        return h

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def run(self):
        self.load_dict()
        self.preprocess()
        y_hat = self.predict_raw(self.data)
        pred_y = (self.sigmoid(y_hat) > 0.5).astype('int')

        pred_df = pd.DataFrame({'preds': pred_y, 'probs': self.sigmoid(y_hat)})
        pred_df.to_csv(self.model_pred, index=False, sep='\t')
        if self.label is not None:
            acc = sum((pred_y == self.y).astype('int')) / self.data.shape[0]
            ks, auc = evaluate_ks_and_roc_auc(self.y, self.sigmoid(y_hat))
            fpr, tpr, threshold = metrics.roc_curve(self.y, self.sigmoid(y_hat))

            evals = {
                "test_acc": acc,
                "test_ks": ks,
                "test_auc": auc,
                "test_fpr": fpr.tolist(),
                "test_tpr": tpr.tolist()
            }

            metrics_buff = json.dumps(evals)

            with open(self.metric_path, 'w') as filePath:
                filePath.write(metrics_buff)
            print("test acc is", evals)

    def get_summary(self):
        return {}

    def get_outputs(self):
        return {}

    def get_status(self):
        return {}


class HeteroLrGuestInfer(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_inputs()
        self.channel = GrpcServers(local_role=self.other_params.party_name,
                                   remote_roles=self.role_params['neighbors'],
                                   party_info=self.node_info,
                                   task_info=self.other_params.task_info)

    def set_inputs(self):
        self.model = self.common_params['model']
        self.task_name = self.common_params['task_name']

        # set role params
        self.data_set = self.role_params['data_set']
        self.id = self.role_params['id']
        self.selected_column = self.role_params['selected_column']
        self.label = self.role_params['label']
        self.model_path = self.role_params['model_path']

        # read from data path
        value = eval(self.other_params.party_datasets[
            self.other_params.party_name].data['data_set'])

        data_path = value['data_path']
        self.data = pd.read_csv(data_path)

    def load_dict(self):
        with open(self.model_path, "rb") as current_model:
            model_dict = pickle.load(current_model)

        self.weights = model_dict['weights']
        self.bias = model_dict['bias']
        self.col_names = model_dict['columns']
        self.std = model_dict['std']

    def preprocess(self):
        if self.label is not None:
            self.y = self.data.pop(self.label).values

        if len(self.col_names) > 0:
            self.data = self.data[self.col_names].values

        if self.std is not None:
            self.data = self.std.transform(self.data)

    def predict_raw(self, x):
        guest_part = np.dot(x, self.weights) + self.bias
        self.channel.sender("guest_part", guest_part)

    def run(self):
        self.load_dict()
        self.preprocess()
        self.predict_raw(self.data)

    def get_summary(self):
        return {}

    def get_outputs(self):
        return {}

    def get_status(self):
        return {}