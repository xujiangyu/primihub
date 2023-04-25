import numpy as np
from sklearn import metrics
from collections import Iterable
from primihub.new_FL.algorithm.utils.net_work import GrpcServer, GrpcServers
from primihub.new_FL.algorithm.utils.base_xus import BaseModel


def dloss(p, y):
    z = p * y
    if z > 18.0:
        return np.exp(-z) * -y
    if z < -18.0:
        return -y
    return -y / (np.exp(z) + 1.0)


def batch_yield(x, y, batch_size):
    for i in range(0, x.shape[0], batch_size):
        yield (x[i:i + batch_size], y[i:i + batch_size])


def trucate_geometric_thres(x, clip_thres, variation, times=2):
    if isinstance(x, Iterable):
        norm_x = np.sqrt(sum(x * x))
        n = len(x)
    else:
        norm_x = abs(x)
        n = 1

    clip_thres = np.max([1, norm_x / clip_thres])
    clip_x = x / clip_thres

    dp_noise = None

    for _ in range(2 * times):
        cur_noise = np.random.normal(0, clip_thres * variation, n)

        if dp_noise is None:
            dp_noise = cur_noise
        else:
            dp_noise += cur_noise

    dp_noise /= np.sqrt(2 * times)

    dp_x = clip_x + dp_noise

    return dp_x


class HeteroLrBase(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def loss(self, y_hat, y_true):
        if self.loss_type == 'log':
            y_prob = self.sigmoid(y_hat)

            return metrics.log_loss(y_true, y_prob)

        elif self.loss == "squarederror":
            return metrics.mean_squared_error(
                y_true, y_hat)  # mse don't activate inputs
        else:
            raise KeyError('The type is not implemented!')

    def get_summary(self):
        """
        """
        return {}

    def set_inputs(self):
        # set common parameters
        self.model = self.kwargs['common_params']['model']
        self.task_name = self.kwargs['common_params']['task_name']
        self.learning_rate = self.kwargs['common_params']['learning_rate']
        self.alpha = self.kwargs['common_params']['alpha']
        self.epochs = self.kwargs['common_params']['epochs']
        self.penalty = self.kwargs['common_params']['penalty']
        self.optimal_method = self.kwargs['common_params']['optimal_method']
        self.momentum = self.kwargs['common_params']['momentum']
        self.random_state = self.kwargs['common_params']['random_state']
        self.scale_type = self.kwargs['common_params']['scale_type']
        self.batch_size = self.kwargs['common_params']['batch_size']
        self.sample_method = self.kwargs['common_params']['sample_method']
        self.sample_ratio = self.kwargs['common_params']['sample_ratio']
        self.loss_type = self.kwargs['common_params']['loss_type']
        self.prev_grad = self.kwargs['common_params']['prev_grad']
        self.model_path = self.kwargs['common_params']['model_path']
        self.metric_path = self.kwargs['common_params']['metric_path']
        self.model_pred = self.kwargs['common_params']['model_pred']

        # set role special parameters
        self.role_params = self.kwargs['role_params']

        # set party node information
        self.node_info = self.kwargs['node_info']

        # set other parameters
        self.other_params = self.kwargs['other_params']

    def run(self):
        pass

    def get_outputs(self):
        return dict()

    def get_status(self):
        return {}


class HeteroLrHost(HeteroLrBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.channel = GrpcServers(local_role=self.other_params.party_name,
                                   remote_roles=self.role_params['neighbors'],
                                   party_info=self.node_info,
                                   task_info=self.other_params.task_info)
        self.add_noise = self.role_params['add_noise']
        self.tol = self.role_params['tol']
        self.data_set = self.role_params['data_set']
        self.selected_column = self.role_params['selected_column']
        self.label = self.role_params['label']
        self.n_iter_no_change = self.role_params['n_iter_no_change']
