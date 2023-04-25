from primihub.new_FL.algorithm.utils.net_work import GrpcServer
# from primihub.new_FL.algorithm.utils.base import BaseModel
from primihub.new_FL.algorithm.utils.base_xus import BaseModel


class ExampleHost(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.common_params = kwargs['common_params']
        self.role_params = kwargs['role_params']
        self.node_info = kwargs['node_info']
        self.other_params = kwargs['other_params']

    def get_summary(self):
        """
        """
        return {}

    def set_inputs(self):
        """
        """

    def run(self):
        print("common_params: ", self.common_params)
        print("role_params: ", self.role_params)
        print("node_info: ", self.node_info)
        print("other_params: ", self.other_params)

    def get_outputs(self):
        return dict()

    def get_status(self):
        return {}