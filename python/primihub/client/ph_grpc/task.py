"""
 Copyright 2022 Primihub

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """
from primihub.client.ph_grpc.event import listener
from primihub.client.tiny_listener import Event
from primihub.client.ph_grpc.service import NodeServiceClient, NODE_EVENT_TYPE, NODE_EVENT_TYPE_TASK_STATUS, \
    NODE_EVENT_TYPE_TASK_RESULT


class Task(object):

    def __init__(self, task_id):
        self.task_id = task_id
        self.task_status = "PENDING"  # PENDING, RUNNING, SUCCESS, FAILED

    def get_status(self):
        pass

    def get_result(self):
        pass

    @listener.on_event("/task/{task_id}/%s" % NODE_EVENT_TYPE[NODE_EVENT_TYPE_TASK_STATUS])
    async def handler_task_status(self, event: Event):
        print("handler_task_status", event.params, event.data)
        # TODO
        # event data
        # {'event_type': 1,
        #      'task_status': {'task_context': {'task_id': '1',
        #                                       'job_id': 'task test status'
        #                                       }
        #                   'status': '' // ? TODO
        #                      }
        #      }
        node = ""  # TODO
        cert = ""  # TODO
        client_id = ""
        client_ip = 6666
        client_port = 12345
        # get node event from other nodes
        grpc_client = NodeServiceClient(node, cert)
        task_id = event.data["task_status"]["task_context"]["task_id"]
        await grpc_client.async_get_node_event(client_id=client_id, client_ip=client_ip, client_port=client_port)

    @listener.on_event("/task/{task_id}/%s" % NODE_EVENT_TYPE[NODE_EVENT_TYPE_TASK_RESULT])
    async def handler_task_result(self, event: Event):
        print("handler_task_result", event.params, event.data)
        # TODO
        # event data
        # {'event_type': 2,
        #  'task_result': {'task_context': {'task_id': '1',
        #                                   'job_id': 'task test result'},
        #                   'result_dataset_url': '' // ? TODO
        #                  }
        #  }