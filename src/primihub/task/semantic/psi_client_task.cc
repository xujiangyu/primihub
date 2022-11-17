/*
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
 */

#include "private_set_intersection/cpp/psi_client.h"

#include "src/primihub/task/semantic/psi_client_task.h"
#include "src/primihub/data_store/factory.h"
#include "src/primihub/util/file_util.h"


using arrow::Table;
using arrow::StringArray;
using arrow::DoubleArray;
//using arrow::Int64Builder;
using arrow::StringBuilder;
using primihub::rpc::VarType;
using std::map;

namespace primihub::task {

PSIClientTask::PSIClientTask(const std::string &node_id,
                             const std::string &job_id,
                             const std::string &task_id,
                             const TaskParam *task_param,
                             std::shared_ptr<DatasetService> dataset_service)
    : TaskBase(task_param, dataset_service), node_id_(node_id),
      job_id_(job_id), task_id_(task_id) {}

int PSIClientTask::_LoadParams(Task &task) {
    auto param_map = task.params().param_map();

    reveal_intersection_ = true;
    try {
        data_index_ = param_map["clientIndex"].value_int32();
        psi_type_ = param_map["psiType"].value_int32();
        dataset_path_ = param_map["clientData"].value_string();
        result_file_path_ = param_map["outputFullFilename"].value_string();
        auto it = param_map.find("sync_result_to_server");
        if (it != param_map.end()) {
            sync_result_to_server = it->second.value_int32();
            VLOG(5) << "sync_result_to_server: " << sync_result_to_server;
        }
        it = param_map.find("server_outputFullFilname");
        if (it != param_map.end()) {
            server_result_path = it->second.value_string();
            VLOG(5) << "server_outputFullFilname: " << server_result_path;
        }
        server_index_ = param_map["serverIndex"];
        server_address_ = param_map["serverAddress"].value_string();
        server_dataset_ = param_map[server_address_].value_string();

    } catch (std::exception &e) {
        LOG(ERROR) << "Failed to load params: " << e.what();
        return -1;
    }
    return 0;
}

int PSIClientTask::_LoadDatasetFromCSV(std::string &filename,
                        int data_col,
                        std::vector <std::string> &col_array) {
    std::string nodeaddr("test address"); // TODO
    std::shared_ptr<DataDriver> driver =
        DataDirverFactory::getDriver("CSV", nodeaddr);
    std::shared_ptr<Cursor> &cursor = driver->read(filename);
    std::shared_ptr<Dataset> ds = cursor->read();
    std::shared_ptr<Table> table = std::get<std::shared_ptr<Table>>(ds->data);

    int num_col = table->num_columns();
    if (num_col < data_col) {
        LOG(ERROR) << "psi dataset colunum number is smaller than data_col";
        return -1;
    }

    auto array = std::static_pointer_cast<StringArray>(
        table->column(data_col)->chunk(0));
    for (int64_t i = 0; i < array->length(); i++) {
        col_array.push_back(array->GetString(i));
    }
    return array->length();
}

int PSIClientTask::_LoadDataset(void) {
    int ret = _LoadDatasetFromCSV(dataset_path_, data_index_, elements_);
    // file reading error or file empty
    if (ret <= 0) {
        LOG(ERROR) << "Load dataset for psi client failed.";
        return -1;
    }
    return 0;
}

int PSIClientTask::_GetIntsection(const std::unique_ptr<PsiClient> &client,
                              ExecuteTaskResponse & taskResponse) {
    psi_proto::Response entrpy_response;
    const std::int64_t num_response_elements =
        static_cast<std::int64_t>(taskResponse.psi_response().encrypted_elements().size());
    for (std::int64_t i = 0; i < num_response_elements; i++) {
        entrpy_response.add_encrypted_elements(
            taskResponse.psi_response().encrypted_elements()[i]);
    }

    psi_proto::ServerSetup server_setup;
    server_setup.set_bits(taskResponse.psi_response().server_setup().bits());
    if (taskResponse.psi_response().server_setup().data_structure_case() ==
        psi_proto::ServerSetup::DataStructureCase::kGcs) {
        auto *ptr_gcs = server_setup.mutable_gcs();
        ptr_gcs->set_div(taskResponse.psi_response().server_setup().gcs().div());
        ptr_gcs->set_hash_range(taskResponse.psi_response().server_setup().gcs().hash_range());
    } else if (taskResponse.psi_response().server_setup().data_structure_case() ==
               psi_proto::ServerSetup::DataStructureCase::kBloomFilter) {
        auto *ptr_bloom_filter = server_setup.mutable_bloom_filter();
        ptr_bloom_filter->set_num_hash_functions(
            taskResponse.psi_response().server_setup().bloom_filter().num_hash_functions()
        );
    } else {
        LOG(ERROR) << "Node psi client get intersection error!";
        return -1;
    }

    std::vector <int64_t> intersection =
        std::move(client->GetIntersection(server_setup, entrpy_response)).value();

    const std::int64_t num_intersection =
        static_cast<std::int64_t>(intersection.size());

    if (psi_type_ == PsiType::DIFFERENCE) {
        map<int64_t, int> inter_map;
        std::int64_t num_elements = static_cast<std::int64_t>(elements_.size());
        for (std::int64_t i = 0; i < num_intersection; i++) {
            inter_map[intersection[i]] = 1;
        }
        for (std::int64_t i = 0; i < num_elements; i++) {
            if (inter_map.find(i) == inter_map.end()) {
                // outFile << i << std::endl;
                result_.push_back(elements_[i]);
            }
        }
    } else {
        for (std::int64_t i = 0; i < num_intersection; i++) {
            // outFile << intersection[i] << std::endl;
            result_.push_back(elements_[intersection[i]]);
        }
    }
    return 0;
}

int PSIClientTask::execute() {
    int ret = _LoadParams(task_param_);
    if (ret) {
        LOG(ERROR) << "Psi client load task params failed.";
        return ret;
    }

    ret = _LoadDataset();
    if (ret) {
        LOG(ERROR) << "Psi client load dataset failed.";
        return ret;
    }

    std::unique_ptr<PsiClient> client =
        std::move(PsiClient::CreateWithNewKey(reveal_intersection_)).value();
    psi_proto::Request client_request =
        std::move(client->CreateRequest(elements_)).value();
    psi_proto::Response server_response;

    grpc::ClientContext context;
    // ExecuteTaskRequest taskRequest;


    std::unique_ptr<VMNode::Stub> stub = VMNode::NewStub(grpc::CreateChannel(
        server_address_, grpc::InsecureChannelCredentials()));
    std::shared_ptr<grpc::ClientReaderWriter<ExecuteTaskRequest, ExecuteTaskResponse>> client_stream(stub->ExecuteTask(&context));

    size_t limited_size = 1 << 21;
    size_t num_elements = client_request.encrypted_elements().size();
    const auto& encrypted_elements = client_request.encrypted_elements();
    VLOG(5) << "num_elements_num_elements: " << num_elements;
    size_t sended_index{0};
    std::vector<ExecuteTaskRequest> send_requests;
    do {
        ExecuteTaskRequest taskRequest;
        PsiRequest* ptr_request = taskRequest.mutable_psi_request();
        ptr_request->set_reveal_intersection(client_request.reveal_intersection());
        size_t pack_size = 0;
        for (size_t i = sended_index; i < num_elements; i++) {
            const auto& element = encrypted_elements[i];
            auto element_len = element.size();
            if (pack_size + element_len > limited_size) {
                break;
            }
            ptr_request->add_encrypted_elements(element);
            pack_size += element_len;
            sended_index++;
        }
        auto *ptr_params = taskRequest.mutable_params()->mutable_param_map();
        ParamValue pv;
        pv.set_var_type(VarType::STRING);
        pv.set_value_string(server_dataset_);
        (*ptr_params)["serverData"] = pv;
        (*ptr_params)["serverIndex"] = server_index_;
        send_requests.push_back(std::move(taskRequest));
        if (sended_index >= num_elements) {
            break;
        }
    } while (true);
    VLOG(5) << "send_requests size: " << send_requests.size();
    // send request to server
    for (const auto& request : send_requests) {
        client_stream->Write(request);
    }
    client_stream->WritesDone();
    VLOG(5) << "client_stream->WritesDone";
    ExecuteTaskResponse taskResponse;
    ExecuteTaskResponse recv_response;
    auto psi_response = taskResponse.mutable_psi_response();
    bool is_initialized{false};
    while (client_stream->Read(&recv_response)) {
        const auto& _psi_response = recv_response.psi_response();
        if (!is_initialized) {
            const auto& res_server_setup = _psi_response.server_setup();
            auto server_setup = psi_response->mutable_server_setup();
            server_setup->set_bits(res_server_setup.bits());
            if (res_server_setup.data_structure_case() ==
                psi_proto::ServerSetup::DataStructureCase::kGcs) {
                auto ptr_gcs = server_setup->mutable_gcs();
                ptr_gcs->set_div(res_server_setup.gcs().div());
                ptr_gcs->set_hash_range(res_server_setup.gcs().hash_range());
            } else if (res_server_setup.data_structure_case() ==
                    psi_proto::ServerSetup::DataStructureCase::kBloomFilter) {
                auto ptr_bloom_filter = server_setup->mutable_bloom_filter();
                ptr_bloom_filter->set_num_hash_functions(
                    res_server_setup.bloom_filter().num_hash_functions()
                );
            }
            is_initialized = true;
        }
        for (const auto& encrypted_element : _psi_response.encrypted_elements()) {
            psi_response->add_encrypted_elements(encrypted_element);
        }
    }
    Status status = client_stream->Finish();
    if (status.ok()) {
        if (taskResponse.psi_response().ret_code()) {
            LOG(ERROR) << "Node psi server process request error.";
            return -1;
        }
        int ret = _GetIntsection(client, taskResponse);
        if (ret) {
            LOG(ERROR) << "Node psi client get insection failed.";
            return -1;
        }
        ret = saveResult();
        if (ret) {
            LOG(ERROR) << "Save psi result failed.";
            return -1;
        }
    } else {
        LOG(ERROR) << "Node push psi server task rpc failed.";
        LOG(ERROR) << status.error_code() << ": " << status.error_message();
        return -1;
    }
    if (this->reveal_intersection_ && this->sync_result_to_server) {
        send_result_to_server();
    }
    return 0;
}

int PSIClientTask::send_result_to_server() {
    grpc::ClientContext context;
    VLOG(5) << "send_result_to_server";
     // std::unique_ptr<VMNode::Stub>
    auto channel = grpc::InsecureChannelCredentials();
    auto stub = VMNode::NewStub(grpc::CreateChannel(server_address_, channel));
    primihub::rpc::TaskResponse task_response;
    std::unique_ptr<grpc::ClientWriter<primihub::rpc::TaskRequest>> writer(stub->Send(&context, &task_response));
    constexpr size_t limited_size = 1 << 22;  // limit data size 4M
    size_t sended_size = 0;
    size_t sended_index = 0;
    bool add_head_flag = false;
    do {
        primihub::rpc::TaskRequest task_request;
        task_request.set_job_id(this->job_id_);
        task_request.set_task_id(this->task_id_);
        task_request.set_storage_type(primihub::rpc::TaskRequest::FILE);
        task_request.set_storage_info(server_result_path);
        size_t pack_size = 0;
        if (!add_head_flag) {
            task_request.add_data("\"intersection_row\"");
            add_head_flag = true;
        }
        for (size_t i = sended_index; i < this->result_.size(); i++) {
            auto& data_item = this->result_[i];
            size_t item_len = data_item.size();
            if (pack_size + item_len > limited_size) {
                break;
            }
            task_request.add_data(data_item);
            pack_size += item_len;
            sended_index++;
        }
        writer->Write(task_request);
        sended_size += pack_size;
        VLOG(5) << "sended_size: " << sended_size << " "
                << "sended_index: " << sended_index << " "
                << "result size: " << this->result_.size();
        if (sended_index >= this->result_.size()) {
            break;
        }
    } while(true);
    writer->WritesDone();
    Status status = writer->Finish();
    if (status.ok()) {
        auto ret_code = task_response.ret_code();
        if (ret_code) {
            LOG(ERROR) << "client Node send result data to server return failed error code: " << ret_code;
            return -1;
        }
    } else {
        LOG(ERROR) << "client Node send result data to server failed. error_code: "
                   << status.error_code() << ": " << status.error_message();
        return -1;
    }
    VLOG(5) << "send result to server success";
}

int PSIClientTask::saveResult() {
    arrow::MemoryPool *pool = arrow::default_memory_pool();
    arrow::StringBuilder builder(pool);

    for (std::int64_t i = 0; i < result_.size(); i++) {
        builder.Append(result_[i]);
    }

    std::shared_ptr<arrow::Array> array;
    builder.Finish(&array);

    std::string col_title =
        psi_type_ == PsiType::DIFFERENCE ? "difference_row" : "intersection_row";
    std::vector<std::shared_ptr<arrow::Field>> schema_vector = {
        arrow::field(col_title, arrow::int64())};
    auto schema = std::make_shared<arrow::Schema>(schema_vector);
    std::shared_ptr<arrow::Table> table = arrow::Table::Make(schema, {array});

    std::shared_ptr<DataDriver> driver =
        DataDirverFactory::getDriver("CSV", "psi result");
    std::shared_ptr<CSVDriver> csv_driver =
        std::dynamic_pointer_cast<CSVDriver>(driver);


    if (ValidateDir(result_file_path_)) {
        LOG(ERROR) << "can't access file path: "
                   << result_file_path_;
        return -1;
    }
    int ret = csv_driver->write(table, result_file_path_);

    if (ret != 0) {
        LOG(ERROR) << "Save PSI result to file " << result_file_path_ << " failed.";
        return -1;
    }

    LOG(INFO) << "Save PSI result to " << result_file_path_ << ".";
    return 0;
}

} // namespace primihub::task
