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

#include "src/primihub/data_store/csv/mysql_driver.h"

#include "src/primihub/data_store/driver.h"

#include <arrow/api.h>
#include <arrow/csv/api.h>
#include <arrow/csv/writer.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/io/api.h>
#include <fstream>
#include <glog/logging.h>
#include <iostream>

namespace primihub {

// =========   Mysql Courser implementation =================
MysqlCursor::MysqlCursor(const std::string &sql, std::shared_ptr<MysqlDriver> driver)
:sql_(sql),dirver_(driver) {
      
}

std::shared_ptr<primihub::Dataset> MysqlCursor::read() {
    // TODO Query the database with sql statement
}

std::shared_ptr<primihub::Dataset> MysqlCursor::read(int64_t offset,
                                                     int64_t limit) {
    // TODO Query the database with sql statement and return limit records
}

int MysqlCursor::write(std::shared_ptr<primihub::Dataset> dataset) {
    // TODO Write the dataset into mysql table
}

void MysqlCursor::close() {
    // TODO close mysql connection

}

// =========   Mysql Driver implementation =================

MysqlDriver::MysqlDriver(const std::string &nodelet_addr)
    : DataDriver(nodelet_addr) {
    driver_type = "MYSQL";
}

std::shared_ptr<Cursor> &MysqlDriver::initCursor(const std::string &conn_str) {
    // TODO
    return nullptr;
}

std::shared_ptr<Cursor> &MysqlDriver::read(const std::string &conn_str) {
    // TODO
    return nullptr;
}

} // namespace primihub
