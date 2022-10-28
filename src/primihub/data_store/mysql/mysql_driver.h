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

#ifndef SRC_PRIMIHUB_DATA_STORE_MYSQL_MYSQL_DRIVER_H_
#define SRC_PRIMIHUB_DATA_STORE_MYSQL_MYSQL_DRIVER_H_

#include "src/primihub/data_store/dataset.h"
#include "src/primihub/data_store/driver.h"

namespace primihub { 

class MysqlDriver;

class MysqlCursor : public Cursor {
public: 
    MysqlCursor(const char* sql, std::shared_ptr<MysqlDriver> driver);
    ~MysqlCursor();
    std::shared_ptr<primihub::Dataset> read() override;
    std::shared_ptr<primihub::Dataset> read(int64_t offset, int64_t limit);
    int write(std::shared_ptr<primihub::Dataset> dataset) override;
    void close() override;

private:
    std::string sql_;
    unsigned long long offset = 0;
    std::shared_ptr<MysqlDriver> driver_;
}; // class MysqlCursor


class MysqlDriver: public DataDriver {
public:
  explicit MysqlDriver(const std::string &nodelet_addr);
  ~MysqlDriver() {}

  std::shared_ptr<Cursor> &initCursor(const std::string &conn_str) override;
  std::shared_ptr<Cursor> &read(const std::string &conn_str) override;
  
  std::string getDataURL() const override;
  int write(std::shared_ptr<arrow::Table> table, std::string &conn_str);

private:
    std::string conn_str_;

}; // class MysqlDriver


} // namespace primihub


#endif  // SRC_PRIMIHUB_DATA_STORE_MYSQL_MYSQL_DRIVER_H_

