import primihub as ph
from primihub import dataset
import logging
import os


@ph.context.function(role='guest', protocol='fill_na', datasets=['data_with_null'], port='9020', task_type="ProcessNAs")
def process_na():

    logging.info("Start processing data with nas.")
    predict_file_path = ph.context.Context.get_predict_file_path()

    dataset_map = ph.context.Context.dataset_map

    data_key = list(dataset_map.keys())[0]

    data = ph.dataset.read(dataset_key=data_key).df_data

    data.fillna(data.median(), inplace=True)

    dir_name = os.path.dirname(predict_file_path)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    data.to_csv(predict_file_path, sep='\t')

    logging.info("Ending processing data with nas.")
