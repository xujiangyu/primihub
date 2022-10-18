import primihub as ph
from primihub import dataset
import logging
import os


@ph.context.function(role='guest', protocol='fill_na', datasets=['data_with_null'], port='9020', task_type="ProcessNAs")
def process_na():

    logging("Start processing data with nas.")
    predict_file_path = ph.context.Context.get_predict_file_path()

    dataset_map = ph.context.Context.dataset_map

    data_key = list(dataset_map.keys())[0]

    data = ph.dataset.read(dataset_key=data_key).df_data

    data.fillna(data.median(), inplace=True)

    # data.to_csv("/app/na_rep_mean.csv", sep='\t')
    if not os.path.exists(predict_file_path):
        os.makedirs(predict_file_path)

    data.to_csv(predict_file_path, sep='\t')

    logging("Ending processing data with nas.")
