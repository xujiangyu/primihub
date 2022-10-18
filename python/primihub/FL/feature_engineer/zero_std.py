import primihub as ph
from primihub import dataset
import logging
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler


@ph.context.function(role='guest', protocol='Standariz', datasets=['data_to_std'], port='9020', task_type="Standardation")
def run_infer():

    logging.info("Start processing data.")
    predict_file_path = ph.context.Context.get_predict_file_path()

    stand_scale = StandardScaler()

    dataset_map = ph.context.Context.dataset_map

    data_key = list(dataset_map.keys())[0]

    data = ph.dataset.read(dataset_key=data_key).df_data

    standard_data = stand_scale.fit_transform(data)

    standard_data_df = pd.DataFrame(standard_data, columns=data.columns)


    dir_name = os.path.dirname(predict_file_path)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # standard_data_df.to_csv("/app/standard_data.csv", sep='\t')
    standard_data_df.to_csv(predict_file_path, sep='\t')

    logging.info("Ending processing data.")
