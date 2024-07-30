class DataIngestion:
    
    def __init__(self, data_config):
        # Load params here
        # Load data from paths
        
        # List of preprocssings to do
        # [std_norm, sample_norm]
        '''
        data_config:

            sample_chunksize: int
            target: str
            
            data_splits:

                full_datapath: path
                tvt_split: **args

                train_datapath:
                val_datapath:
                test_datapath:

            preprocess:
                - name: SampleNorm
                  params: **args

                - name: StandardNorm
                  params: **args
        
        '''
        self.preprocess = []
        pass
    
    # Applicable when `full_datapath` has been provided
    def generate_train_val_test_split(self):
        
        pass
    
    # Expects data_config to have indivudal preprocess configs
    def preprocess_data(self):
        for preprocess_type in self.preprocess:
            # build preprocess class using the builder
            # execute the store_params on train data
            for chunked_data in full_data:
                # execute preprocessing on numpy array
                pass
            pass
        pass
    
    # create int mappings to str labels/targets
    def generate_mappings(self, column_name):
        # create a mapping dict
        pass
    
    def update_config(self, config):
        # update config paths
        return config