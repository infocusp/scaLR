class DataIngestion:
    
    def __init__(self, data_config):
        # Load params here
        # Load paths
        
        # List of preprocssings to do
        # [std_norm, sample_norm]
        # To be read from config
        '''
        data_config:

            preprocess:
                - name: SampleNorm
                  params:
                        **args

                - name: StandardNorm
                  params: 
                        **arg
        
        '''
        self.preprocess = []
        pass
    
    # Applicable when `full_datapath` has been provided
    def generate_train_val_test_split(self):
        
        pass
    
    # Expects data_config to have indivudal preprocess configs
    def preprocess_data(self):
        for preprocess_type in self.preprocess:
            # preprocess_data
            pass
        pass
    
    # useful to append target or batch mappings from `str->int`
    # append directly to the obs of data
    # To decide in depth later
    def generate_mappings(self, column_name):
        # create a mapping dict
        pass
    
    def update_config(self, config):
        # update config paths
        return config