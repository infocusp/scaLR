class FeatureChunking(ScoringBase):
    
    def __init__(self, chunk_model_name, chunk_model_train_config, data_config):
        
        '''
        Feature seclection config has some user definable inputs
        But some have to be constrained by us, we set them here
        
        train_config <- feature selection config
        model_config <- feature selection config
        
        model_config[params] <- feature selection fixed params
            - single layers net [chunksize -> n_cls]
            - model_weights initialize with zero
        
        train_config[opt] = SGD
        train_config[callbacks] <- only nesecary defaults
        
        '''
        feature_selction_model_config = build_chunk_model_config
        feature_selction_train_config = build_chunk_train_config
        
        pass
    
    def build_chunk_model_config():
        pass
    
    def build_chunk_train_config():
        pass

    def train_model(data):
        return trained_model
    
    # If we decide to incorporate multi-layered net with SHAP to extract score
    def extract_classwise_feature_weights(model):
        return weights_dataframe
    
    def generate_scores():
        for feature_chunk in full_data:
            trained_model = trained_model(feature_chunk)
            scores = scorer(trained_model)
            scores.append(feature_weights)
            
        return weights
        
    
    