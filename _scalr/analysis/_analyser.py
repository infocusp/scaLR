class Analysis:
    
    def __init__(self):
        pass
    
    def analyse_and_store_results(self, dirpath):
        pass
    
    
def build_analyser(analysis_config):
    name = analysis_config['name']
    params = analysis_config['params']
    
    analyser = getattr(_scalr.analysis,name)(**params)
    return analyser