class AnalysisBase:
    
    def __init__(self):
        pass
    
    def generate_analysis(self, model, data, **kwargs):
        pass
    
    
def build_analyser(analysis_config):
    name = analysis_config['name']
    params = analysis_config.get('params', None)
    
    analyser = getattr(_scalr.analysis,name)(**params)
    return analyser