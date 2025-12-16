_model_instance = None

def get_model():
    global _model_instance
    if _model_instance is None:
        from .embed import ModelEmbedding
        print("Loading SapBERT model (this may take a few moments)...")
        _model_instance = ModelEmbedding()
        print("Model loaded successfully!")
    return _model_instance