# import torch
# import torch.nn.functional as F
# from transformers import AutoTokenizer, AutoModel


# class ModelEmbedding:

#     # def __init__(self, model_name:str ="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"):
#     #     """
#     #     Initialize the model and tokenizer.

#     #     :param model_name: Hugging Face model name.
#     #     """
#     #     self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#     #     self.model = AutoModel.from_pretrained(model_name, cache_dir="/Users/komalgilani/Desktop/chexo_knowledge_graph/data/models")
#     #     self.model.eval()  



#     _instance = None
#     _model = None
#     _tokenizer = None

#     def __new__(cls, *args, **kwargs):
#         if cls._instance is None:
#             cls._instance = super(ModelEmbedding, cls).__new__(cls)
#         return cls._instance

#     def __init__(self, model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"):
#         if self._model is None:
#             cache_dir = os.path.expanduser("~/.cache/huggingface/models")
#             self._tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
#             self._model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
#             self._model.eval()



#     def get_embedding(self, text: str) -> torch.Tensor:


#         """
#         Given a string 'text', returns a PyTorch tensor
#         representing its embedding using SAPBert.
#         """
#         inputs = self._tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#         with torch.no_grad():
#             outputs = self._model(**inputs)
            
#         # outputs.last_hidden_state has shape: [batch_size, sequence_length, hidden_size]
#         last_hidden_state = outputs.last_hidden_state
#         attention_mask = inputs["attention_mask"]
        
#         # Mean pooling: mask out padding tokens and average
#         mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
#         masked_embeddings = last_hidden_state * mask
#         sum_embeddings = torch.sum(masked_embeddings, dim=1)
#         sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
#         mean_embeddings = sum_embeddings / sum_mask
        
#         return mean_embeddings  # shape: [batch_size, hidden_size]
    

#     def embed_text(self, text: str) -> list:
#         """
#         Returns the embedding as a Python list of floats
#         (suitable for storing in Qdrant).
#         """
#         # get_embedding(text) -> shape: [1, hidden_size]
#         # we take the first row [0] -> shape: [hidden_size]
#         # then detach from graph, move to CPU, and convert to list
#         embedding_tensor = self.get_embedding(text)[0]
#         embedding_list = embedding_tensor.cpu().numpy().tolist()
#         return embedding_list

#     def calculate_similarity(self, label1: str, label2: str) -> float:
#         """
#         Computes and returns the cosine similarity between two labels
#         using the SAPBert model.
#         """
#         emb1 = self.get_embedding(label1)
#         emb2 = self.get_embedding(label2)
        
#         # Cosine similarity in PyTorch (returns a tensor of shape [1])
#         cos_sim = F.cosine_similarity(emb1, emb2)
#         return cos_sim.item()




# # if __name__ == "__main__":
# #     # Example usage
# #     model = ModelEmbedding()
  
    
# #     label1 = "edema"
# #     label2 = "grade of edema"
# #     similarity = model.calculate_similarity(label1, label2)
# #     print(f"Cosine similarity between '{label1}' and '{label2}': {similarity}")


import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class ModelEmbedding:
    _instance = None
    _model = None
    _tokenizer = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ModelEmbedding, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"):
        if self._model is None:
            cache_dir = os.path.expanduser("~/.cache/huggingface/models")
            self._tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/Users/komalgilani/Desktop/cmh/data/models")
            self._model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
            self._model.eval()

    def get_embedding(self, text: str) -> torch.Tensor:
        """
        Given a string 'text', returns a PyTorch tensor
        representing its embedding using SAPBert.
        """
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self._model(**inputs)
            
        # outputs.last_hidden_state has shape: [batch_size, sequence_length, hidden_size]
        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        
        # Mean pooling: mask out padding tokens and average
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked_embeddings = last_hidden_state * mask
        sum_embeddings = torch.sum(masked_embeddings, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        
        return mean_embeddings  # shape: [batch_size, hidden_size]

    def embed_text(self, text: str) -> list:
        """
        Returns the embedding as a Python list of floats
        (suitable for storing in Qdrant).
        """
        embedding_tensor = self.get_embedding(text)[0]
        embedding_list = embedding_tensor.cpu().numpy().tolist()
        return embedding_list

    def calculate_similarity(self, label1: str, label2: str) -> float:
        """
        Computes and returns the cosine similarity between two labels
        using the SAPBert model.
        """
        emb1 = self.get_embedding(label1)
        emb2 = self.get_embedding(label2)
        cos_sim = F.cosine_similarity(emb1, emb2)
        return cos_sim.item()


# if __name__ == "__main__":
#     model = ModelEmbedding()
#     label1 = "potassium in blood "
#     label2 = "potassium in serum/plasma"
#     similarity = model.calculate_similarity(label1, label2)
#     print(f"Cosine similarity between '{label1}' and '{label2}': {similarity}")