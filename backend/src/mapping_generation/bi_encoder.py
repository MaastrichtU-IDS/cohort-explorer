import os
from typing import List

import torch
from langchain_core.embeddings import Embeddings
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

from .param import *
from .utils import global_logger as logger

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
os.environ["TOKENIZERS_PARALLELISM"] = "True"


def get_device():
    torch.cuda.empty_cache()
    if torch.cuda.is_available() and torch.cuda.device_count() > 2:
        return torch.device("cuda:0")
    elif torch.cuda.is_available():
        return torch.device("cuda:1")
    else:
        return torch.device("cpu")


# def parse_tokens(text: str, semantic_type: bool = False, domain: str = None) -> Tuple[str, List[str], str]:
#     """Extract fields from formatted text using regular expressions."""

#     # Patterns to match entire fields accounting for possible internal commas
#     concept_name_pattern = re.compile(r'concept name:(.*?)(?=\s*, synonyms:|$)', re.IGNORECASE)
#     synonyms_pattern = re.compile(r'synonyms:(.*?)(?=\s*, domain:|$)', re.IGNORECASE)
#     domain_pattern = re.compile(r'domain:(.*?)(?=\s*, concept class:|$)', re.IGNORECASE)
#     concept_class_pattern = re.compile(r'concept class:(.*?)(?=\s*, vocabulary:|$)', re.IGNORECASE)
#     vocabulary_pattern = re.compile(r'vocabulary:(.*)', re.IGNORECASE)
#     # Extracting fields using regex
#     concept_name = concept_name_pattern.search(text)
#     synonyms = synonyms_pattern.search(text)
#     domain = domain_pattern.search(text)
#     concept_class = concept_class_pattern.search(text)
#     vocabulary = vocabulary_pattern.search(text)
#     if not concept_name or not synonyms:
#         logger.info(f"Required fields missing in text; returning whole text. Text: {text}")
#         return text, []
#     # Cleaning up results and logging
#     entity = concept_name.group(1).strip() if concept_name else 'Unknown'
#     synonyms_list = [syn.strip() for syn in synonyms.group(1).split(',')] if synonyms and synonyms.group(1) else []
#     domain_result = domain.group(1).strip() if domain else 'Unknown'
#     concept_class_result = concept_class.group(1).strip() if concept_class else 'Unknown'
#     vocabulary_result = vocabulary.group(1).strip() if vocabulary else 'Unknown'
#     # semantic_type = False
#     if semantic_type:
#         entity = entity + ' (' + domain_result + ':' + concept_class_result + ')'
#     logger.info(f"Entity: {entity}, synonyms: {synonyms_list}, vocabulary: {vocabulary_result}")
#     return entity, synonyms_list
# def parse_tokens(text: str, semantic_type: bool = False, domain: str = None) -> Tuple[str, List[str], str]:
#     """Extract entity and synonyms from the formatted text, if present."""
#     text_ = text.strip().lower()
#     print(f"text={text_}")
#     # Initialize default values
#     entity = None
#     synonyms = []
#     semantic_type_of_entity = None
#     # Check if the text follows the new format
#     if 'concept name:' in text_ and 'synonyms:' in text_ and 'domain:' in text_ and 'concept class:' in text_ and 'vocabulary:' in text_:
#         parts = text_.split(', ')
#         concept_dict = {}
#         for part in parts:
#             if ':' in part:
#                 key, value = part.split(':',1)
#                 concept_dict[key.strip()] = value.strip()
#             else:
#                 concept_dict['concept name'] = part.strip()
#         entity = concept_dict.get('concept name', None)
#         synonyms_str = concept_dict.get('synonyms', '')
#         synonyms = process_synonyms(synonyms_str, entity)
#         if semantic_type:
#             if domain:
#                 semantic_type_of_entity = f"{concept_dict.get('domain', '')}:{concept_dict.get('concept class', '')}"
#                 entity = f"{entity} ({semantic_type_of_entity})"
#         logger.info(f"Entity: {entity}, Synonyms: {synonyms}, Semantic Type of Entity: {semantic_type_of_entity}")
#     else:
#         # Fallback to the old format handling
#         if '<ent>' in text_ and '<syn>' in text_:
#             entity = text_.split('<ent>')[1].split('</ent>')[0]
#             synonyms_text = text_.split('<syn>')[1].split('</syn>')[0]
#             synonyms = process_synonyms(synonyms_text,entity)
#             logger.info(f"Entity: {entity}, Synonyms: {synonyms}")
#         else:
#             entity = text_
#             synonyms = []
#         # if "||" in entity:
#         #     entity_type_info = entity.split("||")
#         #     entity = ' '.join(entity_type_info[:-1])  # All but last part
#         #     if semantic_type:
#         #         if domain:
#         #             semantic_type_of_entity = entity_type_info[-1].replace(':',' ').strip()
#         #         else:
#         #             semantic_type_of_entity = entity_type_info[-1].split(':')[-1].strip()
#         #         logger.info(f"Entity: {entity}, Semantic Type of Entity: {semantic_type_of_entity}")

#     return entity, synonyms, None


def parse_text(text):
    text = text.strip().lower()
    # Attempt to extract 'ent' and 'syn' content using regular expressions
    ent_match = re.search(r"<ent>(.*?)</ent>", text)
    syn_match = re.search(r"<syn>(.*?)</syn>", text)
    domain = re.search(r"<domain>(.*?)</domain>", text)

    # Check if 'ent' match is found
    if ent_match:
        entity = ent_match.group(1)
        # Check if 'syn' match is found and it has content
        synonyms = syn_match.group(1).split(";;") if syn_match and syn_match.group(1) else None

        parent_term = domain.group(1) if domain and domain.group(1) else None
        if synonyms:
            synonyms = synonyms[:5]
        # print(f"entity={entity}, synonyms={synonyms}, domain={domain}")
        return (entity, synonyms, parent_term) if ent_match else text
    else:
        # If no 'ent' tag is found, return the whole text
        return text, None, None


import re


def combine_ent_synonyms(text: str) -> str:
    """Extract entity and synonyms from the formatted text, if present."""
    text_ = text.strip().lower()
    domain = None
    if "<desc>" in text_:
        description = text_.split("<desc>")[1].split("</desc>")[0]
        logger.info(f"Description: {description}")
        text_ = text_.split("<desc>")[0]

    elif "concept name:" in text_:
        text_ = re.search(r"concept name:([^,]+)", text_).group(1)
        # synonyms = re.search(r"synonyms:([^,]+)", description).group(1)
        domain = re.search(r"domain:([^,]+)", text_).group(1)
        # synonyms_list = synonyms.split(", ")
    return text_, domain


def process_synonyms(synonyms_text: str, entity: str) -> List[str]:
    """
    Processes the synonyms text to split by ';;' if it exists, otherwise returns the whole text as a single item list.
    If synonyms_text is empty, returns an empty list.
    """
    entity = entity.lower()
    if synonyms_text:
        if ";;" in synonyms_text:
            return [syn for syn in synonyms_text.split(";;") if syn != "" and syn.lower() != entity]
        else:
            return [synonyms_text]
    return []


# from adapters import AutoAdapterModel
# class SPECTEREmbeddings(Embeddings):
#     def __init__(self, model_id: str='allenai/specter2', device: str = None, **kwargs):
#         """
#         Initializes the Custom embedding class by loading the specified transformer model.

#         Parameters:
#             model_id (str): The model identifier from Hugging Face's transformer models.
#             device (str, optional): The device to run the model on ("cuda" or "cpu").
#                                     Defaults to automatically choosing CUDA if available.
#         """
#         self.device = get_device()
#         self.tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base',clean_up_tokenization_spaces=True)
#         self.model = AutoAdapterModel.from_pretrained('allenai/specter2_base', cache_dir=CACHE_DIR)
#         self.model.load_adapter(model_id, source="hf", load_as="specter2", set_active=True)
#         self.model  = self.model.to(self.device)
#         self.model.eval()
#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         """Embed multiple documents. For documents with synonyms, the embeddings of the entities and their synonyms are averaged. If only the entity is present, its embedding is used as is."""
#         embeddings = []
#         pbar = tqdm(total=len(texts), desc="Embedding Documents", unit="doc")
#         for text in texts:
#             text, domain = combine_ent_synonyms(text)
#             text = text + ", " + domain
#             embeddings.append(self.embed_query(text))
#             pbar.update(1)
#         return embeddings
#     def embed_query(self, text_batch: str) -> List[float]:
#         torch.cuda.empty_cache()
#         inputs = self.tokenizer(text_batch, padding=True, truncation=True,
#                                    return_tensors="pt", return_token_type_ids=False, max_length=25)
#         inputs = {key: value.to(self.device) for key, value in inputs.items()}

#         output = self.model(**inputs)
#         embeddings = output.last_hidden_state[:, 0, :].squeeze(0)
#         #convert into list of float
#         embedding_list =embeddings.cpu().tolist()
#         return  embedding_list


class MedCPT_Embeddings(Embeddings):
    def __init__(self, model_id: str = "ncbi/MedCPT-Query-Encoder", device: str = None, **kwargs):
        """
        Initializes the Custom embedding class by loading the specified transformer model.

        Parameters:
            model_id (str): The model identifier from Hugging Face's transformer models.
            device (str, optional): The device to run the model on ("cuda" or "cpu").
                                    Defaults to automatically choosing CUDA if available.
        """
        self.device = torch.device(device if device else "cuda:2" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, clean_up_tokenization_spaces=True)
        self.model = AutoModel.from_pretrained(model_id, cache_dir=CACHE_DIR)
        self.model.to(self.device)
        self.model.eval()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        with torch.no_grad():
            # tokenize the queries
            encoded = tokenizer(
                texts,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=64,
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            # encode the queries (use the [CLS] last hidden states as the representations)
            # check if  model(**encoded) has last_hidden_state
            outputs = self.model(**encoded)
            if hasattr(outputs, "last_hidden_state"):
                embeds = outputs.last_hidden_state[:, 0, :]
            elif hasattr(outputs, "hidden_states"):
                embeds = outputs.hidden_states[-1][:, 0, :]  # Get the last hidden state
            else:
                raise ValueError("Model output does not contain last hidden state or hidden states")
            embeddings = embeds.cpu().tolist()
            return embeddings

    def embed_query(self, text: str) -> List[float]:
        with torch.no_grad():
            encoded = tokenizer(
                [text],
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=64,
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            outputs = self.model(**encoded)
            if hasattr(outputs, "last_hidden_state"):
                embeds = outputs.last_hidden_state[:, 0, :]
            elif hasattr(outputs, "hidden_states"):
                embeds = outputs.hidden_states[-1][:, 0, :]  # Get the last hidden state
            else:
                raise ValueError("Model output does not contain last hidden state or hidden states")
            embeds = embeds.squeeze(0)
            return embeds.cpu().tolist()


class SAPEmbeddings(Embeddings):
    """
    A class for creating embeddings using a specified transformer model based on Hugging Face's implementations.
    cambridgeltl/SapBERT-from-PubMedBERT-fulltext or xlreator/biosyn-biobert-snomed"""

    def __init__(self, model_id: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext", device: str = None, **kwargs):
        """
        Initializes the Custom embedding class by loading the specified transformer model.

        Parameters:
            model_id (str): The model identifier from Hugging Face's transformer models.
            device (str, optional): The device to run the model on ("cuda" or "cpu").
                                    Defaults to automatically choosing CUDA if available.
        """
        self.device = get_device()
        self.model = AutoModel.from_pretrained(
            model_id, trust_remote_code=False, cache_dir=CACHE_DIR, attn_implementation="eager", output_attentions=True
        )
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, do_lower_case=True, cache_dir=CACHE_DIR)
        self.model.eval()  # Ensure the model is in evaluation mode
        self.semantic_type = kwargs.get("semantic_type", False)
        self.include_domain = kwargs.get("include_domain", False)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents by averaging the embeddings of entities and their synonyms."""
        embeddings = []
        # pbar = tqdm(total=len(texts), desc="Embedding Documents", unit="doc")
        for text in texts:
            entity, synonyms, _ = parse_text(text)
            entity_embedding = self.embed_text(entity)  # Ensure it's two-dimensional for stacking
            if synonyms:
                synonym_embeddings = [self.embed_text(syn) for syn in synonyms]
                all_embeddings = torch.cat([entity_embedding] + synonym_embeddings, dim=0)
            else:
                all_embeddings = entity_embedding

            # print(f"Entity Embedding: {entity_embedding.shape}")
            mean_embedding = torch.mean(all_embeddings, dim=0)  # Average the embeddings
            # print(f"Mean Entity Embedding: {entity_embedding.shape}")
            embeddings.append(mean_embedding.cpu().tolist())  # Convert tensor to list
            # pbar.update(1)

        # pbar.close()
        return embeddings

    def embed_text(self, text: str) -> torch.Tensor:
        """Embed a single text using the [CLS] token embedding."""
        encoded_input = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(
            self.device
        )
        with torch.no_grad():
            outputs = self.model(**encoded_input)
            return outputs.last_hidden_state[:, 0, :]  # Take the [CLS] token embedding

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using the same approach as for documents."""
        entity_embedding = self.embed_documents([text])[0]
        return entity_embedding

    # def embed_documents(self, texts: List[str], agg_mode="mean_all_tok") -> List[List[float]]:
    #     """
    #     Embeds a batch of documents into dense representations using Hugging Face pipeline for feature extraction.

    #     Parameters
    #     ----------
    #     texts : List[str]
    #         A list of document texts to be embedded.
    #     agg_mode : str
    #         The aggregation mode for pooling the embeddings ('cls', 'mean_all_tok').

    #     Returns
    #     -------
    #     embeddings : List[List[float]]
    #         A list of embeddings for the documents.
    #     """

    #     self.model.eval()
    #     # Use Hugging Face pipeline to extract token embeddings for each document
    #     new_texts = []
    #     for text in texts:
    #         entity, synonyms, parent_term = parse_text(text.lower())
    #         if synonyms and parent_term:
    #             synonyms.append(entity)
    #             synonyms.append(parent_term)
    #             text = ','.join(synonyms)
    #             new_texts.append(text)
    #         else:
    #             new_texts.append(text)

    #     all_embs = []

    #     # Tokenize the batch of texts
    #     toks = self.tokenizer.batch_encode_plus(
    #         texts,  # Processing multiple samples at once
    #         padding="max_length",
    #         add_special_tokens=True,
    #         max_length=25,
    #         truncation=True,
    #         return_tensors="pt"  # Return as PyTorch tensors
    #     )

    #     # Move tokenized data to the GPU
    #     toks_cuda = {k: v.to(self.device) for k, v in toks.items()}

    #     # Get CLS representation as the embedding for each input
    #     with torch.no_grad():  # Disable gradients since this is inference
    #         cls_rep = self.model(**toks_cuda)[0][:, 0, :]  # CLS token is at index 0

    #     # cls_rep is now of shape (batch_size, 768)
    #     # print(f"shape of cls_rep: {cls_rep.shape}")  # Should be (batch_size, 768)

    #     # Convert each tensor to a list of floats and append to all_embs
    #     all_embs = cls_rep.cpu().detach().numpy().tolist()
    #     print(f"is it list of list of floats: {isinstance(all_embs[0], list)}")
    #     # Store embeddings in a TSV file if needed
    #     store_embedding_tsv_file(all_embs)

    #     return all_embs  # Return list of list of floats
    # Iterate over your input texts
    # for i in tqdm(np.arange(0, len(new_texts))):
    # Tokenize the current batch of texts
    # toks = self.tokenizer.batch_encode_plus(
    #     new_texts,  # Processing one sample at a time
    #     padding="max_length",
    #     add_special_tokens=True,
    #     max_length=25,
    #     truncation=True,
    #     return_tensors="pt"
    # )
    # # Move tokenized data to the GPU
    # toks_cuda = {k: v.to(self.device) for k, v in toks.items()}

    # # Get CLS representation as the embedding
    # cls_rep = self.model(**toks_cuda)[0][:, 0, :]  # CLS token is at index 0
    # cls_rep = cls_rep.squeeze(0)  # Remove the batch dimension
    # print(f"shape of cls_rep: {cls_rep.shape}")
    # # Convert the tensor to a list of floats and append to all_embs
    # all_embs.append(cls_rep.cpu().detach().numpy().tolist())
    # store_embedding_tsv_file(all_embs)
    # return all_embs
    # embeddings = self.pipeline(new_texts)
    # # for i, emd in enumerate(embeddings):
    # #     print(f"shape of embeddings: {np.array(emd).shape}")
    # # Apply pooling to get a single vector for each document
    # pooled_embeddings = []
    # for embedding in embeddings:
    #     embedding = np.array(embedding)
    #     embedding = np.squeeze(embedding, axis=0)
    #     print(f"shape of embedding: {embedding.shape}")
    #     if agg_mode == "cls":
    #         # CLS token (first token) pooling
    #         pooled_embedding = embedding[0]  # CLS token is the first token
    #     elif agg_mode == "mean_all_tok":
    #         # Mean pooling across all tokens
    #         pooled_embedding = np.mean(embedding, axis=0)
    #     else:
    #         raise ValueError(f"Unsupported aggregation mode: {agg_mode}")
    #     pooled_embedding = pooled_embedding.tolist()
    #     print(f"length of pooled_embedding: {len(pooled_embedding)}")
    #     pooled_embeddings.append(pooled_embedding)
    # print(f"is it list of list: {isinstance(pooled_embeddings[0], list)}")
    # return pooled_embeddings

    # def embed_documents(self, texts: List[str], show_progress=True, agg_mode="cls") -> List[List[float]]:
    #     """
    #     Embeds a batch of documents into dense representations.

    #     Parameters
    #     ----------
    #     texts : List[str]
    #         A list of document texts to be embedded (already batched).
    #     show_progress : bool
    #         Whether to display a progress bar.
    #     agg_mode : str
    #         The aggregation mode for pooling the embeddings ('cls', 'mean_all_tok', 'mean', or 'none').

    #     Returns
    #     -------
    #     embeddings : List[List[float]]
    #         A list of embeddings for the documents.
    #     """
    #     self.model.eval()  # Ensure the model is in evaluation mode

    #     # Tokenize the entire batch of documents
    #     batch_tokenized = self.tokenizer.batch_encode_plus(
    #         texts, add_special_tokens=True,
    #         truncation=True, max_length=25,
    #         padding="max_length", return_tensors='pt'
    #     )

    #     batch_tokenized_cuda = {k: v.to(self.device) for k, v in batch_tokenized.items()}

    #     with torch.no_grad():
    #         # Pass the tokenized input through the model
    #         last_hidden_state = self.model(**batch_tokenized_cuda).last_hidden_state

    #         # Aggregation mode for embeddings
    #         if agg_mode == "cls":
    #             embeddings = last_hidden_state[:, 0, :]  # CLS token
    #         elif agg_mode == "mean_all_tok":
    #             embeddings = last_hidden_state.mean(1)  # Mean pooling of all tokens
    #         elif agg_mode == "mean":
    #             embeddings = (last_hidden_state * batch_tokenized_cuda['attention_mask'].unsqueeze(-1)).sum(1) / batch_tokenized_cuda['attention_mask'].sum(-1).unsqueeze(-1)
    #         elif agg_mode == "none":
    #             embeddings = last_hidden_state  # No pooling, use all token embeddings
    #         else:
    #             raise ValueError(f"Unsupported aggregation mode: {agg_mode}")

    #     # Move the embeddings to CPU and convert to list
    #     embeddings = embeddings.cpu().detach().numpy()
    #     store_embedding_tsv_file(embeddings)
    #     return embeddings.tolist()

    # # def embed_documents(self, texts: List[str]) -> List[List[float]]:
    # #     embeddings = []
    # #     pbar = tqdm(total=len(texts), desc="Embedding Documents", unit="doc")
    # #     logger.info(f"Embedding documents\n{texts}")
    # #     for text in texts:
    # #         entity_embedding = self.embed_text(text).squeeze(0)
    # #         # text_embeddings = [entity_embedding]
    # #         # entity_embedding = torch.mean(torch.stack(text_embeddings), dim=0).squeeze(0)
    # #         embeddings.append(entity_embedding.cpu().tolist())  # This should be a list of 768 elements
    # #         pbar.update(1)
    # #     pbar.close()
    # #     return embeddings

    # def embed_synonyms(self,texts: List[str]) -> List[List[float]]:

    #     toks = self.tokenizer.batch_encode_plus(
    #             texts, add_special_tokens=True,
    #             truncation=True, max_length=None,  # Remove or adjust max length
    #             padding="max_length", return_tensors='pt'
    #         )
    #     toks_cuda = {}
    #     for k,v in toks.items():
    #         toks_cuda[k] = v.cuda()
    #     cls_rep = model(**toks_cuda)[0][:,0,:] # use CLS representation as the embedding
    #     return cls_rep
    # def embed_text(self, text: str) -> torch.Tensor:
    #     """Embed a single text using the [CLS] token embedding."""
    #     encoded_input = self.tokenizer(text, return_tensors='pt', padding="max_length", truncation=True,add_special_tokens=True
    #                                    ).to(self.device)
    #     with torch.no_grad():
    #         outputs = self.model(**encoded_input)
    #     cls_embedding = outputs.last_hidden_state[:, 0, :]  # Take the [CLS] token embedding
    #     return cls_embedding # Take the [CLS] token embedding

    # def save_txt_file(self, total_token_count):
    #     """
    #     Save the total token count to a text file.
    #     """
    #     with open('/workspace/mapping_tool/data/output/total_token_count_agreegated_ent_synonym_embed.txt', 'w') as f:
    #         f.write(str(total_token_count))

    # def embed_query(self, text: str) -> List[float]:
    #     """Embed a single query using the same approach as for documents."""
    #     entity_embedding = self.embed_documents([text])[0]
    #     # logger.info(f"Final Embedding: {entity_embedding.shape}")
    #     return entity_embedding


# SPARSE MODEL
model_name = "naver/splade-cocondenser-ensembledistil"
tokenizer = AutoTokenizer.from_pretrained(
    model_name, cache_dir=CACHE_DIR, use_fast=True, max_length=25, truncation=True
)
model = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=CACHE_DIR)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


def sparse_encoder(text: str) -> tuple[list[int], list[float]]:
    entity, _, parent_term = parse_text(text)
    if parent_term:
        entity = entity + ", " + parent_term

    # entity =  entity + ", " + domain
    # logger.info(f"Entity: {entity}")
    query_vec, _ = compute_vector(text=entity, tokenizer=tokenizer, model=model, device=device)
    query_indices = query_vec.nonzero().cpu().numpy().flatten()
    query_values = query_vec.detach().cpu().numpy()[query_indices]
    query_indices = list(query_indices)
    query_values = list(query_values)
    # logger.info(f"Query Indices: {len(query_indices)}")
    return query_indices, query_values


def compute_vector(text, tokenizer, model, device):
    """
    Computes a vector from logits and attention mask using ReLU, log, and max operations.
    """

    tokens = tokenizer(text, return_tensors="pt", max_length=25, truncation=True)  # Ensure truncation
    # Tokenize the input text and return PyTorch tensors
    tokens = {k: v.to(device) for k, v in tokens.items()}  # Move tensors to the device

    # Pass the tokenized input through the pre-trained language model
    with torch.no_grad():
        output = model(**tokens)

    # Extract logits and attention mask from the model output
    logits, attention_mask = output.logits, tokens["attention_mask"]

    # Apply ReLU and log operations to the logits
    relu_log = torch.log(1 + torch.relu(logits))

    # Weight the logits using the attention mask
    weighted_log = relu_log * attention_mask.unsqueeze(-1)

    # Perform max pooling operation along the sequence dimension
    max_val, _ = torch.max(weighted_log, dim=1)

    # Squeeze the result to obtain the final vector
    vec = max_val.squeeze()

    # Return the computed vector and the tokens
    return vec, tokens


def store_embedding_tsv_file(embeddings: List[List[float]]):
    """
    Store the embeddings in a TSV file.
    """
    with open("/workspace/mapping_tool/data/output/embeddings_aggregated_ent_synonym.tsv", "a") as f:
        for emb in embeddings:
            f.write("\t".join(map(str, emb)) + "\n")
