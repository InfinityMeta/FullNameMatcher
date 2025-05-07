import logging
from pathlib import Path
import json
import pickle
from typing import Dict, List, Union

import numpy as np
import faiss
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from src.const import (
    SEARCH_FIELD_NAME,
    EMBEDDINGS_DIM,
    ENCODER_NAME,
    DATASET_PATH,
    PROCESSED_DATASET_PATH,
    FAISS_INDEX_PATH
)


class Matcher:
    """Class for searching fullname matches.
    
    Attributes:
        logger: Logging object.
        search_field_name: Name of field which describes fullname.
        embeddings_dim: Dimension of embeddings produced by encoder model.
        encoder_path: Path to encoder model.
        dataset_path: Path to raw dataset.
        processed_dataset_path: Path to processed dataset.
        faiss_index_path: Path to faiss index.
    """
    def __init__(self, mode: str) -> None:
        """Initialize a Matcher instance.
        
        Args:
            mode: Mode of model. Must be either train or eval.
        """
        self.logger = logging.getLogger('Matcher logger')
        self.search_field_name = SEARCH_FIELD_NAME
        self.embeddings_dim = EMBEDDINGS_DIM

        if mode not in ['train', 'eval']:
            raise ValueError('Matcher mode must be either train or eval.')
        
        self.encoder_name = ENCODER_NAME
        self.dataset_path = Path(DATASET_PATH)
        self.processed_dataset_path = Path(PROCESSED_DATASET_PATH)
        self.faiss_index_path = Path(FAISS_INDEX_PATH)

        if mode == 'train' and not self.dataset_path.exists():
            raise ValueError(f'File for dataset does not exist: {self.dataset_path}')

        elif mode == 'eval':

            if not self.processed_dataset_path.exists():
                raise ValueError(f'File for processed dataset does not exist: {self.processed_dataset_path}')

            if not self.faiss_index_path.exists():
                raise ValueError(f'File for faiss index does not exist: {self.faiss_index_path}')
            
        self.logger.info('Matcher has been initialized.')
            
    def train(self) -> None:
        """Train Matcher model."""
        self._process_dataset()
        self._prepare_faiss_index()

        self.logger.info('Matcher has been trained.')

    def _prepare_faiss_index(self) -> None:
        """Trains faiss index on encoder embeddings."""
        with open(self.processed_dataset_path, 'rb') as f:
            processed_dataset = pickle.load(f)

        dataset_embeddings = self._get_embeddings(processed_dataset)

        nlist = 128
        quantizer = faiss.IndexFlatIP(self.embeddings_dim)
        faiss_index = faiss.IndexIVFFlat(quantizer, self.embeddings_dim, nlist)
        faiss_index.train(dataset_embeddings)
        faiss_index.add(dataset_embeddings)

        faiss.write_index(faiss_index, str(self.faiss_index_path))

        self.logger.info(f'Faiss index has been saved to {self.faiss_index_path}.')

    def _process_dataset(self) -> None: 
        """Process raw dataset: select only data that includes fullnames."""
        with open(self.dataset_path, 'r') as file:
            dataset = json.load(file)

        processed_dataset = []
        for item in dataset:
            item_value = item[self.search_field_name]
            if item_value is not None:
                processed_dataset.append(item_value.lower())

        filtered_items_num = len(dataset) - len(processed_dataset)
        if filtered_items_num > 0:
            self.logger.warning(f'{filtered_items_num} objects in dataset have not specified fullname.')

        with open(self.processed_dataset_path, 'wb') as f:
            pickle.dump(processed_dataset, f)

        self.logger.info(f'Dataset has been processed and saved to {self.processed_dataset_path}.')

    def match(self, candidates: List[Dict[str, str]], matches_num: int = 1) -> Dict[str, str]:
        """Searches relevant matches for provided candidates.
        
        Args:
            candidates: A single fullname or list of fullnames.
            matches_num: Number of matches to return for each candidate.

        Returns:
            List of matches for each candidate in input.
        """
        self.logger.info(f'Matching has been started.')

        candidates_names = [d[self.search_field_name] for d in candidates]

        processed_candidates = self._process_candidates(candidates)

        embeddings = self._get_embeddings(processed_candidates)

        match_indices = self._get_match_indices(embeddings, matches_num)
        matches = self._get_matches(match_indices)

        candidates_matches = {candidate_name: match for candidate_name, match in zip(candidates_names, matches)}

        self.logger.info(f'Matches has been successfully found.')

        return candidates_matches
    
    def _process_candidates(self, candidates: Union[str, List[str]]) -> Union[str, List[str]]:
        """Processes candidates for matching.
        
        Args:
            candidates: Candidates for matching.

        Returns:
            Processed candidates.
        """
        candidates = [item[self.search_field_name] for item in candidates]
        candidates = list(map(str.lower, candidates))

        return candidates
    
    def _get_match_indices(self, embeddings: np.array, matches_num: int = 1) -> np.array:
        """Returns indices of matches for provided embeddings.
        
        Args:
            embeddings: Encoded fullnames.
            matches_num: Number of matches to return for each candidate.

        Returns:
            Numpy array of indices.
        """
        faiss_index = faiss.read_index(str(self.faiss_index_path))
        faiss_index.nprobe = 16 
        _, indices = faiss_index.search(embeddings, matches_num)

        self.logger.info(f'Indices of matches have been found.')

        return indices

    def _get_matches(self, indices: np.array) -> List[List[str]]:
        """Returns matches from processed dataset by provided indices.
        
        Args:
            indices: Indices of matches.

        Returns:
            List of matches for each candidate.
        """
        with open(self.processed_dataset_path, 'rb') as f:
            dataset = pickle.load(f)

        matches = np.take(dataset, indices).tolist()

        matches = [[match.title() for match in lst] for lst in matches]

        self.logger.info(f'Matches have been found.')
        
        return matches
    
    def _get_embeddings(self, input: Union[str, List[str]]) -> np.array:
        """Transforms strings to embeddings.
        
        Args:
            input: A single string or list of strings.

        Returns:
            Numpy array of embeddings.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)
        encoder = AutoModel.from_pretrained(self.encoder_name)

        batch_dict = tokenizer(input, max_length=512, padding=True, truncation=True, return_tensors='pt')
        encoder_outputs = encoder(**batch_dict)
        embeddings = self.average_pool(encoder_outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        embeddings = embeddings.numpy()

        self.logger.info(f'Embeddings have been computed.')

        return embeddings
    
    @staticmethod
    def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]