import logging
from pathlib import Path
import json
import pickle
from typing import Dict, List
from itertools import chain

import numpy as np
import faiss
import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from src.services.fullname_bigram_vectorizer import FullnamesBigramVectorizer
from src.const import (
    SEARCH_FIELD_NAME,
    EMBEDDINGS_DIM,
    RERANKER_NAME,
    DATASET_PATH,
    VOCABULARY_PATH,
    PROCESSED_DATASET_PATH,
    ANN_INDEX_PATH,
    BIGRAMS_MATCHES_NUM,
    RERANKER_THRESHOLD,
    ANN_PARAMS
)


class Matcher:
    """Class for searching fullname matches.
    
    Attributes:
        logger: Logging object.
        search_field_name: Name of field which describes fullname.
        embeddings_dim: Dimension of embeddings produced by reranker model.
        bigrams_matches_num: Number of matches to fetch for each candidate based ob bigrams vectors.
        reranker_threshold: Distance threshold for determination if match if is found or not.
        ann_params: Parameters for ANN index.
        reranker_name: Name of reranker model.
        dataset_path: Path to raw dataset.
        processed_dataset_path: Path to processed dataset.
        vocabulary_path: Path to vocabulary of bigrams.
        ann_index_path: Path to faiss index.
    """
    def __init__(self, mode: str) -> None:
        """Initializes a Matcher instance.
        
        Args:
            mode: Mode of model. Must be either train or eval.
        """
        self.logger = logging.getLogger('Matcher logger')
        self.search_field_name = SEARCH_FIELD_NAME
        self.embeddings_dim = EMBEDDINGS_DIM
        self.bigrams_matches_num = BIGRAMS_MATCHES_NUM
        self.reranker_threshold = RERANKER_THRESHOLD
        self.ann_params = ANN_PARAMS

        if mode not in ['train', 'eval']:
            raise ValueError('Matcher mode must be either train or eval.')
        
        self.reranker_name = RERANKER_NAME
        self.dataset_path = Path(DATASET_PATH)
        self.processed_dataset_path = Path(PROCESSED_DATASET_PATH)
        self.vocabulary_path = Path(VOCABULARY_PATH)
        self.ann_index_path = Path(ANN_INDEX_PATH)

        if mode == 'train' and not self.dataset_path.exists():
            raise ValueError(f'File for dataset does not exist: {self.dataset_path}')

        elif mode == 'eval':

            if not self.processed_dataset_path.exists():
                raise ValueError(f'File for processed dataset does not exist: {self.processed_dataset_path}')
            
            if not self.vocabulary_path.exists():
                raise ValueError(f'File for vocabulary does not exist: {self.vocabulary_path}')

            if not self.ann_index_path.exists():
                raise ValueError(f'File for ann index does not exist: {self.ann_index_path}')
            
        self.logger.info('Matcher has been initialized.')
            
    def train(self) -> None:
        """Train Matcher model."""
        self._process_dataset()
        self._train_vectorizer()
        self._prepare_ann_index()

        self.logger.info('Matcher has been trained.')

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

    def _train_vectorizer(self) -> None:
        """Trains bigram vectorizer and saves vocabulary of bigrams."""
        with open(self.processed_dataset_path, 'rb') as f:
            processed_dataset = pickle.load(f)

        fbv = FullnamesBigramVectorizer()
        fbv.fit(processed_dataset)
        vocabulary = fbv.vectorizer.vocabulary_

        with open(self.vocabulary_path, 'wb') as f:
            pickle.dump(vocabulary, f)

    def _prepare_ann_index(self) -> None:
        """Trains ANN index on bigram vectors."""
        with open(self.processed_dataset_path, 'rb') as f:
            processed_dataset = pickle.load(f)

        bigram_vectors = self._get_bigram_vectors(processed_dataset)

        centroids_num_fraction = self.ann_params['centroids_num_fraction']
        nprobe_num_fraction = self.ann_params['nprobe_num_fraction']

        vectors_num = len(bigram_vectors)
        centroids_num = int(vectors_num ** centroids_num_fraction)
        nprobe_num = int(centroids_num ** nprobe_num_fraction)

        vectors_dim = bigram_vectors.shape[1]

        index_name = self.ann_params['index_name'].format(centroids_num=centroids_num)
        index_metric = self.ann_params['index_metric']

        ann_index = faiss.index_factory(vectors_dim, index_name, index_metric)

        ann_index.train(bigram_vectors)
        ann_index.add(bigram_vectors)
        ann_index.nprobe = nprobe_num

        faiss.write_index(ann_index, str(self.ann_index_path))

        self.logger.info(f'ANN index has been saved to {self.ann_index_path}.')

        self.logger.info(f'Dataset has been processed and saved to {self.processed_dataset_path}.')

    def _get_bigram_vectors(self, processed_dataset: List[str]) -> np.array:
        """Transforms received dataset of fullnames to bigram vectors.
        
        Args:
            dataset: List of fullnames.

        Returns:
            Bigram vectors for dataset.
        """
        with open(self.vocabulary_path, 'rb') as f:
            vocabulary = pickle.load(f)

        fbv = FullnamesBigramVectorizer(vocabulary=vocabulary)
        bigram_vectors = fbv.transform(processed_dataset)
        return bigram_vectors


    def match(
        self,
        candidates: List[Dict[str, str]],
        matches_num: int = 1,
        use_threshold: bool = True
        ) -> Dict[str, str]:
        """Searches relevant matches for provided candidates.
        
        Args:
            candidates: A single fullname or list of fullnames.
            matches_num: Number of matches to return for each candidate.
            use_reranker_threshold: If it is necessary to use reranker threshold.

        Returns:
            List of matches for each candidate in input.
        """
        self.logger.info(f'Matching has been started.')

        processed_candidates = self._process_candidates(candidates)

        bigram_vectors = self._get_bigram_vectors(processed_candidates)

        match_indices = self._get_match_indices(bigram_vectors)
        matches = self._get_matches(match_indices)

        reranked_matches = self._rerank_matches(
            processed_candidates, matches, matches_num, use_threshold
        )

        candidates_matches = {}

        for candidate_name, matches in zip(processed_candidates, reranked_matches):
            candidate_name = candidate_name.title()
            if len(matches) == 0:
                response = 'Matches not found'
            else:
                response = list(map(str.title, matches))

            candidates_matches[candidate_name] = response

        self.logger.info(f'Matching procedure has been finished.')

        return candidates_matches
    
    def _process_candidates(self, candidates: List[Dict[str, str]]) -> List[str]:
        """Processes candidates for matching.
        
        Args:
            candidates: Candidates for matching.

        Returns:
            Processed candidates.
        """
        candidates = [item[self.search_field_name] for item in candidates]
        candidates = list(map(str.lower, candidates))

        return candidates
    
    def _get_match_indices(self, vectors: np.array) -> np.array:
        """Returns indices of matches for provided bigram vectors.
        
        Args:
            vectors: Bigram vectors for fullnames.

        Returns:
            Numpy array of indices.
        """
        ann_index = faiss.read_index(str(self.ann_index_path))
        _, indices = ann_index.search(vectors, self.bigrams_matches_num)

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

        self.logger.info(f'Matches have been found.')
        
        return matches
    
    def _rerank_matches(
        self,
        candidates: List[str],
        matches: List[List[str]], 
        matches_num: int, 
        use_reranker_threshold: bool
        ) -> List[List[str]]:
        """Reranks obtained matches by means of LLM model.
        
        Args:
            candidates: List of candidates for searching matches.
            matches: Matches obtained based on bigrams vectors.
            matches_num: Number of matches to return for each candidate.
            use_reranker_threshold: If it is necessary to use reranker threshold.

        Returns:
            List of reranked matches.
        """
        candidates_embeddings = self._get_embeddings(candidates)

        chained_matches = list(chain(*matches))
        matches_embeddings = self._get_embeddings(chained_matches)

        candidates_num = candidates_embeddings.shape[0]

        candidates_embeddings = candidates_embeddings.reshape(candidates_num, -1, self.embeddings_dim)
        matches_embeddings = matches_embeddings.reshape(candidates_num, -1, self.embeddings_dim)

        cosine_similarities = np.sum(candidates_embeddings * matches_embeddings, axis=2)

        reranked_matches_indices = np.argsort(cosine_similarities)[:, ::-1]
        reranked_matches = [[] for _ in range(candidates_num)]

        for candidate_idx, reranked_indices in enumerate(reranked_matches_indices):
            candidate_cosine_similarities = cosine_similarities[candidate_idx]

            if use_reranker_threshold:
                reranked_cosine_similarities = candidate_cosine_similarities[reranked_indices]
                reranked_indices = reranked_indices[reranked_cosine_similarities >= self.reranker_threshold]

            candidate_matches = matches[candidate_idx]
            reranked_matches[candidate_idx] = [candidate_matches[match_idx] for match_idx in reranked_indices][:matches_num]

        return reranked_matches
    
    def _get_embeddings(self, input: List[str]) -> np.array:
        """Transforms strings to embeddings.
        
        Args:
            input: List of strings.

        Returns:
            Numpy array of embeddings.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.reranker_name)
        encoder = AutoModel.from_pretrained(self.reranker_name)
        encoder.eval()

        batch_dict = tokenizer(input, max_length=512, padding=True, truncation=True, return_tensors='pt')

        with torch.no_grad():
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
