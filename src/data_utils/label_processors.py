"""Label processors handle the preprocessing of multilabel-sets to adapt them for using retrieval-like tasks."""
from typing import Tuple, Union, Dict, List, Any

from src.configs.constants import *

class LabelProcesser:
    """Parent label-processor for Ledgar and Eurlex label processors"""
    def __init__(
            self,
            pos_thres:float = 0.97,
            neg_thres:float = 0.9,
            min_similarity_matrix_pos:float =0.34,
            max_similarity_matrix_pos:float = 0.30,
            examples=None,
            seed:int=SEED,
            textname='text',
            labelname='label'
    ):
        self.pos_thres = pos_thres # jaccard similarity index max
        self.neg_thres = neg_thres # jaccard similarity index max
        self.min_similarity_matrix = min_similarity_matrix_pos # threshold the similarity matrix by this, else 0
        self.max_similarity_matrix = max_similarity_matrix_neg # threshold the similarity matrix by this
        #self.lemmatizer = WordNetLemmatizer()
        #self.stemmer = PorterStemmer()
        #self.stop_words = set(stopwords.words('english'))
        #self.random = np.random.RandomState(seed)
        self.label_corpus =None
        self.label2stem =None
        self.textname=textname
        self.labelname=labelname

        if examples is not None and len(examples)>0:

            # build corpus from examples
            label_corpus, label2stem = self.build_corpus_by_labels(examples)
            self.label_corpus = label_corpus
            self.label2stem = label2stem

            # build label-similarity matrix
            self.SimMat = self.compute_similarity_matrix(list(self.label_corpus.keys()))

    def preprocess_label(self, text):
        pass

    @staticmethod
    def jaccard_similarity(tokens1, tokens2):
        set1 = set(tokens1)
        set2 = set(tokens2)
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        similarity_score = len(intersection) / len(union)
        return similarity_score

    def build_corpus_by_labels(self, list_of_dict_with_labels_and_text):
        """Makes a dictionary of (tokenized/stemmed) labels:List[str] as the corpus by labels"""
        pass

    def _compute_similarity_for_processor_func(self, pair):
        """to be used internally with Pool map similarity functions"""
        idx, j, tokens1, tokens2 = pair
        return idx, j, self.jaccard_similarity(tokens1, tokens2)

    def compute_similarity_matrix(self, corpus):
        """Csompute similarity using calculate_similarity"""
        corpus_size = len(corpus)

        # Create an empty similarity matrix
        similarity_matrix = np.zeros((corpus_size, corpus_size))

        # Generate all pairwise combinations of indices and texts
        pairs = [(i, j, corpus[i], corpus[j]) for i in range(corpus_size) for j in range(i + 1, corpus_size)]

        # Use parallel processing to compute similarities efficiently
        with Pool() as pool:
            results = pool.map(self._compute_similarity_for_processor_func, pairs)

        # Fill in the similarity matrix
        for i,j, similarity in results:
            #i, j = divmod(idx, corpus_size)
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity

        # threshold the similarity matrx -- no, because that will creat positives in the negatives
        return similarity_matrix

    @staticmethod
    def is_in(tuple1, tuple2):
        """is a in b or b in a"""
        s1=set(tuple1); s2 = set(tuple2)
        if not bool(s1.difference(s2)):
            return True
        return not bool(s2.difference(s1))

    @staticmethod
    def _quick_text_hash(text):
        return re.sub("\W+","",text.lower())

    def find_positive(
        self,
        query_text, # text of anchor/query (used to ensure not too similar, like an exact match)
        query_labelstem, # processed label (often a multi-label)
        corpus_keys, # corpus keys of other labels to find matches
        max_candidates=15
    ):
        """find positive match, based on best overlap of multi-label"""
        # first, check if there are other text with same label
        query_label_hash = self._quick_text_hash(query_text)

        # get all text with same label
        best_candidates_text = [
            s for s in self.label_corpus[query_labelstem] if self._quick_text_hash(s)!=query_label_hash
        ]
        if len(best_candidates_text)==0:
            # no similar text: need to find text with overlapping labelss
            kidx = corpus_keys.index(query_labelstem)
            # get similarities with other keys
            k_similarities = self.SimMat[kidx]
            if k_similarities.max()==0:
                #print("%s has no matches:" % '-'.join(query_labelstem))
                return []
            else:
                idx_bests = np.argsort(-1*k_similarities)[:max_candidates]
                # get most similar labels
                label_candidates = [
                    corpus_keys[j] for j in idx_bests if k_similarities[j]>= self.min_similarity_matrix
                ]
                # assert that the labels are AT LEAST inside of each other -- otherwise, no match
                label_candidates = [
                    lab for lab in label_candidates if self.is_in(lab, query_labelstem)
                ]
                if len(label_candidates)==0:
                    #print("%s has no matches:" % '-'.join(query_labelstem))
                    return []

                # get the text of the top candidate text
                best_candidates_text = [subs for s in [
                    self.label_corpus[lab] for lab in label_candidates
                ] for subs in s][:100]

                # ensure candidate texts are not the same
                best_candidates_text = [
                  s for s in self.label_corpus[query_labelstem] if self._quick_text_hash(s)!=query_label_hash
                ]
                if len(best_candidates_text)==0:
                    #print("%s has no matches:" % '-'.join(query_labelstem))
                    return []

        # grab first candidate text htat is NOT a high jaccard similarity
        best_candidates_text = best_candidates_text[::-1]
        top_match = None
        query_text_tokenized = [w for w in query_text.split(" ") if bool(re.search("\w+",w))]
        while top_match is None and len(best_candidates_text)>0:
            candidate_text = best_candidates_text.pop()
            # check that they aren't too similar in text
            candidate_text_tokenized = [w for w in candidate_text.split(" ") if bool(re.search("\w+",w))]
            candidate_sim_score = self.jaccard_similarity(query_text_tokenized, candidate_text_tokenized)
            if candidate_sim_score < self.pos_thres:
                top_match = candidate_text
                return [top_match]
        #print("%s has no matches:" % '-'.join(query_labelstem))
        #print('Its candidate pool was:')
        #print(best_candidates_text[:4])
        return []

    def find_positives(self, examples):
        if True:
            # find positives
            for idx, example in enumerate(examples):
                pos = self.find_positive(
                    query_text=example[self.textname],
                    query_labelstem=self.label2stem[tuple(example[self.labelname])],
                    corpus_keys = list(self.label_corpus.keys()),
                )
                example.update({'positives':pos})
                examples[idx] = example

        return examples

    def find_negative(self, query_text, query_labelstem, corpus_keys, max_candidates=15, n_negatives=1):
        # first, check if there are other text with same label
        query_label_hash = self._quick_text_hash(query_text)
        # get similarities with other keys
        kidx = corpus_keys.index(query_labelstem)
        k_similarities = self.SimMat[kidx]
        if k_similarities.max()==0:
            best_candidate_label = query_labelstem
            while best_candidate_label == query_labelstem:
                #try:
                #    assert isinstance(corpus_keys,list), 'expected corpus_keys to be list: %s' % type(corpus_keys)
                #    assert all([isinstance(s,str) for s in corpus_keys])
                #except:
                #    print('Something wrong with corpus keys')
                #    print(corpus_keys)
                # randomly select keys for the best candidate
                idx_best_candidate_label = self.random.choice(np.arange(len(corpus_keys)))
                best_candidate_label = corpus_keys[idx_best_candidate_label]
        else:
            idx_bests = np.argsort(-1*k_similarities)[:max_candidates]
            # get most similar labels
            label_candidates = [
                corpus_keys[j] for j in idx_bests if (k_similarities[j]!=0 and k_similarities[j] <= self.max_similarity_matrix)
            ]
            # assert that the labels have some disjoint labels
            label_candidates = [
                lab for lab in label_candidates if not self.is_in(lab, query_labelstem)
            ] # disjoint entirely
            # sample randomly from candidate labels
            if len(label_candidates)>0:
                best_candidate_label_idx = self.random.choice(np.arange(len(label_candidates)))
                best_candidate_label = label_candidates[best_candidate_label_idx]
            # sample randomly from entire corpus
            elif len(label_candidates)==0:
                # pick random
                best_candidate_label = query_labelstem
                while best_candidate_label == query_labelstem:
                    best_candidate_label_idx = self.random.choice(np.arange(len(corpus_keys)))
                    best_candidate_label = corpus_keys[best_candidate_label_idx]

        # grab best text
        best_candidates_text = self.label_corpus[best_candidate_label]
        if len(best_candidates_text)==0:
            return []

        # ensure texts and query are not the same
        best_candidates_text = [
            s for s in best_candidates_text if self._quick_text_hash(s)!=query_label_hash
        ]
        if len(best_candidates_text)==0:
            return []

        # ensure texts are not very similar
        top_matches = []
        query_text_tokenized = [w for w in query_text.split(" ") if bool(re.search("\w+",w))]
        while len(top_matches) < n_negatives and len(best_candidates_text)>0:
            candidate_text = best_candidates_text.pop()
            # check that they aren't too similar in text
            candidate_text_tokenized = [w for w in candidate_text.split(" ") if bool(re.search("\w+",w))]
            candidate_sim_score = self.jaccard_similarity(query_text_tokenized, candidate_text_tokenized)
            if candidate_sim_score < self.neg_thres:
                top_matches.append(candidate_text)
                if len(top_matches)==n_negatives:
                    return top_matches
        # no matches
        return []

    def find_negatives(self, examples, n_negatives=1):
        if True:
            # find negatives
            for idx, example in enumerate(examples):
                neg = self.find_negative(
                    query_text=example[self.textname],
                    query_labelstem=self.label2stem[tuple(example[self.labelname])],
                    corpus_keys = list(self.label_corpus.keys()),
                    n_negatives=1
                )
                example.update({'negatives':neg})
                examples[idx] = example

        return examples


class LabelProcesserLedgar(LabelProcesser):
    """Preprocesses labels of LEDGAR for semantic similarity, as well as functionality for finding positive and negative pairs"""

    def __init__(
        self,
        pos_thres = 0.97,
        neg_thres = 0.9,
        min_similarity_matrix_pos =0.33,
        max_similarity_matrix_neg=0.3,
        examples=None,
        seed=42,
        textname='text',
        labelname='label'
    ):
        self.pos_thres = pos_thres # jaccard similarity index max
        self.neg_thres = neg_thres # jaccard similarity index max
        self.min_similarity_matrix = min_similarity_matrix_pos # threshold the similarity matrix by this, else 0
        self.max_similarity_matrix = max_similarity_matrix_neg # threshold the similarity matrix by this, else 0
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.random = np.random.RandomState(seed)
        self.label_corpus =None
        self.label2stem =None
        self.textname=textname
        self.labelname=labelname
        #print(self.preprocess_label("The Borrowers’ obligation"))
        #print(self.preprocess_label("The Borrower's obligations"))

        if examples is not None and len(examples)>0:

            # build corpus from examples
            label_corpus, label2stem = self.build_corpus_by_labels(examples)
            self.label_corpus = label_corpus
            self.label2stem = label2stem

            # build label-similarity matrix
            self.SimMat = self.compute_similarity_matrix(list(self.label_corpus.keys()))

    def preprocess_label(self, text):
        if isinstance(text,str):
            tokens = word_tokenize(text.lower())
            # Remove stop words
            filtered_tokens = [token for token in tokens if token not in self.stop_words]
            # Perform lemmatization and stemming
            processed_tokens = [self.lemmatizer.lemmatize(self.stemmer.stem(token)) for token in filtered_tokens]
            processed_tokens = [w for w in processed_tokens if w not in ["'", "’", "’s", "'s", "(",")", ",", "."]]
            # Return the lemmatized and stop word-free tokens as a string
            return sorted(processed_tokens)

        elif isinstance(text,list):
            if len(text)==1:
                return self.preprocess_label(text[0])
            all_labels = [self.preprocess_label(l) for l in text]
            return sorted([subl for l in all_labels for subl in l])
        else:
            raise NotImplementedError(text)

    def build_corpus_by_labels(self, list_of_dict_with_labels_and_text):
        """Makes a dictionary of (tokenized/stemmed) labels:List[str] as the corpus by labels"""
        label_corpus = {}
        label2lem = {}
        for example in list_of_dict_with_labels_and_text:
            label = example[self.labelname]
            s = example[self.textname]
            if tuple(label) not in label2lem:
                labelstemmed = tuple(self.preprocess_label(label))
                label2lem[tuple(label)] = labelstemmed
            else:
                labelstemmed = label2lem[tuple(label)]
            if labelstemmed not in label_corpus.keys():
                label_corpus[labelstemmed] = []
            if s not in label_corpus[labelstemmed]:
                label_corpus[labelstemmed].append(s)

        # next, calculate the similarities between all pairs of keys
        return label_corpus, label2lem


class DatasetTripletsSimilarityByCoLabel(DatasetTriplets):

    def process(self, list_of_data):
        """Makes (query,pos,neg)-triplets, converts samples to dataframe for pytorch iteration"""

        # initialize the LabelProcessor
        label_processor = self.label_processor_class(
            examples = list_of_data,
            textname = self.focal_text_name
        )

        # find positives
        list_of_data = label_processor.find_positives(list_of_data)

        # only do ones with positives (otherwise no point)
        #list_of_data = [example for example in list_of_data if len(example['positives'])>0]
        #print(len(list_of_data))

        # find negatives
        list_of_data = label_processor.find_negatives(list_of_data, n_negatives=self.n_negatives)
        print(len(list_of_data))

        # loop through the data and add each triplets
        self._loop_through_list_of_data_and_add_to_selfdata(list_of_data = list_of_data)

        # harden the dataset to pandas dataframe
        df = self.sample_data_and_make_static_dataframe(self.data)
        return df #pd.DataFrame({})

    def _build_corpus_of_potential_negatives(self):
        pass

    def _find_negative(self):
        pass

    def _find_positives_and_add_to_data(self):
        """For data that has a label, this can be used to artifically find and create synthetic positives"""
        pass

    def _find_negatives_and_add_to_data(self):
       pass



class LabelProcesserEurlex(LabelProcesser):
    """Preprocesses labels of EURLEX for semantic similarity, as well as functionality for finding positive and negative pairs"""

    def __init__(self, pos_thres = 0.97, neg_thres = 0.9, min_similarity_matrix_pos =0.33, max_similarity_matrix_neg =0.30,  examples=None, seed=42, textname='text',labelname='label'):
        self.pos_thres = pos_thres # jaccard similarity index max
        self.neg_thres = neg_thres # jaccard similarity index max
        self.min_similarity_matrix = min_similarity_matrix_pos # threshold the similarity matrix by this, else 0
        self.max_similarity_matrix = max_similarity_matrix_neg # threshold the similarity matrix by this, else 0
        self.random = np.random.RandomState(seed)
        self.label_corpus =None
        self.label2stem =None
        self.textname=textname
        self.labelname=labelname
        #print(self.preprocess_label("The Borrowers’ obligation"))
        #print(self.preprocess_label("The Borrower's obligations"))

        if examples is not None and len(examples)>0:

            # build corpus from examples
            label_corpus, label2stem = self.build_corpus_by_labels(examples)
            self.label_corpus = label_corpus
            self.label2stem = label2stem

            # build label-similarity matrix
            self.SimMat = self.compute_similarity_matrix(list(self.label_corpus.keys()))

    def preprocess_label(self, text):
        # eurlex labels are already "tokenized" into integers of concepts
        if isinstance(text,str):
            return text
        elif isinstance(text,list):
            if len(text)==1:
                return text
            return sorted(list(set(text)))
        else:
            raise NotImplementedError(text)

    def build_corpus_by_labels(self, list_of_dict_with_labels_and_text):
        """Makes a dictionary of (tokenized/stemmed) labels:List[str] as the corpus by labels"""
        label_corpus = {}
        label2lem = {}
        for example in list_of_dict_with_labels_and_text:
            label = example[self.labelname]
            s = example[self.textname]
            if tuple(label) not in label2lem:
                labelstemmed = tuple(self.preprocess_label(label))
                label2lem[tuple(label)] = labelstemmed
            else:
                labelstemmed = label2lem[tuple(label)]
            if labelstemmed not in label_corpus.keys():
                label_corpus[labelstemmed] = []
            if s not in label_corpus[labelstemmed]:
                label_corpus[labelstemmed].append(s)

        # next, calculate the similarities between all pairs of keys
        return label_corpus, label2lem
