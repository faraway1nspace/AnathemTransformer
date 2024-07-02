"""Dataloaders for sampling data for huggingface Trainers."""
import os
import pandas as pd
from pandas import DataFrame
from rank_bm25 import BM25Okapi
import torch.utils.data as torch_data
from typing import Union, List, Dict, Optional, Any, Tuple

from src.configs.constants import *
from src.data_utils.preprocess import negative_example_generator


class DatasetTriplets(torch_data.Dataset):
    """Torch dataset for triplet-like data with anchor, pos, neg examples, like QA or STS."""
    def __init__(
            self,
            list_of_data=None,
            n_negatives:int = TRIPLETS_N_NEGATIVES,
            topk_negatives_discard:int = TRIPLETS_TOPK_NEGATIVES, # get top kth most-similar results, discard first k, to use as negative
            focal_text_name:str ='query',
            positives_text_name:str ='positives',
            negativess_text_name:str ='negatives',
            seed:int = SEED,
            negative_corpus_method:str = 'bm25', # how to sample (pseudo)negatives internally
            label_processor_class = None # (optional) function to process negatives
    ):
        self.n_negatives = n_negatives
        self.topk_negatives_discard = topk_negatives_discard
        self.data = {}
        self.focal_text_name =focal_text_name
        self.positives_text_name = positives_text_name
        self.negativess_text_name = negativess_text_name
        self.seed = seed
        self.random = np.random.RandomState(self.seed)
        self.label_processor_class = label_processor_class
        self.negative_corpus_method = negative_corpus_method
        assert negative_corpus_method in ['bm25','ann-tfidf']

        if list_of_data is not None and len(list_of_data)>0:

            # loop through the data and add each triplets: export a panda df as final data
            self.df = self.process(list_of_data)

    def process(self, list_of_data:list)->DataFrame:
        """Makes (query,pos,neg)-triplets, converts samples to dataframe for pytorch iteration"""

        # loop through the data and add each triplets
        self._loop_through_list_of_data_and_add_to_selfdata(
            list_of_data = list_of_data
        )

        # add positives to self.data
        self._find_positives_and_add_to_data()

        # add negatives to self.data
        self._find_negatives_and_add_to_data()

        # harden the dataset to pandas dataframe
        df = self.sample_data_and_make_static_dataframe(self.data)
        return df

    def _loop_through_list_of_data_and_add_to_selfdata(
        self,
        list_of_data
    ):
        """loops through and adds the positive/focal texts and negatives"""
        for raw_example in list_of_data:
            # add each element to the data
            self._add_triplet_to_data(
                focal_texts=raw_example[self.focal_text_name],
                positve_texts=raw_example[self.positives_text_name],
                negative_texts=raw_example[self.negativess_text_name],
            )
        self.focal_texts_as_keys = list(self.data.keys())

    def _add_triplet_to_data(
        self,
        focal_texts:Union[str,List[str]],
        positve_texts:List[str],
        negative_texts=List[str]
    )->None:
        """add focal text to the data"""
        do_add_focals = False
        if isinstance(focal_texts,list):
            focal_text = sort(focal_texts)[0]
            do_add_focals = True
        elif isinstance(focal_texts, str):
            focal_text = focal_texts
        if focal_text not in self.data.keys():
            self.data[focal_text] = {'positives':[], 'negatives':[]}
        self.data[focal_text]['positives'] += [p for p in positve_texts if p not in self.data[focal_text]['positives']]
        #if negative_texts is None:
        #    print(focal_texts)
        #    print(positve_texts)
        #    print(negative_texts)
        self.data[focal_text]['negatives'] += negative_texts if (negative_texts is not None) else []
        if do_add_focals:
            self.data[focal_text]['positives'] += focal_texts[1:]

    def _build_corpus_of_potential_negatives(self)->Dict[str,Any]:
        # grab positives as default negatives
        potential_corpus = [
            self.data[k]['positives'][:1] for k in self.focal_texts_as_keys
        ]
        # insert NEGATIVE if empty for an entry
        potential_corpus = [
            'NEGATIVE' if (not bool(s)) else s[0] for s in potential_corpus
        ]

        # negatives by BM25
        if self.negative_corpus_method == 'bm25':

            # tokenize for BM25
            print('building negatives via BM25')
            tokenized_corpus = [s.lower().split(" ") for s in potential_corpus]
            # compile BM25 corpus
            bm25 = BM25Okapi(tokenized_corpus)
            return {'retriever':bm25, 'corpus':potential_corpus}

        elif self.negative_corpus_method == 'ann-tfidf':
            print('building negatives via ANN-TFIDF')
            potential_corpus = [
                s for s in potential_corpus
                if len(s)>40 and len(s.split(" "))>10
            ]
            negative_example_generator= NegativeExampleGenerator(
                n_reps = 1, #
                tfidf_nfeatures = 4000,
                nchar_max_paragraph=3000,
                nword_max=100,
                nchar_max_word=4,
                save_cache = 'negative_sampler_%d-%s.pkl' % (len(potential_corpus), potential_corpus[0][0]),
                corpus = potential_corpus
            )
            return {'retriever':negative_example_generator, 'corpus':potential_corpus}

    def _find_negative(
        self,
        focal_text_as_query,
        positive_examples=None,
        use_focal_text = True,
        use_positives=True,
        neg_retriever=None,
        corpus = None
    ):
        """Given a query, uses BM25 to find similar but wrong answers, to serve as triplet negatives; for a single query"""
        bmquery = (focal_text_as_query if use_focal_text else "") + " " + ("" if (not use_positives) else positive_examples[0])
        bmquery = bmquery.strip()
        if self.negative_corpus_method == 'bm25':
            # make the query tokens
            bmquery_tokenized = bmquery.lower().split(" ")
            # search by BM25
            top_results = neg_retriever.get_top_n(
                bmquery_tokenized, corpus, n=self.topk_negatives_discard + self.n_negatives
            )
        elif self.negative_corpus_method == 'ann-tfidf':
            # query the ANN index
            top_results,_ = neg_retriever.find_negative(
                bmquery, k=self.n_negatives+2, skip=self.topk_negatives_discard
            )

        top_results = [
            s for s in top_results
            if (
                s not in positive_examples+[focal_text_as_query]
            )
        ]
        # remove any text that is equivalent to the query / focal texts
        potential_negatives = top_results[-1*self.n_negatives:]
        return potential_negatives

    def _find_positives_and_add_to_data(self):
        """For data that has a label, this can be used to artifically find and create synthetic positives"""
        pass

    def _find_negatives_and_add_to_data(self):
        """Uses BM25 to find similar but wrong answers, to serve as triplet negatives; loop over data"""

        # build bm25 corpus or tfidf-ANN index
        neg_corpus = self._build_corpus_of_potential_negatives()

        # loop through data, find examples which don't have negatives
        for k,d in self.data.items():
            if not bool(d['negatives']):
                negatives = self._find_negative(
                    focal_text_as_query=k,
                    positive_examples=d['positives'],
                    use_focal_text = True,
                    use_positives=bool(d['positives']),
                    neg_retriever=neg_corpus['retriever'],
                    corpus = neg_corpus['corpus']
                )
                d['negatives']+= negatives
        print('done finding negatives')

    def sample_data_and_make_static_dataframe(self, seed:int = SEED)->DataFrame:
        focals =[]
        pos =[]
        neg = []
        for query,d in self.data.items():
            for j in range(min(self.n_negatives, len(d['negatives']))):
                if len(d['positives'])==0:
                    continue
                dpos = [p for p in d['positives'] if bool(p.strip())]
                if not bool(dpos):
                    continue
                if len(dpos)==1:
                    pos+=dpos
                elif len(dpos)>1:
                    pos.append(self.random.choice(dpos))
                neg.append(d['negatives'][j])
                focals.append(query)
        df = pd.DataFrame({'query':focals, 'pos':pos, 'neg':neg})
        return df

    def add_dataset_to_self(self, other_dataset: Union['DatasetTriplets',List['DatasetTriplets']])->None:
        """Takes another instance of DatasetTriplets and adds to internal dataset."""
        if not isinstance(other_dataset, list):
            other_dataset = [other_dataset]
        # concatenate the other datasets to the self.df
        self.df = pd.concat([self.df]+[dataset.df for dataset in other_dataset], axis=0)

    def __len__(self)->int:
        return len(self.df)

    def __getitem__(self,idx)->dict:
        #key = self.focal_texts_as_keys[idx]
        #return {**{'query':key}, **self.data[key]}
        return self.df.iloc[idx].to_dict()


class DatasetTripletsSimilarityByCoLabel(DatasetTriplets):
    """DatasetTriplet but for datasets organized by multilabel labels (like Ledgar or Eurlex)."""
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
