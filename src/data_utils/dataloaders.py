"""Dataloaders for sampling data for huggingface Trainers."""
import os
import pandas as pd
from pandas import DataFrame
from rank_bm25 import BM25Okapi
import torch.utils.data as torch_data
from typing import Union, List, Dict, Optional, Any, Tuple, Callable

from src.configs.constants import *
from src.dataclasses import (
    TaskDataPerEpoch, MLMDataPerEpoch, NextSentenceDataPerEpoch, QADataPerEpoch, CLSDataPerEpoch, STSDataPerEpoch
)
from src.data_utils.preprocess import negative_example_generator
from src.data_utils.label_processors import LabelProcesser, LabelProcesserLedgar, LabelProcesserEurlex


class DatasetTriplets(torch_data.Dataset):
    """Torch dataset for triplet-like data with anchor, pos, neg examples, like QA or STS."""
    def __init__(
            self,
            list_of_data=List[Any],
            n_negatives:int = TRIPLETS_N_NEGATIVES,
            topk_negatives_discard:int = TRIPLETS_TOPK_NEGATIVES, # get top kth most-similar results, discard first k, to use as negative
            focal_text_name:str ='query',
            positives_text_name:str ='positives',
            negativess_text_name:str ='negatives',
            seed:int = SEED,
            negative_corpus_method:str = 'bm25', # how to sample (pseudo)negatives internally
            label_processor_class:Optional[LabelProcesser] = None # (optional) function to process negatives
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

    def _add_dataset_to_self(
            self,
            other_dataset: Union['DatasetTriplets',List['DatasetTriplets']],
    )->None:
        """Takes another instance of DatasetTriplets and adds to internal dataset."""
        if not isinstance(other_dataset, list):
            other_dataset = [other_dataset]
        # concatenate the other datasets to the self.df
        self.df = pd.concat([self.df]+[dataset.df for dataset in other_dataset], axis=0)

    def extend(self, other_dataset: Union['DatasetTriplets',List['DatasetTriplets']])->None:
        self._add_dataset_to_self(other_dataset)

    def __len__(self)->int:
        return len(self.df)

    def __getitem__(self,idx)->dict:
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



class DatasetPairClassification(torch_data.Dataset):
    """Torch dataset for PairClassification or MultiLabel Classification type data."""
    def __init__(
        self,
        list_of_data=None,
        text1_name ='pair1',
        text2_name ='pair2',
        label_name = 'label',
        datasetname_name = 'cls_id',
        classificationtype_name = 'type',
        nlabels_name = 'n_labels',
        seed = 42
    ):
        self.data = {} # internal data preprocessed
        self.datasets = [] # list of names of datasets in Dataset class
        self.label2int = {} #maps {label:int} dictionary
        self.label2dataset = {} #maps {label:mask}
        self.label2mask = {}
        self.dataset_classification_types = {} # dataset types (pair-classification, classificaiton)
        self.text1_name = text1_name
        self.text2_name = text2_name
        self.label_name = label_name#'label',
        self.datasetname_name = datasetname_name #'cls_id',
        self.classificationtype_name = classificationtype_name#'type',
        self.nlabels_name = nlabels_name #'n_labels'
        self.seed = seed

        # random state
        self.np_random = np.random.RandomState(seed)

        if list_of_data is not None and len(list_of_data)>0:

            # loop through the data and add each triplets: export a panda df as final data
            self.df = self.process(list_of_data, False)

    def process(self, list_of_data, inplace=True):
        """convert the raw examples to dataset"""
        # loop through the data and add each triplets
        self._loop_through_list_of_data_and_add_to_selfdata(
            list_of_data = list_of_data
        )

        # add positives to self.data
        self._find_positives_and_add_to_data()

        # add negatives to self.data
        self._find_negatives_and_add_to_data()

        # make mask for loss function
        self._convert_labelint_to_vectors()

        # make mask for loss function
        self._make_mask()

        # harden the dataset to pandas dataframe
        data_flatten = self.flatten_data(self.data)
        if not inplace:
            return data_flatten
        self.df = data_flatten

    def _loop_through_list_of_data_and_add_to_selfdata(
            self,
            list_of_data
        ):
        """loops through and adds the text pair and label"""
        for raw_example in list_of_data:

            # add each element to the data
            self._add_unit_to_data(
                text1 = raw_example[self.text1_name],
                text2= raw_example[self.text2_name],
                label= raw_example[self.label_name],
                n_labels= raw_example[self.nlabels_name],
                dataset_name= raw_example[self.datasetname_name],
                method = raw_example[self.classificationtype_name]
            )

    def _find_positives_and_add_to_data(self):
        """Finds data with the same label, and adds them as positives"""
        which_clsdatasets_lack_positives = [
            datasetname for datasetname, datasettype
            in self.dataset_classification_types.items()
            if datasettype == 'classification'
        ]
        for datasetname in which_clsdatasets_lack_positives:

            # all unique labels in subdataset
            ulabels_in_clsdataset = sorted(list(set([
                (d['class'],d['label']) for d in self.data[datasetname]
                if d['label'] == self.label2int['%s_%d' % (datasetname, 1)]
            ])))

            # loop through label classes
            for labelclass,label in ulabels_in_clsdataset:

                # other samples with the same class (and positive)
                # `class` is the original dataset class, label = {different, same}
                idx_this_class = [
                    i for i,d
                    in enumerate(self.data[datasetname])
                    if d['class'] == labelclass and d['label']==label
                ]

                idx_this_class_need_positives = [
                    i for i,d
                    in enumerate(self.data[datasetname])
                    if d['class'] == labelclass and d['label']==label
                    and d['text2'] is None
                ]

                # subsample within by permutation
                idx_sample_within = self.np_random.permutation(idx_this_class)

                # get text of permuted-indicies, assign as positive for each sample
                for i,j in zip(idx_this_class_need_positives, idx_sample_within[:len(idx_this_class_need_positives)]):

                    self.data[datasetname][i]['text2'] = self.data[datasetname][j]['text1']

    def _find_negatives_and_add_to_data(self):
        """Finds data with the same label, and adds them as positives"""
        which_clsdatasets_lack_negatives = [
            datasetname for datasetname, datasettype
            in self.dataset_classification_types.items()
            if datasettype == 'classification'
        ]
        for datasetname in which_clsdatasets_lack_negatives:

            # all unique labels in subdataset
            ulabels_in_clsdataset = sorted(list(set([
                (d['class'],d['label']) for d in self.data[datasetname]
                if d['label'] == self.label2int['%s_%d' % (datasetname, 0)]
            ])))

            # loop through label classes
            for labelclass,label in ulabels_in_clsdataset:

                # other samples with the same class (and positive)
                # `class` is the original dataset class, label = {different, same}
                idx_this_class = [
                    i for i,d
                    in enumerate(self.data[datasetname])
                    if d['class'] == labelclass and d['label']==label
                ]
                # indices of all other data
                idx_this_other_class = [
                    i for i,d
                    in enumerate(self.data[datasetname])
                    if d['class'] != labelclass
                ]

                # subsample within by permutation
                idx_sample_otherlabels= self.np_random.choice(idx_this_other_class, size =len(idx_this_class))

                # get text of permuted-indicies, assign as positive for each sample
                for i,j in zip(idx_this_class, idx_sample_otherlabels):

                    self.data[datasetname][i]['text2'] = self.data[datasetname][j]['text1']

    def _convert_labelint_to_vectors(self):
        """Loops through data and converts each labelinteger into a vector for multi-label loss"""
        for datasetname, dataset in self.data.items():
            for example in dataset:
                example.update({
                    'labelvector':self._convert_labelint_to_vector([example['label']])
                })

    def _convert_labelint_to_vector(self, labelints):
        """Loops through data and converts each labelinteger into a vector for multi-label loss"""
        v = np.zeros(len(self.label2int))
        for labelint in labelints:
            v[labelint]=1
        return v

    def _make_mask(self):
        """for each sample, the loss should only pertain to labels within the same dataset, not other datasets -- by masking"""
        if (
            len(self.label2mask)!=self.label2dataset
        ) or bool(
            set(list(self.label2mask.keys())).symmetric_differnce(set(list(self.label2dataset.keys())))
        ):
            # make the self.label2mask
            for label,dataset in self.label2dataset.items():
                #
                self.label2mask[self.label2int[label]] = self._convert_labelint_to_vector([
                    self.label2int[l] for l,dset in self.label2dataset.items() if dset==dataset
                ])

        # loop through data and insert mask into each sample
        for datasetname, dataset in self.data.items():
            for example in dataset:
                example.update({
                    'mask':self.label2mask[example['label']]
                })

    def _add_labels_to_label2int(self, dataset_labels_as_globalname, dataset_name):
        for globallabel in dataset_labels_as_globalname:
            if globallabel not in self.label2int.keys():
                next_label_int = len(self.label2int)
                self.label2int[globallabel] = next_label_int
                self.label2dataset[globallabel] = dataset_name

    def _add_unit_to_data(
        self,
        text1,
        text2,
        label,
        n_labels,
        dataset_name,
        method
    ):
        """Adds one unit of processed data to the internal self.data"""
        if method == 'pair_classification':

            # pair classification: two texts with a label of the relationship between pair
            self._add_text_pair_to_data(
                text1,
                text2,
                label,
                n_labels,
                dataset_name
            )

        elif method == 'classification':

            # classification: single texts, with negatives needing to be deduced later
            self._add_textclass_to_data(
                text1,
                label,
                dataset_name
            )

    def _add_text_pair_to_data(
        self,
        text1,
        text2,
        label,
        n_labels,
        dataset_name
    ):
        """add a text pair to the self data: specifically for pair_classification"""
        if dataset_name not in self.data.keys():
            print('encountered new dataset for pair-classification: %s' % dataset_name)
            self.data[dataset_name] = []
            self.datasets += [dataset_name]
            self.dataset_classification_types[dataset_name] = 'pair_classification'

            # common naming for all labels across all datasets
            dataset_labels_as_globalname = [
                "%s_%d" % (dataset_name, l) for l in range(n_labels)
            ]

            self._add_labels_to_label2int(dataset_labels_as_globalname, dataset_name)

        if text2 is not None:

            self.data[dataset_name].append({
                'text1':text1,
                'text2':text2,
                'label':self.label2int["%s_%d" % (dataset_name, label)],
                'mask':None
            })

    def _add_textclass_to_data(
        self,
        text,
        classlabel,
        dataset_name
    ):
        """add a text to the self data: specifically for classification"""
        if dataset_name not in self.data.keys():
            print('encountered new dataset for classification: %s' % dataset_name)
            self.data[dataset_name] = []
            # register dataset to list of datasets
            self.datasets += [dataset_name]
            # map datset classification types
            self.dataset_classification_types[dataset_name] = 'classification'

            # common naming for all labels across all datasets
            dataset_labels_as_globalname = [
                "%s_%d" % (dataset_name, l) for l in [0,1]
            ]

            self._add_labels_to_label2int(dataset_labels_as_globalname, dataset_name)

        # positives and negatives must be added seperately (same label, different label)
        for label in [0,1]:

            self.data[dataset_name].append({
                'text1':text,
                'text2':None,
                'mask':None,
                'label':self.label2int["%s_%d" % (dataset_name, label)],
                'class':classlabel
            })

    def flatten_data(self, data):
        """Converts data to a giant list"""
        data_all_flat = []
        for datasetname, subdataset in self.data.items():
            data_all_flat += subdataset
        return data_all_flat

    def _integrate_another_dataset(
        self,
        list_of_newdata:list,
        function_to_reformatdata:Callable,
        dataset_name:str
    ):
        """
        Adds new data to the existing self.df
        Arguments:
        :param list_of_newdata: list of data to integrate/add
        :function_to_reformatdata: function that converts the data in list_of_newdata[idx]
        """
        # check that the unit of data has the required fields
        newtestdata = function_to_reformatdata(list_of_newdata[0])
        assert isinstance(newtestdata, list), 'function_to_reformatdata must output a list of reformated data'
        assert not bool(set(['text1','text2','class']).difference(set(newtestdata[0].keys()))), 'new data must have `text1`, `text2`,`class`'
        classlabels = set()
        newdata_converted_all = []

        # loop through and convert all data to acceptable format for internal datasets
        for newdata in list_of_newdata:
            newdata_converted = function_to_reformatdata(newdata)
            for unit in newdata_converted:
                classlabels |= set([unit['class']])
            newdata_converted_all += newdata_converted

        # loop through and ingest all converted data into the self.data internal dataset
        for unit in newdata_converted_all:
            self._add_text_pair_to_data(
                text1=unit['text1'],
                text2=unit['text2'],
                label=unit['class'],
                n_labels=len(classlabels),
                dataset_name=dataset_name
            )
        # remake the mask for ALL data, given the new label sets
        self._convert_labelint_to_vectors()
        # make mask for loss function
        self._make_mask()
        # harden the dataset to pandas dataframe
        data_flatten = self.flatten_data(self.data)
        self.df = data_flatten
        print('done integrating new dataset %s' % dataset_name)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df[idx]


class MLMDataset(torch_data.Dataset):
    """Dataset for MLM task, returning string."""
    def __init__(self,text:List[str]):
        self.df = text
    
    def __len__(self)->int:
        return len(self.df)
    
    def __getitem__(self,idx)->str:
        return self.df[idx]


class NextSentenceDataset(torch_data.Dataset):
    """Dataset for MLM task, returning string."""
    def __init__(self,text:List[Dict[str,str]]):
        self.df = text
    
    def __len__(self)->int:
        return len(self.df)
    
    def __getitem__(self,idx)->dict:
        return self.df[idx]

def reformat_nextsentence_for_cls_task(x):
    """reformats a triplet into one positive pair and one negative pair"""
    return [
        {"text1":x['anchor'], "text2":x['next'], "class":1},
        {"text1":x['anchor'], "text2":x['opposite'], "class":0},
    ]
    
    
def make_torch_datasets(
        task_data:TaskDataPerEpoch,
        other_data:Optional[TaskDataPerEpoch]=None,
) -> Dict[str,Union[DatasetTriplets, DatasetPairClassification, MLMDataset]]:
    """Universal wrapper to make a torch dataset from the raw task data.
    The `task_data` can be any of the `TaskDataPerEpoch` classes specific for tasks such 
    QA, STS, CLS, MLM, and nextSentence datasets."""

    # make the troch datasset for MLM tasks
    if isinstance(task_data, MLMDataPerEpoch):
        assert task_data.taskname == 'mlm'
        return {
            "val":MLMDataset(text = task_data.val),
            "train":MLMDataset(text = task_data.train)
        }
    
    # make torch dataset for QA tasks
    if isinstance(task_data, QADataPerEpoch):
        assert task_data.taskname == 'qa'
        return {
            'val':DatasetTriplets(
                list_of_data = task_data.val,
                n_negatives= TRIPLETS_N_NEGATIVES,
                topk_negatives_discard=TRIPLETS_TOPK_NEGATIVES, # to get similar but different negatives, use BM25 and discard these topk
                negative_corpus_method = NEGATIVE_CORPUS_METHOD_QA
            ),
            'train':DatasetTriplets(
                list_of_data = task_data.train,
                n_negatives= TRIPLETS_N_NEGATIVES,
                topk_negatives_discard=TRIPLETS_TOPK_NEGATIVES,
                negative_corpus_method = NEGATIVE_CORPUS_METHOD_QA
            )
        }

    # make torch dataset for QA tasks for CLS task (DatasetPairClassification)
    if isinstance(task_data, CLSDataPerEpoch):
        assert task_data.taskname == 'cls'
        
        # initialize the classification training dataset
        torch_data = {'val':None,'train':None}
        for split_name in ['val','train']:
            
            # make the base torch data
            torch_data[split_name] = DatasetPairClassification(
                list_of_data=None, seed = SEED
            )
            # process the CLS datax
            torch_data[split_name].process(
                getattr(task_data,split_name), inplace=True
            )
        
        return torch_data
    
    # make torch dataset for STS/retrieval-like task
    if isinstance(task_data, STSDataPerEpoch):
        
        # special cases for multilabelled datasets (need to convert to triplets)
        has_ledgar = any(list(map(lambda x:x['subtype']=='ledgar', task_data['train'])))
        has_eurlex = any(list(map(lambda x:x['subtype']=='eurlex', task_data['train'])))

        # generic case with no special datasets
        torch_data = {
            "val":DatasetTriplets(
                list_of_data = [
                    x for x in task_data.val
                    if x.get('type','na') == 'sts_triplet'
                ],
                n_negatives= TRIPLETS_N_NEGATIVES,
                topk_negatives_discard=TRIPLETS_TOPK_NEGATIVES, # to get similar but different nega
                negative_corpus_method = NEGATIVE_CORPUS_METHOD_STS
            ),
            "train":DatasetTriplets(
                list_of_data = [
                    x for x in task_data.train
                    if x.get('type','na') == 'sts_triplet'
                ],
                n_negatives= TRIPLETS_N_NEGATIVES,
                topk_negatives_discard=TRIPLETS_TOPK_NEGATIVES, # to get similar but different nega
                negative_corpus_method = NEGATIVE_CORPUS_METHOD_STS
            )
        }
        
        # convert to torch dataset (train)
        if has_ledgar:
            # special case for Ledgar
            torch_data_train_ledgar = DatasetTripletsSimilarityByCoLabel(
                list_of_data=[
                    example for example in task_data.train,
                    if (
                        example['type']=='sts_by_textlabel'
                        and
                        example['subtype']=='ledgar'
                    )
                ],
                n_negatives= TRIPLETS_N_NEGATIVES,
                label_processor_class = LabelProcesserEurlex,
                seed = SEED
            )
            torch_data_val_ledgar = DatasetTripletsSimilarityByCoLabel(
                list_of_data=[
                    example for example in task_data.val
                    if (
                        example['type']=='sts_by_textlabel'
                        and
                        example['subtype']=='ledgar'
                    )
                ],
                n_negatives= TRIPLETS_N_NEGATIVES,
                label_processor_class = LabelProcesserLedgar,
                seed = SEED
            )
            # extend the basic (sts-triplet) data with ledgar examples
            torch_data['train'].extend([torch_data_train_ledgar])
            torch_data['val'].extend([torch_data_val_ledgar])
            assert len(torch_data['train'])>len(torch_data_train_ledgar)
            assert len(torch_data['val'])>len(torch_data_val_ledgar)
        
        if has_eurlex:
            # special case for eurlex
            torch_data_train_eurlex = DatasetTripletsSimilarityByCoLabel(
                list_of_data=[
                    example for example in task_data.train,
                    if (
                        example['type']=='sts_by_textlabel'
                        and
                        example['subtype']=='eurlex'
                    )
                ],
                n_negatives= TRIPLETS_N_NEGATIVES,
                label_processor_class = LabelProcesserEurlex,
                seed = SEED
            )
            torch_data_val_eurlex = DatasetTripletsSimilarityByCoLabel(
                list_of_data=[
                    example for example in task_data.val
                    if (
                        example['type']=='sts_by_textlabel'
                        and
                        example['subtype']=='eurlex'
                    )
                ],
                n_negatives= TRIPLETS_N_NEGATIVES,
                label_processor_class = LabelProcesserEurlex,
                seed = SEED
            )            
            # extend the basic (sts-triplet) data with ledgar examples
            torch_data['train'].extend([torch_data_train_eurlex])
            torch_data['val'].extend([torch_data_val_eurlex])
            assert len(torch_data['val'])>len(torch_data_val_eurlex)
        
        return torch_data
    
    raise NotImplementedError(f'returned unrecognized class {str(task_data)}')
