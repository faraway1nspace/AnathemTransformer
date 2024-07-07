from typing import Tuple, Union, Dict, List, Any
import lzma
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from datetime import datetime
from itertools import islice
import json
from datasets import interleave_datasets # for interweaving streaming datasets
from spacy.lang.en import English
import spacy
try:
    nlp = spacy.load("en_core_web_sm")
except:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
import re
import random
import numpy as np
import os
import pickle
import copy
from math import prod

from src.configs.constants import *
from src.configs.dataset_configs import *
from src.configs.dataset_templates import TEXTSEPARATOR
from src.dataclasses import (
    TaskDataPerEpoch,
    MLMDataPerEpoch, 
    NextSentenceDataPerEpoch, 
    QADataPerEpoch, 
    STSDataPerEpoch, 
    CLSDataPerEpoch
)
from src.data_utils.data_utils import (
    check_language, check_is_code, nwords_quick, flatten
)
from src.data_utils.example_processor import ExampleProcessor, NegativeExampleGenerator



def convert_streaming_dataset_to_static_corpus(
    streaming_dataset:Dataset,
    skip:int=0,
    take:int=1000
)->Tuple[List[str], List[str]]:
    """Takes a streaming_dataset and converts it into a list of examples"""
    if skip !=0:
        dataset_to_make_static = streaming_dataset.skip(skip).take(take)
    else:
        dataset_to_make_static = streaming_dataset.take(take)

    examples_static_mlm = [] # data for MLM objective
    examples_static_nextsentence = [] # data for next sentence task
    for i, example in enumerate(dataset_to_make_static):
        # chunk text into ~512 text-strings, and sentences
        examples_processed = example_processor(text = example['text'])
        # chunk, accept/reject, sentences
        example_parsed, do_accept, parsed_sentences = examples_processed.values()
        if is_do_acceptgood:
            # mlm gets the chunks of text-strings
            examples_static_mlm.extend(example_parsed)
            if len(parsed_sentences)>15:
                # sentences for next sentence prediction: make triplet of s1,s2,opposite, where opposites get label=1
                examples_static_nextsentence.extend(
                    convert_sequence_into_nextsentence_pairs(parsed_sentences)
                )
                #FOOFU - STOPPED HERE TO FIGURE OUT WHY MY NEXT-SENTENCE STUFF IS SO LONG
        if (i+1)%100==0:
            print("...streaming size: " % len(examples_static_mlm))

    return examples_static_mlm, examples_static_nextsentence


def convert_sequence_into_nextsentence_pairs(list_of_sentences:List[str]) -> Dict[str,str]:
    """Converts a list of sentences into a list of dicts, with next-sentence pairs."""
    n = len(list_of_sentences)

    def opposite(i,n):
        return (i + round(n/2+1)) % n

    list_of_nextsentence_pairs = []
    # loop through sequence, make quadruplets of anchor1+anchor2+anchor3..., next and an opposite
    #for o1a, o1b, o2 in zip(range(0,n-2), range(1,n-1), range(2,n)):
    for o1a, o1b, o1c, o1d, o2a, o2b in zip(
        range(0,n-5), range(1,n-4), range(2,n-3), range(3,n-2), range(4,n-1), range(5,n)
    ):
        # anchor text is three sentences
        s_anchor = list_of_sentences[o1a] + " " + list_of_sentences[o1b] + " " +  list_of_sentences[o1c] + " " + list_of_sentences[o1d]
        # target is the fourth (next-sentence)
        s_next = list_of_sentences[o2a] + " " + list_of_sentences[o2b]
        idx_opposite = opposite(o1c,n)
        string_opposite = list_of_sentences[idx_opposite]
        if (idx_opposite+1)< len(list_of_sentences):
            string_opposite+= (" " + list_of_sentences[idx_opposite+1])
        list_of_nextsentence_pairs.append(
            {
                "anchor":s_anchor,
                "next":s_next,
                "opposite":string_opposite
            }
        )
    return list_of_nextsentence_pairs


def chunk_docs_into_chunks_and_sentences(
    list_of_strings:List[str],
    nlp=None,
    config_chunking:dict=None,
    seed:int = SEED,
    fieldname:str='text',
    min_number_of_sentence_for_nextsentence_prediction:int = NEXTSENTENCE_MIN_N_SENTENCES
):
    """Splits long docs into chunks that do next exceet max_seq_len, as well as sentences for next-sentence-prediction."""
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe("sentencizer")

    if config_chunking is None:
        config_chunking = {
            'max_seq_length':512,
            'min_seq_length':48,
            'max_chunk_size':6,
            'min_sentence_len':20,
            'seed':seed
        }
    else:
        config_chunking.update({'seed':seed})

    # initialize the example processor
    example_processor = ExampleProcessor(
        config=config_chunking, char_per_word = CHAR_PER_WORD, nlp =nlp
    )

    examples_static_chunks = [] # data for MLM objective
    examples_static_nextsentence = [] # data for next sentence task
    for i, example in enumerate(list_of_strings):
        # chunk text into ~512 text-strings, and sentences
        if isinstance(example,str):
            examples_processed = example_processor(text = example)
        elif isinstance(example,dict):
            examples_processed = example_processor(text = example[fieldname])
        # chunk, accept/reject, sentences
        example_parsed, do_accept, parsed_sentences = examples_processed.values()
        if do_accept:
            # mlm gets the text-strings chunked to size 512
            examples_static_chunks.extend(example_parsed)
            if len(parsed_sentences)> min_number_of_sentence_for_nextsentence_prediction: 
                # sentences for next sentence prediction: make triplet of s1,s2,opposite, where opposites get label=1
                examples_static_nextsentence.extend(
                    convert_sequence_into_nextsentence_pairs(parsed_sentences)
                )

    return examples_static_chunks, examples_static_nextsentence


def initialize_and_get_mlm_streaming_datasets(
        data_streaming_config:Dict[str,Any],
        streaming_cleaning_functions:List[Any],
        start_proportion:float = None,
        epoch:int=0,
        seed=SEED,
        path_to_val_cache:str = PATH_CACHE_MLM_VAL,
        path_to_train_cache_epoch:str = PATH_CACHE_MLM_TRAIN,
        do_check_english:bool = True
) -> Tuple[MLMDataPerEpoch, NextSentenceDataPerEpoch]:
    """Converts stream of unlabelled text data into static datasets for: MLM task and next-sentence-prediction task."""
    # list of files to stream
    files = data_streaming_config['files']
    # number of examples to take from stream for validation set
    val_size = data_streaming_config['val_size']
    # number of examples to take from stream for training set
    train_chunk_size = data_streaming_config['train_chunk_size']
    min_seq_len = data_streaming_config['min_seq_length']
    # normalization constant for normalizing the weights into probabilities
    probability_normalization_const = sum([x[2] for x in files])

    # where to initialize start-stream for training data
    if start_proportion is None:
        start_proportion = np.random.RandomState(seed+epoch).uniform()*0.99

    # reload cached files
    path_to_train_cache = None if not '%03g' in path_to_train_cache_epoch else path_to_train_cache_epoch % epoch
    do_make_valset = not os.path.isfile(path_to_val_cache)
    do_make_trainset = not os.path.isfile(path_to_train_cache)
    if not do_make_valset:
        print('RELOADING VAL-MLM SET: iter=%s' % path_to_val_cache)
        with open(path_to_val_cache,'rb') as pcon:
            datalist_val_mlm_static = pickle.load(pcon)
            datalist_val_sentences_static = pickle.load(pcon)
            epoch = pickle.load(pcon)
            log_source_val = pickle.load(pcon)
        print('VAL-MLM SET SIZE: %d' % len(datalist_val_mlm_static))
    else:
        datalist_val_mlm_static, datalist_val_sentences_static, log_source_val = [],[],{}
    if not do_make_trainset:
        print('RELOADING TRAIN-MLM SET: iter=%s' % path_to_val_cache)
        with open(path_to_train_cache,'rb') as pcon:
            datalist_train_mlm_static = pickle.load(pcon)
            datalist_train_sentences_static = pickle.load(pcon)
            epoch = pickle.load(pcon)
            log_source_train = pickle.load(pcon)
        print('TRAIN-MLM EPOCH-%d SET SIZE: %d' % (epoch, len(datalist_train_mlm_static)))
    else:
        datalist_train_mlm_static, datalist_train_sentences_static,log_source_train = [],[],{}

    if (do_make_trainset or do_make_valset):

        # initialize the nlp-sentencizer for chunking
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe("sentencizer")

        # loop through datasets
        for (mlm_nm, set_nm, prob, dataset_size, special_handling, partition_shuffle, threshold_specialchar), dataset_key in zip(
            files, streaming_cleaning_functions.keys()
        ):
            if prob ==0:
                continue
            prob /= probability_normalization_const

            # get cleaning & filter functions for streaming data functionality
            clean_func, filter_func, removefeature_names = streaming_cleaning_functions[dataset_key]

            # set arguments for the load_dataset (huggingface repos)
            load_dataset_args = {
                'path':mlm_nm, 'name':set_nm, 'split':'train', 'streaming':True, 'trust_remote_code':True
            }
            # for other non-huggingface repos, path needs to be a "builder"
            if mlm_nm.endswith('.jsonl') or mlm_nm.endswith('.jsonl.zip') or mlm_nm.endswith('.jsonl.zst'):
                load_dataset_args.update({'path':'json','data_files':mlm_nm})

            # special proecssing of datasets with multiple partitions
            if bool(partition_shuffle): # or str(epoch)=='val':

                n_files, n_per_file = partition_shuffle
                dataset_size = n_per_file
                print('trying %s initialization (shuffling through %d files)' % (mlm_nm, n_files))

                # whether there is a filter
                if filter_func is None:
                    dset_stream = load_dataset(**load_dataset_args)
                else:
                    dset_stream = load_dataset(**load_dataset_args).filter(filter_func)

                # validation set
                if do_make_valset:
                    # take from stream
                    n_valset_take = max(int(prob*val_size), 1)
                    print('take %d from %s validation'% (n_valset_take, mlm_nm))
                    dset_stream_val = dset_stream.take(n_valset_take).map(clean_func).remove_columns(removefeature_names)
                    # convert stream to a static set (and check english language)
                    dset_static_val_thisset =[
                        e['text'] for e in dset_stream_val
                        if bool(re.search(r"\w+",e['text'][:200])) and (nwords_quick(e['text'][:10000])>min_seq_len)
                    ]
                # training set
                if do_make_trainset:
                    # randomly skip a bunch from this set
                    skip_to_start = int(start_proportion*n_per_file)
                    take_from_this_set = max(int(round(train_chunk_size*prob)),1)
                    print('take %d from %s training'% (take_from_this_set, mlm_nm))
                    # shuffle: take a random data partition (from the dataset's list of files)
                    dset_stream_train = dset_stream.shuffle(
                        seed = seed+epoch, buffer_size = skip_to_start+take_from_this_set,
                    )
                    dset_stream_train = dset_stream_train.skip(
                        skip_to_start # random skip through dataset to new start position
                    ).take(
                        take_from_this_set # take this amount for the training ste
                    ).map(clean_func).remove_columns(removefeature_names)
                    # convert training to static dataset
                    dset_static_train_thisset =[
                        e['text'] for e in dset_stream_train
                        if bool(re.search(r"\w+",e['text'][:200])) and (nwords_quick(e['text'][:10000])>min_seq_len)
                    ]
            else:
                # regular streaming
                print('trying %s initialization' % mlm_nm)
                # whether there is a filter
                if filter_func is None:
                    dset_stream = load_dataset(**load_dataset_args).map(clean_func).remove_columns(removefeature_names)
                else:
                    dset_stream = load_dataset(**load_dataset_args).filter(filter_func).map(clean_func).remove_columns(removefeature_names)
                # take from stream
                n_valset_take = max(int(prob*val_size), 1) # size of valset
                print('take %d from %s validation'% (n_valset_take, mlm_nm))
                skip_to_start = int(start_proportion*(dataset_size-n_valset_take)) # random point to skip to
                n_train_take = max(int(round(train_chunk_size*prob)),1) # size of train set
                print('take %d from %s train'% (n_train_take, mlm_nm))
                if do_make_valset:
                    dset_stream_val = dset_stream.take(n_valset_take)
                    # checking for: existence of any words and ii) size of sequence meets minimum criteria
                    dset_static_val_thisset = [
                        e['text'] for e in dset_stream_val
                        if bool(re.search(r"\w+",e['text'][:200])) and (nwords_quick(e['text'][:10000])>min_seq_len)
                    ]
                if do_make_trainset:
                    dset_stream_train = dset_stream.skip(n_valset_take+skip_to_start).take(n_train_take)
                    # checking for: existence of any words and ii) size of sequence meets minimum criteria
                    dset_static_train_thisset = [
                        e['text'] for e in dset_stream_train
                        if bool(re.search(r"\w+",e['text'][:200])) and (nwords_quick(e['text'][:10000])>min_seq_len)
                    ]
            print('Done getting streams/reloading from %s' % mlm_nm)
            # check language, chunk sentences
            if do_make_valset:
                # discard non-english
                dset_static_val_thisset =[
                    e for e in dset_static_val_thisset
                    if check_language(e, threshold_specialchar)[0]
                ]
                print('done val language check')
                # split multi-answers that I want made into separate texts
                dset_static_val_thisset = [
                    e for e in dset_static_val_thisset
                    if TEXTSEPARATOR not in e
                ] + flatten([
                    e.split(TEXTSEPARATOR) for e in dset_static_val_thisset
                    if TEXTSEPARATOR in e
                ])
                # chunk the docs (512-tokens and next-sentence prediction sentences)
                dset_val_chunked_for_mlm, dset_val_nextsentence = chunk_docs_into_chunks_and_sentences(
                    list_of_strings=dset_static_val_thisset,
                    config_chunking=copy.deepcopy(data_streaming_config),
                    seed=seed+epoch,
                    nlp=nlp
                )
                print('done val longtext chunking')
                # add to val set
                datalist_val_mlm_static.extend(dset_val_chunked_for_mlm)
                datalist_val_sentences_static.extend(dset_val_nextsentence)
                # log the sources of text
                log_source_val[dataset_key] = len(dset_val_chunked_for_mlm)

            # check language, chunk sentences
            if do_make_trainset:
                # discard non-english
                dset_static_train_thisset =[
                    e for e in dset_static_train_thisset
                    if check_language(e, threshold_specialchar)[0]
                ]
                print('done train language check')
                # split multi-answers that I want made into separate texts
                dset_static_train_thisset = [
                    e for e in dset_static_train_thisset
                    if TEXTSEPARATOR not in e
                ] + flatten([
                    e.split(TEXTSEPARATOR) for e in dset_static_train_thisset
                    if TEXTSEPARATOR in e
                ])
                assert not any([TEXTSEPARATOR in e for e in dset_static_train_thisset])
                # chunk the docs (512-tokens and next-sentence prediction sentences)
                dset_train_chunked_for_mlm, dset_train_nextsentence = chunk_docs_into_chunks_and_sentences(
                    list_of_strings=dset_static_train_thisset,
                    config_chunking=copy.deepcopy(data_streaming_config),
                    seed=seed+epoch,
                    nlp=nlp
                )
                print('done trains longtext chunking')

                # ensure that none of the examples in the traning set are in the validation set
                if do_make_valset:
                    dset_train_chunked_for_mlm = [
                        s for s in dset_train_chunked_for_mlm
                        if s not in dset_val_chunked_for_mlm
                    ]
                    dset_train_nextsentence = [
                        tlt for tlt in dset_train_nextsentence
                        if (
                            tlt['anchor'] not in [
                                vtlt['anchor'] for vtlt in dset_val_nextsentence
                            ]
                        )
                    ]

                # add to training set
                datalist_train_mlm_static.extend(dset_train_chunked_for_mlm)
                datalist_train_sentences_static.extend(dset_train_nextsentence)
                # log the sources of text
                log_source_train[dataset_key] = len(dset_train_chunked_for_mlm)

        print('Done collecting streaming data')

    if do_make_valset:
        print('saving streamed validation data: %s' % path_to_val_cache)
        with open(path_to_val_cache,'wb') as pcon:
            pickle.dump(datalist_val_mlm_static, pcon)
            pickle.dump(datalist_val_sentences_static, pcon)
            pickle.dump(epoch,pcon)
            pickle.dump(log_source_val, pcon)
    if do_make_trainset:
        print('saving streamed training for epoch %d: %s' % (epoch, path_to_train_cache))
        with open(path_to_train_cache,'wb') as pcon:
            pickle.dump(datalist_train_mlm_static, pcon)
            pickle.dump(datalist_train_sentences_static, pcon)
            pickle.dump(epoch,pcon)
            pickle.dump(log_source_train,pcon)
    # return the training and validation sets
    #return {
    #    'train':{
    #        'mlm':datalist_train_mlm_static,
    #        'nextsentence':datalist_train_sentences_static
    #    },
    #    'val':{
    #        'mlm':datalist_val_mlm_static,
    #        'nextsentence':datalist_val_sentences_static
    #    },
    #    'epoch':epoch,
    #    'index_stream':start_proportion,
    #    'log_source':{'train':log_source_train, 'val':log_source_val}
    #}
    return (
        MLMDataPerEpoch(
            train=datalist_train_mlm_static,
            val=datalist_val_mlm_static,
            epoch=epoch,
            index_stream=start_proportion,
            taskname='mlm',
            log_source={'train':log_source_train, 'val':log_source_val}
        ),
        NextSentenceDataPerEpoch(
            train=datalist_train_sentences_static,
            val=datalist_val_sentences_static,
            epoch=epoch,
            index_stream=start_proportion,
            taskname='mlm',
            log_source={'train':log_source_train, 'val':log_source_val}
        )            
    )


def make_report_about_mlm_datasets(mlm_task_dataset:dict, dir_out:str=DIR_LOG)->None:
    """Creates a report about the report about the sources of data for MLM task."""
    epoch = mlm_task_dataset.get('epoch','unknown')
    out_path = os.path.join(dir_out, f'log_source_mlm_epoch-{epoch}.json')
    sums = {}
    for setnm,setcnt in mlm_task_dataset['log_source'].items():
        for dnm, dcnt in setcnt.items():
            if dnm not in sums: sums[dnm]=0
            sums[dnm]+=dcnt
    
    # proportion of data dedicated to each dataset
    out = {
        "date":datetime.today().strftime('%Y-%m-%d'),
        "epoch":epoch,
        "dataset_proportions":{
            k:round(v/sum([a for a in sums.values()]),4) for k,v in sums.items()
        }
    }
    # save
    json.dumps(out, out_path, indent=3)



def initialize_qa_streaming_datasets(
        data_streaming_config,
        streaming_cleaning_functions
):
    files = data_streaming_config['files']
    qa_streaming_datsets, qa_probabilities, qa_datasizes = [],[],[]
    for (qa_nm, set_nm, prob, dataset_size, special_handling, partition_shuffle) in files:

        if prob ==0:
            continue
        # get cleaning & filter functions for streaming data / map & filters
        clean_func, filter_func, feature_names, removefeature_names = streaming_cleaning_functions[qa_nm]

        # arguments for the load_dataset (huggingface repos)
        load_dataset_args = {
            'path':qa_nm, 'name':set_nm, 'split':'train', 'streaming':True, 'trust_remote_code':True
        }
        # for other non-huggingface repos, path needs to be a "builder"
        if qa_nm.endswith('.jsonl') or qa_nm.endswith('.jsonl.zip') or qa_nm.endswith('.jsonl.zst'):
            load_dataset_args.update({'path':'json','data_files':qa_nm})

        print('trying %s' % qa_nm)
        if filter_func is None:
            dset_stream = load_dataset(**load_dataset_args).map(clean_func).remove_columns(removefeature_names)
        else:
            dset_stream = load_dataset(**load_dataset_args).filter(filter_func).map(clean_func).remove_columns(removefeature_names)

        qa_streaming_datsets.append(dset_stream)
        qa_probabilities.append(prob);
        qa_datasizes.append(dataset_size)

    print('done initializing the QA streaming datasets')
    return qa_streaming_datsets, qa_probabilities, qa_datasizes


def streaming_skip(skip, list_of_streaming_datasets, probabilities, datasizes, seed=42, convert_to_static = False):
    """Function loops through a list of streaming datasets, skips a first K values based on the probabilities, and returns them"""
    out = []
    normalized_p = [p/sum(probabilities) for p in probabilities]
    for dset, p, size in list_of_streaming_datasets, normalized_p, datasizes:
        skip_in_this_set = max(0,int(p)*skip)
        out.append(dset.skip(skip_in_this_set))
    return out


def streaming_take(skip, start_proportion, chunksize, list_of_streaming_datasets, probabilities, datasizes,  convert_to_static = False):
    """Takes some examples based on a starting point within the dataset, as a proportion of its total size"""
    out = []
    normalized_p = [p/sum(probabilities) for p in probabilities]
    for j, (dset, p, size) in enumerate(zip(list_of_streaming_datasets, normalized_p, datasizes)):
        #print(type(dset))
        #print(type(p))
        #print(type(size))
        # skip for valset
        skip_in_this_set = int(round(p*skip))
        # afterwards, where to start?
        skip_to_start = int(start_proportion*(size-skip_in_this_set))
        take_from_this_set = int(round(chunksize*p))
        if skip_to_start>0:
            dset_skipped = dset.skip(skip_in_this_set+skip_to_start).take(take_from_this_set)
        else:
            dset_skipped = dset.take(take_from_this_set)

        if not convert_to_static:
            # option to return the streaming dataset
            out.append(dset_skipped)
        else:
            # option just to convert the streaming dataset to static outputs
            for example in dset_skipped:
                example['source_id'] = j
                out.append(example)
        print('done %d' % j)
    return out


def train_test_splits_from_stream_qa(
        streaming_dataset,
        val_size:int = 100,#2000,
        epoch:int = 0,
        chunk_size = 500,#6000,
        path_to_val_cache = PATH_CACHE_QA_VAL,
        probabilities = None,
        datasizes = None,
        seed=SEED
):
    """
    val_size = 2000, number of streaming-iter to skip, reserved for the val-sze
    epoch = 0, epoch will change the seed when sampling the chunk idx for making the training set
    chunk_size = 5000, # number of streaming-iter to select the training data chunk
    max_chunk_start = 2000000, # randomly sample within this interval for streaming chunks
    """
    if os.path.isfile(path_to_val_cache):
        print('RELOADING VAL-QA SET: iter=%s' % path_to_val_cache)
        with open(path_to_val_cache,'rb') as pcon:
            val_corpus_list = pickle.load(pcon)
        print('VAL-QA SET SIZE: %d' % len(val_corpus_list))
    else:
        # stream validation set
        print('STREAMING VAL-QA DATA: %d' % val_size)
        val_corpus_list = streaming_take(
            skip=0,
            start_proportion=0,
            chunksize=val_size,
            list_of_streaming_datasets=streaming_dataset,
            probabilities=probabilities,
            datasizes=datasizes,
            convert_to_static = True
        )
        print('REALIZED VAL-QA DATA: %d' % len(val_corpus_list))
        # save the validation corpus
        print('SAVING VAL-QA SET: %s' % path_to_val_cache)
        with open(path_to_val_cache,'wb') as pcon:
            pickle.dump(val_corpus_list, pcon)

    # take a random interger to start the streaming of training data
    # starts at a random position
    train_start_proportion = np.random.RandomState(seed + epoch).random()*0.99
    print(train_start_proportion)

    # stream training data
    print('STREAMING TRAIN QA-DATA: %d STARTING AT: %0.3f' % (chunk_size,train_start_proportion))
    train_corpus_list = streaming_take(
            skip=val_size,
            start_proportion=train_start_proportion,
            chunksize=chunk_size,
            list_of_streaming_datasets=streaming_dataset,
            probabilities=probabilities,
            datasizes=datasizes,
            convert_to_static = True
        )

    print('REALISED TRAIN QA-DATA SIZE: %d' % len(train_corpus_list))
    return {
        'train':train_corpus_list,
        'val':val_corpus_list,
        'epoch':0,
        'index_stream':train_start_proportion,
    }


def initialize_and_get_triplet_streaming_datasets(
    data_streaming_config,
    streaming_cleaning_functions,
    start_proportion:float = None,
    epoch:int=0,
    seed:int=SEED,
    path_to_val_cache:str = 'cache_val_qa.pkl',
    path_to_train_cache_epoch:str = 'cache_train_qa_%03g.pkl',
    do_check_english = True,
    name = 'qa' #
)->Union[QADataPerEpoch, STSDataPerEpoch]:
    """Converts stream of unlabelled text data into static datasets for: for Triplet data tasks (QA-task/IR-task)"""
    # list of files to stream
    print('Initializing the streaming-QA to static-dataset procedure...')
    files = data_streaming_config['files']
    # number of examples to take from stream for validation set
    val_size = data_streaming_config['val_size']
    # number of examples to take from stream for training set
    train_chunk_size = data_streaming_config['train_chunk_size']
    min_seq_len = data_streaming_config.get('min_seq_length', 48)
    # normalization constant for normalizing the weights into probabilities
    probability_normalization_const = sum([x[2] for x in files])

    # where to initialize start-stream for training data
    if start_proportion is None:
        start_proportion = np.random.RandomState(seed+epoch).uniform()*0.99

    # reload cached files
    path_to_train_cache = None if not '%03g' in path_to_train_cache_epoch else path_to_train_cache_epoch % epoch
    do_make_valset = not os.path.isfile(path_to_val_cache)
    do_make_trainset = not os.path.isfile(path_to_train_cache)
    if not do_make_valset:
        print(f'RELOADING VAL-{name} SET: iter=%s' % path_to_val_cache)
        with open(path_to_val_cache,'rb') as pcon:
            datalist_val_triplet_static = pickle.load(pcon)
        print(f'VAL-{name} SET SIZE: %d' % len(datalist_val_triplet_static))
    else:
        datalist_val_triplet_static = []
    if not do_make_trainset:
        print(f'RELOADING VAL-{name} SET: iter=%s' % path_to_val_cache)
        with open(path_to_train_cache,'rb') as pcon:
            datalist_train_triplet_static = pickle.load(pcon)
        print(f'TRAIN-{name} EPOCH-%d SET SIZE: %d' % (epoch, len(datalist_train_triplet_static)))
    else:
        datalist_train_triplet_static = []

    if (do_make_trainset or do_make_valset):

        # loop through datasets
        for (data_nm, set_nm, prob, dataset_size, special_handling, partition_shuffle), dataset_key in zip(
            files, streaming_cleaning_functions.keys()
        ):
            if prob ==0:
                continue
            prob /= probability_normalization_const

            # get cleaning & filter functions for streaming data functionality
            clean_func, filter_func, feature_names, removefeature_names = streaming_cleaning_functions[dataset_key]

            # set arguments for the load_dataset (huggingface repos)
            load_dataset_args = {
                'path':data_nm, 'name':set_nm, 'split':'train', 'streaming':True,  'trust_remote_code':True
            }
            # for other non-huggingface repos, path needs to be a "builder"
            if data_nm.endswith('.jsonl') or data_nm.endswith('.jsonl.zip') or data_nm.endswith('.jsonl.zst'):
                load_dataset_args.update({'path':'json','data_files':data_nm})

            # special proecssing of datasets with multiple partitions
            if bool(partition_shuffle): # or str(epoch)=='val':

                n_files, n_per_file = partition_shuffle
                dataset_size = n_per_file
                print('trying %s initialization (shuffling through %d files)' % (data_nm, n_files))

                # whether there is a filter
                if filter_func is None:
                    dset_stream = load_dataset(**load_dataset_args)
                else:
                    dset_stream = load_dataset(**load_dataset_args).filter(filter_func)

                # validation set
                if do_make_valset:
                    # take from stream
                    n_valset_take = max(int(prob*val_size), 1)
                    if n_valset_take==1:
                        print(prob)
                        print(val_size)
                    print('take %d from %s validation'% (n_valset_take, data_nm))
                    dset_stream_val = dset_stream.take(n_valset_take).map(clean_func).remove_columns(removefeature_names)
                    # convert stream to a static set and do check
                    dset_static_val_thisset = [
                        e for e in dset_stream_val if (
                            (e['query'] is not None)
                            and
                            bool(re.search(r"\w+",e['query'][:200]))
                        )
                    ]
                # training set
                if do_make_trainset:
                    # randomly skip a bunch from this set
                    skip_to_start = int(start_proportion*n_per_file)
                    take_from_this_set = max(int(round(train_chunk_size*prob)),1)
                    print('take %d from %s training'% (take_from_this_set, data_nm))
                    # shuffle: take a random data partition (from the dataset's list of files)
                    dset_stream_train = dset_stream.shuffle(
                        seed = seed+epoch, buffer_size = skip_to_start+take_from_this_set,
                    )
                    dset_stream_train = dset_stream_train.skip(
                        skip_to_start # random skip through dataset to new start position
                    ).take(
                        take_from_this_set # take this amount for the training ste
                    ).map(clean_func).remove_columns(removefeature_names)
                    # convert training to static dataset
                    dset_static_train_thisset = [
                        e for e in dset_stream_train if (
                            (e['query'] is not None)
                            and
                            bool(re.search(r"\w+",e['query'][:200]))
                        )
                    ]
            else:
                # regular streaming
                print('trying %s initialization' % data_nm)
                # whether there is a filter
                if filter_func is None:
                    dset_stream = load_dataset(**load_dataset_args).map(clean_func).remove_columns(removefeature_names)
                else:
                    dset_stream = load_dataset(**load_dataset_args).filter(filter_func).map(clean_func).remove_columns(removefeature_names)
                # take from stream
                n_valset_take = max(int(prob*val_size), 1) # size of valset
                if n_valset_take==1:
                    print(prob)
                    print(val_size)
                print('take %d from %s validation'% (n_valset_take, data_nm))
                skip_to_start = int(start_proportion*(dataset_size-n_valset_take)) # random point to skip to
                n_train_take = max(int(round(train_chunk_size*prob)),1) # size of train set
                print('take %d from %s train'% (n_train_take, data_nm))
                if do_make_valset:
                    dset_stream_val = dset_stream.take(n_valset_take)
                    dset_static_val_thisset = [
                        e for e in dset_stream_val if (
                            (e['query'] is not None)
                            and
                            bool(re.search(r"\w+",e['query'][:200]))
                        )
                    ]
                if do_make_trainset:
                    dset_stream_train = dset_stream.skip(
                        n_valset_take+skip_to_start
                    ).take(
                        n_train_take
                    )
                    dset_static_train_thisset = [
                        e for e in dset_stream_train if (
                            (e['query'] is not None)
                            and
                            bool(re.search(r"\w+",e['query'][:200]))
                        )
                    ]
            print('Done getting streams/reloading from %s' % data_nm)
            # check language, chunk sentences
            if do_make_valset:
                # discard non-english
                dset_static_val_thisset =[
                    e for e in dset_static_val_thisset
                    if check_language(e['query'])[0] 
                ]
                print('done val language check')
                # add to val set
                datalist_val_triplet_static.extend(dset_static_val_thisset)

            # check language, chunk sentences
            if do_make_trainset:
                # discard non-english
                dset_static_train_thisset =[
                    e for e in dset_static_train_thisset
                    if check_language(e['query'])[0] 
                ]
                print('done train language check')

                # ensure that none of the examples in the traning set are in the validation set
                if do_make_valset:
                    val_queries = set([q['query'] for q in dset_static_val_thisset])
                    dset_static_train_thisset = [
                        s for s in dset_static_train_thisset
                        if s['query'] not in val_queries
                    ]

                # add to training set
                datalist_train_triplet_static.extend(dset_static_train_thisset)

        print(f'Done collecting {name} streaming data')

    if do_make_valset:
        print('saving streamed %s validation data: %s' % (name, path_to_val_cache))
        with open(path_to_val_cache,'wb') as pcon:
            pickle.dump(datalist_val_triplet_static, pcon)

    if do_make_trainset:
        print('saving streamed %s training for epoch %d: %s' % (name, epoch, path_to_train_cache))
        with open(path_to_train_cache,'wb') as pcon:
            pickle.dump(datalist_train_triplet_static, pcon)

    return QADataPerEpoch(
        train=datalist_train_triplet_static,
        val=datalist_val_triplet_static,
        epoch=epoch,
        index_stream=start_proportion,
        log_source={},
        taskname=name
    )


def initialize_and_get_classification_streaming_datasets(
        data_streaming_config,
        streaming_cleaning_functions,
        start_proportion:float = None,
        epoch:int=0,
        seed:int=SEED,
        path_to_val_cache:str = 'cache_val_cls.pkl',
        path_to_train_cache_epoch:str = 'cache_train_cls_%03g.pkl',
        do_check_english:bool = True,
        name = 'cls' #
) -> CLSDataPerEpoch:
    """Converts stream of unlabelled text data into static datasets for: pair-classification tasks"""
    # list of files to stream
    files = data_streaming_config['files']
    # number of examples to take from stream for validation set
    val_size = data_streaming_config['val_size']
    # number of examples to take from stream for training set
    train_chunk_size = data_streaming_config['train_chunk_size']
    min_seq_len = data_streaming_config.get('min_seq_length', 48)
    # normalization constant for normalizing the weights into probabilities
    probability_normalization_const = sum([x[2] for x in files])

    # where to initialize start-stream for training data
    if start_proportion is None:
        start_proportion = np.random.RandomState(seed+epoch).uniform()*0.99

    # reload cached files
    path_to_train_cache = None if not '%03g' in path_to_train_cache_epoch else path_to_train_cache_epoch % epoch
    do_make_valset = not os.path.isfile(path_to_val_cache)
    do_make_trainset = not os.path.isfile(path_to_train_cache)
    if not do_make_valset:
        print(f'RELOADING VAL-{name} SET: iter=%s' % path_to_val_cache)
        with open(path_to_val_cache,'rb') as pcon:
            datalist_val_triplet_static = pickle.load(pcon)
        print(f'VAL-{name} SET SIZE: %d' % len(datalist_val_triplet_static))
    else:
        datalist_val_triplet_static = []
    if not do_make_trainset:
        print(f'RELOADING VAL-{name} SET: iter=%s' % path_to_val_cache)
        with open(path_to_train_cache,'rb') as pcon:
            datalist_train_triplet_static = pickle.load(pcon)
        print(f'TRAIN-{name} EPOCH-%d SET SIZE: %d' % (epoch, len(datalist_train_triplet_static)))
    else:
        datalist_train_triplet_static = []

    if (do_make_trainset or do_make_valset):

        # loop through datasets
        for (data_nm, set_nm, prob, dataset_size, special_handling, partition_shuffle), dataset_key in zip(
            files, streaming_cleaning_functions.keys()
        ):
            if prob ==0:
                continue
            prob /= probability_normalization_const

            # get cleaning & filter functions for streaming data functionality
            clean_func, filter_func, feature_names, removefeature_names = streaming_cleaning_functions[dataset_key]

            # set arguments for the load_dataset (huggingface repos)
            load_dataset_args = {
                'path':data_nm, 'name':set_nm, 'split':'train', 'streaming':True,  'trust_remote_code':True
            }
            # for other non-huggingface repos, path needs to be a "builder"
            if data_nm.endswith('.jsonl') or data_nm.endswith('.jsonl.zip') or data_nm.endswith('.jsonl.zst'):
                load_dataset_args.update({'path':'json','data_files':data_nm})

            # special proecssing of datasets with multiple partitions
            if bool(partition_shuffle): # or str(epoch)=='val':

                n_files, n_per_file = partition_shuffle
                dataset_size = n_per_file
                print('trying %s initialization (shuffling through %d files)' % (data_nm, n_files))

                # whether there is a filter
                if filter_func is None:
                    dset_stream = load_dataset(**load_dataset_args)
                else:
                    dset_stream = load_dataset(**load_dataset_args).filter(filter_func)

                # validation set
                if do_make_valset:
                    # take from stream
                    n_valset_take = max(int(prob*val_size), 1)
                    print('take %d from %s validation'% (n_valset_take, data_nm))
                    dset_stream_val = dset_stream.take(n_valset_take).map(clean_func).remove_columns(removefeature_names)
                    # convert stream to a static set and do check
                    dset_static_val_thisset = [
                        e for e in dset_stream_val if bool(re.search(r"\w+",e['pair1'][:200]))
                    ]
                # training set
                if do_make_trainset:
                    # randomly skip a bunch from this set
                    skip_to_start = int(start_proportion*n_per_file)
                    take_from_this_set = max(int(round(train_chunk_size*prob)),1)
                    print('take %d from %s training'% (take_from_this_set, data_nm))
                    # shuffle: take a random data partition (from the dataset's list of files)
                    dset_stream_train = dset_stream.shuffle(
                        seed = seed+epoch, buffer_size = skip_to_start+take_from_this_set,
                    )
                    dset_stream_train = dset_stream_train.skip(
                        skip_to_start # random skip through dataset to new start position
                    ).take(
                        take_from_this_set # take this amount for the training ste
                    ).map(clean_func).remove_columns(removefeature_names)
                    # convert training to static dataset
                    dset_static_train_thisset = [
                        e for e in dset_stream_train if bool(re.search(r"\w+",e['pair1'][:200]))
                    ]
            else:
                # regular streaming
                print('trying %s initialization' % data_nm)
                # whether there is a filter
                if filter_func is None:
                    dset_stream = load_dataset(**load_dataset_args).map(clean_func).remove_columns(removefeature_names)
                else:
                    dset_stream = load_dataset(**load_dataset_args).filter(filter_func).map(clean_func).remove_columns(removefeature_names)
                # take from stream
                n_valset_take = max(int(prob*val_size), 1) # size of valset
                print('take %d from %s validation'% (n_valset_take, data_nm))
                skip_to_start = int(start_proportion*(dataset_size-n_valset_take)) # random point to skip to
                n_train_take = max(int(round(train_chunk_size*prob)),1) # size of train set
                print('take %d from %s train'% (n_train_take, data_nm))
                if do_make_valset:
                    dset_stream_val = dset_stream.take(n_valset_take)
                    dset_static_val_thisset = [
                        e for e in dset_stream_val if bool(re.search(r"\w+",e['pair1'][:200]))
                    ]
                if do_make_trainset:
                    dset_stream_train = dset_stream.skip(n_valset_take+skip_to_start).take(n_train_take)
                    dset_static_train_thisset = [
                        e for e in dset_stream_train if bool(re.search(r"\w+",e['pair1'][:200]))
                    ]
            print('Done getting streams/reloading from %s' % data_nm)
            # check language
            if do_make_valset:
                # discard non-english
                dset_static_val_thisset =[
                    e for e in dset_static_val_thisset if check_language(e['pair1'])[0] #detect(e['pair1'][:200]+" hello")=='en'
                ]
                print('done val language check')
                # add to val set
                datalist_val_triplet_static.extend(dset_static_val_thisset)

            # check language
            if do_make_trainset:
                # discard non-english
                dset_static_train_thisset =[
                    e for e in dset_static_train_thisset if check_language(e['pair1'])[0]
                ]
                print('done train language check')

                # ensure that none of the examples in the traning set are in the validation set
                def hashtest(text1,text2):
                    texthash = text1.lower()
                    texthash+= "" if text2 is None else text2[:1000].lower()
                    return texthash

                if do_make_valset:
                    val_queries = set([hashtest(q['pair1'],q['pair2']) for q in dset_static_val_thisset])
                    dset_static_train_thisset = [
                        s for s in dset_static_train_thisset if hashtest(s['pair1'],s['pair2']) not in val_queries
                    ]

                # add to training set
                datalist_train_triplet_static.extend(dset_static_train_thisset)

        print(f'Done collecting {name} streaming data')

    if do_make_valset:
        print('saving streamed %s validation data: %s' % (name, path_to_val_cache))
        with open(path_to_val_cache,'wb') as pcon:
            pickle.dump(datalist_val_triplet_static, pcon)

    if do_make_trainset:
        print('saving streamed %s training for epoch %d: %s' % (name, epoch, path_to_train_cache))
        with open(path_to_train_cache,'wb') as pcon:
            pickle.dump(datalist_train_triplet_static, pcon)

    return CLSDataPerEpoch(
        train=datalist_train_triplet_static,
        val=datalist_val_triplet_static,
        epoch=epoch,
        index_stream=start_proportion,
        log_source={},
        taskname=name
    )




# initialize the MLM streaming sets
# EPOCH 1
def preprocess_mlm_data(epoch:int, seed:int=SEED)-> Dict[str, Union[MLMDataPerEpoch,NextSentenceDataPerEpoch]]:
    """Wrapper for initialize_and_get_mlm_streaming_datasets to intialize MLM task."""
    datasets_static_mlm = initialize_and_get_mlm_streaming_datasets(
        data_streaming_config = data_streaming_config_mlm,
        streaming_cleaning_functions = mlm_streaming_cleaning_functions, 
        start_proportion = None,
        epoch=epoch,
        seed=seed,
        path_to_val_cache = data_streaming_config_mlm['path_cache_mlm_val'],
        path_to_train_cache_epoch = data_streaming_config_mlm['path_cache_mlm_train'],
        do_check_english=True,
    )

    # log the creation
    try:
        make_report_about_mlm_datasets(datasets_static_mlm, dir_out=DIR_LOG)
    except:
        print('make_report_about_mlm_datasets failed')
    print('DONE MLM PREPROCESSING')
    return datasets_static_mlm


# initiate the CLS pair classification datasets
def preprocess_cls_data(epoch:int, seed:int=SEED)-> CLSDataPerEpoch:
    """Wrapper for initialize_and_get_classification_streaming_datasets to intialize CLS task."""
    datasets_static_cls = initialize_and_get_classification_streaming_datasets(
        data_streaming_config=data_streaming_config_cls,
        streaming_cleaning_functions=cls_streaming_cleaning_functions,
        start_proportion = None,
        epoch=epoch,
        seed=seed, #SEED,
        path_to_val_cache = data_streaming_config_cls['path_cache_cls_val'],
        path_to_train_cache_epoch = data_streaming_config_cls['path_cache_cls_train'],
        do_check_english = True,
        name = 'cls' #
    )
    print('DONE CLS PREPROCESSING')
    return datasets_static_cls


# initiate the QA streaming sets
def preprocess_qa_data(epoch:int, seed:int=SEED)-> QADataPerEpoch:
    """Wrapper for initialize_and_get_classification_streaming_datasets to intialize CLS task."""
    datasets_static_qa = initialize_and_get_triplet_streaming_datasets(
        data_streaming_config = data_streaming_config_qa,
        streaming_cleaning_functions = qa_streaming_cleaning_functions,
        start_proportion = None,
        epoch=epoch,
        seed=seed,
        path_to_val_cache = data_streaming_config_qa['path_cache_qa_val'],
        path_to_train_cache_epoch = data_streaming_config_qa['path_cache_qa_train'],
        do_check_english = True,
        name = 'qa' #
    )
    print('DONE QA PREPROCESSING')
    return datasets_static_qa


# initiate the STS/retrieval/paraphrase sets
def preprocess_sts_data(epoch:int, seed:int=SEED)-> TaskDataPerEpoch:
    """Wrapper for initialize_and_get_classification_streaming_datasets to intialize CLS task."""
    datasets_statics_sts = initialize_and_get_triplet_streaming_datasets(
        data_streaming_config = data_streaming_config_sts,
        streaming_cleaning_functions = sts_streaming_cleaning_functions,
        start_proportion = None,
        epoch=epoch,
        seed=seed,
        path_to_val_cache = data_streaming_config_sts['path_cache_sts_val'],
        path_to_train_cache_epoch = data_streaming_config_sts['path_cache_sts_train'],
        do_check_english = True,
        name = 'sts' #
    )
    print('DONE STS PREPROCESSING')
    return datasets_statics_sts


# initialize the negative corpus
negative_example_generator = NegativeExampleGenerator()

