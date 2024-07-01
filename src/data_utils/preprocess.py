from typing import Tuple, Union, Dict, List, Any
import lzma
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from itertools import islice
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
from src.data_utils.data_utils import (
    check_language, check_is_code, nwords_quick, flatten
)
from src.data_utils.example_processor import ExampleProcessor, NegativeExampleGenerator



def convert_streaming_dataset_to_static_corpus(
    streaming_dataset:Dataset,
    skip:int=0,
    take:int=1000
):
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


def convert_sequence_into_nextsentence_pairs(list_of_sentences:List[str]):
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


TEXTSEPARATOR = "%0XTEXTXEPARAT0RX%0"


def chunk_docs_into_chunks_and_sentences(
    list_of_strings,
    nlp=None,
    config_chunking=None,
    seed = 42,
    fieldname='text',
    min_number_of_sentence_for_nextsentence_prediction = 17
):
    """Splits long docs into chunks that do next exceet max_seq_len, as well as sentences for next-sentence-prediction """
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
            if len(parsed_sentences)> min_number_of_sentence_for_nextsentence_prediction: #4:
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
    path_to_val_cache:str = 'cache_val_mlm.pkl',
    path_to_train_cache_epoch:str = 'cache_train_mlm_%03g.pkl',
    do_check_english:bool = True
):
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
        print('RELOADING VAL-QA SET: iter=%s' % path_to_val_cache)
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
    return {
        'train':{
            'mlm':datalist_train_mlm_static,
            'nextsentence':datalist_train_sentences_static
        },
        'val':{
            'mlm':datalist_val_mlm_static,
            'nextsentence':datalist_val_sentences_static
        },
        'epoch':epoch,
        'index_stream':start_proportion,
        'log_source':{'train':log_source_train, 'val':log_source_val}
    }


print(convert_sequence_into_nextsentence_pairs(['a','b','c','d','e','f','g']))

def gather_mlm_datsets_from_remote_repositories(
        data_streaming_config_mlm = data_streaming_config_mlm
):
    pass


# generates the negative corpus
negative_example_generator= NegativeExampleGenerator()


# initialize the streaming sets
# EPOCH 1
initialize_and_get_mlm_streaming_datasets(
    data_streaming_config = data_streaming_config_mlm,
    streaming_cleaning_functions = mlm_streaming_cleaning_functions, 
    start_proportion = None,
    epoch=0,
    seed=SEED,
    path_to_val_cache = data_streaming_config_mlm['path_cache_mlm_val'],
    path_to_train_cache_epoch = data_streaming_config_mlm['path_cache_mlm_train'],
    do_check_english=True,
)



print('DONE PREPROCESS')
