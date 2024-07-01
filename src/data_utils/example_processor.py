import datasets
from math import prod
import numpy as np
import os
import pickle
import pynndescent
import random
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple, Union, Dict, List, Any

try:
    nlp = spacy.load("en_core_web_sm")
except:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

nlp.add_pipe("sentencizer")

from src.configs.constants import *
from src.configs.dataset_configs import example_processor_config, negative_config
from src.data_utils.data_utils import check_language, check_is_code

class ExampleProcessor:
    """Class to handle splitting example-texts into chunks and sentences."""
    def __init__(
            self,
            config:dict=example_processor_config,
            char_per_word:int = CHAR_PER_WORD,
            nlp =nlp,
    ):
        self.nlp = nlp
        self.char_per_word = char_per_word
        self.max_seq_length = config.get('max_seq_length', 512) # maximum word-length for chunks for mlm objective (else split)
        self.min_seq_length = config.get('min_seq_length', 128) # min sequence length for chunks (else discard
        self.max_chunk_size = config.get('max_chunk_size', 5) # maximum number of chunks of text to take (each ~512 in length)
        self.min_sentence_len = config.get('min_sentence_len', 20) # for next-sentence, min sentence size to merge together
        self.seed = config.get('seed', 42)
        self.max_chunk_length = self.max_chunk_size * self.max_seq_length
        self.max_chunk_length_char = int(self.max_chunk_length*self.char_per_word)
        self.min_seq_length_char = int(self.min_seq_length*self.char_per_word)
        self.min_sentence_length_char = int(self.min_sentence_len*self.char_per_word)

    @staticmethod
    def split_into_chunks(text:str, chunk_char_size:int, overlapping_size:int = 50)->List[str]:
        chunks = []
        start = 0
        end = chunk_char_size + overlapping_size
        while start < len(text):
            chunk = text[start:end]
            period_index = chunk.find(". ")
            if period_index != -1:
                chunk = chunk[period_index + 1:]
            else:
                first_space_index = chunk.find(" ")
                if first_space_index != -1:
                    chunk = chunk[first_space_index + 1:]
            # Check if the chunk has been split and contains more than one word
            #if start > 0 and " " in chunk:
            if end < len(text) and " " in chunk and chunk[-1]!=" ":
                last_space_index = chunk.rfind(" ")
                chunk = chunk[:last_space_index]
            chunks.append(chunk)
            start += chunk_char_size
            end += chunk_char_size
        return chunks

    def split_chunk_into_sentences(self, chunk, discard_first_sentence=True, discard_last_sentence=True ):
        doc = self.nlp(chunk)
        MAX_CHAR_LEN = int(self.max_seq_length*self.char_per_word)
        sentences = [sent.text for sent in doc.sents]
        if discard_first_sentence:
            sentences = sentences[1:]
        if discard_last_sentence:
            sentences = sentences[:-1]

        super_list_concatenated = [] # accumulates concatenated sentences
        super_list_raw_sentences = [] # accumulates raw sentences (for next-sentence prediction)
        buffer = []
        buffer_len = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            if buffer_len + sentence_len > MAX_CHAR_LEN:
                super_list_concatenated.append(" ".join(buffer))
                super_list_raw_sentences.extend(buffer)
                buffer = []
                buffer_len = 0

            buffer.append(sentence)
            buffer_len += sentence_len

        if buffer:  # If there are any remaining sentences in the buffer
            super_list_concatenated.append(" ".join(buffer))
            super_list_raw_sentences.extend(buffer)

        return super_list_concatenated, super_list_raw_sentences

    def _sample_chunk_span(self, text, max_chunk_length_char):
        chunks = self.split_into_chunks(text, max_chunk_length_char)
        # randomly sample from the chunks
        #FOOBAR SAMPLE FROM CHUNKS
        return random.choice(chunks)

    def is_too_small_quickcheck(self, text, textlen=None):
        if textlen is None: textlen = len(text.strip())
        return textlen < self.min_seq_length_char*0.9

    def is_too_small(self, nwords):
        return nwords < self.min_seq_length

    def is_larger_than_max_chunk_quickcheck(self, text, textlen):
        """if it is larger than a chunksize, then we need to sample chunks"""
        if textlen is None: textlen = len(text.strip())
        return textlen > self.max_chunk_length_char

    def is_short_than_a_chunk(self, text, textlen):
        """if it is shorter than a chunk, then we'll take all text, in chunks"""
        if textlen is None: textlen = len(text.strip())
        return textlen < self.max_chunk_length_char

    def is_smaller_than_two_paragraphs(self, text):
        charlen = len(text)
        if charlen < (1.5*self.max_seq_length*self.char_per_word):
            return True, re.split(r"[\s\n\r]+",text.strip())
        if charlen > (2.5*self.max_seq_length*self.char_per_word):
            return False, None
        # inbetween cases, split and calculate the number of words
        textsplit = re.split(r"[\s\n\r]+",text.strip())
        nwords = len(textsplit)
        if nwords < 1.2*self.max_seq_length:
            return True, textsplit
        return False, textsplit

    @staticmethod
    def preprocess_sentences(list_of_sentences, min_sentence_char_length):
        """Merges small sentences in a sequence of sentence, until the strings are greater than `min_sentence_char_length`"""
        processed_sentences = []
        buffer = ""

        for sentence in list_of_sentences:
            if len(sentence) < min_sentence_char_length:
                buffer = buffer + " " + sentence
                if (len(buffer)>=min_sentence_char_length):
                    processed_sentences.append(buffer.strip())
                    buffer = ""
            else:
                if (len(buffer)<min_sentence_char_length):
                    to_add = buffer + " " + sentence
                    processed_sentences.append(to_add.strip())
                    buffer = ""
                else:
                    processed_sentences.extend([buffer.strip(), sentence.strip()])

        if buffer:  # If there are any remaining sentences in the buffer
            processed_sentences.append(buffer)

        return processed_sentences

    def process(self, text:str)-> dict:
        """Chunks and samples large portions of text"""

        charlen = len(text.strip())

        # DISCARD if it is too small for copus
        if self.is_too_small_quickcheck(text, charlen):

            return {'text':[], 'do_accept':False, 'sentences':[]}

        # sample span of chunks: if it larger than our max chunk size
        if self.is_larger_than_max_chunk_quickcheck(text, charlen):
            text_span_chunks = self._sample_chunk_span(text, self.max_chunk_length_char)
        else:
            text_span_chunks = text

        # check if it smaller, than 1.5 seqlen, then we just accept it all as one unit to truncate later in tokenizer
        is_smaller_than_2_paras, textsplit = self.is_smaller_than_two_paragraphs(text_span_chunks)

        if is_smaller_than_2_paras:

            # check if less than minsize
            if self.is_too_small(len(textsplit)):
                # if too small, return nothing
                return {'text':[], 'do_accept':False, 'sentences':[]}

            # return text to be truncated
            return {'text':[text_span_chunks], 'do_accept':True, 'sentences':[]}

        # leftover cases: text that needs to be chunked into ~512 / max_seq_len
        text_to_return, sentences_to_return = self.split_chunk_into_sentences(text_span_chunks)

        # return text strings as list of chunks, flag
        return {
            'text':text_to_return,
            'do_accept':True,
            'sentences':self.preprocess_sentences(sentences_to_return, self.min_sentence_length_char),
        }

    def __call__(self, text):
        return self.process(text)


class NegativeExampleGenerator:
    """Builds a queryable corpus of negative examples using ANN and approximate TFIDF vectors"""
    def __init__(
            self,
            n_reps:int = negative_config["n_reps"],# 1,
            n_takes:int = negative_config["n_takes"],# 5000,
            #dataset_name = 'cerebras/SlimPajama-627B',
            tfidf_nfeatures:int = negative_config["tfidf_nfeatures"],# 3000,
            nchar_max_paragraph:int= negative_config["nchar_max_paragraph"],#3000,
            nword_max:int= negative_config["nword_max"],#100,
            nchar_max_word:int= negative_config["nchar_max_word"],#4,
            max_sent_total:int = negative_config["max_sent_total"],# 5,
            corpus = negative_config["corpus"], #None,
            save_cache = negative_config["save_cache"],#'/tmp/negative_corpus_cache.pkl'
    ):
        self.stopwords =  [
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
            'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
            'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
            'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but',
            'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
            'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
            'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
            'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will',
            'just', 'don', 'should', 'now'
        ]
        self.n_reps = n_reps
        self.n_takes = n_takes
        self.tfidf_nfeatures = tfidf_nfeatures
        self.nchar_max_paragraph = nchar_max_paragraph
        self.nword_max = nword_max
        self.nchar_max_word = nchar_max_word
        self.max_sent_total = max_sent_total
        self.save_cache = save_cache
        if corpus is None:
            # fetch the corpus from streaming data (or reload from cache if available)
            print('warning: `corpus` is empty. Generating default corpus from RedPajama')
            self.corpus_static = self.fetch_default_corpus(self.save_cache)
        else:
            assert isinstance(corpus,list)
            assert len(corpus)>0
            print('using predefined corpus of length: %s' % len(corpus))
            self.corpus_static = corpus

        # build an ann index
        self.build_ann_index(self.corpus_static)

    def fetch_default_corpus(self, cache_file):
        """fetches streaming corpus and converts to a static list of data"""
        corpus_static = []
        if os.path.isfile(cache_file):
            print('reloading negative corpus %s for NegativeExampleGenerator' % cache_file)
            with open(cache_file, 'rb') as pcon:
                corpus_static = pickle.load(pcon)
                self.n_reps = pickle.load(pcon)
                self.n_takes = pickle.load(pcon)
                self.tfidf_nfeatures = pickle.load(pcon)
                self.nchar_max_paragraph = pickle.load(pcon)
                self.nword_max = pickle.load(pcon)
                self.nchar_max_word = pickle.load(pcon)
                self.max_sent_total = pickle.load(pcon)
                #self.max_sent_total = 5
        else:
            print('fetching streaming corpus for negatives (RedPajama)(%s reps)' % self.n_reps)
            # first do random draws from the corpora
            redpajama_set_name_support = ["RedPajamaCommonCrawl", "RedPajamaC4", "RedPajamaStackExchange", "RedPajamaWikipedia","RedPajamaBook", "RedPajamaArxiv"]
            for i_rep in range(self.n_reps):
                # load the streaming datasets (RedPajama)
                corpus_streaming = datasets.load_dataset(
                    'cerebras/SlimPajama-627B',
                    split="train",
                    streaming=True
                ).shuffle(
                    buffer_size = self.n_takes
                ).filter(
                    lambda x : x['meta']['redpajama_set_name'] in redpajama_set_name_support
                ).take(
                    self.n_takes
                ).remove_columns('meta')
                # convert streaming data to static and check language
                this_corpus_static = [
                    e['text'] for e in corpus_streaming #if langdetect(e['text'][:200]+' hello')=='en'
                    if check_language(e['text'])[0]
                ]
                # take only a few sentences per text
                this_corpus_static = [
                    self.limit_text_to_k_sentences(s, k=self.max_sent_total) for s in this_corpus_static
                ]
                # filtering again non-english
                this_corpus_static = [
                    s for s in this_corpus_static
                    if check_language(s)[0]
                ]
                # add
                corpus_static += this_corpus_static
                if (i_rep % 5)==0:
                    print('size of negative corpus: %d' % len(corpus_static))

            print('finished collecting streaming examples for negative corpus. Saving to %s' % self.save_cache)
            # save the cache
            with open(self.save_cache, 'wb') as pcon:
                pickle.dump(corpus_static, pcon)
                pickle.dump(self.n_reps, pcon)
                pickle.dump(self.n_takes, pcon)
                pickle.dump(self.tfidf_nfeatures, pcon)
                pickle.dump(self.nchar_max_paragraph, pcon)
                pickle.dump(self.nword_max, pcon)
                pickle.dump(self.nchar_max_word, pcon)
                pickle.dump(self.max_sent_total, pcon)

        return corpus_static

    def build_ann_index(self, corpus):
        """vectorizes a corpus and builds an ann index"""
        # stem words in preparation for tfidf vectorizer
        corpus_processed = [
            self.preprocess_text_to_index(s) for s in corpus
        ]
        # convert the corpus into tfidfvectors
        self.tfidfvectorizer = TfidfVectorizer(max_features=self.tfidf_nfeatures)
        self.tfidfvectorizer.fit(corpus_processed)
        self.corpus_vectors = self.tfidfvectorizer.transform(corpus_processed)

        # build the ann index
        self.ann_index = pynndescent.NNDescent(self.corpus_vectors)
        print('finished building the ANN index')

    @staticmethod
    def limit_text_to_k_sentences(text, k=5):
        """splits text into sentences, then limits the paragraph to just `k` sentences"""
        if len(text)<400:
            return text
        text = text[:10000]
        sentences = [s for s in re.split(r"(?<=\w\w\.)\s+",text) if len(s)>1]
        n_sent = len(sentences)
        if n_sent<=k:
            return text
        # if larger than limit, pick a (pseudo)random set of sentences
        random_sent_start_max_offset = n_sent-k
        random_sent_start_offset = ord(sentences[-1][:10][-1]) % random_sent_start_max_offset
        random_sent_end_offset = random_sent_start_offset + k
        return " ".join(sentences[random_sent_start_offset:random_sent_end_offset])

    def preprocess_text_to_index(self, text):
        """converts text into small k-character word stems before passing to TFIDF"""
        ptext = text[:self.nchar_max_paragraph].lower()
        ptext = ' '.join([
            w[:self.nchar_max_word] for w in ptext.split(' ')[:self.nword_max]
            if (w not in self.stopwords)
        ])
        ptext = re.sub("\W+",' ',ptext).strip()
        return ptext

    def process_query(self,text):
        """Vectorizes query text for retrieval"""
        query_processed = self.preprocess_text_to_index(text)
        return self.tfidfvectorizer.transform([query_processed]), query_processed

    def find_negative(self, query_text, k=1, skip=1):
        """Finds similar text to the query text, skipping the first `skip` and returning `k` top matches"""
        query_vector, query_processed = self.process_query(query_text)
        ann_idx,scores = self.ann_index.query(query_vector, k = k+skip)
        retrieved_text = [
            self.corpus_static[i] for i in ann_idx[0][skip:]
        ]
        retrieved_text = [
            s for s in retrieved_text
            if (
                s.lower().replace(" ","")[:100]!=query_text.lower().replace(" ","")[:100]
            )
        ]
        if len(retrieved_text)>0:
            return retrieved_text, scores
        # check that the texts are different
        skip+=1
        return self.find_negative(query_text, k=k, skip=skip)



    
