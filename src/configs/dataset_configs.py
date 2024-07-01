from src.configs.constants import *
from src.configs.dataset_cleaners import *


# entries: url, subset, probability, size, option(name of postprocess subsetting), shuffle?
MLM_FILES = [
    ('Pavithree/askHistorians',None, 0.6, 51300,'mlm',False, 0.8),
    ("Cohere/wikipedia-22-12", 'en', 42.0, 8590000, "mlm",(351, 100000), 0.16), # wikipedia has 351 files (each with 100000 examples)
    ('wikimedia/wikipedia','20231101.en', 7.0, 41*156288,"mlm",(41,156288),0.1), # wikipedia long docs
    ('EleutherAI/the_pile_deduplicated', None, 16.0, 134000000, 'mlm', (1650, 81405), 0.09), # 1650 files each with ~?
    # monology/pile-uncopyrighted -> this seems like the original pile that I was using, with book3 removed
    ("tiiuae/falcon-refinedweb", None, 17.5, 968000000, "mlm", (5534, 174000), 0.09), # CC; has 5534 files as parquet (each with ~174919)
    ('Skylion007/openwebtext', None, 5.5, 4000000, 'mlm', (21, 213000), 0.1),
    #("Multi-Domain-Expert-Layers/the_pile_books3_packed_128k", None, 4.8/2, 34500, "mlm", (15, 9900), 0.15), # has 15 files (each with with ~9978/9983)
    ("P1ayer-1/books-3-textbooks", None, 5.2/4, 5439, "mlm", (7, 777), 0.15), # has 7 files (each with with 777)
    #("nRuaif/book2-lite-cleaned", None, 4.8/2, 81500, "mlm", (818, 100), 0.1), # I
    ("Chat-Error/book2-lite-cleaned", None, 4.6/2, 81500, "mlm", (816, 100), 0.1), # I
    ("macrocosm/arxiv_abstracts", None, 3.8, 2250000, "mlm", (23, 2250000//23), 0.12), # set to zero because in PILE (has 23 parquet files)
    ("ccdv/pubmed-summarization", None, 0, 120000, "mlm", False, 0.12), # 3.75 set to zero because elsiever and pubmed in Pile below
    ('big_patent', 'all', 0.80, 154000, 'mlm', False, 0.15), # use as an alternative to /NIH_ExPORTER_awarded_grant_text.jsonl.zst
    ("pile-of-law/pile-of-law",'euro_parl', 0.46, 7254, "mlm", False, 0.1),
    # I think I should remove the hackernews because it was originally included as a discussion-tree in pile
    ('kerinin/hackernews-stories', None, 0, 31300, 'mlm', (8, 52220), 0.1), # 1.7 hackernews stories alternative: this was originally included because of discussion
    ("https://the-eye.eu/public/AI/pile_v2/data/NIH_ExPORTER_awarded_grant_text.jsonl.zst", None, 0, 985651, "mlm", False, 0.15), # still works, but may fail eventually
    ("https://drive.switch.ch/index.php/s/j9S0GRMAbGZKa1A/download?path=%2F&files=LEDGAR_2016-2019.jsonl.zip", None, 9.0, 1200000, "mlm", False, 0.2),
    ("albertvillanova/legal_contracts", None, 1.1, 106000, 'mlm', False, 0.15),
    ("pile-of-law/pile-of-law",'r_legaladvice', 1.63, 109740, "mlm", False, 0.15),
    ("pile-of-law/pile-of-law",'exam_outlines', 0.1, 12, "mlm",False, 0.2), # useless (but interesting)
    ("pile-of-law/pile-of-law",'cc_casebooks',0.5, 59 ,"mlm",False, 0.2),
    ("eloukas/edgar-corpus", "full", 1.9, 47000, "mlm",(28, 4000), 0.15), # has 28 files each with 1k-5k (variable amount of data: 1styear 1060 vs 5508 in 2018
    ("Rahmaa/ElsevieR_ClEaN", None, 1.7, 31600, "mlm", False, 0.15),
    ('ashraq/financial-news-articles', None, 1.0, 306000, "mlm", (2, 153100), 0.1), # has 2 files (each with 153121)
    ('pile-of-law/pile-of-law','courtlistener_opinions',  0.8, 1000000 , "mlm", (16, 229000), 0.1), # has 16 files (each with 229678 to 526543)
    ('pile-of-law/pile-of-law',"sec_administrative_proceedings", 0.9, 10805, "mlm", False, 0.1), # 118.4 MiB
    ('pile-of-law/pile-of-law',"irs_legal_advice_memos", 0.76/4, 442, "mlm", False,0.18), # 35.8 MiB
    ('Aukandseal/IRSpubs_2020-2024',None,0.76/4,2379,"mlm",False, 0.20), # my own documents
    ('launch/gov_report','plain_text',0.65, 17500, 'mlm', False, 0.1),
    ('izumi-lab/open-text-books',None,  3.5, 150000, 'mlm', False, 0.15),
    ('gigant/ted_descriptions',None, 0, 5705, 'mlm', False, 0.2), # too small and irrelevant
    ('Skelebor/book_titles_and_descriptions', None, 2.24, 1000000,'mlm', (2, 1000000//2), 0.2),
    ('joelito/legal_case_document_summarization',None, 2.2, 7700, 'mlm', False, 0.2),
    ('joelito/legal-mc4','en', 1.1, 180000, 'mlm', False, 0.1),
    ('Hellisotherpeople/DebateSum', None, 1.58, 24647, 'mlm',False, 0.9),
    ('lukesjordan/worldbank-project-documents', None, 0.45, 15700, 'mlm', False, 0.08),
    ('64bits/lex_fridman_podcast_for_llm_vicuna',None, 0.7, 17200,'mlm',False,0.5),
    ('nid989/EssayFroum-Dataset',None, 0.88, 25600,'mlm',False,0.5),
    ('nlpaueb/finer-139',None, 1.2, 179195, 'mlm',False, 0.8),
    ('squad',None, 2.0/2, 87600, 'mlm', False, 0.2),
    ("Isotonic/human_assistant_conversation",None,1.4, 58700, 'mlm',(3, 195590),0.09),
    ("hugfaceguy0001/stanford_plato",None, 0.33, 1776,'mlm', False, 0.4),
    ("nvidia/ChatQA-Training-Data", "synthetic_convqa", 2*2/3, 40000,'mlm',False,0.1), # same source
    ("nvidia/ChatQA-Training-Data", "tatqa", 2*1/3, 11500,'mlm',False,0.3), # # same source
] #


data_streaming_config_mlm = {
    'files':MLM_FILES,
    'val_size':VAL_CORPUS_SIZE, #2000,
    'min_seq_length':MIN_SEQ_LENGTH,
    'max_seq_length':MAX_SEQ_LENGTH,
    'max_chunk_size':MAX_CHUNK_SIZE,
    'train_chunk_size':TRAINING_CORPUS_SIZE_PER_EPOCH,
    'max_chunk_start':1000000,
    "seed":SEED,
    "path_cache_mlm_val":PATH_CACHE_MLM_VAL,
    "path_cache_mlm_train":PATH_CACHE_MLM_TRAIN,
}



# configuration for the streaming mlm set
clsdata_streaming_config = {
    'files':'cls_files',
    'max_seq_length':MAX_SEQ_LENGTH,
    'val_size':500,
    'train_chunk_size':TRAINING_CORPUS_SIZE_PER_EPOCH,
    'seed':SEED,
}


# example processor config
example_processor_config = {
    'max_seq_length':MAX_SEQ_LENGTH,
    'min_seq_length':MIN_SEQ_LENGTH,
    'max_chunk_size':MAX_CHUNK_SIZE,
    'min_sentence_len':20,
    'seed':SEED
}


# for building the negative corpus
negative_config = {
    "n_reps": 1,
    "n_takes": NEGATIVE_CORPUS_SIZE,
    "tfidf_nfeatures": 3000,
    "nchar_max_paragraph":3000,
    "nword_max":100,
    "nchar_max_word":4,
    "max_sent_total": 5,
    "corpus" : None,
    "save_cache":'/tmp/negative_corpus_cache.pkl'
}



# cleaning function for the MLM task
mlm_streaming_cleaning_functions = {
    'Pavithree/askHistorians':(clean_askhistorians, filter_askhistorians, ['q_id','title','selftext','document','subreddit','url','answers']),
    "Cohere/wikipedia-22-12":(lambda x : x, None, ['id', 'title', 'url', 'wiki_id', 'views', 'paragraph_id', 'langs']),
    'wikimedia/wikipedia':(lambda x: x, None, ['id', 'url', 'title']),
    #'EleutherAI/pile/all':(lambda x: x, filter_pileall_mlm, ['meta']), # GONE
    'EleutherAI/the_pile_deduplicated':(lambda x: x, filter_notcodelike, []),
    # monology/pile-uncopyrighted -> this seems like the original pile that I was using, with book3 removed
    "tiiuae/falcon-refinedweb":(clean_stream_refinedweb, None, ['url', 'timestamp', 'dump', 'segment', 'image_urls','content']),
    'Skylion007/openwebtext':(lambda x : x, None, []),
    #"Multi-Domain-Expert-Layers/the_pile_books3_packed_128k":(lambda x: x, None, ['meta']),
    "P1ayer-1/books-3-textbooks":(clean_player1book3, None, ['title', 'authors']),
    #"nRuaif/book2-lite-cleaned":(lambda x: {'text':x['text'][1000:]}, None, []), # old book2 now Chat-Error/book2-lite-cleaned
    "Chat-Error/book2-lite-cleaned":(lambda x: {'text':x['text'][1000:]}, None, []),
    "macrocosm/arxiv_abstracts":(clean_stream_arxiv, None, ['embeddings', 'doi','abstract']),
    "ccdv/pubmed-summarization":(clean_stream_pubmedsum, None, ['abstract','article']),
    'big_patent':(clean_bigpatent, None, ['description', 'abstract']),
    "pile-of-law/pile-of-law/euro_parl":(lambda x : x, filter_europarl_mlm, ['created_timestamp', 'downloaded_timestamp', 'url']),
    #"philArchive": fails, but available as subset in eloukas/edgar-corpus as domain=='PhilPapers'
    'kerinin/hackernews-stories':(clean_hackernews, filter_hackernews, ['Title','Text','labels']),
    "https://the-eye.eu/public/AI/pile_v2/data/NIH_ExPORTER_awarded_grant_text.jsonl.zst":(lambda x:x, None,['meta']),
    "https://drive.switch.ch/index.php/s/j9S0GRMAbGZKa1A/download?path=%2F&files=LEDGAR_2016-2019.jsonl.zip":(clean_ledgarmlm,None,['provision','source']),
    "albertvillanova/legal_contracts":(clean_legalcontractslong, None,[]),
    "pile-of-law/pile-of-law/r_legaladvice":(lambda x : x, None, ['created_timestamp', 'downloaded_timestamp', 'url']),
    "pile-of-law/pile-of-law/exam_outlines":(lambda x : x, None, ['created_timestamp', 'downloaded_timestamp', 'url']),
    "pile-of-law/pile-of-law/cc_casebooks":(clean_casetextbook, None, ['created_timestamp', 'downloaded_timestamp', 'url']), # clean_casetextbook
    "eloukas/edgar-corpus":(
        clean_edgarcorpus, None, [
            'filename', 'cik', 'year', 'section_1A', 'section_1B', 'section_4', 'section_1', 'section_2', 'section_3', 'section_7',
            'section_5', 'section_6', 'section_8', 'section_9', 'section_10', 'section_7A', 'section_9A', 'section_9B',
            'section_11', 'section_12', 'section_13', 'section_14', 'section_15'
        ]),
    "Rahmaa/ElsevieR_ClEaN":(clean_elseiver_mlm, None, ['Unnamed: 0', 'Clean_Title', 'Clean_Text', 'Clean_Summary']),
    'ashraq/financial-news-articles':(clean_financial_news_mlm, None, ['title','url']),
    'pile-of-law/pile-of-law/courtlistener_opinions':(clean_courtlistener, None, ['created_timestamp', 'downloaded_timestamp', 'url']),
    "pile-of-law/pile-of-law/sec_administrative_proceedings":(clean_secproceedings_mlm, None, ['created_timestamp', 'downloaded_timestamp', 'url']),
    "pile-of-law/pile-of-law/irs_legal_advice_memos":(clean_irs_advice_mlm, None, ['created_timestamp', 'downloaded_timestamp', 'url']),
    'Aukandseal/IRSpubs_2020-2024':(lambda x: x, None, []),
    'launch/gov_report':(clean_govreport, None, ['id','document','summary']),
    'izumi-lab/open-text-books':(lambda x: x, None, []),
    'gigant/ted_descriptions':(lambda x: {'text':x['descr']}, None, []),
    'Skelebor/book_titles_and_descriptions':(lambda x : {'text': x['description']},lambda x : len(str(x['description']))>80, []),
    'joelito/legal_case_document_summarization':(lambda x: {'text':x['summary']}, None, []),
    'joelito/legal-mc4/en':(lambda x:x, None, ['url','timestamp','matches']),
    'Hellisotherpeople/DebateSum':(clean_debatesum, filter_debatesum,[
        '#CharsAbstract', '#CharsDocument', '#CharsExtract', '#WordsAbstract', '#WordsDocument', '#WordsExtract', 'AbsCompressionRatio', 'Abstract', 'Citation',
        'DebateCamp', 'ExtCompressionRatio', 'Extract', 'Tag', 'Unnamed: 0', 'Year', 'Full-Document','OriginalDebateFileName'
    ]),
    'lukesjordan/worldbank-project-documents':(clean_worldbank, None, ['project_id','document_text','document_type']),
    '64bits/lex_fridman_podcast_for_llm_vicuna':(clean_lexfridmanchat, None, ['conversations','id']),
    'nid989/EssayFroum-Dataset':(clean_essayforum, None, ['Cleaned Essay','Correct Grammar']),
    "nlpaueb/finer-139":(clean_finer139_for_mlm, filter_finer139, ['ner_tags','tokens','id']),
    'squad':(clean_squad, None, ['context','question','answers','title','id']),
    "Isotonic/human_assistant_conversation":(clean_isotonicconversations, None, ["prompt","response"]),
    "hugfaceguy0001/stanford_plato":(clean_stanfordplato, None, [
        'shorturl', 'title', 'pubinfo', 'preamble', 'toc', 'main_text', 'bibliography', 'related_entries'
    ]),
    "nvidia/ChatQA-Training-Data/synethetic_convqa":(clean_nvidiaqa, None, ['document', 'messages', 'answers']),
    "nvidia/ChatQA-Training-Data/tatqa":(clean_nvidiaqa, None, ['document', 'messages', 'answers'])
}
