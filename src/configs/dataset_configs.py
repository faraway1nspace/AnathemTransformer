from src.configs.constants import *
from src.configs.dataset_cleaners import *

# MLM TASK Datasets configs per huggingface data: each entry is a: url, subset, probability, size, option(name of postprocess subsetting), shuffle?
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


# QA TASK Datasets configs per huggingface data: each entry is a: url, subset, probability, size, option(name of postprocess subsetting), shuffle?
QA_FILES = [
    ('embedding-data/PAQ_pairs',None, DEFAULT_PROB_QA, 7.29*10**6, 'qa_triplet', False), # wikipedia pop culture pairs # get from 'set', 7.29*10**6
    ('gbharti/finance-alpaca',None, DEFAULT_PROB_QA, 6.89*10**5, 'qa_triplet', False), # Stanford's Alpaca (https://github.com/tatsu-lab/stanford_alpaca) and FiQA (https://sites.google.com/view/fiqa/) with another 1.3k pairs custom generated using GPT3.5
    ('wiki_qa',None, DEFAULT_PROB_QA, 20.4*10**3, 'qa_triplet', False), # Wiki Question Answering corpus from Microsoft. with multiple negatives that are similar!
    ('donfu/oa-stackexchange',None, DEFAULT_PROB_QA*2, 6330000, 'qa_triplet', (14, int(6330000//14))), # stack-exchange question-answer pairs, across lots of domains; notice the original is 6.6 million, but there is a filter
    ('gart-labor/eclassTrainST', None, 0.02, 450912, 'qa_triplet', False), # questions about trade / business stuff
    ('THUDM/webglm-qa', None, DEFAULT_PROB_QA, 43600, 'qa_triplet', False),
    ('sciq',None, DEFAULT_PROB_QA, 11679, 'qa_triplet', False), # science questions from Allenai, with a question and support
    ('LLukas22/lfqa_preprocessed', None, DEFAULT_PROB_QA*1.5, 226000,'qa_triplet',False),# REDDIT QUESTION ANSWERS (ASK historians, ask me like I'M FIVE)
    ('npvinHnivqn/EnglishDictionary',None, DEFAULT_PROB_QA/4, 30864, 'qa_triplet',False), # 0.05 original size: 11200, post-file 30865
    ('alzoubi36/policy_qa', None, DEFAULT_PROB_QA/4, 17100,  'qa_triplet',False),
    ('sc2qa/sc2q_commoncrawl',None, DEFAULT_PROB_QA, 44500, 'qa_triplet', False),
    ('yahoo_answers_topics', None, DEFAULT_PROB_QA, 401357,'qa_triplet', False),
    ('launch/gov_report_qs','paragraph', DEFAULT_PROB_QA/5, 4878, 'qa_triplet', False),
    ('theoldmandthesea/17k_business_book', None, DEFAULT_PROB_QA/4, 17480, 'qa_triplet', False),
    ('sayhan/strix-philosophy-qa', None, DEFAULT_PROB_QA/2, 133799, 'qa_triplet',False), # philosophy questions
    ('BoltMonkey/psychology-question-answer',None, DEFAULT_PROB_QA/2, 197000, 'qa_triplet', False), # psychology questions
    ("sharsh02/Investopedia-Insights", None, DEFAULT_PROB_QA/2, 10558, 'qa_triplet', False)
]


STS_FILES = [
    # dataset name, subset, take_probability, dataset size
    ('xsum', None, DEFAULT_PROB_QA, 204000, 'sts_by_triplet', False),
    ('embedding-data/simple-wiki',None, DEFAULT_PROB_QA, 102000, 'sts_by_triplet', False), # wikipedia paraphrases
    ('embedding-data/coco_captions_quintets',None, DEFAULT_PROB_QA/2,82800, 'sts_by_triplet', False), # caption paraphrases
    ('embedding-data/SPECTER',None, DEFAULT_PROB_QA/2,684000, 'sts_by_triplet', False), # ?
    ('paws','labeled_final',DEFAULT_PROB_QA, 49400, 'sts_by_triplet', False), #
    ('embedding-data/QQP_triplets',None,DEFAULT_PROB_QA, 102000, 'sts_by_triplet', False), # quora?
    #("allenai/scirepeval", 'cite_prediction_new', DEFAULT_PROB_QA/2, 1300000, 'sts_by_triplet', False), # ? this takes a long time...maybe use smaller verion
    ("allenai/scirepeval", 'cite_prediction_aug2023refresh', DEFAULT_PROB_QA/2, 480000, 'sts_by_triplet', False), # ? this takes a long time...maybe use smaller verion
    ("lighteval/legal_summarization","BillSum", DEFAULT_PROB_QA, 18900, 'sts_by_triplet', False),
    ('https://drive.switch.ch/index.php/s/j9S0GRMAbGZKa1A/download?path=%2F&files=LEDGAR_2016-2019.jsonl.zip', None, DEFAULT_PROB_QA, 1000000, 'sts_by_label', False),
    ('eurlex', None, DEFAULT_PROB_QA, 45000, 'sts_by_label', False),
    ('humarin/chatgpt-paraphrases',None, DEFAULT_PROB_QA, 172059, 'sts_by_triplet', False),
    ('gigaword', None, DEFAULT_PROB_QA*1.5, 2000000, 'sts_by_triplet',False),
    ('pszemraj/govreport-summarization-8192',None, DEFAULT_PROB_QA/4, 10019, 'sts_by_triplet',False),
]


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


data_streaming_config_qa = {
    'files':QA_FILES,
    'max_seq_length':MAX_SEQ_LENGTH,
    'val_size':VAL_CORPUS_SIZE,
    'train_chunk_size':TRAINING_CORPUS_SIZE_PER_EPOCH,
    'seed':SEED,
    "path_cache_qa_val":PATH_CACHE_QA_VAL,
    "path_cache_qa_train":PATH_CACHE_QA_TRAIN,
}


data_streaming_config_sts = {
    'files':STS_FILES,
    'max_seq_length':MAX_SEQ_LENGTH
    'val_size':VAL_CORPUS_SIZE,
    'train_chunk_size':TRAINING_CORPUS_SIZE_PER_EPOCH,
    'seed':SEED,
    "path_cache_sts_val":PATH_CACHE_STS_VAL,
    "path_cache_sts_train":PATH_CACHE_STS_TRAIN,
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
    'seed':SEED,
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


# cleaning functions for the QA task
qa_streaming_cleaning_functions = {
    'embedding-data/PAQ_pairs':(clean_stream_PAQ_pairs, None, ['query','positives','negatives'],['set']),
    'gbharti/finance-alpaca':(clean_stream_finance_alpaca,None, ['query','positives','negatives'],['input', 'output', 'text', 'instruction']),
    'wiki_qa':(clean_stream_wiki_qa, None, ['query','positives','negatives'],['question_id', 'question', 'document_title', 'answer', 'label']),
    'donfu/oa-stackexchange':(clean_stream_oa_stackexchange, filter_os_stackexchange, ['query','positives','negatives'], ['INSTRUCTION', 'RESPONSE', 'SOURCE', 'METADATA']),
    'gart-labor/eclassTrainST':(clean_eclassTrainST, None, ['query','positives','negatives'], ['text', 'entailment', 'contradiction', 'label']),
    'THUDM/webglm-qa':( clean_webglmqa, None, ['query','positives','negatives'], ['question','answer','references']),
    'sciqa': (clean_stream_sciqa, None, ['query','positives','negatives'], ['question', 'distractor3', 'distractor1', 'distractor2', 'correct_answer', 'support']),
    'LLukas22/lfqa_preprocessed':(clean_lfqa, None, ['query','positives','negatives'], ['question','answer','context']), #REDDIT QUESTION ANSWERS (ASK historians, ask me like I'M FIVE)
    'npvinHnivqn/EnglishDictionary':(clean_dictionary, filter_dictionary, ['query','positives','negatives'], ['word','definition']), # dictionaries
    'alzoubi36/policy_qa':(clean_policyqa, None, ['query','positives','negatives'],  ['id', 'title', 'context', 'question', 'answers'] ), # PRIVACYGLUE
    'sc2qa/sc2q_commoncrawl':(clean_sc2qa, None, ['query','positives','negatives'], ['question','article','url']),
    'yahoo_answers_topics':(clean_yahooanswers, filter_yahooanswers, ['query','positives','negatives'], ['id', 'topic', 'question_title', 'question_content', 'best_answer']),
    'launch/gov_report_qs':(clean_govreportqa, None, ['query','positives','negatives'],['doc_id', 'summary_paragraph_index', 'document_sections', 'question_summary_pairs']),
    'theoldmandthesea/17k_business_book':(clean_businessbookqa, None, ['query','positives','negatives'], ['question','answer','book']),
    'sayhan/strix-philosophy-qa':(clean_strixphilosophyqa, None, ['query','positives','negatives'], ['category','question','answer']),
    "BoltMonkey/psychology-question-answer":(clean_psychologyquestionanswer,None, ['query','positives','negatives'], ['question','answer']),
    "sharsh02/Investopedia-Insights":(clean_investopediaqa, None, ['query','positives','negatives'], ['prompts','response'])
}


# cleaning functions for the STS/retrieval task
sts_streaming_cleaning_functions = {
    'xsum':(clean_xsum, None, ['query','positives','negatives'],['summary','id','document']),
    'embedding-data/simple-wiki':(clean_simple_wiki, None, ['query','positives','negatives'],['set']),
    'embedding-data/coco_captions_quintets':(clean_coco_captions_quintets,None, ['query','positives','negatives'],['set']),
    'embedding-data/SPECTER':(clean_specter,None, ['query','positives','negatives'],['set']),
    'paws':(clean_paws,None, ['query','positives','negatives'],['id', 'sentence1', 'sentence2', 'label']), # NOTE: these are adversarial paraphrases -- the negatives should be ignored
    'embedding-data/QQP_triplets':(clean_qqp,None, ['query','positives','negatives'],['set']),
    "allenai/scirepeval":(clean_allenai_citeprediction, None,  ['query','positives','negatives'], ['pos','neg']),
    "lighteval/legal_summarization":(clean_legalsum, None, ['query','positives','negatives'], ['article', 'summary']),
    "https://drive.switch.ch/index.php/s/j9S0GRMAbGZKa1A/download?path=%2F&files=LEDGAR_2016-2019.jsonl.zip":(
        clean_ledgarlabelled, None, ['query','label'], ['provision','source']
    ),
    "eurlex":(clean_eurlex, None,  ['query','positives','negatives'], ['celex_id', 'title', 'text', 'eurovoc_concepts']),
    'humarin/chatgpt-paraphrases':(clean_chatgptparaphrases, filter_chatgptparaphrases, ['query','positives','negatives'], ['text','paraphrases','category','source']),
    'gigaword':(clean_gigaword, None, ['query','positives','negatives'], ['document','summary']),
    'pszemraj/govreport-summarization-8192':(clean_govreportsumm, None, ['query','positives','negatives'],['report', 'summary', 'input_token_len', 'summary_token_len'])
}
