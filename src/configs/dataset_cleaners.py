import re
from math import prod
from typing import Tuple, Union, Dict, List, Any
from src.configs.constants import *
from src.configs.dataset_templates import *


def random_by_char(text:str, take:int=3, charlim:int=10):
    """Pseudo random number based on an imput text. Determinisitic."""
    nums = [ord(ch) for ch in 'xqz'+text.replace(' ','')[:charlim]][(-1*take):]
    return 3*prod(nums[:2])-nums[-1]


def check_is_code(text:str)->float:
    """Estimates a ratio of special char (that may indicate math/code notation); less than 10% is code for normal text."""
    nchar = min(5000,len(text))
    nchar_after_removespecialchar = len(re.sub(r"[\<\>\_\@\^\=\+\*\$\{\[\]\}\(\)\/\\\.]",'',text[:5000]))
    ratio_specialchar = 1-nchar_after_removespecialchar/nchar
    return ratio_specialchar


def clean_squad(x:Dict[str,str])->Dict[str,str]:
    """Converst squad triplet (q,context, a) into a pseudo-conversation using templates."""
    passagetext = x['context']
    template = TEMPLATES_SQUAD[random_by_char(passagetext) % len(TEMPLATES_SQUAD)]
    text = template.replace(
        "{QUESTION}", x['question']
    ).replace(
        "{ANSWER}", x['answers']['text'][0]
    ).replace(
        "{PASSAGE}", passagetext
    )
    return {'text':text}


def filter_finer139(x:Dict[str,str])->bool:
    return sum(x['ner_tags'])>0


def clean_finer139_for_mlm(x:Dict[str,Any])->Dict[str,str]:
    tokens,tags = x['tokens'], x['ner_tags']
    passage = re.sub("(?<=\w)\s+(?=[\’\,\;\.\”\)])","",re.sub("(?<=\d)\s\%","%",re.sub("\$\s(?=\d)","$"," ".join(tokens))))
    #print(passage)
    concept_answers = [
        (w,FINER139_CLASSES[t].strip(),i)
        for i,(w,t) in enumerate(zip(tokens, tags)) if t!=0
    ]
    which_take = ord(passage[:20].replace(" ","")[-1])
    ans_triplet = concept_answers[which_take % len(concept_answers)]
    # get all answers of same classe
    other_answers_of_same_class = [a for a in concept_answers if a[1]==ans_triplet[1]]
    if len(other_answers_of_same_class)==1:
        tnumber,ansclass,idx = ans_triplet
        is_dollar = "$" if tokens[idx-1]=='$' else ""
        is_unit = tokens[idx+1] if tokens[idx+1].lower() in ['million','billion','hundred','thousand','percent','%','hundred-thousand'] else ""
        tnumber = (is_dollar + tnumber + " "+is_unit).strip().replace(' %',"%")
        ansprefix = {0:'a',1:'an',2:"the",3:'the'}[int(ansclass[0] in ['a','e','i','o','u','y'])+2*(ansclass[-1]=='s')]
        ansclass = {0:ansclass.title(), 1:ansclass}[which_take % 2]
        template = TEMPLATE_FINER139[which_take % len(TEMPLATE_FINER139)]
        text = template.replace("{NUMBER}",tnumber).replace("{ANSWER}",ansclass).replace("{PREFIX}",ansprefix).replace('{PASSAGE}',passage)
        return {'text':text}
    else:
        multi_number= []
        for ans_triplet in other_answers_of_same_class:
            tnumber,ansclass,idx = ans_triplet
            is_dollar = "$" if tokens[idx-1]=='$' else ""
            is_unit = tokens[idx+1] if tokens[idx+1].lower() in ['million','billion','hundred','thousand','percent','%','hundred-thousand'] else ""
            tnumber = (is_dollar + tnumber + " "+is_unit).strip().replace(' %',"%")
            multi_number.append(tnumber)
        multi_number = list(set(multi_number))
        # combine the multi numbers
        sep = {0:'/',1:' / ', 2:' and ', 3: ' & ', 4:' + '}[which_take % 5]
        tnumber = ", ".join(multi_number[:-1]) + sep + multi_number[-1]
        ansprefix = {0:'a',1:'an',2:"the",3:'the'}[int(ansclass[0] in ['a','e','i','o','u','y'])+2*(ansclass[-1]=='s')]
        ansclass = {0:ansclass.title(), 1:ansclass}[which_take % 2]
        template = TEMPLATE_FINER139[which_take % len(TEMPLATE_FINER139)]
        text = template.replace("{NUMBER}",tnumber).replace("{ANSWER}",ansclass).replace("{PREFIX}",ansprefix).replace('{PASSAGE}',passage)
        return {'text':text}


def clean_stream_refinedweb(x):
    x['text'] = x['content']
    return x


def clean_stream_arxiv(x):
    x['text'] = x['abstract']
    return x


def clean_stream_pubmedsum(x):
    x['text'] = x['article']
    return x


def remove_first_http_url(text):
    """Removes http strings from hackersnews"""
    pattern = r'http[s]*://[^ ]+'
    return re.sub(pattern, '', text, 1)


def clean_ledgarmlm(x:Dict[str,Any])->Dict[str,Any]:
    x['text'] = x['provision']
    return x


def clean_casetextbook(example:Dict[str,Any])->Dict[str,Any]:
    """Removes tables and excess \n includes somes specifics for Saylor books footmatter."""
    example = clean_irs_advice_mlm(example)
    # discard the first 8 lines ~ they are usually boilerplate text
    example['text'] = '\n'.join(example['text'].split('\n')[8:])
    example['text'] = example['text'].replace("Saylor URL: http://www.saylor.org/books"," ").replace("Saylor.org", " ").replace('Saylor Books', " ")
    return example


def clean_edgarcorpus(example:Dict[str,Any])->Dict[str,Any]:
    """Clean and process the edgar corpus."""
    example['text'] = example['section_1'] + "\n" + example['section_2'] + "\n" + example['section_3'] + "\n" + example['section_7']
    return example


def clean_elseiver_mlm(example:Dict[str,Any])->Dict[str,Any]:
    """Clean and process the edgar corpus."""
    example['text'] = example['Clean_Title'] + " - " + example['Clean_Summary'] + "\n" + example['Clean_Text']
    return example


def clean_financial_news_mlm(example):
    example['text'] = example['title'] + "\n" + example['text']
    return example


def filter_pileall_mlm(x):
    return x['meta']['pile_set_name'] in ['NIH ExPorter','OpenWebText2','PubMed Abstracts','StackExchange','Wikipedia (en)','ArXiv']


def filter_europarl_mlm(x):
    return len(x['text'])>60*7 # at least a small paragraphs


def clean_courtlistener(x:Dict[str,Any])->Dict[str,Any]:
    """Clean the courlistener dataset"""
    text = x['text']
    text = text.replace(".\n",'XxXx').replace(":\n",'YyYy').replace("-\n",'').replace("\n"," ").replace('XxXx','.\n').replace('YyYy',':\n')
    return {'text':text}


def clean_irs_advice_mlm(x):
    text = x['text']
    pattern = r'\x0C'
    text = re.sub(pattern, "", text) # ^L characters
    text = re.sub(r'^[\d,.%$+\-\s\=]+\n?$',"",text,flags=re.MULTILINE | re.DOTALL)
    text = re.sub(r'\-{10,}',"",text)
    text = re.sub(r'^(.*)?[Pp]age\s\d+\n?$',"",text,flags=re.MULTILINE)
    #x['text'] = text.replace("\n"," ").strip()
    text = text.replace(".\n",'XxXx').replace(":\n",'YyYy').replace("-\n",'').replace("\n"," ").replace('XxXx','.\n').replace('YyYy',':\n')
    # find tables and remove
    text = "\n".join([s for s in text.split('\n') if not is_potential_table(s)])
    return {'text':text}


def clean_secproceedings_mlm(x):
    text = x['text']
    if 'I.\n' in text:
        text = "".join(re.split(r"^I.\n", text, flags=re.MULTILINE)[1:])
    else:
        text = '\n'.join(text.split('\n')[10:])
    # I don't remember what this removes
    pattern = r'\x0C'
    s = re.sub(pattern, "", text) # ^L characters
    # removes a number ( or (12)) that is just a line with no text
    text = re.sub(r'^(\()*\d+[\.\)]?\n?$', '', text,flags=re.MULTILINE)
    # remove sentence-breaks
    s = s.replace(".\n",'XxXx').replace(":\n",'YyYy').replace("-\n",'').replace("\n"," ").replace('XxXx','.\n').replace('YyYy',':\n')
    s = s.replace('¶',' ')
    x['text'] = s
    return x


def filter_notcodelike(x):
    """checks if text has a lot of non-alphanumeric characters that indicates it is probably computer code / math notation"""
    ratio_specialchar = check_is_code(x['text'])
    return ratio_specialchar<0.1


def clean_hackernews(x):
    x['text']= x['Title'] + ' ' + remove_first_http_url(x['Text'])
    return x


def filter_hackernews(x):
    return (len(x['Text']) > 60) and (check_is_code(x['Text'])<0.1)


def clean_bigpatent(x):
    start_offset=0; end_offset=1
    if 'BACKGROUND OF THE INVENTION' in x['description']:
        start_offset = x['description'].index('BACKGROUND OF THE INVENTION')+27
    elif 'BACKGROUND OF INVENTION' in x['description']:
        start_offset = x['description'].index('BACKGROUND OF INVENTION')+23
    elif 'BACKGROUND' in x['description']:
        start_offset = x['description'].index('BACKGROUND')+10
    if 'SUMMARY OF THE INVENTION' in x['description']:
        end_offset = x['description'].index('SUMMARY OF THE INVENTION')
    elif 'SUMMARY OF INVENTION' in x['description']:
        end_offset = x['description'].index('SUMMARY OF INVENTION')
    elif 'SUMMARY' in x['description']:
        end_offset = x['description'].index('SUMMARY')
    if end_offset < (start_offset+20):
        return {
            'text': '\n'.join(x['description'].split('\n')[:4]) + x['abstract']
        }
    background_text = x['description'][start_offset:end_offset]
    # remove all [xxxx] number breaks
    background_text = re.sub("\[[0-9]+\]\s*","", background_text)
    return {
        'text': (background_text.strip() + "\n"+ x['abstract'])
    }

def clean_govreport(x):
    x['text'] = x['document']
    return x

def filter_debatesum(x):
    """fiters out extremist/hateful content from the debatesum dataset (auto-labelled, poor precision)"""
    if len(str(x['Full-Document']).split(' '))<400:
        return False
    if x['OriginalDebateFileName'] in DEBATESUM_EXTREMIST_FILTER_OUT1:
        return False
    if x['OriginalDebateFileName'] in DEBATESUM_EXTREMIST_FILTER_OUT2:
        return False
    if x['OriginalDebateFileName'] in DEBATESUM_EXTREMIST_FILTER_OUT3:
        return False
    if x['OriginalDebateFileName'] in DEBATESUM_EXTREMIST_FILTER_OUT4:
        return False
    if x['OriginalDebateFileName'] in DEBATESUM_EXTREMIST_FILTER_OUT5:
        return False
    if x['OriginalDebateFileName'] in DEBATESUM_EXTREMIST_FILTER_OUT6:
        return False
    if x['OriginalDebateFileName'] in DEBATESUM_EXTREMIST_FILTER_OUT7:
        return False
    return True


def clean_debatesum(x):
    x['text'] = re.sub(r'\s+'," ",str(x['Full-Document']))
    return x


## world bank processing functions
def is_potential_table(line):
    """Checks if a text is a table"""
    num_and_special_count = sum([len(w) for w in re.findall(r'[0-9.,=\(\)$€£*\%\-\/\:]+', line)])
    total_chars = len(line)
    if total_chars==0:
        return True
    num_and_special_ratio = num_and_special_count / total_chars
    if num_and_special_ratio > 0.25:
        return True
    words = line.split()
    upper_title_words = [word for word in words if word.isupper() or word.istitle()]
    ratio = len(upper_title_words) / len(words) if len(words) > 0 else 0
    if ratio > 0.5:
        return True
    # Heuristic 3: Check if the line starts with a number (common in tables)
    if re.search("^[0-9]", line) and re.search("[0-9]$", line):
        return True
    nchar = len(re.sub(r"\s+","",line))
    nspace = len(re.sub(r"\w+","",line))
    if nspace !=0 and nchar/nspace <3.2:
        return True
    return False


def clean_worldbank(x):
    # remove weird characters
    pattern = r'\x0C'
    s = re.sub(pattern, "", x['document_text']) # ^L characters
    s = re.sub(r'\\\.', ".", s) # \. artifacts
    # remove excess/inccorect \n breaks
    s = s.replace(".\n",'XxXx').replace(":\n",'YyYy').replace("-\n",'').replace("\n"," ").replace('XxXx','.\n').replace('YyYy',':\n')
    s = s.replace('¶',' ')
    # discard text that is clearly tabular
    s_cleaned = "\n".join([
        p for p in s.split('\n') if not is_potential_table(p)
    ])
    return {'text':s_cleaned}



def clean_lexfridmanchat(x):
    convo = x['conversations']
    nm1,nm2 = LIST_OF_HOSTGUEST_PAIRS[ord(convo[0]['value'].replace(' ',"")[:20][-1]) % len(LIST_OF_HOSTGUEST_PAIRS)]
    map_to_names = {'human':nm1, 'lex':nm1,'gpt':nm2,'guest':nm2}
    text = '\n'.join([
        '%s: "%s"' % (map_to_names[talkfrag['from']], talkfrag['value']) for talkfrag in convo
    ])
    return {'text':text}


def clean_essayforum(x):
    return {'text':x['Correct Grammar']}



def filter_askhistorians(x):
    if len(x['answers']['text'])==0:
        return False
    if len(x['title'])<10:
        return False
    if '?' not in x['title']:
        return False
    return x

def make_text_for_askhistorians(question, answers, selftext):
    """splits answers into multiple separable texts
    [print(k+'\n-----\n') for k in make_text_for_askhistorians("Hello?", ["a1",'a2','a3'], "this is ome text").split('%0XTEXTXEPARAT0RX%0')]
    """
    if len(answers)>1:
        out_texts_listed = []
        for answer in answers:
            out_texts_listed.append(make_text_for_askhistorians(
                question, [answer], selftext
            ))
        return TEXTSEPARATOR.join(out_texts_listed)

    # no self text
    if len(selftext)<2:
        # find a template
        template = TEMPLATES_ASKHISTORIANS['no_context'][
                random_by_char(answers[0]) % len(TEMPLATES_ASKHISTORIANS['no_context'])
        ]
        out_text = template.replace("{QUESTION}",question).replace("{ANSWER}",answers[0])
        return out_text
    # yes self-text
    out_texts_listed = [make_text_for_askhistorians(question, answers, "")]
    # find a template
    template = TEMPLATES_ASKHISTORIANS['context'][
        random_by_char(answers[0]) % len(TEMPLATES_ASKHISTORIANS['context'])
    ]
    out_texts_listed += [template.replace("{QUESTION}",question).replace("{ANSWER}",answers[0]).replace("{SELFTEXT}", selftext)]
    return TEXTSEPARATOR.join(out_texts_listed)


def clean_askhistorians(x):
    out_text=  make_text_for_askhistorians(
        x['title'], x['answers']['text'], x['selftext']
    )
    return {'text':out_text}



def clean_isotonicconversations(x):
    """Randomizes the names of speakers and question-askers, as well as removes other non-natural language text"""
    text = x['text']
    if len(list(re.findall("####Human####:",text)) + list(re.findall(r"\n+human\:",text))) == 1:
         names_of_agents = [('Question', "Answer")]*2 + [("Q",'A')] + [('Question', "Response")]
    else:
        names_of_agents = PAIRS_OF_USERASSISTANT_NAMES
    name_human, name_bot = names_of_agents[random_by_char(text,charlim=20) % len(names_of_agents)]
    # easy replace: expect template of ####human###
    text = text.replace("####Human####",name_human).replace("####Assistant####",name_bot)
    # more difficult for weird follow-up questions
    if 'human:' in text.lower() or 'humans:' in text.lower():
        text = re.sub('\n+Huma(n|ns)\:',"\n\n"+name_human+":", text, flags=re.MULTILINE)
    if 'assistant:' in text.lower():
        text = re.sub('\n+Assistant\:',"\n\n"+name_bot+":", text, flags=re.MULTILINE)
    text = text.replace('<|stop|>',"").replace('\nOutput:',"")
    # remove lines that are just a single number
    #text = re.sub("\:\n(?=\d)","^XxXx^",text,flag).
    return {'text':text}



def clean_nvidiaqa(x, max_len = 512, fudge_factor=1.37):
    questions = x['messages']
    question = questions[0]['content']
    context = x['document']
    if len(questions)==1:
        answer = x['answers'][0]
    elif len(questions)>1 and (questions[1]['role']=="assistant"):
        answer = questions[1]['content']
    # find a template
    template = TEMPLATES_NVIDIAQA[
        random_by_char(question+answer) % len(TEMPLATES_NVIDIAQA)
    ]
    template_text_half = template.replace("{QUESTION}",question).replace("{ANSWER}",answer)
    # calculate the amount of tokens consumed for Q+A+template
    ntokens_used_for_qa_and_template = len(template_text_half.split(" "))*fudge_factor
    # calculate differnce we need to subtract
    n_remaining_free_tokens = int(max_len-ntokens_used_for_qa_and_template)
    context_split = context.split(" ")
    if (len(context_split)*fudge_factor) > n_remaining_free_tokens:
        context = " ".join(context.split(" ")[:int(n_remaining_free_tokens/fudge_factor-1)]) + "..."
    filled_text = template_text_half.replace("{CONTEXT}", context)
    return {'text':filled_text}


def clean_legalcontractslong(x):
    # remove pagination like 5 -----
    text = re.sub(r"\s*\d+\s*\-+","",x['text'])
    # remove extended -------
    text = re.sub(r"\s*\-{4,}\s*","",text)
    # remove extended ——————————————
    text = re.sub(r"\s*\—{4,}\s*","",text)
    # remove extended __
    text = re.sub(r"\s*\_{3,}\s*","___",text)
    # remove multiple line breaks
    text = re.sub(r"\n+","\n",text,flags=re.MULTILINE)
    # calculate the word count per line
    lines = [l.strip() for l in text.split('\n') if len(l.strip())>0]
    if len(lines)<5:
        return {'text':text}
    # calculate word count
    wc = [len([w for w in l.strip().split(' ') if len(w)>0]) for l in lines]
    # calculate densities
    density = [wc[0]] + [
        sum(l)/3 for l in zip(wc[:-2], wc[1:-1], wc[2:])
    ] + [wc[-1]]
    # threshold
    threshold_on_density = 3.01
    # remove sections likely to be pagination/signatures and other table-like-stuff
    filtered_text = "\n".join([
        l for d,l in zip(density[:-2], lines) if d>=threshold_on_density
    ])
    # attach sentences that are incorrectly split into paragras
    filtered_text = filtered_text.replace(
        ".\n",'XxXx'
    ).replace(":\n",'YyYy').replace(";\n",'ZzZz').replace("-\n",'').replace("\n"," ").replace(
        'XxXx','.\n'
    ).replace('YyYy',':\n').replace('ZzZz',';\n')
    return {'text':filtered_text}


def clean_stanfordplato(x):
    """clean the stanford encyclopedia of philosophy"""
    # get the preamble
    text = " ".join([l.replace("\n", " ") for l in x['preamble']])
    for s in x['main_text']:
        # get section title
        text +"\n%s\n"%s['section_title']
        # get section main text
        text += " ".join([l.replace("\n", " ") for l in s['main_content']])
        # loop through all subsections
        for subs in s['subsections']:
            text += "\n%s\n" % subs['subsection_title']
            text += " ".join([l.replace("\n", " ") for l in subs['content']])
    return {'text':text}


def clean_player1book3(x):
    text = x['text'].replace("_","")
    if len(text)>=4000:
        text = text[3000:]
    x.update({'text':text})
    return x

def clean_govreportqa(x):
    """Clean govreport for QA task."""
    q_raw = x['question_summary_pairs']['question']
    a_raw = x['question_summary_pairs']['summary']
    if len(q_raw)==1:
        q_concat = q_raw[0]
        a_concat = a_raw[0]
    elif len(q_raw)<=3 and len(q_raw)>1:
        q_proc = [q[0].lower() + q[1:].strip('?') for q in q_raw]
        q_concat = ', '.join(q_proc[:-1]) + ', and ' + q_proc[-1] + '?'
        a_concat = ' '.join(a_raw)
    else:
        a_concat = ' '.join(a_raw) # answer will include ALL of summaries
        # randomly select a third question
        q_random = q_raw[2:][random_by_char(a_concat,3,10) % (len(q_raw)-2)]
        # combine first 2 questions and the random one as a list
        q_selected = q_raw[:2] + [q_random]
        q_proc = [q[0].lower() + q[1:].strip('?') for q in q_selected]
        q_concat = ', '.join(q_proc[:-1]) + ', and ' + q_proc[-1] + '?'
    x['query']=q_concat
    x['positives']=[a_concat]
    x['negatives']=[]
    x['type'] = 'qa_triplet'
    return x


def filter_dictionary(x):
    """Get definitions of only medium sized words with large definitions."""
    if x['word'] is None:
        return False
    return len(x['definition'])>100 and len(x['word'].replace(" ",''))>=4


def clean_dictionary(x):
    """Converts a dictionary term into a question, sampling randomly from 20 template questions."""
    idx_random_question_template = ord(x['definition'].replace(' ','')[-6]) % len(LIST_OF_DICTIONARY_PARAPHRASES)
    question_template =LIST_OF_DICTIONARY_PARAPHRASES[idx_random_question_template]
    x['query'] = question_template % x['word']
    x['positives'] = [x['definition']]
    x['negatives'] = []
    x['type'] = 'qa_triplet'
    return x


def clean_webglmqa(x):
    x['query']=x['question']
    x['positives'] = [x['answer']]
    x['negatives'] = []
    x['type'] = 'qa_triplet'
    return x


def clean_stream_PAQ_pairs(x):
    x['query'] = x['set'][0]+'?'
    x['positives'] = [x['set'][1]]
    x['negatives'] = []
    x['type'] = 'qa_triplet'
    return x

def clean_stream_finance_alpaca(x):
    x['query'] = x['instruction']
    x['positives'] = [x['output']]
    x['negatives'] = []
    x['type'] = 'qa_triplet'
    return x

def clean_stream_wiki_qa(x):
    x['query'] = x['question']
    is_pos = x['label']
    answer = x['answer']
    pos = [answer] if is_pos else []
    neg = [answer] if (not is_pos) else []
    x['positives'] = pos
    x['negatives'] = neg
    x['type'] = 'qa_triplet'
    return x

def clean_stream_oa_stackexchange(x):
    x['query'] = x['INSTRUCTION']
    x['positives'] = [x['RESPONSE']]
    x['negatives'] = []
    x['type'] = 'qa_triplet'
    return x

def clean_stream_sciqa(x):
    x['query'] = x['question']
    x['positives'] = [x['support']]
    x['negatives'] = []
    x['type'] = 'qa_triplet'
    return x

def clean_lfqa(x):
    x['query'] = x['question']
    x['positives'] = [x['answer']]
    x['negatives'] = []
    x['type'] = 'qa_triplet'
    return x

def filter_os_stackexchange(x):
    return x['SOURCE'] in STACKEXCHANGE_NONQUANT_DOMAINS

def get_name_and_description_eclassTrainST(text):
    description, name = text.split("; Name:")
    return description.replace("Description: ","").strip(), name.strip()

def clean_eclassTrainST(x):
    """This set isn't really about entailment/contradiction; it is really a dictionary"""
    description, name = get_name_and_description_eclassTrainST(x['text'])
    pos, _ = get_name_and_description_eclassTrainST(x['entailment'])
    extra, _ = get_name_and_description_eclassTrainST(x['contradiction'])
    x['query'] = 'What is a "%s"?' % name
    x['positives'] = [pos]
    x['negatives'] = []
    # add the entailment as positive, contradiction as negatives
    if x['label'] == 'entailment':
        x['positives'].append(extra)
    else:
        x['negatives'] = [extra]
    x['type'] = 'qa_triplet'
    return x

# do to: alzoubi36/policy_qa - policy questions
def clean_policyqa(x):
    """Adds more context to the questions about data security in the alzoubi36/policy_qa qa set """
    idx_random_question_template = ord(x['context'].replace(' ','')[-5]) % len(POLICYQA_PREPEND)
    question_template =POLICYQA_PREPEND[idx_random_question_template]
    q = x['question']
    q = q[0].lower() + q[1:]
    x['query'] = question_template % q # ['id', 'title', 'context', 'question', 'answers']
    x['positives'] = [x['context']]
    #negatives_random, _ = negative_example_generator.find_negative(x['context'], k = 1, skip=10)
    x['negatives'] = []
    x['type'] = 'qa_triplet'
    return x

def clean_sc2qa(x):
    x['query'] = x['question']
    x['positives'] = [x['article']]
    x['negatives'] = []
    x['type'] = 'qa_triplet'
    return x

def clean_yahooanswers(x):
    x['query'] = (x['question_title'] + " " + x['question_content']).strip() # 'question_title', 'question_content'
    x['positives'] = [x['best_answer']]
    x['negatives'] = []
    x['type'] = 'qa_triplet'
    return x

def filter_yahooanswers(x):
    """Yahoo news filtering (filter for 6=business; 3=education; 9=govt)"""
    return x['topic'] in [3,6,9] and len(x['question_title'])>10 and len(x['best_answer'])>10


def clean_businessbookqa(x):
    """17k business books cleaning"""
    x['query'] = x['question']
    x['positives'] = [x['answer']]
    x['negatives'] = []
    x['type'] = 'qa_triplet'
    return x

def clean_strixphilosophyqa(x):
    """Cleans the sayhan/strix-philosophy-qa dataset for triplet-loss"""
    return {
        'query':x['question'],
        'positives':[x['answer']],
        'negatives':[],
        'type':'qa_triplet'
    }

def clean_psychologyquestionanswer(x):
    return {
        'query':x['question'],
        'positives':[x['answer']],
        'negatives':[],
        'type':'qa_triplet'
    }


def clean_investopediaqa(x):
    return {
        'query':x['prompts'],
        'positives':[x['response']],
        'negatives':[],
        'type':'qa_triplet'
    }


def clean_legalsum(x):
    max_char_len_billsum = int(CHAR_PER_WORD*MAX_SEQ_LENGTH)
    text = x['article'][:max_char_len_billsum]
    if 'SEC. 2.' in text:
        text = ".".join(text.split('SEC. 2.')[1].split('.')[1:])
    else:
        if 'SHORT TITLE' in text:
             text = text.split('SHORT TITLE')[1]
    x['query'] = x['summary']
    x['positives'] = [text.strip()]
    x['negatives'] = []
    x['type'] = 'sts_triplet'
    return x


def clean_xsum(x):
    x['query'] = x['summary']
    x['negatives'] = []
    x['positives'] = [x['document']]
    x['type'] = 'sts_triplet'
    return x


def clean_eurlex(x):
    x['query'] = x['text']
    x['negatives'] = []
    x['positives'] = []
    x['type'] = 'sts_by_textlabel'
    x['label'] = x['eurovoc_concepts']
    x['subtype'] = 'eurlex'
    return x


def clean_allenai_citeprediction(x):
    x['query'] = x['query']['abstract']
    pos = x['pos']['abstract']
    x['positives'] = [pos] if pos is not None else []
    neg = x['neg']['abstract']
    x['negatives'] = [neg] if neg is not None else []
    x['type'] = 'sts_triplet'
    return x


def clean_simple_wiki(x):
    x['query'] = x['set'][0]
    x['positives'] = [x['set'][1]]
    x['negatives'] = []
    x['type'] = 'sts_triplet'
    return x


def clean_coco_captions_quintets(x):
    x['query'] = x['set'][0]
    x['positives'] = x['set'][1:]
    x['negatives'] = []
    x['type'] = 'sts_triplet'
    return x


def clean_specter(x):
    x['query'] = x['set'][0]
    x['positives'] = [x['set'][1]]
    x['negatives'] = [x['set'][2]]
    x['type'] = 'sts_triplet'
    return x


def clean_paws(x):
    x['query'] = x['sentence1']
    x['positives'] = [x['sentence2']]
    x['negatives'] = []
    x['type'] = 'sts_triplet'
    return x


def clean_qqp(x):
    x['query'] = x['set']['query']
    x['positives'] = x['set']['pos']
    x['negatives'] = x['set']['neg']
    x['type'] = 'sts_triplet'
    return x


def clean_ledgarlabelled(x):
    x['query'] = x['provision']
    x['negatives'] = []
    x['positives'] = []
    x['type'] = 'sts_by_textlabel'
    x['subtype'] = 'ledgar'
    return x


def clean_debatesum_sts(x):
    x['query'] = x['Abstract']
    x['positives'] = [x['Extract']]
    x['negatives'] = []
    x['type'] = 'sts_triplet'
    return x


def filter_chatgptparaphrases(x):
    return x['category']=='sentence'


def clean_chatgptparaphrases(x):
    x['query'] = x['text']
    x['positives'] = eval(x['paraphrases'])
    x['negatives'] = []
    x['type'] = 'sts_triplet'
    return x


def clean_gigaword(x):
    x['query'] = x['summary']
    x['positives'] = [x['document']]
    x['negatives'] = []
    x['type'] = 'sts_triplet'
    return x


def clean_govreportsumm(x):
    MAX_TOKEN_CHAR_LEN = int(CHAR_PER_WORD*MAX_SEQ_LENGTH/TOKEN_FUDGE_FACTOR)
    text2 = x['report'][:MAX_TOKEN_CHAR_LEN]
    text1 = x['summary'][:MAX_TOKEN_CHAR_LEN]
    x['query'] = text1.strip()
    x['positives'] = [text2.strip()]
    x['negatives'] = []
    x['type'] = 'sts_triplet'
    return x


def clean_snli(x):
    x['pair1'] = x['premise']
    x['pair2'] = x['hypothesis']
    x['type'] = 'pair_classification'
    x['cls_id'] = 'snli'
    x['n_labels'] = 3
    return x


def clean_contractnli(x):
    x['pair1'] = x['premise']
    x['pair2'] = x['hypothesis']
    x['type'] = 'pair_classification'
    x['cls_id'] = 'contractnli'
    x['n_labels'] = 3
    return x


def clean_mnli(x):
    x['pair1'] = x['premise']
    x['pair2'] = x['hypothesis']
    #x['label'] = []
    x['type'] = 'pair_classification'
    x['cls_id'] = 'mnli'
    x['n_labels'] = 3
    return x


def clean_cannotdatast(x):
    x['pair1'] = x['premise']
    x['pair2'] = x['hypothesis']
    x['type'] = 'pair_classification'
    x['cls_id'] = 'cannotdataset'
    x['n_labels'] = 2
    return x


def clean_newscategory(x):
    x['pair1'] = x['headline'] + ". " + x['short_description']
    x['pair2'] = None
    x['label'] = NEWSCATEGORIES.get(x['category'],NEWSCATEGORIES['OTHER'])
    x['type'] = 'classification'
    x['cls_id'] = 'newscategory'
    x['n_labels'] = 40
    return x

def clean_doceeevents(x):
    x['pair1'] = x['text']
    x['pair2'] = None
    x['label'] = DOCEEEVENTS.get(x['event_type'],DOCEEEVENTS['other'])
    x['type'] = 'classification'
    x['cls_id'] = 'doceeevents'
    x['n_labels'] = 61
    return x


def clean_dbpedia_l2(x):
    x['pair1'] = x['text']
    x['pair2'] = None
    x['label'] = DBPEDIA_L2.get(x['l2'],DBPEDIA_L2['other'])
    x['type'] = 'classification'
    x['cls_id'] = 'dbpedia_l2'
    x['n_labels'] = 71 # 219
    return x


def clean_dbpedia_l3(x):
    x['pair1'] = x['text']
    x['pair2'] = None
    x['label'] = DBPEDIA_L3.get(x['l3'],DBPEDIA_L3['other'])
    x['type'] = 'classification'
    x['cls_id'] = 'dbpedia_l3'
    x['n_labels'] = 220
    return x


def clean_casehold_positives(x):
    x['pair1'] = x['citing_prompt'].split('(<HOLDING>)')[0]
    correct_holding_id = int(x['label'])
    correct_holding_text = x['holding_%d' % correct_holding_id]
    x['pair2'] = correct_holding_text
    x['label'] = 1
    x['type'] = 'pair_classification'
    x['cls_id'] = 'casehold'
    x['n_labels'] = 2
    return x


def clean_casehold_negatives(x):
    x['pair1'] = x['citing_prompt'].split('(<HOLDING>)')[0]
    correct_holding_id = int(x['label'])
    incorrect_holding_id = (correct_holding_id+1) % 4
    incorrect_holding_text = x['holding_%d' % incorrect_holding_id]
    x['pair2'] = incorrect_holding_text
    x['label'] = 0
    x['type'] = 'pair_classification'
    x['cls_id'] = 'casehold'
    x['n_labels'] = 2
    return x


def filter_snli(x):
    return x['label']!=-1


def filter_newscategory(x):
    return x['category'] not in ['LATINO VOICES',"QUEER VOICES", "BLACK VOICES"]


def clean_mtopintent(x):
    # id (int64)	text (string)	label (int32)	label_text (string)
    x['pair1'] = x['text']
    x['pair2'] = None
    x['type'] = 'classification'
    x['cls_id'] = 'mtopintent'
    x['n_labels'] = 113
    return x


def clean_syntheticpiifinance(x):
    """We'll use the expanded type as 1600 labels for these contracts"""
    x['pair1'] = x['generated_text'].replace('**'," ")
    x['pair2'] = None
    x['label'] = (x['document_type']+"_"+x['expanded_type']).lower().replace(" ","-")
    x['type'] = 'classification'
    x['cls_id'] = 'syntheticpiifinance'
    x['n_labels'] = 1679
    return x

def filter_syntheticpiifinance(x):
    """Filter to only english-language examples in ppi-finace dataset."""
    if x['language'].lower()!='english':
        return False
    label2 = x['document_type']+"|"+x['expanded_type']
    return label2 not in [
        'XBRL|Financial Statement Footnotes',
        'XBRL|Taxonomy Extension',
        'XBRL|Financial Statement',
        'XBRL|Compliance Data Validation',
        'FpML|Index Amortizing Swap',
        'XBRL|Credit Rating Analysis',
        'XBRL|Merger & Acquisition Analysis',
        'FpML|Credit Spread Swaps',
        'Financial Data Feed|Fixed Income Data',
        'FpML|Basis Swap', 'FpML|Inflation Swaps',
        'XBRL|Regulatory Filing', 'XBRL|Audit Report', 'XBRL|Risk Assessment',
        'Financial Data Feed|Financial Index Data', 'Financial Data Feed|Market Volatility',
        'XBRL|Presentation Linkbase Design', 'FIX Protocol|Risk Management',
        'XBRL|Industry-Specific Metrics', 'FIX Protocol|QuoteRequest', 'XBRL|Tax Filing', 'FpML|Asian Option',
        'Financial Data Feed|Regulatory Reporting',
        'Financial Data Feed|Interest Rate Swaps', 'XBRL|Financial Forecast', 'FpML|Commodity Swap', 'FpML|Forward Rate Agreement',
        'XBRL|Metadata Annotation Framework', 'XBRL|Comparative Financial Analysis', 'XBRL|Risk and Compliance Data Integration', 'FpML|Binary Option',
        'XBRL|Nonprofit Organization Financials', 'FpML|Mortgage-Backed Security Swaps', 'XBRL|Sustainability Reporting Metrics', 'FpML|FX Derivative',
        'XBRL|Business Rules Validation', 'XBRL|Label Linkbase Generation', 'XBRL|XBRL Instance Document Creation', 'FpML|Range Accrual Swap',
        'XBRL|Quarterly Earnings', 'FpML|Cross Currency Swap', 'FpML|Equity Swap', 'FpML|Convertible Bond Swaps',
        'FpML|Interest Rate Cap', 'FpML|Foreign Exchange Swaps', 'FpML|Constant Maturity Swap', 'XBRL|Quarterly Financial Statement', 'Financial Statement|XBRL',
        'XBRL|Extension Taxonomy Development', 'XBRL|Financial Statement Mapping', 'XBRL|Data Transformation Automation', 'FpML|Basket Option',
        'EDI|XML', 'XBRL|International Financial Reporting Standards (IFRS)', 'XBRL|Custom XBRL Schema', 'Financial Data Feed|Currency Exchange', 'FpML|Variance Swap',
        'FpML|Barrier Option', 'FpML|Basket Swaps', 'FpML|Interest Rate Swap', 'XBRL|Financial Ratio Analysis', 'XBRL|Calculation Linkbase Creation', 'XBRL|Regulatory Compliance Mapping',
        'FpML|Commodity Swaps', 'FpML|Credit Spread Option', 'EDI|X12', 'FIX Protocol|Quote', 'XBRL|Audit Trail Generation', 'XBRL|Business Valuation',
        'XBRL|Small Business Financial Statements', 'XBRL|Data Quality Assessment Framework',
        'XBRL|Investor Presentation',
        'XBRL|Validation Rule Set Definition', 'FpML|Credit Default Swap', 'FpML|Interest Rate Floor', 'XBRL|Financial Data Aggregation',
        'XBRL|Entity-specific Disclosure Creation', 'XBRL|Industry Benchmarking', 'XBRL|Cross-domain Mapping Strategy', 'XBRL|Investor Relations Reporting',
        'XBRL|Financial Compliance Review', 'FpML|Inflation-Linked Swaps', 'XBRL|Instance Document Generation', 'Financial Data Feed|Exchange Traded Funds',
        'FpML|Digital Option', 'XBRL|Inline XBRL Implementation', 'FpML|Energy Swaps', 'XBRL|XBRL Rendering and Visualization', 'FpML|Equity Swaps', 'CSV|XML',
        'Financial Data Feed|Risk Management', 'XBRL|Data Integration Framework', 'XBRL|Sustainability Report', 'FpML|Yield Curve Swaps',
        'FpML|Inflation Swap', 'Financial Data Feed|Futures Market Data',
        'XBRL|Consolidated Financial Reports',
        'FpML|Credit Default Swaps', 'XBRL|Taxonomy Extension Development', 'FpML|Volatility Swaps', 'XBRL|Corporate Governance', 'XBRL|Government Regulatory Filing',
        'XBRL|Reference Linkbase Construction'
    ]

