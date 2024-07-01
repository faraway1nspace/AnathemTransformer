import re
from math import prod
from typing import Tuple, Union, Dict, List, Any
from src.configs.dataset_templates import *

def random_by_char(text:str, take:int=3, charlim:int=10):
    """Pseudo random text based on an imput text."""
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
    """Removes tables and excess \n includes somes specifics for Saylor books footmatter"""
    # discards the first 8 percent
    #discard = int(0.08*len(example['text']))
    #example['text'] = example['text'][discard:].replace('\n'," ")
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