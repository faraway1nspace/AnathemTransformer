from langdetect import detect
import re
from typing import Tuple, Union, Dict, List, Any

def check_is_code(text:str)->float:
    """Estimates a ratio of special char (that may indicate math/code notation); less than 10% is code for normal text."""
    nchar = min(5000,len(text))
    nchar_after_removespecialchar = len(re.sub(r"[\<\>\_\@\^\=\+\*\$\{\[\]\}\(\)\/\\\.]",'',text[:5000]))
    ratio_specialchar = 1-nchar_after_removespecialchar/nchar
    return ratio_specialchar


def check_language(text:str, special_char_threshold:float=0.10)->Tuple[bool, float]:
    """Verifies that a string is: i) English, and ii) not overly mathematical/code."""
    ratio_specialchar = check_is_code(text)
    if ratio_specialchar>=special_char_threshold:
        return False, ratio_specialchar
    try:
        is_eng = detect(text[:200]+" hello")=='en'
        return is_eng, -1
    except:
        return False, -1


def nwords_quick(text):
    """Quick estimate of the number of words."""
    return len([w for w in text.split(" ") if len(w)>0])


def flatten(list_of_lists):
    """Converts list of list of strings to a list of strings."""
    return [subl for l in list_of_lists for subl in l]
    
