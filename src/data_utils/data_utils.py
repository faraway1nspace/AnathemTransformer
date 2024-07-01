from langdetect import detect
import re
from typing import Tuple, Union, Dict, List, Any
from src.configs.dataset_cleaners import check_is_code


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
    
