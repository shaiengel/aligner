import re
from docx import Document
import requests
from align.services.utils import create_folder, format_text
from align.services.docx_util import write_docx
from pathlib import Path
import logging
from html.parser import HTMLParser
from align.services.docx_util import remove_nikud, read_docx

def normalize_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()

def strip_html_tags(text):
    # Remove all HTML tags but keep inner text
    clean = re.sub(r'<[^>]+>', '', text)
    # Collapse multiple spaces/newlines into one space
    clean = re.sub(r'\s+', ' ', clean)
    return normalize_spaces(clean)

STEIN_URL = "https://www.sefaria.org.il/api/v3/texts/{}.{}{}?version=primary&version=translation&fill_in_missing_segments=1&return_format=wrap_all_entities"

massechets = [
    #{'name': 'Berakhot', 'end': '64', 'end_amud': '1'},
    #{'name': 'Shabbat', 'end': '157', 'end_amud': '2'},
    #{'name': 'Eruvin', 'end': '105', 'end_amud': '1'},
    #{'name': 'Pesachim', 'end': '121', 'end_amud': '2'},
    #{'name': 'Rosh_Hashanah', 'end': '35', 'end_amud': '1'},
    #{'name': 'Yoma', 'end': '88', 'end_amud': '1'},
    #{'name': 'Sukkah', 'end': '56', 'end_amud': '2'},
    #{'name': 'Beitzah', 'end': '40', 'end_amud': '2'},
    #{'name': 'Taanit', 'end': '31', 'end_amud': '1'},
    #{'name': 'Megillah', 'end': '32', 'end_amud': '1'},
    #{'name': 'Moed_Katan', 'end': '29', 'end_amud': '1'},
    #{'name': 'Chagigah', 'end': '27', 'end_amud': '1'},
    #{'name': 'Yevamot', 'end': '122', 'end_amud': '2'},
    #{'name': 'Ketubot', 'end': '112', 'end_amud': '2'},
    #{'name': 'Nedarim', 'end': '91', 'end_amud': '2'},
    #{'name': 'Nazir', 'end': '66', 'end_amud': '2'},
    #{'name': 'Sotah', 'end': '49', 'end_amud': '2'},
    #{'name': 'Gittin', 'end': '90', 'end_amud': '2'},
    #{'name': 'Kiddushin', 'end': '82', 'end_amud': '2'},
    #{'name': 'Bava_Kamma', 'end': '119', 'end_amud': '2'},
    #{'name': 'Bava_Metzia', 'end': '119', 'end_amud': '1'},
    #{'name': 'Bava_Batra', 'end': '176', 'end_amud': '2'},
    #{'name': 'Sanhedrin', 'end': '113', 'end_amud': '2'},
    #{'name': 'Makkot', 'end': '24', 'end_amud': '2'},
    #{'name': 'Shevuot', 'end': '49', 'end_amud': '2'},
    #{'name': 'Avodah_Zarah', 'end': '76', 'end_amud': '2'},
    #{'name': 'Horayot', 'end': '14', 'end_amud': '1'},
    #{'name': 'Zevachim', 'end': '120', 'end_amud': '2'},
    #{'name': 'Menachot', 'end': '110', 'end_amud': '1'},
    #{'name': 'Chullin', 'end': '142', 'end_amud': '1'},
    #{'name': 'Bekhorot', 'end': '61', 'end_amud': '1'},
    #{'name': 'Arakhin', 'end': '34', 'end_amud': '1'},
    #{'name': 'Temurah', 'end': '34', 'end_amud': '1'},
    #{'name': 'Keritot', 'end': '28', 'end_amud': '2'},
    #{'name': 'Meilah', 'end': '22', 'end_amud': '1'},
    {'name': 'Tamid', 'end': '33', 'end_amud': '2', 'start': '25', 'start_amud': '2'},    
    #{'name': 'Niddah', 'end': '73', 'end_amud': '1'}    
]

def process_hadran(text):
    text = remove_nikud(text)
    
    # Rule 1: Extract all words after 'מסכת'
    words_after_masechet = []
    # Split text into words and look for מסכת starting from the end
    words = text.strip().split()
    for i, word in enumerate(reversed(words)):
        if 'מסכת' in word:
            # Calculate the actual position from the beginning
            actual_position = len(words) - 1 - i
            # Take all words after this position
            words_after_masechet = words[actual_position + 1:]
            massecht = ' '.join(words_after_masechet)
            break
    
    # Rule 2: Extract word(s) between 'הדרן עלך' and 'וסליקא'
    middle_words = []
    middle_pattern = r'הדרן עלך\s+(.+?)\s+וסליקא'
    middle_match = re.search(middle_pattern, text)
    if middle_match:
        middle_words = middle_match.group(1).strip().split()
    
    # Rule 3: Insert 'פרק [middle_words]' between 'הדרן עלך' and the rest
    if middle_words:
        # Replace the middle words with "פרק [middle_words]"
        middle_words_text = " ".join(middle_words)
        replacement_pattern = rf'הדרן עלך\s+{re.escape(middle_words_text)}\s+וסליקא'
        replacement_text = f'הדרן עלך פרק {middle_words_text} והדרן עלך וסליקא'
        modified_text = re.sub(replacement_pattern, replacement_text, text)
    else:
        modified_text = text

    end = read_docx('end.docx') 
    end = remove_nikud(end) 
    end = end.replace("פלונית", massecht)
    text = modified_text + " " + end      
    
    return text 
    


def read_amud(name, daf, amud):
    if amud == 1:
        amud = 'A'
    elif amud == 2:
        amud = 'B'        
    url = STEIN_URL.format(name, daf, amud)
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise error for bad HTTP status codes
        data = response.json()
        clean_lines = [strip_html_tags(line) for line in data["versions"][0]["text"]]
        text = "".join(clean_lines)
        text = text.replace("\n", " ") 
        text = format_text(text)
        return text
    except Exception as e:
        logging.error(f"Failed to fetch {url}: {e}")
        raise e
    
def read_daf(name, daf):
    text = read_amud(name, daf, 1)
    text += " "
    text += read_amud(name, daf, 2)
    text += " "
    return text

def read_masechet(name, end, end_amud):
    for i in range(34, end):
        print(f"Processing daf {i} of {name}...")
        text = read_daf(name, i)
        write_docx(f"repo\\{name}\\{name}_{i}.docx", text)
    
    print(f"Processing end daf {end} of {name}...")
    if end_amud == 2: 
        text = read_daf(name, end) 
        text = process_hadran(text)
        write_docx(f"repo\\{name}\\{name}_{end}.docx", text)
    elif end_amud == 1:
        text = read_amud(name, end, 1)
        text = process_hadran(text)
        write_docx(f"repo\\{name}\\{name}_{end}.docx", text)     



def main():
    create_folder("repo")
    
    for masechet in massechets:
        #text = read_amud(masechet['name'], 25, 2)
        #write_docx(f"repo\\{masechet['name']}\\{masechet['name']}_{25}.docx", text) 
        #text, massechet = process_hadran(text)
        create_folder(f"repo\\{masechet['name']}")
        print(f"Processing {masechet['name']}...")
        read_masechet(masechet['name'], int(masechet['end']), int(masechet['end_amud']))
        '''if masechet['end_amud'] == '1':
            amud = 'A'
        elif masechet['end_amud'] == '2':
            amud = 'B'        
        url = STEIN_URL.format(masechet['name'], masechet['end'], amud)
        
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise error for bad HTTP status codes
            data = response.json()
            clean_lines = [strip_html_tags(line) for line in data["versions"][0]["text"]]
            text = "".join(clean_lines)
            text = text.replace("\n", " ") 
            logging.info(text)
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch {url}: {e}")'''
        


if __name__ == '__main__':    
    main()
    