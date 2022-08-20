import sys
import torch
import re

from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer
from asian_bart import AsianBartForConditionalGeneration
from unicodedata import normalize

app = Flask(__name__)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if "cuda" in DEVICE.type :
    torch.cuda.set_device(DEVICE)

kor2eng_tokenizer = AutoTokenizer.from_pretrained("hyunwoongko/asian-bart-ecjk", src_lang = "ko_KR", tgt_text = "en_XX")
kor2eng_tokenizer.add_tokens("<tag>")
eng2kor_tokenizer = AutoTokenizer.from_pretrained("hyunwoongko/asian-bart-ecjk", src_lang = "en_XX", tgt_text = "ko_KR")
eng2kor_tokenizer.add_tokens("<tag>")

kor2eng_translator = AsianBartForConditionalGeneration.from_pretrained("/workspace/Translator/Model/kor2eng/tagged_bt_kor2eng")
kor2eng_translator.eval()
kor2eng_translator.to(DEVICE)

eng2kor_translator = AsianBartForConditionalGeneration.from_pretrained("/workspace/Translator/Model/eng2kor/tagged_bt_eng2kor")
eng2kor_translator.eval()
eng2kor_translator.to(DEVICE)

def translate_2_eng(text) -> str :
    inputs = kor2eng_tokenizer(text, return_tensors = "pt")
    result_tokens = kor2eng_translator.generate(inputs["input_ids"].to(DEVICE),
                                                decoder_start_token_id = kor2eng_tokenizer.lang_code_to_id["en_XX"],
                                                max_length = 100,
                                                num_beams = 7,
                                                no_repeat_ngram_size = 2)
    result_string = kor2eng_tokenizer.decode(result_tokens[0], skip_special_tokens = True, clean_up_tokenization_spaces = True).replace(' ', '').replace("▁", ' ').strip()
    return result_string

def translate_2_kor(text) -> str :
    inputs = eng2kor_tokenizer(text, return_tensors = "pt")
    result_tokens = eng2kor_translator.generate(inputs["input_ids"].to(DEVICE),
                                                decoder_start_token_id = eng2kor_tokenizer.lang_code_to_id["ko_KR"],
                                                max_length = 100,
                                                num_beams = 7,
                                                no_repeat_ngram_size = 2)
    result_string = eng2kor_tokenizer.decode(result_tokens[0], skip_special_tokens = True, clean_up_tokenization_spaces = True).replace(' ', '').replace("▁", ' ').strip()
    return result_string

@app.route('/',methods = ['GET', 'POST'])
def index():
    source = request.form.get("string")
    input_lang = request.form.get("input_lang_type")
    to_slang = request.form.get("do_slang")
    
    if not source :
        translated = "문장을 입력해주세요."
    else :
        if to_slang :
            source = "<tag>"+source
        if input_lang == "kor" :
            translated = translate_2_eng(source).replace("@user", '')
        elif input_lang == "eng" :
            translated = translate_2_kor(source)
        else :
            translated = "언어를 선택해주세요"
    translated = re.sub(r'\([^)]*\)', '', translated)    
    if translated.startswith(r"\u") :
        translated = ' '.join(translated.split()[1:])
    if translated[0] in [',', '.', '?', '!'] :
        translated = translated[1:].strip()
    
    return render_template('index.html', translated_string = translated)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug = True)
