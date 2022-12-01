"""
## App: NLP App with Streamlit (NLPiffy)
Author: [Seongjin Lee(GirinMan)](https://github.com/GirinMan))\n
Source: [Github](https://github.com/GirinMan/HAI-DialectTranslator/)
Credits: 2022-Fall HAI Team 1 project

실행 방법: streamlit run app.py

"""
# Core Pkgs
import streamlit as st 
import os
import requests
import torch
import numpy as np
import pandas as pd
import clipboard

from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(layout="wide")

@st.cache(show_spinner=False, allow_output_mutation=True, suppress_st_warning=True, max_entries=1)
def load_gpt():
    with st.spinner("Loading gpt3 model for generation..."):
        tokenizer = AutoTokenizer.from_pretrained(
        'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
        bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
        )
        model = AutoModelForCausalLM.from_pretrained(
        'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
        pad_token_id=tokenizer.eos_token_id,
        torch_dtype='auto', low_cpu_mem_usage=False
        ).to(device='cuda', non_blocking=True)
        model.eval()

    return model, tokenizer


def generate_sentences(model, tokenizer, text, length, num_output):
    tokenized_text = tokenizer(text, return_tensors='pt').to('cuda')
    generated_ids = model.generate(
        tokenized_text.input_ids,
        do_sample=True, #샘플링 전략 사용
        max_length=(len(tokenized_text.input_ids[0]) + length), # 최대 디코딩 길이
        top_k=10, # 확률 순위 밖인 토큰은 샘플링에서 제외
        top_p=0.6, # 누적 확률 이내의 후보집합에서만 생성
        no_repeat_ngram_size = 4,
        num_return_sequences = num_output # 한 번에 출력할 다음 문장 선택지의 개수
    ).detach().to('cpu')
    
    generated_texts = [tokenizer.decode(id, skip_special_tokens=True) for id in generated_ids]
    return generated_texts

def main():
    """ NLP Based App with Streamlit """
    
    # Title
    st.title("Auto Korean Eassay Writer by GirinMan")
    st.subheader("한국어 기반 자동 작문 프로그램\nhttps://github.com/GirinMan/Auto-Korean-Eassay-Writer")
    st.markdown("""
        #### Description
        - 한국어로 학습된 GPT-3 모델을 이용하여 입력된 문장에 이어지는 다음 문장들을 자동으로 생성해주어 좋은 품질의 소감문이나 보고서 등을 빠르게 작성할 수 있습니다.
        - 기존 텍스트를 입력한 뒤 다음 문장 생성 버튼을 누르면 자연스럽게 이어지는 문장 몇 개를 생성해 줍니다.
        - 생성된 문장을 기존 텍스트에 이어 붙이고 적당히 수정한 뒤 다음 문장을 생성하는 것을 반복하여 짧은 시간에 사람이 작성한 것 같은 많은 양의 텍스트를 생성할 수 있습니다.
        - 사랑의실천 과제, 봉사활동 소감문 등 글의 퀄리티는 크게 중요하지 않지만 채워야 하는 분량이 많은 경우에 유용합니다.
        - 모델 서빙 및 현재 보여지는 웹 페이지는 Streamlit을 활용하여 구현되었습니다.
        - Reference: KoGPT: KakaoBrain Korean(hangul) Generative Pre-trained Transformer (https://github.com/kakaobrain/kogpt)
        """)

    model, tokenizer = load_gpt()

    input_area = st.empty()
    input_area.text_area(f"텍스트 입력", "여기에 입력", key='input_txt' ,height=200)

    new_sentence_length = 64
    num_output = 5

    col1, col2 = st.columns(2, gap="small", )
    area = st.container()

    with col1:
        if st.button("글자 수 세기"):
            text = st.session_state.input_txt
            total_count = len(text)
            char_count = len(text.replace(" ", "").replace("\n", "").replace("\t", ""))
            area.subheader(f"공백 포함 {total_count}자 / 공백 제외 {char_count}자")

    with col2:
        if st.button("다음 문장 생성"):

            area.subheader("문장 생성 결과")
            input_txt = st.session_state.input_txt

            with area:
                with st.spinner("Generating next sentences..."):
                    new_sentences = generate_sentences(model, tokenizer, input_txt, new_sentence_length, num_output)
                    new_sentences = [sent.split(input_txt)[-1].replace('\n', ' ') for sent in new_sentences]

            for i, sent in enumerate(new_sentences):
                area.code(sent)

if __name__ == '__main__':
	main()
