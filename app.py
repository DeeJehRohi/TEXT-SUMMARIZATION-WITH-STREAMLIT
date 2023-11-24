import streamlit as st 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.chains.summarize import load_summarize_chain
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64
import re

#model and tokenizer loading
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)

#file loader and preprocessing
def file_preprocessing(file):
    loader =  PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        print(text)
        final_texts = final_texts + text.page_content
    return final_texts

#LLM pipeline PDF
def llm_pipeline(filepath):
    pipe_sum = pipeline(
        'summarization',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 500, 
        min_length = 50)
    input_text = file_preprocessing(filepath)
    result = pipe_sum(input_text)
    result = result[0]['summary_text']
    return result


#function to display the PDF of a given file 
@st.cache_data
def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)


#streamlit code 
st.set_page_config(layout="wide")
st.title("My Summarization App using Langauge Model")

### encode the input paragraph to summarize
text_input = st.text_area(label ="Text to summarize", height = 250)

if text_input is not None:
###text counter function
    res = len(re.findall(r'\w+' , str(text_input)))
    st.write(':red[The total number of words are :]' + str(res))
###end of text counter function
    if st.button("summarize"):
        tokenized_text = tokenizer.encode_plus(
            str(text_input),
            return_attention_mask = True, 
            return_tensors="pt")

### pass the tokenized paragraph in model
        generated_token = base_model.generate(
            input_ids = tokenized_text["input_ids"],
            attention_mask=tokenized_text["attention_mask"],
            max_length = 500,
            use_cache=True,
            )

### decode the summarized paragraph token
        summarized_paragraph = [
            tokenizer.decode(token_ids=ids, skip_special_tokens=True) for ids in generated_token  
        ]

        st.write("## Summarized Text")
        st.write(" ".join(summarized_paragraph))
        ###text counter function
        res = len(re.findall(r'\w+' , str(summarized_paragraph)))
        st.write(':red[The total number of words are :]' + str(res))

##end of text_input
def main():
    

    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])
    
    if uploaded_file is not None:
        if st.button("Summarize"):
            col1, col2 = st.columns(2)
            filepath = "data/"+uploaded_file.name
            with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())
            with col1:
                st.info("Uploaded File")
                pdf_view = displayPDF(filepath)

            with col2:
                summary = llm_pipeline(filepath)
                st.info("Summarization Complete")
                st.success(summary)




if __name__ == "__main__":
    main()