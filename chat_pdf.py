# pip install langchain
# pip install openai
# pip install PyPDF2
# pip install faiss-cpu
# pip install tiktoken
# pip install python-docx
import textract
import docx
import os
import gradio as gr
from pathlib import Path
from pdfminer.high_level import extract_text
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS

os.environ["OPENAI_API_KEY"] = ""

# def chat_pdf(file_path, query,chain_type="refine"):
def chat_file(file_path,query,chain_type="refine"):
    file_ext = file_path.name.split('.')[-1]
    raw_text = ""
    if file_ext == 'pdf':
        try:
            text = extract_text(file_path.name)
            if text:
                raw_text += text
        except Exception as e:
            raw_text = f"Error occurred while reading PDF: {str(e)}"
    elif file_ext == 'doc' or 'docx':
        try:
            text = textract.process(file_path.name)
            if text:
                raw_text = text.decode('utf-8')
        except Exception as e:
            raw_text = f"Error occurred while reading DOC/DOCX file: {str(e)}"
    elif file_ext == 'txt':
        try:
            with open(file_path, 'r') as file:
                raw_text = file.read()
        except Exception as e:
            raw_text = f"Error occurred while reading TXT file: {str(e)}"
    else:
        raw_text = "文件类型错误，必须是PDF或DOC或txt文件！"
    return raw_text

    # text_splitter = CharacterTextSplitter(
    #     separator="\n",
    #     chunk_size=1000,
    #     chunk_overlap=200,
    #     length_function=len,
    # )
    # texts = text_splitter.split_text(raw_text)
    # embeddings = OpenAIEmbeddings()
    # docsearch = FAISS.from_texts(texts, embeddings)
    # chain = load_qa_chain(OpenAI(model_name="text-davinci-003", max_tokens=800), chain_type=chain_type)
    #
    # docs = docsearch.similarity_search(query)
    # result = chain.run(input_documents=docs, question=query)
    return raw_text+file_ext

# iface = gr.Interface(
#     fn=chat_pdf,
#     inputs=["file"],
#     outputs="text",
#     layout="vertical",
#     title="PDF Question Answering",
#     description="Upload a PDF or DOC file and ask a question about its content.",
#)
# iface = gr.Interface(
#     fn=chat_pdf,
#     inputs=gr.inputs.File(label="上传一个DOC文件"),
#     outputs=gr.outputs.Textbox(label="提取的文本内容")
# )
iface = gr.Interface(fn=chat_file,
                     inputs=["file","text"],
                     outputs="text",
                     title="放入所需要读取的文件",
                     description="Upload a PDF or DOC file and ask a question about its content.")

iface.launch()
# # 每20秒问一次
# file_gpt("C:/Users/scofi/Desktop/2023_GPT4All_Technical_Report.pdf","who are the authors of the article?")
# chat_pdf("C:/Users/scofi/Desktop/Audio gpt.docx","what's the point about this article?")
# file_gpt("C:/Users/scofi/Desktop/2023年新员工-业务实践安排（技术线）2.pdf","根据这一份文件，帮我用400字左右总结几点学习到的知识",chain_type ="refine")#"stuff""map_reduce" "refine"

# pdf还没有实现，要换成chatbot
