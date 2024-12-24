import weaviate
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import weaviate.classes as wvc
import os
import requests
import json
from langchain_community.document_loaders import AsyncChromiumLoader, OnlinePDFLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from urllib.parse import quote
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from langchain.memory import ConversationBufferMemory
from selenium.common.exceptions import TimeoutException 

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

print("====== Connecting to Weaviate Client ======")

client = weaviate.connect_to_wcs(
    cluster_url=os.getenv("WCS_CLUSTER_URL"),
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WCS_API_KEY")),
    headers={
        "X-OpenAI-Api-Key": os.environ["OPENAI_APIKEY"] 
    },
    skip_init_checks=True
)
client.collections.delete("Question")
questions = client.collections.create(
    name="Question",
    vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(), 
    generative_config=wvc.config.Configure.Generative.openai()
)

questions = client.collections.get("Question")
memory = ConversationBufferMemory()

def get_relevant_links_from_university_website(query: str):
    print("called get_relevant_links_from_university_website")
    query = quote(query)
    url = f"https://www.snhu.edu/search?q={query}"


    chrome_options = Options()
    chrome_options.add_argument("--headless")

    driver = webdriver.Chrome(options=chrome_options)

    driver.get(url)

    wait_time = 10

    try:
        element = WebDriverWait(driver, wait_time).until(
            EC.presence_of_element_located((By.ID, "ss360-heading-all_results"))
        )
    except TimeoutException:
        print("Element not found")
    

    html_document = driver.page_source

    driver.quit()
    soup = BeautifulSoup(html_document, "html.parser")
    html_search_results = []
    pdf_search_results = []
    for result in soup.find_all("a"):
        link = result["href"]
        if 'pdf' in link and 'http' in link and 'snhu.edu' in link:
            pdf_search_results.append(link)
        if 'pdf' not in link and 'http' in link and 'snhu.edu' in link:
            html_search_results.append(link)
    return html_search_results, pdf_search_results


def get_document_from_university_website_and_add_to_weaviate(query: str):
    html_links, pdf_links = get_relevant_links_from_university_website(query)

    print("====== Loading html documents ======")
    loader = AsyncChromiumLoader(html_links[:10])
    html_documents = loader.load()

    print("====== Extracting content From html documents ======")
    bs_transformer = BeautifulSoupTransformer()
    text_documents = bs_transformer.transform_documents(html_documents, tags_to_extract=["p"])

    print("====== Loading pdf documents ======")
    pdf_documents = []

    if pdf_links:
        for pdf_link in pdf_links[:10]:
            pdf_documents.append(OnlinePDFLoader(pdf_link).load())  


    text_documents.extend(pdf_documents)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=100,
    )

    print("====== Splitting text documents ======")
    docs_split = splitter.transform_documents(text_documents)


    print("====== Adding documents to Weaviate ======")
    docs_to_add_to_weaviate = []
    for doc in docs_split:
        docs_to_add_to_weaviate.append({
            "question": query,
            "text": doc.page_content
        })


    questions.data.insert_many(docs_to_add_to_weaviate)

def ask_weaviate(query: str):
    history = memory.load_memory_variables({})

    print("====== Generating response from Weaviate ======")
    ANSWER_NOT_FOUND = "ANSWER NOT FOUND"
    prompt = f"""
    You are a helpful assistant in the SNHU website and you are asked to help a student with a question.
    Below is the chat history you already had with the student:
    {history["history"]}
    First try and use the chat history to help you answer the student's question,which is {query} if
    not possible then summarise the information below to assist the student if it is related to the student question,
      if you find that
    the student question can't be answered by either the chat history or the text below, the last resort should be returning the words {ANSWER_NOT_FOUND} if you can't.
    """
    try:
        response = questions.generate.hybrid(query=query, grouped_task=prompt, limit=5)

        if ANSWER_NOT_FOUND in response.generated:
            get_document_from_university_website_and_add_to_weaviate(query)
            response = questions.generate.hybrid(query=query, grouped_task=prompt, limit=5)

    except Exception as e:
        get_document_from_university_website_and_add_to_weaviate(query)
        response = questions.generate.hybrid(query=query, grouped_task=prompt, limit=5)

        if ANSWER_NOT_FOUND in response.generated:
            get_document_from_university_website_and_add_to_weaviate(query)
            response = questions.generate.hybrid(query=query, grouped_task=prompt, limit=5)

    
    memory.save_context(
        {"input": query}, {"output": response.generated}
    )
    print(memory.load_memory_variables({}))
    return response.generated

