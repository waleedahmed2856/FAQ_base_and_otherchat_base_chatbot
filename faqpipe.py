from pathlib import Path
import pandas as pd
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_groq import ChatGroq
import chromadb
import re
from dotenv import load_dotenv
load_dotenv(".env.txt")

file_path = str(Path.cwd()/"resourse/faq_data.csv")

def ingest_faq():

    data = pd.read_csv(file_path)
    
    return data.head()


collection_name = "faq"
ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.Client()


def initialize_components(path):
    global collection
    
    if collection_name not in [c.name for c in chroma_client.list_collections()]:
        
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=ef
        )

        data = pd.read_csv(path)

        docs = data['question'].to_list()
        metadata = [{"answer": ans} for ans in data['answer'].to_list()]
        ids = [f"id{i}" for i in range(len(docs))]

        collection.add(
            documents=docs,
            metadatas=metadata,
            ids=ids
        )

        return "Collection successfully added to ChromaDB."
    
    else:
        # LOAD existing collection
        collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=ef
        )
        return f"Collection {collection_name} loaded."

llm = ChatGroq(model="openai/gpt-oss-120b",
                temperature=0.3, 
                max_tokens=150)

def get_relevent_embaddings(query):

    collection = chroma_client.get_collection(name=collection_name)

    result = collection.query(query_texts=[query],
                              n_results=2)
    return result

def get_embadding_context(query):

    context = " ".join(r["answer"] for r in query["metadatas"][0])

    return context

import re

def get_relivant_answer(query, context):

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.3, 
        max_tokens=150
    )
    
    prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{query}

If the answer is not in the context, say "I don't know contact our team for further details".
"""
    
    response = llm.invoke(prompt)

    return re.sub(r'\s+', ' ', response.content)

if __name__ == "__main__":

    # data = ingest_faq()

    saving_components = initialize_components(file_path)
    
    query = "how would I recieve my order from you would you delivered it"

    embaddings = get_relevent_embaddings(query)

    context = get_embadding_context(embaddings)

    answer = get_relivant_answer(query, context)

    print(answer)
    