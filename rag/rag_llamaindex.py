## An example of using RAG with LlamaIndex and Bedrock

from llama_index.core import Settings, load_index_from_storage, VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.retrievers.bedrock import AmazonKnowledgeBasesRetriever
from llama_index.llms.bedrock import Bedrock
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.core import SimpleDirectoryReader

import numpy as np
from tqdm import tqdm
import concurrent.futures

# Initialize Bedrock embedding model and LLM
embed_model = BedrockEmbedding(
    aws_access_key_id="YOUR_ACCESS_KEY_ID",
    aws_secret_access_key="YOUR_SECRET_ACCESS_KEY",
    region_name="us-east-1",
    model="amazon.titan-embed-text-v2:0"
)
llm = Bedrock(model="anthropic.claude-3-5-sonnet-20240620-v1:0")
Settings.embed_model = embed_model
Settings.llm = llm

# Load or create index
persist_dir = "./indexes"
try:
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
except:
    # If index doesn't exist, create a new one
    loader = SimpleDirectoryReader(input_dir="./sources/book1")
    documents = loader.load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=persist_dir)

# Function to process each arrhythmia
def process_arrhythmia(arrhythmia, query_engine):
    response = query_engine.query(f"How is {arrhythmia} reflected in ECG?")
    return response.response

# Main RAG function
def perform_rag(ecg_conditions, top_k=3):
    query_engine = index.as_query_engine(similarity_top_k=top_k)
    responses = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_arrhythmia, arrhythmia, query_engine): arrhythmia 
                   for arrhythmia in ecg_conditions}

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                response = future.result()
                responses.append(response)
            except Exception as e:
                print(f"Error processing {futures[future]}: {e}")

    return responses

# Example usage
if __name__ == "__main__":
    ecg_conditions = [
        "atrial fibrillation",
        # ... other ECG conditions
    ]

    responses = perform_rag(ecg_conditions)
    for arrhythmia, response in zip(ecg_conditions, responses):
        print(f"\n{arrhythmia}:\n{response}")