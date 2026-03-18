import os
import re
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from PyPDF2 import PdfReader
from docx import Document
from config.azure_config import (
    AZURE_STORAGE_CONNECTION_STRING,
    CONTAINER_NAME,
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_INDEX,
    AZURE_SEARCH_ADMIN_KEY,
    AZURE_OPENAI_KEY
)

client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint="https://hakunamatata-255666.cognitiveservices.azure.com/",
    api_version="2025-01-01-preview"
)

class IngestionAgent:
    def __init__(self):
        self.blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        self.container_client = self.blob_service_client.get_container_client(CONTAINER_NAME)

    def upload_file(self, file_path):
        file_name = os.path.basename(file_path)
        with open(file_path, "rb") as data:
            self.container_client.upload_blob(name=file_name, data=data, overwrite=True)
        print(f"Uploaded {file_name} to Blob Storage!")

    def extract_text(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        text = ""
        if ext == ".pdf":
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        elif ext == ".docx":
            doc = Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        else:
            raise ValueError("Unsupported file type! Only PDF or DOCX allowed.")
        return text.strip()

    def split_into_clauses(self, text):
        clauses = re.split(r'\n{2,}|(?<=\.)\s*\n|(?<=:)\s*\n|(?<=;)\s*\n', text)
        return [c.strip() for c in clauses if c.strip()]

    def chunk_text(self, text, max_chars=1000):
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunks.append(text[start:end])
            start = end
        return chunks

    def get_embedding(self, text):
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def upload_clauses_to_search(self, clauses, index_endpoint, index_name, admin_key):
        search_client = SearchClient(
            endpoint=index_endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(admin_key)
        )
        for i, clause in enumerate(clauses):
            chunks = self.chunk_text(clause)
            for j, chunk in enumerate(chunks):
                embedding = self.get_embedding(chunk)
                doc = {
                    "id": f"{i}_{j}",
                    "content": chunk,
                    "source": "uploaded_doc",
                    "embedding": embedding
                }
                search_client.upload_documents(documents=[doc])
        print("Clauses uploaded to Azure AI Search with embeddings!")

    def search_clauses(self, query, top_k=5):
        search_client = SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=AZURE_SEARCH_INDEX,
            credential=AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)
        )
        query_embedding = self.get_embedding(query)
        vector_search_results = search_client.search(
            vector={"value": query_embedding, "fields": "embedding", "k": top_k}
        )
        top_clauses = [r.get("content", "") for r in vector_search_results]
        return top_clauses

def main():
    try:
        agent = IngestionAgent()
        sample_file = "data/sample_docs/testdoc.docx"
        agent.upload_file(sample_file)
        text = agent.extract_text(sample_file)
        print("\nExtracted text (first 500 chars):\n", text[:500], "...\n")
        clauses = agent.split_into_clauses(text)
        print(f"Total clauses extracted: {len(clauses)}\n")
        agent.upload_clauses_to_search(
            clauses,
            index_endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=AZURE_SEARCH_INDEX,
            admin_key=AZURE_SEARCH_ADMIN_KEY
        )
        query = input("Enter your legal query (e.g., 'shortfall closing'): ").strip()
        if not query:
            print("No query entered. Exiting.")
            return
        top_clauses = agent.search_clauses(query, top_k=5)
        if not top_clauses:
            print("No matching clauses found. Try broader terms or synonyms.")
            return
        print("\nTop matching clauses:")
        for clause in top_clauses:
            print("-", clause, "\n")
        combined_text = "\n".join(top_clauses)
        prompt = f"Summarize the following legal clauses into key compliance points:\n{combined_text}"
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        summary = response.choices[0].message.content
        print("Grounded Summary:\n")
        print(summary)
    except Exception as ex:
        print(f"Error: {ex}")

if __name__ == "__main__":
    main()
