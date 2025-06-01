import sys
import os
import re
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import numpy as np
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import tiktoken  # For token counting
import PyPDF2
from src.embedding_models import BaseEmbeddingModel, OpenAIEmbeddingModel, MiniEmbeddingModel
import pickle 
import html2text

from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class VectorDB:
    def __init__(self, directory="documents", vector_file="embeddings.npy", max_words_per_chunk=2000, embedding_model: BaseEmbeddingModel = OpenAIEmbeddingModel()):
        """
        Initializes the in-memory database by processing text files and generating embeddings.
        """
        self.directory = directory
        self.vector_file = vector_file
        self.chunks_file = os.path.splitext(vector_file)[0] + "_chunks.pkl"
        self.max_words_per_chunk = max_words_per_chunk
        self.embedding_model = embedding_model

        if os.path.exists(self.vector_file) and os.path.exists(self.chunks_file):
            print(f"[VectorDB] Found existing {self.vector_file} and {self.chunks_file}, loading them instead of regenerating.")
            with open(self.chunks_file, 'rb') as f:
                self.chunks = pickle.load(f)
            self.embeddings = np.load(self.vector_file)
            print(f"[VectorDB] Loaded {len(self.embeddings)} embeddings from {self.vector_file}.")
            return


        '''
        # Create mathlib_docs.txt + metadata
        self.extract_mathlib_chunks(
            lean_dir=".lake/packages/mathlib/Mathlib",
            output_txt=os.path.join(self.directory, "mathlib_docs.txt"),
            output_metadata=os.path.join(self.directory, "mathlib_docs_metadata.jsonl")
        )
        '''

        # Load all .txt documents (including mathlib_docs.txt)
        docs = self.read_text_files()
        print(f"[VectorDB] Reading all of the documents in {self.directory}.")
        
        self.chunks = self.embedding_model.split_documents(docs)
        print(f"[VectorDB] Splitting the knowledge base into {len(self.chunks)} chunks. Saving the chunks")
        with open(self.chunks_file, 'wb') as f:
            pickle.dump(self.chunks, f)
            print(f"[VectorDB] Chunks saved to {self.chunks_file}")

        # Generate embeddings from just the text
        if isinstance(self.chunks[0], dict) and "chunk" in self.chunks[0]:
            texts = [r["chunk"] for r in self.chunks]
        else:
            texts = self.chunks
        
        # Get embeddings from the model
        embeddings_data = self.embedding_model.get_embeddings_batch(texts)
        self.embeddings = embeddings_data
        print(f"[VectorDB] Generated {len(self.embeddings)} embeddings with shape {self.embeddings.shape}. Stored at {self.vector_file}")
        
        self.store_embeddings()

    
    @staticmethod
    def scrape_website(url: str, output_file: str):
        """
        Fetches the main content of a given URL and saves it to an output file.
        Handles both HTML pages and PDF files.
        """
        try:
            response = requests.get(url)
            response.raise_for_status()
            content_type = response.headers.get('Content-Type', '')

            if 'application/pdf' in content_type or url.endswith('.pdf'):
                with open(output_file, 'wb') as file:
                    file.write(response.content)
                try:
                    with open(output_file, 'rb') as pdf_file:
                        reader = PyPDF2.PdfReader(pdf_file)
                        text = "\n".join([page.extract_text() for page in reader.pages])
                    text_output_file = os.path.splitext(output_file)[0] + '.txt'
                    with open(text_output_file, 'w', encoding='utf-8') as text_file:
                        text_file.write(text)
                    print(f"PDF content extracted and saved to {text_output_file}")
                except Exception as e:
                    print(f"Error extracting text from PDF: {e}")
            elif 'text/html' in content_type:
                soup = BeautifulSoup(response.text, "html.parser")
                text_content = soup.get_text(separator="\n", strip=True)
                with open(output_file, 'w', encoding='utf-8') as file:
                    file.write(text_content)
                print(f"HTML content saved to {output_file}")
    
            else:
                print(f"Unsupported content type: {content_type}")

                '''
                                        elif 'text/html' in content_type:
                markdown_text = html2text.html2text(response.text)
                #markdown_output_file = os.path.splitext(output_file)[0] + '.md'
                markdown_output_file = output_file
                with open(markdown_output_file, 'w', encoding='utf-8') as file:
                    file.write(markdown_text)
                print(f"HTML content converted to Markdown and saved to {markdown_output_file}")

            '''

        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching the URL: {e}")

    @staticmethod
    def scrape_website_html_to_md(url: str, output_file: str):
        try:
            response = requests.get(url)
            response.raise_for_status()
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' in content_type:
                markdown_text = html2text.html2text(response.text)
                #markdown_output_file = os.path.splitext(output_file)[0] + '.md'
                markdown_output_file = output_file
                with open(markdown_output_file, 'w', encoding='utf-8') as file:
                    file.write(markdown_text)
                print(f"HTML content converted to Markdown and saved to {markdown_output_file}")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching the URL: {e}")

    def read_text_files(self):
        """
        Reads all text files in the directory and returns their content as a list.
        """
        documents = []
        for filename in os.listdir(self.directory):
            if filename.endswith(".txt"):
                print(f"[VectorDB.read_text_files] Reading {filename}.")
                with open(os.path.join(self.directory, filename), 'r', encoding='utf-8') as file:
                    documents.append(file.read())
        return documents
    
    def store_embeddings(self):
        """
        Saves the numpy matrix of embeddings to a .npy file.
        """
        np.save(self.vector_file, self.embeddings)
        print(f"[VectorDB.store_embeddings] Embeddings saved to {self.vector_file}")
    
    @staticmethod
    def cosine_similarity(vector1, vector2):
        """
        Calculate cosine similarity between two vectors.
        Returns a value between -1 and 1, where 1 means identical vectors.
        """
        dot_product = np.dot(vector1, vector2)
        
        magnitude1 = np.sqrt(np.sum(np.square(vector1)))
        magnitude2 = np.sqrt(np.sum(np.square(vector2)))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        return dot_product / (magnitude1 * magnitude2)
    
    @staticmethod
    def chunk_markdown(file_path: str, max_words: int = 4000):
        """
        Splits text into chunks of a specified maximum number of words.
        """
        words = text.split()
        return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]



    def chunk_markdown_with_metadata(md_path, section_title, out_pkl_path):
        source = os.path.basename(md_path)

        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split by ## or ### headers
        chunks_raw = re.split(r'(?:^##\s+|^###\s+)', content, flags=re.MULTILINE)

        records = []
        for i, chunk in enumerate(chunks_raw):
            chunk = chunk.strip()
            if len(chunk.split()) < 50:
                continue

            records.append({
                "id": f"{section_title.lower().replace(' ', '_')}_{i}",
                "title": section_title,
                "source": source,
                "order": i,
                "chunk": chunk + "\n<EOC>"
            })

        with open(out_pkl_path, "wb") as f:
            pickle.dump(records, f)

        print(f" {len(records)} chunks with metadata saved to {out_pkl_path}")



    @staticmethod    
    def extract_mathlib_chunks(lean_dir="mathlib4/Mathlib", output_txt="documents/mathlib_docs.txt", output_metadata="documents/mathlib_docs_metadata.jsonl"):
        chunks = []
        metadata = []
        
        for root, _, files in os.walk(lean_dir):
            for file in files:
                if file.endswith(".lean"):
                    full_path = os.path.join(root, file)
                    with open(full_path, "r") as f:
                        code = f.read()
                    
                    matches = re.finditer(r'(def|theorem|lemma)\s+[^\n]*[\s\S]*?(?=\n\n|\Z)', code)
                    for i, match in enumerate(matches):
                        chunk_text = match.group().strip()
                        chunk_type = match.group(1)
                        rel_path = os.path.relpath(full_path, lean_dir)
                        chunk_id = f"{rel_path.replace('/', '__')}_{chunk_type}_{i}"
                        
                        chunks.append(chunk_text + "\n<EOC>")
                        metadata.append({
                            "id": chunk_id,
                            "type": chunk_type,
                            "source_file": rel_path,
                            "index": i,
                            "start_offset": match.start(),
                            "end_offset": match.end()
        })
        
        # Save text chunks for embedding
        with open(output_txt, "w") as f:
            f.write("\n\n".join(chunks))
            
        # Save metadata
        with open(output_metadata, "w") as f:
            for entry in metadata:
                json.dump(entry, f)
                f.write("\n")

        print(f"[extract_mathlib_chunks] Extracted {len(chunks)} chunks from Mathlib.")


    @staticmethod
    def get_top_k(npy_file: str, embedding_model: BaseEmbeddingModel, query: str, k: int = 5, verbose: bool = False):
        """
        Returns top-k most similar chunks and their scores.
        """
        # Load embeddings from .npy file
        stored_embeddings = np.load(npy_file)
        
        # Load chunks from corresponding .pkl file
        chunks_file = os.path.splitext(npy_file)[0] + "_chunks.pkl"
        with open(chunks_file, 'rb') as f:
            chunks = pickle.load(f)
        
        # Generate embedding for the query
        query_embedding = embedding_model.get_embedding(query)
        
        # Calculate similarities
        similarities = [VectorDB.cosine_similarity(query_embedding, embedding) 
                        for embedding in stored_embeddings]
        
        # Get top k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        top_k_indices = top_k_indices.flatten()
        
        # Retrieve top-k chunks and their similarities
        top_k_chunks = [chunks[index] for index in top_k_indices]
        top_k_scores = [similarities[index] for index in top_k_indices]
        
        # Output the chunks and scores
        if verbose:
            for i, (chunk, score) in enumerate(zip(top_k_chunks, top_k_scores)):
                print(f"Result #{i+1} (Score: {score:.4f})")
                print(f"{chunk[:200]}...")
                print("-" * 50)

        return top_k_chunks, top_k_scores
    
    @staticmethod
    def run(query: str = None):
        """
        Main entry point for manual or scripted execution.
        Initializes the vector DB and optionally performs a sample query.
        """
        print("[VectorDB.run] Starting vector DB setup...")

        openai_embedding_model = OpenAIEmbeddingModel()
        embedding_database_file = "database.npy"
        
        vector_db = VectorDB(
            directory="documents", 
            vector_file=embedding_database_file, 
            embedding_model=openai_embedding_model
        )
        
        top_k_chunks, top_k_scores = VectorDB.get_top_k(
            embedding_database_file, openai_embedding_model, query, k=3, verbose=True
        )
        
        print("[VectorDB.run] Completed vector DB setup and query.")
        return top_k_chunks, top_k_scores

if __name__ == "__main__":
    '''
    # Create and save the database
    openai_embedding_model = OpenAIEmbeddingModel()
    mini_embedding_model = MiniEmbeddingModel()

    # Name of the file that stores the embeddings in memory. 
    embedding_database_file = "database.npy"

    vector_db = VectorDB(directory="documents", 
                         vector_file=embedding_database_file, 
                         embedding_model=openai_embedding_model)

    # Example usage of querying the database. 
    query = "What is reinforcement learning?"
    top_k_results = VectorDB.get_top_k(embedding_database_file, openai_embedding_model, query, k=3, verbose=True)

    '''

    input_websites = [
        [
            "documents/lean_prover_01.txt",
            "https://leanprover.github.io/theorem_proving_in_lean4/introduction.html"
        ],
        [
            "documents/lean_prover_02.txt",
            "https://leanprover.github.io/theorem_proving_in_lean4/dependent_type_theory.html"
        ],
        [
            "documents/lean_prover_03.txt",
            "https://leanprover.github.io/theorem_proving_in_lean4/propositions_and_proofs.html"
        ],
        [
            "documents/lean_prover_04.txt",
            "https://leanprover.github.io/theorem_proving_in_lean4/quantifiers_and_equality.html"
        ],
        [
            "documents/lean_prover_05.txt",
            "https://leanprover.github.io/theorem_proving_in_lean4/tactics.html"
        ],
        [
            "documents/lean_prover_06.txt",
            "https://leanprover.github.io/theorem_proving_in_lean4/interacting_with_lean.html"
        ],
        [
            "documents/lean_prover_07.txt",
            "https://leanprover.github.io/theorem_proving_in_lean4/inductive_types.html"
        ],
        [
            "documents/lean_prover_08.txt",
            "https://leanprover.github.io/theorem_proving_in_lean4/induction_and_recursion.html"
        ],
        [
            "documents/lean_prover_09.txt",
            "https://leanprover.github.io/theorem_proving_in_lean4/structures_and_records.html"
        ],
        [
            "documents/lean_prover_10.txt",
            "https://leanprover.github.io/theorem_proving_in_lean4/type_classes.html"
        ],
        [
            "documents/lean_prover_11.txt",
            "https://leanprover.github.io/theorem_proving_in_lean4/conv.html"
        ],
        [
            "documents/lean_prover_12.txt",
            "https://leanprover.github.io/theorem_proving_in_lean4/axioms_and_computation.html"
        ],
    ]

    for output_file, url in input_websites:
        print(f"[VectorDB] Scraping {url} to {output_file}")
        VectorDB.scrape_website_html_to_md(url, output_file)

print("run" in dir(VectorDB)) 