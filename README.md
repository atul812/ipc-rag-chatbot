# IPC RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built using LangChain, Cohere, FAISS, and HuggingFace Embeddings for answering questions from the Indian Penal Code (IPC).

## Overview

This project demonstrates how Large Language Models can be combined with semantic search to answer legal questions based on source documents rather than relying solely on model memory.

The chatbot processes the IPC PDF, converts it into vector embeddings, stores them in a FAISS vector database, retrieves relevant sections for a user query, and generates grounded responses using Cohere's LLM.

## Features

* PDF document ingestion
* Intelligent document chunking
* Semantic search using embeddings
* FAISS vector database
* Retrieval-Augmented Generation (RAG)
* Custom legal response formatting
* Reduced hallucinations through document grounding

## Tech Stack

* Python
* LangChain
* Cohere
* FAISS
* HuggingFace Sentence Transformers
* PyPDF
* Recursive Character Text Splitter

## Architecture

User Question

↓

Retriever (FAISS)

↓

Relevant Chunks

↓

Prompt Template

↓

Cohere LLM

↓

Grounded Answer


## Document Processing Pipeline:

IPC PDF

↓

PDF Loader

↓

Text Chunking

↓

Embeddings

↓

FAISS Vector Store

## How It Works

### Step 1: Load PDF

The IPC document is loaded using PyPDFLoader.

### Step 2: Chunking

The document is divided into smaller chunks using RecursiveCharacterTextSplitter.

* Chunk Size: 1000
* Overlap: 200

### Step 3: Embeddings

Each chunk is converted into vector embeddings using the all-MiniLM-L6-v2 sentence transformer model.

### Step 4: Vector Database

Embeddings are stored inside a FAISS vector store for efficient similarity search.

### Step 5: Retrieval

For every user query, the top relevant document chunks are retrieved.

### Step 6: Generation

The retrieved context is provided to Cohere's LLM, which generates a grounded response.

## Example Queries

* What is murder under IPC?
* What is the punishment for theft?
* Explain Section 420.
* What is criminal breach of trust?
* What is the difference between theft and robbery?

## Challenges and Learnings

During development, I experimented with:

* Different chunk sizes
* Chunk overlap values
* Retrieval depth (k value)
* Prompt engineering techniques
* Hallucination reduction strategies

These experiments significantly improved answer relevance and consistency.

## Future Improvements

* Source citation support
* Streamlit web interface
* Conversation memory
* Multi-document support
* Hybrid retrieval
* Reranking pipeline

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

