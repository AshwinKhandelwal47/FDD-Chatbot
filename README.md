# FDD-Chatbot
Financial Due Diligence Assistant powered by RAG and provider-agnostic LLMs

A Python-based AI assistant for financial analysts. Upload financial documents (annual reports, income statements, balance sheets), and ask intelligent due diligence questions via a chat interface — backed by whichever LLM you have access to, with automatic fallback if one goes down.

## What It Does

1. Ingests financial PDFs and Excel/CSV files — extracts text and tables with dual-method parsing
2. Embeds extracted content into a ChromaDB vector store for semantic retrieval
3. Answers financial due diligence questions via RAG-powered chat with cited sources
4. Calculates financial ratios deterministically in Python — the LLM interprets, never computes
5. Detects financial red flags automatically from extracted data
6. Switches LLM providers (Groq, DeepSeek, GLM, NVIDIA, Ollama) via a dropdown — no code changes required


## Architecture
<img width="487" height="636" alt="image" src="https://github.com/user-attachments/assets/9e366088-3e67-4bca-abcf-291d1cfc209c" />

## Key Design Decisions

#### 1. Provider agnosticism

All LLM interaction goes through the LLMProvider abstract interface. chatbot.py and extractor.py never import a concrete provider class — only src.llm.factory. Adding a new provider is a config change, not a code change.

#### 2. Fallback chain

ProviderUnavailableError (network/auth/rate-limit) triggers automatic fallback to the next provider in LLM_FALLBACK_ORDER. ExtractionError (bad schema/validation) does NOT — a data problem won't be fixed by a different provider.

#### 3. LLM interprets, never computes

All financial ratios (gross margin, CAGR, DSO, current ratio, etc.) are computed deterministically in src/calculations/ratios.py. The LLM receives pre-computed values and explains them in natural language. This is your zero-hallucination-on-numbers guarantee.

#### 4. Pydantic validation gates

Extracted financial data passes through validators that check internal consistency — e.g. gross_profit ≈ revenue - cogs within 0.1% tolerance. Extraction failures are loud and never silent.

#### 5. Financial-aware chunking

Tables are never split mid-row. Consecutive pages with matching column headers are merged into one chunk. Text and table chunks carry identical metadata keys so retrieval handles both uniformly.

## Supported Financial Documents

**1. PDF(text-based)** PyMuPDF + pdfplumber Primary method for annual reports                              
**2. PDF(complex tables)** camelot-py (lattice → stream fallback) Requires Ghostscript on Windows                               
**3. Excel(.xlsx /.xls)** openpyxl via pandas Each sheet becomes one table chunk                                                         
**4. CSV** pandas Full file as one table chunk

## Tech Stack

1. Language: Python 3.12+
2. UI: StreamlitLLM
3. Client: huggingface_hub InferenceClient (OpenAI-compatible)
4. Embeddings: sentence-transformers / all-MiniLM-L6-v2
5. Vector DB: ChromaDB
6. PDF Text: PyMuPDF (fitz)
7. PDF Tables: pdfplumber + camelot-py
8. Excel/CSV: pandas + openpyxl
9. Data Validation: Pydantic v2
10. Financial Math: Python + scipy
11. Testing: pytest + unittest.mock
