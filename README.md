# \# AUTOSAR HLD AI Analyzer

# 

# An AI-powered document analysis application designed to help users explore AUTOSAR High-Level Design (HLD) and technical documents using Retrieval-Augmented Generation (RAG).

# 

# \## Overview

# 

# AUTOSAR HLD AI Analyzer combines document processing, semantic search, vector embeddings, and a local language model to answer questions using relevant information retrieved from technical documents.

# 

# The goal is to make large engineering documents easier to navigate, understand, and analyze.

# 

# \## Key Features

# 

# \- \*\*PDF document processing\*\* — extract text from technical PDF documents.

# \- \*\*Semantic retrieval\*\* — retrieve relevant document passages based on the meaning of a question.

# \- \*\*RAG-based question answering\*\* — generate responses using retrieved context.

# \- \*\*Vector database\*\* — store and search document embeddings using ChromaDB.

# \- \*\*Local LLM integration\*\* — use Ollama to run a supported language model locally.

# \- \*\*Interactive interface\*\* — interact with the analyzer through a Streamlit application.

# 

# \*Feature availability depends on the current implementation and configuration.\*

# 

# \## Technology Stack

# 

# | Component | Technology |

# |---|---|

# | Programming language | Python |

# | User interface | Streamlit |

# | PDF processing | PyMuPDF |

# | Embeddings | Sentence Transformers |

# | Vector database | ChromaDB |

# | Local language model runtime | Ollama |

# | AI architecture | Retrieval-Augmented Generation (RAG) |

# 

# \## Project Structure

# 

# ```text

# AUTOSAR-HLD-AI-Analyzer/

# ├── app.py

# ├── requirements.txt

# ├── README.md

# ├── .gitignore

# └── src/

# &#x20;   ├── embeddings/

# &#x20;   ├── ingestion/

# &#x20;   ├── preprocessing/

# &#x20;   ├── rag/

# &#x20;   └── vectorstore/

# ```

# 

# \## Prerequisites

# 

# Before running the project, install:

# 

# \- Python 3.11 or another version compatible with the project's dependencies.

# \- Git.

# \- Ollama.

# \- A compatible local language model.

# \- Sufficient RAM and disk space for the selected model and document index.

# 

# \## Installation and Setup

# 

# \### 1. Clone the repository

# 

# ```bash

# git clone https://github.com/babli0510/AUTOSAR-HLD-AI-Analyzer.git

# cd AUTOSAR-HLD-AI-Analyzer

# ```

# 

# \### 2. Create a virtual environment

# 

# \*\*Windows:\*\*

# 

# ```cmd

# python -m venv .venv

# .venv\\Scripts\\activate

# ```

# 

# \### 3. Install dependencies

# 

# ```bash

# python -m pip install -r requirements.txt

# ```

# 

# \### 4. Set up Ollama

# 

# Install Ollama from https://ollama.com/.

# 

# Download the model configured for your application. For example, if the application uses `qwen2.5:0.5b`:

# 

# ```bash

# ollama pull qwen2.5:0.5b

# ```

# 

# Make sure Ollama is running before using the application.

# 

# \### 5. Start the application

# 

# ```bash

# python -m streamlit run app.py --server.fileWatcherType none --server.runOnSave false

# ```

# 

# Streamlit will display a local URL in the terminal. Open that URL in your browser.

# 

# \## How It Works

# 

# 1\. A technical PDF is processed to extract its text.

# 2\. Extracted content is divided into searchable chunks.

# 3\. An embedding model converts chunks into vector representations.

# 4\. ChromaDB stores and retrieves relevant chunks.

# 5\. The retrieved context is passed to the language model.

# 6\. The application presents an answer based on the available context.

# 

# \## Important Notes

# 

# \- The vector database and source PDFs are not included in this repository.

# \- Document indexing may be required before querying documents.

# \- Model names, database paths, and ingestion settings must match the application configuration.

# \- AI-generated responses may contain errors. Verify important engineering details against the original AUTOSAR documentation.

# \- Do not upload confidential documents, credentials, API keys, or copyrighted material without authorization.

# 

# \## Future Improvements

# 

# \- Improve retrieval accuracy and answer grounding.

# \- Add source citations and page-level references.

# \- Add automated evaluation for question-answer quality.

# \- Improve document chunking and ranking.

# \- Add clearer setup and deployment instructions.

# 

# \## License

# 

# No license has been specified yet. All rights are reserved by default unless a license is added by the project owner.

# 

# \## Author

# 

# \*\*Project:\*\* AUTOSAR HLD AI Analyzer  

# \*\*GitHub:\*\* https://github.com/babli0510/AUTOSAR-HLD-AI-Analyzer



