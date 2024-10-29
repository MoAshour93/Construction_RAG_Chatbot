# 🏗️ Construction RAG Chatbot with Mistral 7B 🗂️💬

Welcome to the **Construction RAG Chatbot** repository! This project enables interactive querying of PDF documents using the **Mistral 7B** model. By combining **Retrieval-Augmented Generation (RAG)** techniques and the Mistral model, users can upload PDFs and ask questions, receiving the most contextually accurate responses. Built with LangChain and Streamlit, this chatbot is designed to simplify document analysis tasks within the construction industry, making information retrieval fast and intuitive.

## 📑 Table of Contents
1. [📋 Project Overview](#project-overview)
2. [🚀 Key Features](#key-features)
3. [🔧 Installation & Setup](#installation--setup)
4. [🛠️ Usage Guide](#usage-guide)
5. [🏗️ Construction Use Cases](#construction-use-cases)
6. [📈 Limitations & Future Enhancements](#limitations--future-enhancements)
7. [🔗 General Links & Resources](#general-links--resources)

---

## 📋 Project Overview

This chatbot utilizes the **Mistral 7B** language model to perform document-based question answering with PDF files. Powered by **Retrieval-Augmented Generation (RAG)**, the chatbot extracts relevant information from the uploaded PDFs, processes the content into embeddings, and uses the Mistral 7B model to respond with contextually accurate answers. This setup makes it ideal for handling large volumes of information, as commonly found in construction project documentation.

---

## 🚀 Key Features

- **PDF Querying**: Upload PDF documents to ask questions and get precise answers from the content.
- **Retrieval-Augmented Generation (RAG)**: Combines document embeddings with LLM-based response generation for enhanced accuracy.
- **Optimized for Construction**: Tailored for industry use, particularly useful for RICS APC submission preparation, project documentation, and compliance verification.
- **Streamlit Interface**: A user-friendly interface for easy interaction and querying.
  
---

## 🔧 Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/YourUsername/Construction_RAG_Chatbot.git
   cd Construction_RAG_Chatbot
   ```

2. **Set Up Environment**:
   - **Python 3.10** or later is recommended.
   - Create and activate a virtual environment:
     ```bash
     python -m venv rag_env
     source rag_env/bin/activate  # On Windows, use `rag_env\Scripts\activate`
     ```

3. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Directory Setup**:
   - Create a folder named `data` in the project directory and place the PDFs for querying within this folder.

---

## 🛠️ Usage Guide

1. **Run the Streamlit App**:
   ```bash
   streamlit run Mistral_7B_Chatbot_Cuda_enabled.py
   ```

2. **Upload a PDF**:
   - Drag and drop a PDF document into the app’s upload section.

3. **Ask Questions**:
   - Enter a question in the text input field about the document’s content (e.g., “What safety guidelines are mentioned?”).
   - The chatbot will respond with the most relevant information from the document, leveraging the power of Mistral 7B for accurate and context-aware answers.

---

## 🏗️ Construction Use Cases

- **RICS APC Submissions**: Streamline RICS APC preparation by asking competency-related questions from guidelines and previous submissions.
- **Project Document Analysis**: Query project documents to quickly access information on compliance, safety protocols, and resource management.
- **Bid Preparation**: Improve bid responses by retrieving specific contract details and project requirements from bid-related documents.
- **Construction Compliance**: Validate document compliance by querying directly on standards and guidelines within uploaded PDFs.

---

## 📈 Limitations & Future Enhancements

### Limitations
- **Hardware Requirements**: Running the Mistral 7B model locally may require significant computational resources.
- **Response Time**: Processing large documents may increase response latency.

### Future Enhancements
- **Enhanced Document Type Support**: Extend support to additional formats like DOCX and TXT.
- **Cloud Deployment**: Deploy on cloud platforms for accessibility and scalability.
- **GUI Improvements**: Incorporate more customizable options for document management and query handling.

---

## 🔗 General Links & Resources

- **Our Website**: [www.apcmasterypath.co.uk](https://www.apcmasterypath.co.uk)
- **LinkedIn**: [Mohamed Ashour](https://www.linkedin.com/in/mohamed-ashour-0727/)
