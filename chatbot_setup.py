# F1 AI Chatbot - RAG System Setup
# This creates the vector database for intelligent Q&A

import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("🤖 Setting up F1 AI Race Engineer...\n")

# Step 1: Load all knowledge documents
print("📚 Loading knowledge base documents...")
loader = DirectoryLoader('race_knowledge/', glob="*.txt", loader_cls=TextLoader)
documents = loader.load()

print(f"✅ Loaded {len(documents)} documents\n")

# Step 2: Split documents into chunks
print("✂️ Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
texts = text_splitter.split_documents(documents)

print(f"✅ Created {len(texts)} text chunks\n")

# Step 3: Create embeddings and vector store
print("🧠 Creating vector database with OpenAI embeddings...")
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(texts, embeddings)

print("✅ Vector database created!\n")

# Step 4: Save vector store
print("💾 Saving vector database...")
vectorstore.save_local("faiss_index")

print("✅ Saved to 'faiss_index/' folder\n")

# Step 5: Test the chatbot
print("🔮 Testing chatbot with sample question...\n")

# Load the LLM
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7
)

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Test question
test_question = "What was Verstappen's tire strategy at Monaco?"

print(f"❓ Question: {test_question}\n")

# Get relevant docs
docs = retriever.invoke(test_question)

# Create prompt with context
context = "\n\n".join([doc.page_content for doc in docs])
prompt = f"Based on this context:\n\n{context}\n\nAnswer this question: {test_question}"

# Get answer
response = llm.invoke(prompt)

print(f"🏎️ AI Race Engineer Response:")
print(f"{response.content}\n")

print("🏁 Chatbot setup complete!")
print("✅ Ready to integrate into dashboard!")