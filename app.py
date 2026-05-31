import os, warnings
warnings.filterwarnings("ignore")

from langchain_community.document_loaders   import PyPDFLoader
from langchain_text_splitters               import RecursiveCharacterTextSplitter
from langchain_community.vectorstores       import FAISS
from langchain_community.embeddings         import HuggingFaceEmbeddings
from langchain_core.prompts                 import PromptTemplate
from langchain_cohere                       import ChatCohere


COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not COHERE_API_KEY:
    raise ValueError(
        "COHERE_API_KEY environment variable not found."
    )

PDF_PATH = "data/THE_INDIAN_PENAL_CODE.pdf"

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(
        f"PDF not found: {PDF_PATH}"
    )
print(f"PDF ready → {PDF_PATH}")

loader   = PyPDFLoader(PDF_PATH)
pages    = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks   = splitter.split_documents(pages)
print(f" {len(pages)} pages → {len(chunks)} chunks")

print("Creating embeddings — using free HuggingFace model ...")
print("   (First run downloads the model ~90MB, takes 1-2 min) ...")

embeddings   = HuggingFaceEmbeddings(
    model_name = "all-MiniLM-L6-v2"   # free, fast, no API key needed
)
vector_store = FAISS.from_documents(chunks, embeddings)
print("Vector store ready")

prompt = PromptTemplate(
  input_variables=["context", "question"],
  template="""
  You are a legal assistant specializing in the Indian Penal Code.

  Use only the provided context.

  If the answer is not found in the context, say so.

  Provide the response in this format:

  Section:
  Explanation:
  Punishment:

  Context:
  {context}

  Question:
  {question}

  Answer:
  """
)

llm          = ChatCohere(model="command-a-03-2025", temperature=0.2)
retriever    = vector_store.as_retriever(search_kwargs={"k": 4})
chain        = prompt | llm

def chat(question):
    docs    = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    result  = chain.invoke({"context": context, "question": question})
    pages = []

    for doc in docs:
      page = doc.metadata.get("page")
      if page is not None:
        pages.append(page + 1)

    print(result.content)
    print(f"\nSources: {sorted(set(pages))}")


if __name__ == "__main__":
  print("IPC RAG Chatbot Ready")
  print("Type 'exit' to quit.\n")

  chat("What is murder under the IPC?")

  while True:
    question = input("You: ")

    if question.lower() == "exit":
      break
    chat(question)
