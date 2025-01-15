
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

DATA_PATH = '../data/'
DB_FAISS_PATH = 'vectorstore_faiss/db'
print("paths defined")
# Create vector database
def create_vector_db():
    # Load documents from the directory
    loader = DirectoryLoader(DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)

    documents = loader.load()
    print("documents loaded")
    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    texts = text_splitter.split_documents(documents)
    print("text splited")
    # Load embedding model
    embed_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large",
                                       model_kwargs={'device': 'cpu'})
    print("model taken")
    # Create FAISS vectorstore
    print("creating vector store")
    db = FAISS.from_documents(texts, embed_model)
    print("done creation")
    db.save_local(DB_FAISS_PATH)
    print("saved")
    print(f"Vector database saved at: {DB_FAISS_PATH}")

if __name__ == "__main__":
    create_vector_db()
