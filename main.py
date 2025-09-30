import os
import warnings
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

warnings.filterwarnings("ignore")

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("⚠️ Falta la variable OPENAI_API_KEY en el archivo .env")

llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

pdf_path = "document.pdf"
reader = PdfReader(pdf_path)

# Extraer texto de todas las páginas
text = ""
for page in reader.pages:
    text += page.extract_text() or ""

# Dividir en chunks manejables
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
chunks = text_splitter.split_text(text)

print(f"✂️ Se generaron {len(chunks)} chunks de texto")

vectorstore = Chroma.from_texts(chunks, embeddings)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

while True:
    query = input("\n❓ Pregunta ('salir' para terminar): ")
    if query.lower() == "salir":
        break

    result = qa.invoke({"query": query})
    respuesta = result["result"]

    print("\n🤖 Respuesta:", respuesta)

    print("\n📄 Fuentes relacionadas:")
    for i, doc in enumerate(result["source_documents"], start=1):
        preview = doc.page_content[:200].replace("\n", " ")
        print(f"  {i}. {preview}...")
