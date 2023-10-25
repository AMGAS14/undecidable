from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

#Document loader
loader = TextLoader("abc.txt")
documents = loader.load()

#Document Transoformer
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

#Embedding model
embeddings = OpenAIEmbeddings()

#Vector DB to store embeddings
docsearch = Chroma.from_documents(texts, embeddings)

#Retriever being used using parameter retriever
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())

query = "What this document about? Summarize it in a paragraph"
qa.run(query)