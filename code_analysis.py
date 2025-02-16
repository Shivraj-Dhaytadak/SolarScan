from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter , Language
from langchain_ollama.chat_models import ChatOllama
from langchain.memory import ConversationSummaryMemory
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import Agent ,AgentExecutor, create_tool_calling_agent

def loader(PATH):
    return GenericLoader.from_filesystem(PATH,
                                        glob = "**/*",
                                       suffixes=[".py"],
                                       show_progress=True,
                                       parser = LanguageParser(language=Language.PYTHON, 
                                                               parser_threshold=500)
                                        ).load()

def vectorrizer(PATH):
    final_docs = RecursiveCharacterTextSplitter(
        chunk_size=1000,chunk_overlap=200
        ).split_documents(loader(PATH))
    
    embeddings = OllamaEmbeddings(model = "qwen2.5-coder:3b")
    return Chroma.from_documents(documents= final_docs , embedding=embeddings, persist_directory='./vector')

def retrival_chain(PATH ,INPUT):

    llm = ChatOllama(model="qwen2.5-coder:3b" , num_predict=30000)
    PROMPT = """
    Youre a software engineer and you have to explain the given code based on the context from retrieved documents.
    Explain every thing in a detailed manner like youre having a Knowledge Transfer session with a colleague.
    Things you need to do give explanation of the code start from classes and their methods and working and so on. DO NOT SUMMARIZE.
    explain each function in detail and how it works and what it does. if asked only a function explain that function only with reference to the file it is used.
    DO NOT SUMMARIZE.
    context: {context}
    Question:{input}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(PROMPT)

    stuff_chain = create_stuff_documents_chain(llm=llm,prompt=prompt, output_parser=StrOutputParser())
    retrieval_chain = create_retrieval_chain(
        retriever=vectorrizer(PATH=PATH).as_retriever(
            search_type="mmr",
            search_kwargs={'k': 6}
            ),combine_docs_chain=stuff_chain)

    # return retrieval_chain.invoke({'input': INPUT})
    for i in retrieval_chain.stream({'input': INPUT}):
        try:
            if i['answer']:
                yield i['answer']
        except:
            pass