from flask import Blueprint, jsonify, request
from mongo_client import get_database
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models.azure_openai import AzureChatOpenAI
from bson.json_util import dumps
from typing import List
from langchain.schema import Document
from langchain.vectorstores.base import VectorStoreRetriever
from pydantic import Field
from langchain.llms.openai import AzureOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.agents.agent_toolkits import (
    create_vectorstore_router_agent,
    VectorStoreRouterToolkit,
    VectorStoreInfo,
)
import weaviate
from langchain.vectorstores import Weaviate
import os
import uuid

indexRoutes = Blueprint('index_route', __name__)

dbname = get_database()
collection_name = dbname["esg"]


class FilteredRetriever(VectorStoreRetriever):
    vectorstore: VectorStoreRetriever
    search_type: str = "similarity"
    search_kwargs: dict = Field(default_factory=dict)
    filter_prefix: List[str]
    key: str

    def get_relevant_documents(self, query: str) -> List[Document]:
        results = self.vectorstore.get_relevant_documents(query=query)
        return [doc for doc in results if doc.metadata[self.key] in self.filter_prefix]


@indexRoutes.route("/")
def get():
    item_details = collection_name.find()
    list_cur = list(item_details)
    json_data = dumps(list_cur)
    return jsonify(json_data), "200"


@indexRoutes.route("/create-vec", methods=['POST'])
def vec():
    try:
        os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["OPENAI_API_BASE"] = "https://usedsatqs2aoa01.openai.azure.com/"
        os.environ["OPENAI_API_KEY"] = "9268bc1f2d534e459f61f70d1bae86d1"
        os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
        loader = PyPDFLoader("./docs/Fannie_Mae_ESG_2021.pdf")
        documents = loader.load_and_split()
        unique_id = uuid.uuid1()
        for document in documents:
            document.metadata = {"file_id": str(unique_id)}
        text_splitter = CharacterTextSplitter(
            chunk_size=6000, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        print("document generated")
        embeddings = OpenAIEmbeddings(chunk_size=1)
        print("embeded")
        docsearch = MongoDBAtlasVectorSearch.from_documents(
            docs, embeddings, collection=collection_name, index_name="esg-search-index")
        print("docsearch done")
        print(docsearch)
        return "200"
    except Exception as e:
        print(e)
        return "400"


@indexRoutes.route("/ans", methods=['POST'])
def answer():
    try:
        question = request.form["question"]
        os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["OPENAI_API_BASE"] = "https://usedsatqs2aoa01.openai.azure.com/"
        os.environ["OPENAI_API_KEY"] = "9268bc1f2d534e459f61f70d1bae86d1"
        os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
        embeddings = OpenAIEmbeddings(chunk_size=1)
        # total vector store from db and collection
        vectorstore = MongoDBAtlasVectorSearch(
            collection_name, embeddings, index_name="esg-search-index")
        print("parent vectorstore done")
        # for multiple selected files -> create a vector combination
        fannie_vector = FilteredRetriever(
            vectorstore=vectorstore.as_retriever(), filter_prefix=["aef3a936-26db-11ee-b209-44e517a137fd"], key="file_id")
        print("fannie filtered vectorstore done")
        retreivalQAFM = ConversationalRetrievalChain.from_llm(AzureChatOpenAI(
            deployment_name="gpt-4", model_name="gpt-4", temperature=0), fannie_vector)
        print("fannioe mae CQA done")
        newyork_vector = FilteredRetriever(
            vectorstore=vectorstore.as_retriever(), filter_prefix=["4eef720f-26db-11ee-a3a0-44e517a137fd"], key="file_id")
        print("new york filtered vectorstore done")
        retreivalQABNY = ConversationalRetrievalChain.from_llm(AzureChatOpenAI(
            deployment_name="gpt-4", model_name="gpt-4", temperature=0), newyork_vector)
        print("BNY CQA done")
        tools = [
            Tool(
                name="Fannie Mae",
                func=retreivalQAFM.run,
                description="useful for when you need to answer questions about Fannie Mae ESG Report.",
            ),
            Tool(
                name="Bank Of New York Melon",
                func=retreivalQABNY.run,
                description="useful for when you need to answer questions about Bank Of New York Melon ESG Report.",
            ),
        ]
        print("tools done")
        agent = initialize_agent(
            tools, llm=AzureChatOpenAI(
                deployment_name="gpt-4", model_name="gpt-4", temperature=0), agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
        )
        print("agent init done")
        # create combination of vectors
        # perform a similarity search between a query and the ingested documents
        # docs = vectorstore.similarity_search(question)
        # for single selected docs -> retrieve only single vector
        # filtered_retriever = FilteredRetriever(
        #     vectorstore=vectorstore.as_retriever(), filter_prefix=`1  q["aef3a936-26db-11ee-b209-44e517a137fd"], key="file_id")
        # qa = ConversationalRetrievalChain.from_llm(AzureChatOpenAI(
        #     deployment_name="gpt-4", model_name="gpt-4", temperature=0), vectorstore.as_retriever())

        # agent.run(question)
        print(agent.run(f"{'question': {question}, 'chat_history': {[]}}"))
        # response = system_response['answer']
        # final_res = {
        #     "answer": response
        # }
        # return jsonify(final_res), 200
        return 200
    except Exception as e:
        print(e)
        return "400"


@indexRoutes.route("/insert", methods=['POST'])
def create():
    item_1 = {
        "_id": "U1IT00001",
        "item_name": "Blender",
        "max_discount": "10%",
        "batch_number": "RR450020FRG",
        "price": 340,
        "category": "kitchen appliance"
    }

    item_2 = {
        "_id": "U1IT00002",
        "item_name": "Egg",
        "category": "food",
        "quantity": 12,
        "price": 36,
        "item_description": "brown country eggs"
    }
    collection_name.insert_many([item_1, item_2])
    return "200"


@indexRoutes.route("/weaviate-vector", methods=['POST'])
def weaviatevector():
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_BASE"] = "https://usedsatqs2aoa01.openai.azure.com/"
    os.environ["OPENAI_API_KEY"] = "9268bc1f2d534e459f61f70d1bae86d1"
    os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
    index_name = "esg-index"
    loader = PyPDFLoader("./docs/Fannie_Mae_ESG_2021.pdf")
    documents = loader.load_and_split()
    unique_id = uuid.uuid1()
    for document in documents:
        document.metadata = {"file_id": str(unique_id)}
    text_splitter = CharacterTextSplitter(
        chunk_size=6000, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    print("document generated")
    embeddings = OpenAIEmbeddings(chunk_size=1)
    print("embeded")
    db = Weaviate.from_documents(
        docs, embeddings, weaviate_url='', by_text=False)
    print("client ready")
    print("vector created", db)
    return "200"
