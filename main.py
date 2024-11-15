from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model 
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from pydantic import BaseModel
from llama_index.core.output_parsers import PydanticOutputParser 
from llama_index.core.query_pipeline import QueryPipeline
from prompts import context,code_parser_template
from code_reader import code_reader
from dotenv import load_dotenv
import httpx
# import ast
import json
import os

load_dotenv(override=True)
my_var = os.getenv('LLAMA_CLOUD_API_KEY')
print(f"LLAMA_CLOUD_API_KEY: {my_var}")


#Mistral is used for text completions

llm = Ollama(
    model="mistral",
    request_timeout=1800.0
)


parser = LlamaParse(result_type = "markdown")

file_extractor = {".pdf":parser}
documents = SimpleDirectoryReader("./data_new",file_extractor=file_extractor).load_data()


embed_model = resolve_embed_model("local:BAAI/bge-m3")
vector_index=VectorStoreIndex.from_documents(documents,embed_model=embed_model)
query_engine = vector_index.as_query_engine(llm=llm)

tools = [
    QueryEngineTool(
        query_engine=  query_engine,
        metadata=ToolMetadata(
            name="treatment_doc",
            description="this gives documentation about diseases and some treatments. Use this for reading docs for the treatment"
        ),
        
    ),
]
    
agent = ReActAgent.from_tools(tools,llm=llm,verbose= True,context=context)




while (prompt := input("Enter a Prompt(q to quit):")) != "q":
    try:
        result = agent.query(prompt)
        print("Response:", result)
        with open(os.path.join("output", "treatment.txt"), "a") as f:
            
            f.write("#######################################################################"+ str(result))
    except httpx.ReadTimeout:
        print("Request timed out. Please try again.")
    except Exception as e:
        print(f"An error occurred: {e}")



    
    
    

