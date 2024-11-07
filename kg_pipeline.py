import os
import logging
import dotenv

dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO)
logging.info("Starting the data pipeline...")

# Get list of PDF files from 'files' folder
logging.info('Getting list of PDF files from "files" folder')
files_path = "data"
files = [
    files_path + "/" + file for file in os.listdir(files_path) if file.endswith(".md")
]
logging.info(f"List of PDF files: {files}")

# Instantiate the token text splitter
logging.info("Instantiating the token text splitter")
from langchain.text_splitter import TokenTextSplitter

splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)

# Split the PDFs into chunks as Documents
logging.info("Splitting the PDFs into chunks as Documents")
# from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader

documents = []

for file in files:
    # Load the PDF file
    text_loader = TextLoader(file_path=file)
    # Split the PDF into Documents
    files_documents = text_loader.load_and_split(text_splitter=splitter)
    # Add the Documents to the list
    documents.extend(files_documents)
    logging.info(f"Loaded and split {file} into {len(files_documents)} Documents")

# Instantiate LLM to use with the LLMGraphTransformer
logging.info("Instantiating LLM to use with the LLMGraphTransformer")
from langchain_community.llms import Ollama

llm = Ollama(model="llama3", temperature=0.0)

# To use a local LLM we need to create a chat_prompt to solve the AttributeError from the LLMGraphTransformer

# Create a system message to provide the LLM with the instructions
logging.info(
    "Creating a chat_prompt to provide the LLM with the instructions and examples"
)
from langchain_experimental.graph_transformers.llm import SystemMessage

system_prompt = """
You are a data scientist working for the police and you are building a knowledge graph database.
Your task is to extract information from data and convert it into a knowledge graph database.
Provide a set of Nodes in the form [head, head_type, relation, tail, tail_type].
It is important that the head and tail exists as nodes that are related by the relation.
If you can't pair a relationship with a pair of nodes don't add it.
When you find a node or relationship you want to add try to create a generic TYPE for it that describes the entity you can also think of it as a label.
You must generate the output in a JSON format containing a list with JSON objects. Each object should have the keys: "head", "head_type", "relation", "tail", and "tail_type".
"""

system_message = SystemMessage(content=system_prompt)

# Create a human message to combine with the SystemMessage
# We need:
# 1) A parser to provide the LLM with the format instructions
# 2) Examples to provide the LLM with the context we want to extract
# 3) A list of nodes and relationships to provide the LLM with the context we want to extract

# 1) The parser
# To instantiate a parser we need a Pydantic class.
# Instead of using the default langchain_experimental.graph_transformers.llm.UnstructuredRelation,
# we'll build our own class to provide more crime-related context instructions
from langchain_core.pydantic_v1 import BaseModel, Field


class UnstructuredRelation(BaseModel):
    head: str = Field(
        description=(
            "extracted food related entities in Norwegian like Kjøtt, mat, fisk, poteter, koke, etc."
            "Must use human-readable unique identifier in Norwegian."
        )
    )
    head_type: str = Field(
        description="extracted food related entities in Norwegian like Kjøtt, fisk, poteter, koke, etc"
    )
    relation: str = Field(description="relation between the head and the tail entities")
    tail: str = Field(
        description=(
            "extracted food related entities in Norwegian like Kjøtt, and, fisk, poteter, koke, etc."
            "Must use human-readable unique identifier."
        )
    )
    tail_type: str = Field(
        description="type of the extracted head entity in Norwegian like Kjøtt, mat, fisk, poteter, koke, etc"
    )


# Instantiate the parser with our Pydantic class to provide the LLM with the format instructions
from langchain_experimental.graph_transformers.llm import JsonOutputParser

parser = JsonOutputParser(pydantic_object=UnstructuredRelation)

# 2) The examples
examples = [
    {
        "text": (
            "For å lage en deilig spaghetti carbonara, starter du med å koke spaghetti i saltet vann. "
            "Mens pastaen koker, steker du bacon og blander egg med revet parmesan."
        ),
        "head": "Spaghetti carbonara",
        "head_type": "Dish",
        "relation": "MAIN_INGREDIENT",
        "tail": "Spaghetti",
        "tail_type": "Ingredient",
    },
    {
        "text": (
            "For å lage en deilig spaghetti carbonara, starter du med å koke spaghetti i saltet vann. "
            "Mens pastaen koker, steker du bacon og blander egg med revet parmesan."
        ),
        "head": "Bacon",
        "head_type": "Ingredient",
        "relation": "USED_IN",
        "tail": "Spaghetti carbonara",
        "tail_type": "Dish",
    },
    {
        "text": (
            "En klassisk Caesar salat inneholder romaine salat, krutonger, parmesan og en dressing laget av ansjos, "
            "egg, sitron, olivenolje og worcestershire saus."
        ),
        "head": "Caesar salat",
        "head_type": "Dish",
        "relation": "CONTAINS",
        "tail": "Romaine salat",
        "tail_type": "Ingredient",
    },
    {
        "text": (
            "For å lage en perfekt bløtkake, starter du med å bake en sukkerbrødsbunn. "
            "Deretter fylles kaken med krem og bær, før den dekkes med krem og pyntes."
        ),
        "head": "Bløtkake",
        "head_type": "Dish",
        "relation": "PREPARATION_METHOD",
        "tail": "Baking",
        "tail_type": "Cooking Method",
    },
    {
        "text": (
            "For å lage en perfekt bløtkake, starter du med å bake en sukkerbrødsbunn. "
            "Deretter fylles kaken med krem og bær, før den dekkes med krem og pyntes."
        ),
        "head": "Sukkerbrødsbunn",
        "head_type": "Ingredient",
        "relation": "PART_OF",
        "tail": "Bløtkake",
        "tail_type": "Dish",
    },
    {
        "text": (
            "En tradisjonell pizza margherita lages med en tynn bunn, tomatssaus, fersk mozzarella, "
            "basilikum og olivenolje. Den stekes i en steinovn på høy temperatur."
        ),
        "head": "Pizza margherita",
        "head_type": "Dish",
        "relation": "COOKING_METHOD",
        "tail": "Steinovn",
        "tail_type": "Cooking Equipment",
    },
    {
        "text": (
            "En tradisjonell pizza margherita lages med en tynn bunn, tomatssaus, fersk mozzarella, "
            "basilikum og olivenolje. Den stekes i en steinovn på høy temperatur."
        ),
        "head": "Mozzarella",
        "head_type": "Ingredient",
        "relation": "USED_IN",
        "tail": "Pizza margherita",
        "tail_type": "Dish",
    },
    {
        "text": (
            "For å lage en perfekt boeuf bourguignon, brunes kjøttet først i en gryte. "
            "Deretter tilsettes rødvin, grønnsaker og krydder før det hele småkoker i flere timer."
        ),
        "head": "Boeuf bourguignon",
        "head_type": "Dish",
        "relation": "MAIN_INGREDIENT",
        "tail": "Storfekjøtt",
        "tail_type": "Ingredient",
    },
    {
        "text": (
            "For å lage en perfekt boeuf bourguignon, brunes kjøttet først i en gryte. "
            "Deretter tilsettes rødvin, grønnsaker og krydder før det hele småkoker i flere timer."
        ),
        "head": "Rødvin",
        "head_type": "Ingredient",
        "relation": "USED_IN",
        "tail": "Boeuf bourguignon",
        "tail_type": "Dish",
    },
]
# old_examples = [
#     {
#         "text": (
#             "Fisk kan kokes sammen med poteter, for å lage suppe."
#         ),
#         "head": "Fisk",
#         "head_type": "Mat",
#         "relation": "EKSEMPEL_PAA",
#         "tail": "Matrett",
#         "tail_type": "Matrett",
#     },
# ]

# 3) The list of nodes and relationships
# to be coded and experimented with...

# Instantiate the human prompt template using the examples and Pydantic class with format instructions

from langchain_experimental.graph_transformers.llm import PromptTemplate

human_prompt = PromptTemplate(
    template="""
Examples:
{examples}

For the following text, extract entities and relations as in the provided example.
{format_instructions}\nText: {input}""",
    input_variables=["input"],
    partial_variables={
        "format_instructions": parser.get_format_instructions(),
        "node_labels": None,
        "rel_types": None,
        "examples": examples,
    },
)

# Instantiate the human message prompt to provide the LLM with the instructions and examples
from langchain_experimental.graph_transformers.llm import HumanMessagePromptTemplate

human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)

# Create a chat_prompt combining the system mand human messages to use with the LLMGraphTransformer
from langchain_experimental.graph_transformers.llm import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message_prompt])

# Instantiate the LLMGraphTransformer that will extract the entities and relationships from the Documents
logging.info(
    "Instantiating the LLMGraphTransformer that will extract the entities and relationships from the Documents"
)
from langchain_experimental.graph_transformers import LLMGraphTransformer

llm_transformer = LLMGraphTransformer(llm=llm, prompt=chat_prompt)

# Convert the Documents into Graph Documents
# This is the heavy computation part...
logging.info("Converting the Documents into Graph Documents...")
graph_documents = llm_transformer.convert_to_graph_documents(documents)

# Instantiate the Neo4JGraph to persist the data
logging.info("Instantiating the Neo4JGraph to persist the data")
from langchain_community.graphs import Neo4jGraph

graph = Neo4jGraph()

# Persist the Graph Documents into the Neo4JGraph
logging.info("Persisting the Graph Documents into the Neo4JGraph")
graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)

logging.info("Data pipeline completed successfully!")
