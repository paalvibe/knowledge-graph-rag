from langchain_community.llms import Ollama
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
import os
import logging
import dotenv

from chains.querychain import QueryChain


class KG(QueryChain):
    def _setup(self):
        dotenv.load_dotenv()
        logging.basicConfig(level=logging.INFO)
        logging.info('Starting up the Knowledge Graph RAG...')

        # Instantiate the Neo4J connector
        logging.info(f'Instantiating the Neo4J connector for: { os.getenv("NEO4J_URI") }')
        graph = Neo4jGraph()

        # Instantiate LLM to use with the Graph RAG
        logging.info('Instantiating LLM to use with the LLMGraphTransformer')
        llm = Ollama(model='llama3', temperature=0.0)

        # Instantiate the langchain Graph RAG with the Neo4J connector and the LLM
        self.chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm, verbose=True)

        logging.info('Knowledge Graph RAG is ready to go!')
        logging.info('='*50)

    def query(self, *, question):
        result = self.chain.invoke({"query": question})
        print("kg.py:" + repr(33) + ":result:" + repr(result))
        if result['result']:
            ret = result['result']
        else:
            ret = result
        print(f"KG result: {ret}")
        return ret
