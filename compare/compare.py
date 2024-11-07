from compare.fixtures import QUESTIONS
from chains.kg import KG


def compare():
    chains = {"rag_graph": KG()}

    for chain_key in chains.keys():
        print(f"Run queries on chain {chain_key}")
        chain = chains[chain_key]
        for entry in QUESTIONS:
            question = entry["question"]
            expected_reply = entry["expected_reply"]
            print(f"question: {question}")
            print(f"expected_reply: {expected_reply}")
            actual_reply = chain.query(question=question)
            print("actual_reply:" + repr(actual_reply))
            print("TODO: LLM as a judge eval reply")
