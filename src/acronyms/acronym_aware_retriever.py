from typing import List, Union
import dspy
import re
from nltk import sent_tokenize


class AcronymAwareRetriever(dspy.Retrieve):
    """
    Custom retriever for searching documents containing a specific acronym.
    """

    def __init__(self, vectordb, k: int = 3):
        super().__init__(k=k)
        self.vectordb = vectordb

    def forward(self, acronym: Union[str, List[str]], k: int = None) -> dspy.Prediction:
        """
        Retrieve documents containing the provided acronym by directly querying
        the vector database. The documents are filtered to include only those
        where the acronym appears as a standalone word.
        """
        # Creating regex to match the acronym as a standalone word
        regex = r'\b' + re.escape(acronym) + r'\b'

        # Retrieve documents that might contain the acronym
        results = self.vectordb.get(
            where_document={"$contains": acronym},
            limit=k if k else self.k,
            include=['documents', 'embeddings'])

        # Extract relevant information
        documents = results.get('documents', [])
        embeddings = results.get('embeddings', [])

        # Filter documents and their embeddings using regex, avoiding duplicates
        seen_documents = set()
        filtered_documents = []
        filtered_embeddings = []
        filtered_passages = []

        for doc, emb in zip(documents, embeddings):
            # Check if the document contains the acronym as a standalone word and is not already added
            if re.search(regex, doc, re.IGNORECASE) and doc not in seen_documents:
                seen_documents.add(doc)
                filtered_documents.append(doc)
                filtered_embeddings.append(emb)
                # Extract passages for documents that pass the regex check
                filtered_passages.append(self.extract_passages(doc, acronym))

        return dspy.Prediction(
            documents=filtered_documents,
            embeddings=filtered_embeddings,
            passages=filtered_passages
        )

    def extract_passages(self, text, acronym):
        """
        Extract the two preceding and two following sentences after the first appearance of the acronym in
        the text.
        """
        sentences = sent_tokenize(text)
        for i, sentence in enumerate(sentences):
            if acronym.lower() in sentence.lower():
                start = max(i - 2, 0)
                end = min(i + 3, len(sentences))
                return ' '.join(sentences[start:end])
        return ""
