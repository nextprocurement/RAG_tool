import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd

########################################################################
# CLASS DEFINITIONS
########################################################################
class CustomEmbeddings():
    def __init__(self, model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        self.model = SentenceTransformer(model)

    def encode(self, texts):
        return self.model.encode(texts)

    def __call__(self, input):
        return self.encode(input)


class Chunker:
    def __init__(self, context_window=3000, max_windows=5, window_overlap=0.02):
        self.context_window = context_window
        self.max_windows = max_windows
        self.window_overlap = window_overlap

    def __call__(self, paper):
        snippet_idx = 0
        startpos = 0

        while snippet_idx < self.max_windows and len(paper) > startpos:
            endpos = startpos + \
                int(self.context_window * (1.0 + self.window_overlap))
            if endpos > len(paper):
                endpos = len(paper)
            snippet = paper[startpos:endpos]

            next_newline_pos = snippet.rfind('\n')
            if next_newline_pos != -1 and next_newline_pos >= self.context_window // 2:
                snippet = snippet[:next_newline_pos + 1]

            yield snippet_idx, snippet.strip()
            startpos += self.context_window - \
                int(self.context_window * self.window_overlap)
            snippet_idx += 1

########################################################################
# METHODS
########################################################################
def create_vector_store(
    df: pd.DataFrame,
    path_to_index: str = '/export/usuarios_ml4ds/cggamella/RAG_tool/example1',
    embedding_model: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    vector_store_name: str = 'test1',
    context_window: int = 3000,
    max_windows: int = 100,
    window_overlap: float = 0.1
):
    embedding_model = CustomEmbeddings(
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    chunker = Chunker(context_window, max_windows, window_overlap)

    # Create vector store
    client = chromadb.PersistentClient(path=path_to_index)
    collection = client.get_or_create_collection(
        name=vector_store_name, embedding_function=embedding_model)

    def addVectorDataToDb(df, chunker) -> None:
        embeddings: list = []
        metadatas: list = []  # @TODO: Check if necessary
        documents: list = []
        ids: list = []

        # Normalize text
        df['text'] = df['text'].str.lower()

        try:
            for idx, row in df.iterrows():
                # divide in chunks
                for id_chunk, chunk in chunker(row.text):
                    embeddings.append(embedding_model.encode(chunk).tolist())
                    # metadatas.append(f"{str(idx)}_{str(id_chunk)}")
                    documents.append(chunk)
                    ids.append(f"{str(idx)}_{str(id_chunk)}")
            collection.add(
                embeddings=embeddings,
                # metadatas=metadatas,
                documents=documents,
                ids=ids
            )
            print("Data added to collection")
        except Exception as e:
            print("Add data to db failed : ", e)

    # Insert data into collection
    addVectorDataToDb(df, chunker)

    return collection


def searchDataByVector(collection, embedding_model, query):
    try:
        query_vector = embedding_model.encode(query).tolist()
        res = collection.query(
            query_embeddings=[query_vector],
            n_results=1,
            include=['distances', 'embeddings', 'documents'],
        )
        print("Query", "\n--------------")
        print(query)
        print("Result", "\n--------------")
        print(res['documents'][0][0])
        print("Vector", "\n--------------")
        print(res['embeddings'][0][0])
        print("")
        print("")
        print("Complete Response", "\n-------------------------")
        print(res)

    except Exception as e:
        print("Vector search failed : ", e)
