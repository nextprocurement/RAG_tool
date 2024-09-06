from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
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

class GraphVisualizer:
    def __init__(self, kernel='umbral', threshold=0.21, remove_self_links=True):
        self.kernel = kernel
        self.threshold = threshold
        self.remove_self_links = remove_self_links

    def set_embeddings(self, embeddings):
        self.embeddings = np.array(embeddings)

    def _get_similarity_matrix(self):
        if self.embeddings is None:
            raise ValueError("Please set embeddings before trying to get similarity matrix.")
        #print(type(self.embeddings), self.embeddings.shape)
        print(self.embeddings)
        similarity_matrix = cosine_similarity(self.embeddings)
        print("La matriz de similitud es:\n", similarity_matrix)
        if self.kernel == "umbral":
            similarity_matrix[similarity_matrix < self.threshold] = 0
        if self.remove_self_links:
            np.fill_diagonal(similarity_matrix, 0)
        return similarity_matrix

    def visualize_graph(self, plot_graph=True):
        similarity_matrix = self._get_similarity_matrix()
        G = nx.from_numpy_array(similarity_matrix, create_using=nx.DiGraph)

        if plot_graph:
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, node_color='skyblue', with_labels=True, node_size=70, edge_color='blue', width=0.4)
            plt.title("Document Similarity Graph")
            plt.show()

        return {
            'number_of_nodes': G.number_of_nodes(),
            'number_of_edges': G.number_of_edges(),
            'edges_per_node': G.number_of_edges() / G.number_of_nodes()
        }
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
