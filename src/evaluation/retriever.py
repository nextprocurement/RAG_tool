from sentence_transformers import SentenceTransformer, util
import time

class Index(object):
    def __init__(self, corpus: list, doc_ids: list, model_name: str = 'SeyedAli/Multilingual-Text-Semantic-Search-Siamese-BERT-V1'):
        self.model = SentenceTransformer(model_name)
        self.corpus = corpus
        self.doc_ids = doc_ids
        
        # Preprocess the corpus: split each document into sentences
        self.sentences = []
        self.mapping = []
        for doc_id, document in enumerate(corpus):
            
            # Split the document into sentences based on "."
            substrings = document.split(".")
            substrings = [s.strip() for s in substrings if s.strip()]  #
            
            self.sentences.extend(substrings)
            self.mapping.extend([doc_id] * len(substrings))
        
        print("-- -- Encoding corpus...")
        start_time = time.time()
        self.index = self.model.encode(self.sentences)
        print("-- -- Corpus encoded in {} minutes".format((time.time()-start_time)/60))
    
    def retrieve(self, query, topk=5):
        # Encode query
        query_emb = self.model.encode(query)
        
        # Compute dot score between query and all sentence embeddings
        scores = util.dot_score(query_emb, self.index)[0].cpu().tolist()
        
        # Combine sentences & scores
        sentence_score_pairs = list(zip(self.sentences, scores, self.mapping))
        
        # Sort by decreasing score
        sentence_score_pairs = sorted(sentence_score_pairs, key=lambda x: x[1], reverse=True)[:topk]
        
        # Retrieve the topk results with original document mapping
        results = []
        for sentence, score, doc_id in sentence_score_pairs:
            results.append({
                "document_id": self.doc_ids[doc_id],
                "sentence": sentence,
                "score": score,
                "original_document": self.corpus[doc_id]
            })
        
        return results