import chromadb
import numpy as np
import uuid
import os

class FaceDB:
    def __init__(self, db_path=None, collection_name="faces"):
        if db_path is None:
            # Default to a folder in the project root
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(base_dir, "chroma_db_store")
            
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"} # Use cosine similarity
        )

    def add_person(self, person_id, embedding):
        """
        Add a person's face embedding to the database.
        
        Args:
            person_id (str): Unique identifier for the person.
            embedding (list or np.array): Embedding vector.
        """
        if isinstance(embedding, np.ndarray):
            embedding = embedding.flatten().tolist()
            
        record_id = f"{person_id}_{uuid.uuid4()}"
        
        print(f"Adding {person_id} to ChromaDB with embedding dimension: {len(embedding)}")
        self.collection.add(
            embeddings=[embedding],
            metadatas=[{"person_id": person_id}],
            ids=[record_id]
        )
        print(f"✅ Successfully added {person_id} to database with ID {record_id}")
        
        # Log current database size
        total_count = self.collection.count()
        print(f"📊 Total enrolled persons in database: {total_count}")

    def search_person(self, embedding, threshold=0.590):
        """
        Search for a matching person in the database.
        
        Args:
            embedding (list or np.array): Embedding vector to query.
            threshold (float): Distance threshold for a match.
            
        Returns:
            str: person_id if found, else None.
        """
        if isinstance(embedding, np.ndarray):
            embedding = embedding.flatten().tolist()
            
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=1
        )
        
        if not results['ids'] or not results['ids'][0]:
            print("🔍 No matches found in database")
            return None
            
        distance = results['distances'][0][0]
        person_id = results['metadatas'][0][0]['person_id']
        
        print(f"🔍 Closest match: {person_id} (distance: {distance:.3f}, threshold: {threshold})")
        
        # With cosine distance: 0 is identical, 2 is opposite.
        # ArcFace thresholds usually around 0.3-0.5 for cosine distance.
        if distance < threshold:
            print(f"✅ Match found: {person_id}")
            return person_id
        else:
            print(f"❌ No match - distance {distance:.3f} exceeds threshold {threshold}")
            
        return None
