import re
from typing import List, Dict
from openai import OpenAI
from modules.embeddings import embed_text, embed_batch
from modules.vector_store import InMemoryVectorStore, Chunk
import config


class RAGPipeline:
    """Complete RAG system in one file."""
    
    def __init__(self, system_prompt: str = "You are a helpful assistant."):
        self.store = InMemoryVectorStore()
        self.llm = OpenAI(api_key=config.XAI_API_KEY, base_url="https://api.x.ai/v1")
        self.system_prompt = system_prompt
        self.is_built = False
    
    def build_from_markdown(self, filepath: str):
        """Load a markdown file, chop it up, and make it searchable."""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        chunks = self._chunk_by_headers(content, filepath)
        texts = [c.content for c in chunks]
        embeddings = embed_batch(texts)
        
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
        
        self.store.add_chunks(chunks)
        self.is_built = True
        return len(chunks)
    
    def query(self, question: str, top_k: int = 5) -> Dict:
        """Ask a question and get an answer from the documents."""
        q_embedding = embed_text(question)
        results = self.store.search(q_embedding, top_k)
        
        context = "\n\n---\n\n".join([
            f"[Source: {r['metadata'].get('section', 'Unknown')}]\n{r['content']}"
            for r in results
        ])
        
        response = self.llm.chat.completions.create(
            model=config.CHAT_MODEL,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer based only on the context above. Cite sources using [Source: section name]."}
            ],
            temperature=0.3
        )
        
        return {
            'answer': response.choices[0].message.content,
            'sources': [r['metadata'].get('section', '') for r in results],
            'similarities': [r['similarity'] for r in results]
        }
    
    def _chunk_by_headers(self, content: str, source: str) -> List[Chunk]:
        """Split markdown by headers."""
        lines = content.split('\n')
        chunks = []
        current_headers = {}
        current_content = []
        
        for line in lines:
            header_match = re.match(r'^(#{1,4})\s+(.+)$', line)
            if header_match:
                if current_content:
                    header_chain = " > ".join(
                        current_headers[l] for l in sorted(current_headers.keys())
                    )
                    text = '\n'.join(current_content).strip()
                    if text:
                        chunks.append(Chunk(
                            content=f"[{header_chain}]\n\n{text}",
                            metadata={'section': header_chain, 'source': source}
                        ))
                    current_content = []
                
                level = len(header_match.group(1))
                current_headers[level] = header_match.group(2).strip()
                for l in list(current_headers.keys()):
                    if l > level:
                        del current_headers[l]
            else:
                if line.strip():
                    current_content.append(line)
        
        if current_content:
            header_chain = " > ".join(
                current_headers[l] for l in sorted(current_headers.keys())
            )
            text = '\n'.join(current_content).strip()
            if text:
                chunks.append(Chunk(
                    content=f"[{header_chain}]\n\n{text}",
                    metadata={'section': header_chain, 'source': source}
                ))
        
        return chunks