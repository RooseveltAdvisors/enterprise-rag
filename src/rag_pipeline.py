import os
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import (
    EnsembleRetriever,
    ContextualCompressionRetriever,
    MultiQueryRetriever
)
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Any
from typing_extensions import TypedDict
from langchain.schema import Document
from langgraph.graph import END, StateGraph, START
"""
rag_pipeline.py - Core RAG Pipeline Implementation

This module implements an advanced Retrieval-Augmented Generation (RAG) pipeline
that combines vector search, quality control, and web search capabilities.

Key Features:
1. Multi-Stage Retrieval
   - Vector similarity search
   - Metadata-enhanced scoring
   - Document hierarchy awareness
   - Source diversification

2. Quality Control
   - Answer relevance verification
   - Hallucination detection
   - Source validation
   - Multi-attempt generation

3. Web Search Integration
   - Fallback for missing information
   - Real-time data augmentation
   - Source preservation

The pipeline is designed for high accuracy and reliability in enterprise
document processing applications.
"""

from rag_config import (
    RUNTIME_DIR, DATA_DIR, LLM_TYPE, EMBEDDING_TYPE,
    get_llm_config, validate_config,
    ENABLE_ANSWER_GRADER, ENABLE_HALLUCINATION_GRADER
)
from embedding_models import get_embeddings
import logging
from pprint import pprint
from langchain_community.tools.tavily_search import TavilySearchResults
import datetime
import traceback
from collections import deque
import functools

# Get root logger
logger = logging.getLogger()

# Pipeline node names
GENERATE_NODE = "generate"
WEB_SEARCH_NODE = "web_search"
RETRIEVE_NODE = "retrieve"
GRADE_CHUNKS_NODE = "grade_chunks"

# Global caches
embedding_cache = {}
vectorstore_cache = None

def get_cached_embedding(text):
    """Get embedding from cache or compute and cache it."""
    global embedding_cache
    if text not in embedding_cache:
        embeddings = get_embeddings()
        embedding_cache[text] = embeddings.embed_query(text)
    return embedding_cache[text]

def get_cached_vectorstore():
    """Get vectorstore from cache or load it."""
    global vectorstore_cache
    if vectorstore_cache is None:
        try:
            vectorstore_path = os.path.join(RUNTIME_DIR, "vectorstore")
            logger.info("\n🔄 Initializing knowledge base...")
            
            if not os.path.exists(vectorstore_path):
                raise FileNotFoundError(f"❌ Knowledge base not found at {vectorstore_path}")
            
            logger.info("📥 Loading document embeddings...")
            embeddings = get_embeddings()
            logger.info(f"✨ Using embedding model: {type(embeddings).__name__}")
            
            logger.info("\n🔍 Setting up search index...")
            vectorstore_cache = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
            logger.info("✅ Successfully loaded knowledge base")
        except Exception as e:
            logger.error(f"Error loading vectorstore: {str(e)}", exc_info=True)
            raise
    return vectorstore_cache

def get_llm(temperature=None):
    """Get the appropriate LLM based on configuration."""
    config = get_llm_config()
    if temperature is not None:
        config["temperature"] = temperature
        
    if LLM_TYPE.lower() == "azure":
        return AzureChatOpenAI(**config)
    else:
        return ChatOpenAI(**config)

class GradeDocuments(BaseModel):
    """Binary score for document relevance during retrieval phase."""
    binary_score: str = Field(
        description="Chunks are relevant to the question, 'yes' or 'no'"
    )

class GradeHallucinations(BaseModel):
    """Quality control grader that checks if the generated answer is supported by the retrieved chunks.
    Helps prevent the model from making up information or "hallucinating" facts not present in the sources."""
    binary_score: str = Field(
        description="Answer is grounded in the facts from retrieved chunks, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    """Quality control grader that verifies the generated answer actually addresses the user's question.
    Helps ensure responses are relevant and complete rather than off-topic or partial."""
    binary_score: str = Field(
        description="Answer directly addresses and resolves the user's question, 'yes' or 'no'"
    )

class GenerationAttempt(BaseModel):
    """Track generation attempts and status."""
    attempt_count: int = Field(default=0)
    last_generation: str = Field(default="")

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]
    sources: List[Dict[str, Any]]
    generation_attempts: GenerationAttempt 

# Initialize LLMs and tools
llm = get_llm(temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)
web_search_tool = TavilySearchResults(k=3)

# Prompts
system_grader = """You are a grader assessing relevance of a retrieved document to a user question.
Grade chunks liberally - if there's ANY potential relevance or overlapping content with the question, grade it as relevant.
Consider ALL of the following as relevant matches:
1. Exact text matches (even partial)
2. Number matches (including variations with/without commas, or approximate values)
3. Date matches (in any format)
4. Semantic similarity (related concepts or topics)
5. Contextual relevance (information that provides context)
6. Supporting information (facts that help understand the answer)
7. Section or topic relevance (content from relevant sections)
8. Related statistics or data points

When in doubt, prefer to include rather than exclude content.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

system_hallucination = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

system_answer = """You are a grader assessing whether an answer addresses / resolves a question.
Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

# RAG template
rag_template = """
You are a helpful assistant that answers questions based on the following context.
The context is organized by sections, with each section having a header and related content.
Always cite your sources using both the section name and filename.

Context: {context}
Question: {question}

Instructions:
1. Answer the question using information from the provided context
2. Pay attention to the section headers to understand the broader context of each piece of information
3. Cite sources in this format: [Section: section_name] (filename)
4. If multiple sources support a statement, cite all of them
5. If you're unsure about something, say so rather than making assumptions
6. Keep your answers concise and focused on the question
7. Organize your answer to follow the natural flow of sections when possible

Answer:
"""

# Create prompts
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system_grader),
    ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
])

hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", system_hallucination),
    ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
])

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", system_answer),
    ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
])

# Create chains
retrieval_grader = grade_prompt | structured_llm_grader
hallucination_grader = hallucination_prompt | structured_llm_grader
answer_grader = answer_prompt | structured_llm_grader

rag_prompt = ChatPromptTemplate.from_template(rag_template)
rag_chain = rag_prompt | llm

def format_docs(docs):
    """Format documents with their source information and section structure."""
    sections = {}
    for doc in docs:
        section = doc.metadata.get('section', 'Unknown Section')
        if section not in sections:
            sections[section] = {'headers': [], 'content': []}
        
        doc_type = doc.metadata.get('element_types')
        if doc_type == 'header':
            sections[section]['headers'].append(doc)
        else:
            sections[section]['content'].append(doc)
    
    formatted_sections = []
    for section, section_docs in sections.items():
        formatted_sections.append(f"\n=== {section} ===\n")
        
        for header in section_docs['headers']:
            content = header.page_content.strip()
            source = header.metadata.get('filename', 'Unknown source')
            formatted_sections.append(f"Header: {content}\nSource: {source}")
        
        for content_doc in section_docs['content']:
            content = content_doc.page_content.strip()
            source = content_doc.metadata.get('filename', 'Unknown source')
            formatted_sections.append(f"{content}\nSource: {source}")
    
    return "\n\n".join(formatted_sections)

def extract_sources(docs):
    """Extract source information from documents."""
    sources = []
    seen_hashes = set()
    for doc in docs:
        file_hash = doc.metadata.get('file_hash')
        if not file_hash or file_hash not in seen_hashes:
            source_info = {
                'filename': doc.metadata.get('filename', 'Unknown source'),
                'source': doc.metadata.get('source', 'Unknown path'),
                'section': doc.metadata.get('section', 'Unknown section'),
                'context': doc.page_content[:200] + "..."
            }
            sources.append(source_info)
            if file_hash:
                seen_hashes.add(file_hash)
    return sources

def format_answer(pipeline_output):
    """Format the final answer with sources and download links."""
    if not pipeline_output or "generate" not in pipeline_output:
        return "Error: No valid output from pipeline"
        
    output = pipeline_output["generate"]
    generation = output["generation"]
    sources = output.get("sources", [])
    generation_attempts = output.get("generation_attempts")
    
    # Format the answer
    formatted_answer = generation
    
    # Add warning if needed
    if generation_attempts and generation_attempts.attempt_count >= 3:
        formatted_answer = "⚠️ *This answer may need verification*\n\n" + formatted_answer
    
    # Add formatted sources section if available
    if sources:
        formatted_answer += "\n\nSOURCES:\n"
        used_sources = {}
        for source in sources:
            filename = source['filename']
            if filename not in used_sources:
                used_sources[filename] = source
                if filename == "web_search":
                    formatted_answer += f"\n🌐 **{source.get('title', 'Web Result')}**\n"
                    formatted_answer += f"[View Source]({source.get('url', '#')})\n"
                else:
                    formatted_answer += f"\n📄 **{filename}**\n"
                    if 'source' in source:
                        # Create download link for PDF files
                        source_path = source['source']
                        if source_path.lower().endswith('.pdf'):
                            formatted_answer += f"[Download PDF]({source_path}) | "
                    if 'context' in source:
                        formatted_answer += f"*{source['context'][:200]}...*\n"
    
    return formatted_answer

def batch_grade_chunks(chunks, question, batch_size=10):
    """Grade document chunks in batches to reduce API calls."""
    if not chunks:
        logger.info("\n=== Chunk Grading ===")
        logger.info("Status: No chunks to grade")
        logger.info("Action: Returning empty list")
        return []
    
    logger.info("\n=== Chunk Grading ===")
    logger.info(f"Total Chunks: {len(chunks)}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Query: {question}")
    
    # Log initial chunk distribution
    sections = {}
    for chunk in chunks:
        section = chunk.metadata.get('section', 'Unknown Section')
        if section not in sections:
            sections[section] = 0
        sections[section] += 1
    
    logger.info("\nInitial Section Distribution:")
    for section, count in sections.items():
        logger.info(f"- {section}: {count} chunks")
    
    # Combine all chunks into a single batch for one API call
    batch_context = "\n\n".join([
        f"Chunk {i+1}:\n{chunk.page_content}\n\nMetadata: {chunk.metadata.get('section', 'Unknown Section')}"
        for i, chunk in enumerate(chunks)
    ])
    
    # Create batch prompt
    batch_prompt = f"""
    Question: {question}

    Please evaluate the relevance of each text chunk to the question.
    Consider ALL of the following as relevant matches:
    1. Exact text matches (even partial)
    2. Number matches (including variations)
    3. Date matches (in any format)
    4. Semantic similarity
    5. Contextual relevance
    6. Supporting information
    7. Section relevance
    8. Related statistics

    Text chunks to evaluate:
    {batch_context}

    For each document, respond with ONLY 'yes' or 'no' on a new line, in order.
    Example:
    yes
    no
    yes
    """
    
    try:
        # Get batch evaluation in a single API call
        llm = get_llm(temperature=0)
        response = llm.invoke(batch_prompt)
        
        # Parse results
        results = response.content.strip().split('\n')
        relevant_count = len([r for r in results if 'yes' in r.lower()])
        
        logger.info("\n=== Grading Results ===")
        logger.info(f"Total Chunks Graded: {len(chunks)}")
        logger.info(f"Relevant Chunks: {relevant_count}")
        logger.info(f"Relevance Rate: {(relevant_count/len(chunks))*100:.1f}%")
        
        relevant_chunks = [chunk for chunk, result in zip(chunks, results) if 'yes' in result.lower()]
        
        # Log detailed section-wise results
        section_results = {}
        for chunk, result in zip(chunks, results):
            section = chunk.metadata.get('section', 'Unknown Section')
            if section not in section_results:
                section_results[section] = {'total': 0, 'relevant': 0}
            section_results[section]['total'] += 1
            if 'yes' in result.lower():
                section_results[section]['relevant'] += 1
        
        logger.info("\nSection-wise Results:")
        for section, counts in section_results.items():
            relevance_rate = (counts['relevant'] / counts['total']) * 100
            logger.info(f"- {section}:")
            logger.info(f"  • Total: {counts['total']}")
            logger.info(f"  • Relevant: {counts['relevant']}")
            logger.info(f"  • Relevance Rate: {relevance_rate:.1f}%")
        
        return relevant_chunks
    except Exception as e:
        logger.error(f"Error during chunk grading: {str(e)}")
        logger.info("\nStatus: Grading failed, returning all chunks as fallback")
        logger.info(f"Error: {str(e)}")
        return chunks

def web_search(state):
    """Perform web search and preserve source URLs."""
    logger.info("\n🌐 Searching the Web")
    logger.info("I'm looking for relevant information online...")
    question = state["question"]
    
    logger.info("\n=== Web Search Process ===")
    logger.info(f"Query: {question}")
    
    try:
        logger.info("Status: Initiating web search")
        docs = web_search_tool.invoke({"query": question})
        
        if not docs:
            logger.info("Status: No results found")
            logger.info("Reason: Web search API returned empty results")
            logger.info("Action: Will return no-results message")
            return {
                "documents": [],
                "question": question,
                "sources": [],
                "generation": "No relevant information found from web search."
            }
        
        # Preserve both content and URLs in the results
        web_results = []
        sources = []
        for doc in docs:
            try:
                title = doc.get('title', 'No title available')
                content = doc.get('content', '')
                if content:
                    web_results.append(f"{title}\n\n{content}")
                    sources.append({
                        "filename": "web_search",
                        "source": "web",
                        "url": doc.get("url", "No URL available"),
                        "title": title
                    })
            except Exception as e:
                logger.error(f"Error processing web search result: {str(e)}")
                continue
        
        if not web_results:
            return {
                "documents": [],
                "question": question,
                "sources": [],
                "generation": "Error processing web search results."
            }
        
        combined_content = "\n\n---\n\n".join(web_results)
        
        # Generate a summary using the web results
        summary_prompt = f"""
        Based on the following web search results, provide a comprehensive answer to the question: {question}

        Search Results:
        {combined_content}

        Instructions:
        1. Synthesize the information from all sources
        2. Provide specific details and numbers when available
        3. If there are conflicting information, mention the discrepancies
        4. If the information is time-sensitive, mention when the data is from
        """
        
        llm = get_llm(temperature=0)
        generation = llm.invoke(summary_prompt)
        
        return {
            "documents": [Document(page_content=combined_content, metadata={"filename": "web_search"})],
            "question": question,
            "sources": sources,
            "generation": generation.content if hasattr(generation, 'content') else str(generation)
        }
    except Exception as e:
        logger.error(f"Error in web search: {str(e)}")
        return {
            "documents": [],
            "question": question,
            "sources": [],
            "generation": f"Error performing web search: {str(e)}"
        }

def grade_chunks(state):
    """Grade document chunks with optimized batch processing."""
    logger.info("\n📚 Analyzing Document Relevance")
    logger.info("I'm checking which documents are most relevant to your question...")
    question = state["question"]
    chunks = state["documents"]
    
    # Skip grading if both graders are disabled
    if not ENABLE_ANSWER_GRADER and not ENABLE_HALLUCINATION_GRADER:
        logger.info("Chunk grading skipped: both graders are disabled")
        return {
            "documents": chunks,
            "question": question,
            "sources": extract_sources(chunks)
        }
    
    # Group chunks by section
    sections = {}
    for chunk in chunks:
        section = chunk.metadata.get('section', 'unknown')
        if section not in sections:
            sections[section] = []
        sections[section].append(chunk)
    
    # Process headers and content separately
    filtered_chunks = []
    
    # Always include headers
    for section_chunks in sections.values():
        headers = [d for d in section_chunks if d.metadata.get('element_types') == 'header']
        filtered_chunks.extend(headers)
        
        # Batch grade content chunks
        content_chunks = [d for d in section_chunks if d.metadata.get('element_types') == 'content']
        if content_chunks:
            relevant_chunks = batch_grade_chunks(content_chunks, question)
            filtered_chunks.extend(relevant_chunks)
    
    return {
        "documents": filtered_chunks,
        "question": question,
        "sources": extract_sources(filtered_chunks)
    }

def retrieve(state):
    """Retrieve relevant documents using advanced retrieval features."""
    logger.info("\n🔍 Finding Relevant Documents")
    logger.info("I'm searching through our knowledge base...")
    question = state["question"]
    logger.info(f"\nProcessing query: {question}")
    
    logger.info("\n=== Search Process Started ===")
    logger.info(f"Query: {question}")
    
    try:
        logger.info("Status: Loading vectorstore")
        vectorstore = get_cached_vectorstore()
        
        # Create base retriever with weighted search
        base_retriever = vectorstore.as_retriever(
            search_type="mmr",  # Use MMR for diversity
            search_kwargs={
                "k": 30,  # Increased for better coverage
                "fetch_k": 50,  # Fetch more for filtering
                "lambda_mult": 0.7,  # Balance between relevance and diversity
                "filter": None,  # We'll handle filtering in post-processing
            }
        )
        
        # Create multi-query retriever for query expansion
        from langchain.retrievers import MultiQueryRetriever
        
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=get_llm(temperature=0)
        )
        
        # Get initial documents
        logger.info("\n=== Multi-Query Retrieval ===")
        logger.info("Status: Generating query variations")
        documents = multi_query_retriever.invoke(question)
        logger.info(f"Status: Retrieved {len(documents)} initial documents")
        
        # Post-process documents based on metadata
        logger.info("\n=== Document Processing ===")
        logger.info("Status: Starting metadata-based scoring")
        processed_docs = []
        seen_content = set()
        
        for doc in documents:
            # Skip duplicate content
            content_hash = hash(doc.page_content)
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)
            
            # Apply metadata-based scoring
            score = 1.0
            metadata = doc.metadata
            
            # Boost headers and their importance
            if metadata.get('element_types') == 'header':
                score *= 1.5
                # Additional boost for headers with specific styles
                if metadata.get('style', {}).get('font_weight') in ['bold', '700', '800', '900']:
                    score *= 1.2
                if metadata.get('style', {}).get('font_size', 0) > 14:
                    score *= 1.1
            
            # Boost by hierarchy level (higher levels are more important)
            hierarchy_level = metadata.get('hierarchy_level', 0)
            score *= (1 + 0.2 * hierarchy_level)
            
            # Boost by importance score from ingestion
            importance_score = metadata.get('importance_score', 1.0)
            score *= importance_score
            
            # Boost based on relationships
            relationships = metadata.get('relationships', [])
            if relationships:
                # Handle both list and dictionary formats
                if isinstance(relationships, dict):
                    # Dictionary format
                    semantic_relations = relationships.get('semantic', [])
                    spatial_relations = relationships.get('spatial', [])
                    style_relations = relationships.get('style', [])
                else:
                    # List format
                    semantic_relations = [r for r in relationships if r.get('type', '').startswith('semantic')]
                    spatial_relations = [r for r in relationships if r.get('type', '').startswith('spatial')]
                    style_relations = [r for r in relationships if r.get('type', '').startswith('style')]

                # Boost content that's semantically related to the section
                if any(r.get('type') == 'belongs_to' for r in semantic_relations):
                    score *= 1.1
                
                # Boost content with spatial relationships
                if any(r.get('type') == 'aligned' for r in spatial_relations):
                    score *= 1.05
                
                # Boost emphasized content
                if any(r.get('type') == 'emphasis' for r in style_relations):
                    score *= 1.15
            
            # Store the computed score
            doc.metadata['retrieval_score'] = score
            processed_docs.append(doc)
        
        # Sort by score and take top results
        processed_docs.sort(key=lambda x: x.metadata.get('retrieval_score', 0), reverse=True)
        documents = processed_docs[:20]  # Keep top 20 after scoring
        
        logger.info("\n=== Scoring Results ===")
        logger.info(f"Initial documents: {len(processed_docs)}")
        logger.info(f"After deduplication: {len(seen_content)}")
        logger.info(f"Final selected: {len(documents)}")
        
        # Log score distribution
        scores = [doc.metadata.get('retrieval_score', 0) for doc in documents]
        if scores:
            logger.info(f"Score range: {min(scores):.2f} - {max(scores):.2f}")
            avg_score = sum(scores) / len(scores)
            logger.info(f"Average score: {avg_score:.2f}")
        
        # Log detailed document analysis
        sections = {}
        element_types = {}
        
        for doc in documents:
            # Track sections
            section = doc.metadata.get('section', 'Unknown Section')
            if section not in sections:
                sections[section] = 0
            sections[section] += 1
            
            # Track element types dynamically
            doc_type = doc.metadata.get('element_types', 'unknown')
            if doc_type not in element_types:
                element_types[doc_type] = 0
            element_types[doc_type] += 1
        
        logger.info("\n=== Document Analysis ===")
        logger.info("1. Section Distribution:")
        for section, count in sections.items():
            logger.info(f"   - {section}: {count} chunks")
        
        logger.info("\n2. Element Types:")
        for etype, count in element_types.items():
            logger.info(f"   - {etype}: {count} chunks")
        
        # Extract sources
        sources = extract_sources(documents)
        logger.info(f"Total unique sources: {len(sources)}")
        
        return {"documents": documents, "question": question, "sources": sources}
        
    except Exception as e:
        error_msg = f"Error during retrieval: {str(e)}"
        print(f"\n{error_msg}")
        logger.error(error_msg, exc_info=True)
        return {"documents": [], "question": question, "sources": []}

def generate(state):
    """Generate answer with optimized processing."""
    logger.info("\n✍️ Crafting Your Answer")
    logger.info("I'm putting together a comprehensive response based on the information I found...")
    question = state["question"]
    documents = state["documents"]
    
    # If we have a pre-generated answer (from web search), use it
    if "generation" in state:
        return state
    
    # Initialize or get generation attempts
    generation_attempts = state.get("generation_attempts", GenerationAttempt())
    generation_attempts.attempt_count += 1
    
    try:
        # Handle case with no relevant documents
        if not documents:
            logger.info("Generating response with no relevant chunks")
            no_docs_prompt = f"""
            Question: {question}
            
            Important: No relevant chunks were found in our knowledge base for this question.
            
            Please generate a response that:
            1. Clearly states that we don't have this specific information in our knowledge base
            2. Explains what information would typically be needed to answer this question
            3. Suggests specific authoritative sources where the user can find this information
            4. If applicable, mentions any related general information we can provide
            5. Maintains a helpful and professional tone
            
            Format the response with clear sections and bullet points for readability.
            """
            
            llm = get_llm(temperature=0)
            generation = llm.invoke(no_docs_prompt)
            
            # Ensure we have a valid generation
            if not generation or (hasattr(generation, 'content') and not generation.content.strip()):
                generation = "I apologize, but I don't have specific information about this in our knowledge base. I recommend checking official sources or contacting relevant experts for accurate, up-to-date information."
            
            generation_text = generation.content if hasattr(generation, 'content') else str(generation)
        else:
            # Normal generation with documents
            formatted_context = format_docs(documents)
            logger.info("Formatted context length: %d", len(formatted_context))
            
            try:
                # Get raw generation
                generation = rag_chain.invoke({
                    "context": formatted_context,
                    "question": question
                })
                logger.debug("Raw generation result type: %s", type(generation))
                logger.debug("Raw generation result: %s", generation)
                
                # Extract content from generation result
                if isinstance(generation, AIMessage):
                    generation_text = generation.content
                elif isinstance(generation, BaseMessage):
                    generation_text = generation.content
                elif isinstance(generation, str):
                    generation_text = generation
                else:
                    generation_text = str(generation)
                
                logger.debug("Raw generation result: %s", generation)
                
                # Validate generation
                if not generation_text or not generation_text.strip():
                    raise ValueError("Empty generation result")
                
            except Exception as e:
                logger.error(f"Generation error: {str(e)}")
                error_message = str(e)
                if isinstance(error_message, dict):
                    error_message = str(error_message)
                generation_text = f"""I apologize, but I encountered an error while generating a response. 
                This might be due to processing issues with the retrieved documents. 
                Here's what happened: {error_message}"""
        
        # Clean up generation text
        generation_text = generation_text.strip()
        
        if not generation_text:
            logger.warning("Empty generation result")
            generation_text = "I apologize, but I was unable to generate a response. This might be due to processing issues with the retrieved documents."
        
        generation_attempts.last_generation = generation_text
        
        return {
            "documents": documents,
            "question": question,
            "generation": generation_text,
            "sources": state.get("sources", []),
            "generation_attempts": generation_attempts
        }
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        return {
            "documents": documents,
            "question": question,
            "generation": "I apologize, but an error occurred while generating the response.",
            "sources": state.get("sources", []),
            "generation_attempts": generation_attempts
        }

def grade_generation(state):
    """Grade generation with optimized processing."""
    logger.info("\n✅ Quality Check")
    logger.info("I'm verifying the accuracy and completeness of the answer...")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    generation_attempts = state.get("generation_attempts")
    
    logger.info("\n=== Generation Grading ===")
    logger.info(f"Query: {question}")
    logger.info(f"Generation Attempt: {generation_attempts.attempt_count if generation_attempts else 1}")
    logger.info(f"Document Count: {len(documents)}")
    logger.info(f"Answer Grader: {'Enabled' if ENABLE_ANSWER_GRADER else 'Disabled'}")
    logger.info(f"Hallucination Grader: {'Enabled' if ENABLE_HALLUCINATION_GRADER else 'Disabled'}")
    
    # If grading is disabled, accept the generation after first attempt
    if not ENABLE_ANSWER_GRADER and not ENABLE_HALLUCINATION_GRADER:
        logger.info("\nStatus: All graders disabled")
        logger.info("Action: Accepting generation without validation")
        return "end"
    
    # If there are no documents, only check if the generation addresses the question
    if not documents:
        logger.info("\nStatus: No documents available")
        if ENABLE_ANSWER_GRADER:
            logger.info("Action: Checking if generation addresses question")
            prompt = f"""
            Question: {question}
            Generation: {generation}
            
            Does this generation provide a clear response that addresses the question, even if it cannot provide specific details? (yes/no)
            Answer with ONLY 'yes' or 'no'.
            """
            
            llm = get_llm(temperature=0)
            response = llm.invoke(prompt)
            addresses_question = 'yes' in response.content.lower()
            
            if addresses_question or (generation_attempts and generation_attempts.attempt_count >= 2):
                logger.info("\nStatus: Generation accepted")
                logger.info(f"Reason: {'Question addressed' if addresses_question else 'Max attempts reached'}")
                return "end"
        else:
            logger.info("Status: Answer grader disabled")
            logger.info("Action: Accepting generation without validation")
            return "end"
    else:
        # Determine which aspects to check based on configuration
        aspects_to_check = []
        if ENABLE_HALLUCINATION_GRADER:
            aspects_to_check.append("Is the generation grounded in the facts?")
        if ENABLE_ANSWER_GRADER:
            aspects_to_check.append("Does the generation address the question?")
        
        if not aspects_to_check:
            logger.info("All grading disabled: accepting generation with chunks")
            return "end"
        
        # Normal grading with documents
        batch_prompt = f"""
        Question: {question}
        
        Generation to evaluate:
        {generation}
        
        Context:
        {format_docs(documents)}
        
        Evaluate the following aspects and respond with ONLY {'two lines' if len(aspects_to_check) == 2 else 'one line'}:
        {chr(10).join(f"{i+1}. {aspect}" for i, aspect in enumerate(aspects_to_check))}
        """
        
        llm = get_llm(temperature=0)
        response = llm.invoke(batch_prompt)
        results = response.content.strip().split('\n')
        
        # Process results based on enabled graders
        checks_passed = True
        logger.info("\n=== Grading Results ===")
        
        if ENABLE_HALLUCINATION_GRADER and results:
            is_grounded = 'yes' in results[0].lower()
            checks_passed = checks_passed and is_grounded
            logger.info(f"Fact Check: {'Passed' if is_grounded else 'Failed'}")
        
        if ENABLE_ANSWER_GRADER and len(results) > (1 if ENABLE_HALLUCINATION_GRADER else 0):
            addresses_question = 'yes' in results[-1].lower()
            checks_passed = checks_passed and addresses_question
            logger.info(f"Answer Check: {'Passed' if addresses_question else 'Failed'}")
        
        if checks_passed:
            logger.info("\nStatus: Generation accepted")
            logger.info("Reason: All enabled checks passed")
            return "end"
        
        if generation_attempts and generation_attempts.attempt_count >= 3:
            logger.info("\nStatus: Generation accepted")
            logger.info("Reason: Maximum attempts reached")
            logger.info("Note: Answer may need verification")
            return "end"
    
    logger.info("\nStatus: Generation rejected")
    logger.info("Action: Attempting another generation")
    logger.info(f"Attempt {generation_attempts.attempt_count if generation_attempts else 1} of 3")
    return "generate"

def decide_next_step(state):
    """Decide whether to generate an answer or use web search."""
    filtered_chunks = state["documents"]
    question = state["question"].lower()
    
    # Log detailed search path decision
    chunk_count = len(filtered_chunks)
    logger.info("\n=== Search Path Decision ===")
    logger.info(f"Query: {question}")
    logger.info(f"Retrieved Chunks: {chunk_count}")
    
    if not filtered_chunks:
        logger.info("Status: No relevant chunks found in knowledge base")
        logger.info("Action: Falling back to web search")
        logger.info("Reason: Vector search returned 0 chunks that passed relevance criteria")
        logger.info("Next Step: Will query external search API")
        return WEB_SEARCH_NODE
    
    logger.info("Status: Found relevant chunks in knowledge base")
    logger.info(f"Action: Proceeding with local knowledge base ({chunk_count} chunks)")
    logger.info("Reason: Vector search returned sufficient relevant chunks")
    logger.info("Next Step: Will generate answer from local content")
    return GENERATE_NODE

def process_query(query, enable_answer_grader=None, enable_hallucination_grader=None):
    """Process a query through the RAG pipeline."""
    try:
        # Override grading flags if provided
        global ENABLE_ANSWER_GRADER, ENABLE_HALLUCINATION_GRADER
        if enable_answer_grader is not None:
            ENABLE_ANSWER_GRADER = enable_answer_grader
        if enable_hallucination_grader is not None:
            ENABLE_HALLUCINATION_GRADER = enable_hallucination_grader

        # Check for web search request
        if "search online" in query.lower() or "web search" in query.lower():
            logger.info("\n🌐 Searching the web as requested...")
            logger.info("I'll look for the most up-to-date information online.")
            docs = web_search_tool.invoke({"query": query})
            web_results = []
            sources = []
            
            for doc in docs:
                web_results.append(doc["content"])
                sources.append({
                    "filename": "web_search",
                    "source": "web",
                    "url": doc.get("url", "No URL available"),
                    "title": doc.get("title", "No title available")
                })
            
            # Generate summary from web results
            combined_content = "\n\n".join(web_results)
            summary_prompt = f"""
            Based on the following web search results, provide a comprehensive answer to: {query}

            Search Results:
            {combined_content}

            Instructions:
            1. Synthesize the information from all sources
            2. Provide specific details and numbers when available
            3. If there are conflicting information, mention the discrepancies
            4. If the information is time-sensitive, mention when the data is from
            """
            
            llm = get_llm(temperature=0)
            generation = llm.invoke(summary_prompt)
            generation_text = generation.content if hasattr(generation, 'content') else str(generation)
            
            return {
                "generate": {
                    "documents": [Document(page_content=combined_content, metadata={"filename": "web_search"})],
                    "question": query,
                    "generation": generation_text,
                    "sources": sources,
                    "generation_attempts": GenerationAttempt(attempt_count=1, last_generation=generation_text)
                }
            }
        
        # Default RAG pipeline
        inputs = {
            "question": query,
            "generation_attempts": GenerationAttempt()
        }
        
        outputs = []
        for output in app.stream(inputs):
            outputs.append(output)
        
        if outputs:
            final_output = outputs[-1]
            if "generate" in final_output:
                # Extract generation from final output
                generation = final_output["generate"]
                if isinstance(generation, dict):
                    generation_text = generation.get("generation", "")
                elif isinstance(generation, (str, AIMessage, BaseMessage)):
                    generation_text = generation.content if hasattr(generation, 'content') else str(generation)
                else:
                    generation_text = str(generation)
                
                return {
                    "generate": {
                        "documents": final_output.get("documents", []),
                        "question": query,
                        "generation": generation_text,
                        "sources": final_output.get("sources", []),
                        "generation_attempts": final_output.get("generation_attempts", GenerationAttempt())
                    }
                }
            
        # Only return None if we truly have no output
        if not outputs:
            return None
            
        # If we have outputs but no valid generation, return error in expected format
        return {
            "generate": {
                "documents": [],
                "question": query,
                "generation": "Error: Unable to generate a valid response. Please try rephrasing your question.",
                "sources": [],
                "generation_attempts": GenerationAttempt(attempt_count=1, last_generation="Error: Unable to generate a valid response.")
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise

def create_workflow():
    """Create the workflow graph."""
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node(WEB_SEARCH_NODE, web_search)
    workflow.add_node(RETRIEVE_NODE, retrieve)
    workflow.add_node(GRADE_CHUNKS_NODE, grade_chunks)
    workflow.add_node(GENERATE_NODE, generate)
    
    # Add edges
    workflow.add_edge(START, RETRIEVE_NODE)
    workflow.add_edge(RETRIEVE_NODE, GRADE_CHUNKS_NODE)
    workflow.add_conditional_edges(
        GRADE_CHUNKS_NODE,
        decide_next_step,
        {
            GENERATE_NODE: GENERATE_NODE,
            WEB_SEARCH_NODE: WEB_SEARCH_NODE
        }
    )
    workflow.add_edge(WEB_SEARCH_NODE, GENERATE_NODE)
    workflow.add_conditional_edges(
        GENERATE_NODE,
        grade_generation,
        {
            GENERATE_NODE: GENERATE_NODE,
            "end": END
        }
    )
    
    return workflow.compile()

def print_settings():
    """Print important RAG pipeline settings."""
    logger.info("\n🛠️ System Settings")
    logger.info("Here's how I'm configured to help you:")
    logger.info(f"LLM Type: {LLM_TYPE}")
    logger.info(f"Embedding Type: {EMBEDDING_TYPE}")
    logger.info("\nQuality Control Settings:")
    logger.info(f"- Answer Relevance Check: {'Enabled' if ENABLE_ANSWER_GRADER else 'Disabled'}")
    logger.info("  (Ensures responses address the question)")
    logger.info(f"- Source Fact Check: {'Enabled' if ENABLE_HALLUCINATION_GRADER else 'Disabled'}")
    logger.info("  (Verifies responses are supported by documents)")
    logger.info("\nRetrieval Settings:")
    logger.info("- Vector Store: FAISS")
    logger.info("- Initial Fetch: k=30, fetch_k=50")
    logger.info("- MMR Search: lambda=0.7 (balance relevance/diversity)")
    logger.info("- Query Expansion: Multiple query variations")
    logger.info("- Metadata Scoring:")
    logger.info("  • Headers:")
    logger.info("    - Base: 1.5x boost")
    logger.info("    - Bold: Additional 1.2x")
    logger.info("    - Large Font: Additional 1.1x")
    logger.info("  • Hierarchy: Progressive boost by level")
    logger.info("  • Content:")
    logger.info("    - Base: Weighted by importance score")
    logger.info("    - Section Relation: 1.1x boost")
    logger.info("    - Header Alignment: 1.05x boost")
    logger.info("    - Emphasis: 1.15x boost")
    logger.info("- Final Results: Top 20 after scoring")
    logger.info("\nWeb Search:")
    logger.info("- Provider: Tavily")
    logger.info("- Results per search: 3")
    logger.info("\nType 'quit' to exit.")
    logger.info("=" * 25 + "\n")

def interactive_mode():
    """Run the RAG pipeline in interactive mode."""
    logger.info("\n👋 Welcome! I'm here to help answer your questions!")
    logger.info("I can search through our knowledge base and the web to find the most relevant information.")
    print_settings()
    
    while True:
        try:
            query = input("\nEnter your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
                
            pipeline_output = process_query(query)
            if pipeline_output:
                answer = format_answer(pipeline_output)
                logger.info("\n" + answer + "\n")
            else:
                error_msg = """
❌ I wasn't able to generate an answer for your question.

This could be because:
• I couldn't find relevant information in our knowledge base
• The question might need to be more specific
• There might have been a technical issue processing your request

💡 Suggestions:
• Try rephrasing your question
• Break down complex questions into simpler parts
• Provide more context or details if possible

I'm here to help, so please feel free to try again with a different question!
"""
                logger.info(error_msg)
            
        except Exception as e:
            logger.error(f"\nError: {str(e)}")
            continue

# Initialize tools and workflow
llm = get_llm(temperature=0)
web_search_tool = TavilySearchResults(k=3)
app = create_workflow()

if __name__ == "__main__":
    try:
        # Validate configuration
        validate_config()
        
        # Run in interactive mode
        interactive_mode()
        
    except Exception as e:
        logger.error("Error in main process:", exc_info=True)
        logger.error(f"Error: {str(e)}")
        logger.error("\nFull stack trace:")
        logger.error(traceback.format_exc())
        raise
