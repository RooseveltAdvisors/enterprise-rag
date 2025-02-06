"""
document_processor_gpu.py - GPU-Accelerated Document Processing System

This module implements a sophisticated document processing pipeline that leverages GPU
acceleration for efficient processing of large document collections. It provides:

Key Features:
1. Multi-GPU Support
   - Parallel document processing across multiple GPUs
   - Automatic GPU selection based on utilization
   - Dynamic memory management and batch sizing

2. Advanced Document Analysis
   - Layout and style extraction
   - Document hierarchy preservation
   - Relationship detection between elements
   - Visual emphasis recognition

3. Reliable Processing
   - Automatic checkpointing
   - Error recovery mechanisms
   - Progress tracking and statistics
   - Resource cleanup

4. Vector Store Integration
   - Efficient document embedding
   - Incremental updates
   - Automatic index management

The system is designed for enterprise-scale document processing with a focus on
performance, reliability, and document structure preservation.
"""

import os
import signal
import logging
import json
import datetime
import argparse
import traceback
import time
import multiprocessing as mp
import threading
from hashlib import md5
from typing import List, Dict, Any, Optional, Tuple, Union
from contextlib import contextmanager

import torch
import pynvml
from tqdm import tqdm
from unstructured.partition.pdf import partition_pdf
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from rag_config import RUNTIME_DIR, DATA_DIR, BATCH_SIZE
from embedding_models import get_embeddings

# Processing Constants
DEFAULT_TIMEOUT_SECONDS = 900  # 15 minutes
DEFAULT_SAVE_INTERVAL = 300    # 5 minutes
DEFAULT_CHECKPOINT_INTERVAL = 300  # 5 minutes
DEFAULT_GPU_MEMORY_LIMIT = 2 * 1024 * 1024 * 1024  # 2GB
DEFAULT_BATCH_SIZE = 1
MAX_BATCH_SIZE = 8
GPU_MEMORY_UTILIZATION = 0.8  # Use 80% of available memory

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('document_processing.log'),
        logging.StreamHandler()
    ]
)

class TimeoutException(Exception):
    """
    Exception raised when a document processing operation exceeds its time limit.
    
    This exception is used in conjunction with the timeout context manager to ensure
    that individual document processing tasks don't hang indefinitely, which is
    particularly important in a production environment.
    """
    pass

@contextmanager
def timeout(seconds: int):
    """
    Context manager for implementing timeouts on document processing operations.
    
    Args:
        seconds (int): Maximum number of seconds to allow the operation to run
        
    Raises:
        TimeoutException: If the operation exceeds the specified time limit
        
    This context manager uses SIGALRM for timeout implementation, making it
    Unix-specific. For Windows compatibility, a different timeout mechanism
    would need to be implemented.
    """
    def timeout_handler(signum, frame):
        raise TimeoutException(f"Processing timed out after {seconds} seconds")

    original_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)

class GPUManager:
    """
    Manages GPU-related operations and monitoring for the document processing pipeline.
    
    This class provides functionality for:
    - GPU initialization and status monitoring
    - Memory usage tracking and optimization
    - Automatic GPU selection based on utilization
    - Real-time GPU statistics and reporting
    
    The manager uses NVIDIA Management Library (NVML) for direct GPU interaction
    and provides fallback mechanisms for systems without GPU support.
    """
    
    @staticmethod
    def init_nvml() -> bool:
        """Initialize NVIDIA Management Library."""
        try:
            pynvml.nvmlInit()
            return True
        except Exception as e:
            logging.warning(f"Could not initialize NVML for GPU monitoring: {e}")
            return False

    @staticmethod
    def get_gpu_info() -> List[Dict[str, Any]]:
        """Get detailed information about available GPUs."""
        gpu_info = []
        try:
            pynvml.nvmlInit()
            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                device_name = pynvml.nvmlDeviceGetName(handle)
                
                if isinstance(device_name, bytes):
                    device_name = device_name.decode('utf-8')
                
                gpu_info.append({
                    'index': i,
                    'name': str(device_name),
                    'utilization': info.gpu,
                    'memory_total': memory_info.total / 1024**2,
                    'memory_used': memory_info.used / 1024**2,
                    'memory_free': memory_info.free / 1024**2
                })
                logging.info(f"Found GPU {i}: {device_name}")
        except Exception as e:
            logging.warning(f"Could not get GPU information: {str(e)}")
        return gpu_info

    @staticmethod
    def select_best_gpus(num_gpus: Optional[int] = None) -> str:
        """Select the least utilized GPUs."""
        gpu_info = GPUManager.get_gpu_info()
        if not gpu_info:
            return ""
        
        sorted_gpus = sorted(
            gpu_info,
            key=lambda x: (x['utilization'], x['memory_used'])
        )
        
        num_to_select = num_gpus if num_gpus is not None else len(sorted_gpus)
        selected_gpus = sorted_gpus[:num_to_select]
        
        return ",".join(str(gpu['index']) for gpu in selected_gpus)

    @staticmethod
    def print_status():
        """Print current status of all GPUs."""
        try:
            gpu_info = GPUManager.get_gpu_info()
            if not gpu_info:
                print("No GPU information available")
                return
                
            print("\nGPU Status:")
            print("-" * 80)
            print(f"{'GPU ID':<8} {'Name':<20} {'Utilization':<12} {'Memory Used':<12} {'Memory Total':<12}")
            print("-" * 80)
            
            for gpu in gpu_info:
                print(f"{gpu['index']:<8} "
                      f"{gpu['name'][:18]:<20} "
                      f"{gpu['utilization']:>3}%{' '*8} "
                      f"{gpu['memory_used']/1024:>5.1f} GB{' '*4} "
                      f"{gpu['memory_total']/1024:>5.1f} GB")
            print("-" * 80)
        except Exception as e:
            print(f"Error getting GPU status: {str(e)}")
            logging.error(f"Error in print_gpu_status: {str(e)}", exc_info=True)

class DocumentProcessor:
    """
    Core document processing engine with GPU acceleration and vectorstore management.
    
    This class implements a comprehensive document processing pipeline that:
    1. Processes documents in parallel using available GPUs
    2. Maintains document structure and relationships
    3. Manages a vector store for processed documents
    4. Provides checkpointing and recovery mechanisms
    
    Key Features:
    - Multi-GPU support with automatic workload distribution
    - Smart batching based on available GPU memory
    - Incremental processing with checkpoint/resume
    - Comprehensive error handling and recovery
    - Real-time progress tracking and statistics
    
    The processor is designed for production use with features like:
    - Automatic GPU selection and optimization
    - Memory management and cleanup
    - Progress saving and restoration
    - Detailed logging and monitoring
    """
    
    def __init__(
        self,
        runtime_dir: str = RUNTIME_DIR,
        data_dir: str = DATA_DIR,
        batch_size: int = BATCH_SIZE,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        save_interval: int = DEFAULT_SAVE_INTERVAL,
        gpu_devices: Optional[str] = None
    ):
        """Initialize the DocumentProcessor with specified settings."""
        self._init_directories(runtime_dir, data_dir)
        self._init_settings(batch_size, timeout_seconds, save_interval)
        self._init_gpu_configuration(gpu_devices)
        self._init_state()

    def _init_directories(self, runtime_dir: str, data_dir: str) -> None:
        """Initialize directory paths."""
        self.runtime_dir = runtime_dir
        self.data_dir = data_dir
        self.vectorstore_path = os.path.join(runtime_dir, "vectorstore")
        self.processed_files_path = os.path.join(runtime_dir, "processed_files.json")
        self.checkpoint_path = os.path.join(runtime_dir, "documents_checkpoint.pkl")

    def _init_settings(self, batch_size: int, timeout_seconds: int, save_interval: int) -> None:
        """Initialize processing settings."""
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        self.save_interval = save_interval
        self.checkpoint_interval = DEFAULT_CHECKPOINT_INTERVAL

    def _init_gpu_configuration(self, gpu_devices: Optional[str]) -> None:
        """Initialize GPU configuration."""
        if gpu_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
            logging.info(f"Set CUDA_VISIBLE_DEVICES to {gpu_devices}")
        
        self.num_gpus = torch.cuda.device_count()
        self.gpu_memory_limit, self.detection_batch_size = self._calculate_gpu_settings()
        
        if self.num_gpus > 0:
            logging.info(f"Using {self.num_gpus} GPUs with batch size {self.detection_batch_size}")
            for i in range(self.num_gpus):
                gpu_props = torch.cuda.get_device_properties(i)
                logging.info(f"GPU {i}: {gpu_props.name}, Memory: {gpu_props.total_memory / 1024**3:.1f} GB")
        else:
            logging.warning("No GPUs found, using CPU only")

    def _init_state(self) -> None:
        """Initialize state variables."""
        self.last_save_time = time.time()
        self.last_checkpoint_time = time.time()
        self.current_files = {}
        self.documents_buffer = []
        self.embeddings = get_embeddings()
        self.nvml_initialized = GPUManager.init_nvml()

    def _calculate_gpu_settings(self, gpu_index: int = 0) -> Tuple[int, int]:
        """Calculate GPU memory limit and batch size."""
        try:
            if not torch.cuda.is_available():
                return DEFAULT_GPU_MEMORY_LIMIT, DEFAULT_BATCH_SIZE
                
            gpu_mem = torch.cuda.get_device_properties(gpu_index).total_memory
            memory_limit = int(gpu_mem * GPU_MEMORY_UTILIZATION)
            batch_size = min(MAX_BATCH_SIZE, max(1, int(gpu_mem / (1024**3))))
            
            return memory_limit, batch_size
            
        except Exception as e:
            logging.warning(f"Error calculating GPU settings: {e}")
            return DEFAULT_GPU_MEMORY_LIMIT, DEFAULT_BATCH_SIZE

    @staticmethod
    def process_single_pdf(args: Tuple[str, str, int, int]) -> Tuple[List[Document], Dict]:
        """Process a single PDF file with GPU acceleration."""
        file_path, filename, timeout_seconds, gpu_id = args
        processor = PDFProcessor(gpu_id)
        return processor.process_file(file_path, filename, timeout_seconds)

    def process_files_in_batches(
        self,
        files: List[str],
        vectorstore: Optional[FAISS] = None
    ) -> Tuple[List[Document], Optional[FAISS]]:
        """Process multiple files in batches using available GPUs."""
        if not files:
            return [], None

        documents = []
        num_gpus = max(1, torch.cuda.device_count())
        num_processes = max(1, min(num_gpus, mp.cpu_count() // 2))

        processing_args = [
            (os.path.join(self.data_dir, filename), filename, self.timeout_seconds, i % num_processes)
            for i, filename in enumerate(files)
        ]

        try:
            with mp.Pool(num_processes) as pool:
                results = []
                for result in tqdm(
                    pool.imap_unordered(self.process_single_pdf, processing_args),
                    total=len(processing_args),
                    desc="Processing PDFs"
                ):
                    self._handle_processing_result(result, documents, vectorstore)
                    results.append(result)

                    if time.time() - self.last_save_time >= self.save_interval:
                        self._perform_periodic_save(documents, vectorstore)
                        results = []
                        self.last_save_time = time.time()

            return documents, vectorstore

        except Exception as e:
            logging.error(f"Error in batch processing: {str(e)}")
            raise

    def _handle_processing_result(
        self,
        result: Tuple[List[Document], Dict],
        documents: List[Document],
        vectorstore: Optional[FAISS]
    ) -> None:
        """Handle the result of processing a single file."""
        docs, file_info = result
        if docs and all(isinstance(doc, Document) for doc in docs):
            documents.extend(docs)
        
        filename = os.path.basename(file_info['path'])
        self.current_files[filename] = file_info

    def _perform_periodic_save(
        self,
        documents: List[Document],
        vectorstore: Optional[FAISS]
    ) -> None:
        """Perform periodic saving of documents and vectorstore."""
        if documents:
            try:
                vectorstore = self._update_vectorstore(documents, vectorstore)
                documents.clear()
            except Exception as e:
                logging.error(f"Error during periodic vectorstore update: {str(e)}")

        self.save_progress(vectorstore)

    def _update_vectorstore(
        self,
        documents: List[Document],
        vectorstore: Optional[FAISS]
    ) -> Optional[FAISS]:
        """Update vectorstore with new documents."""
        try:
            # If no vectorstore provided but exists on disk, load it
            if not vectorstore and os.path.exists(self.vectorstore_path):
                try:
                    vectorstore = FAISS.load_local(
                        self.vectorstore_path,
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    logging.info("Loaded existing vectorstore from disk")
                except Exception as e:
                    logging.error(f"Error loading existing vectorstore: {str(e)}")
                    vectorstore = None

            # Create new vectorstore if none exists
            if not vectorstore:
                vectorstore = FAISS.from_documents(documents, self.embeddings)
                logging.info("Created new vectorstore")
            else:
                # Add new documents to existing vectorstore
                vectorstore.add_documents(documents)
                logging.info(f"Added {len(documents)} documents to existing vectorstore")
            
            # Save updated vectorstore
            os.makedirs(os.path.dirname(self.vectorstore_path), exist_ok=True)
            vectorstore.save_local(self.vectorstore_path)
            logging.info("Saved vectorstore to disk")
            
            return vectorstore
            
        except Exception as e:
            logging.error(f"Error updating vectorstore: {str(e)}")
            raise

    def load_or_create_vectorstore(self) -> FAISS:
        """Load existing vectorstore or create a new one."""
        try:
            self.documents_buffer, self.current_files = self.load_checkpoint()
            vectorstore = self._load_existing_vectorstore()
            
            new_files = self.get_new_files_to_process()
            if new_files or self.documents_buffer:
                self.start_auto_save_thread(vectorstore)
                newly_processed_docs, vectorstore = self.process_files_in_batches(new_files, vectorstore)
                self.documents_buffer.extend(newly_processed_docs)
                
                if time.time() - self.last_checkpoint_time >= self.checkpoint_interval:
                    self.save_checkpoint(self.documents_buffer, self.current_files)
                    self.last_checkpoint_time = time.time()
                
                if self.documents_buffer:
                    vectorstore = self._finalize_processing(vectorstore)
            
            return vectorstore
            
        except Exception as e:
            logging.error("Error in load_or_create_vectorstore:", exc_info=True)
            if self.documents_buffer:
                self.save_checkpoint(self.documents_buffer, self.current_files)
            raise

    def _load_existing_vectorstore(self) -> Optional[FAISS]:
        """Load existing vectorstore if it exists."""
        if os.path.exists(self.vectorstore_path):
            return FAISS.load_local(
                self.vectorstore_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        return None

    def _finalize_processing(self, vectorstore: Optional[FAISS]) -> Optional[FAISS]:
        """Finalize document processing and update vectorstore."""
        try:
            vectorstore = self._update_vectorstore(self.documents_buffer, vectorstore)
            self.documents_buffer.clear()
            self.cleanup_checkpoint()
            logging.info("Document buffer cleared and checkpoint cleaned up")
            return vectorstore
        except Exception as e:
            logging.error(f"Failed to process documents: {str(e)}")
            self.save_checkpoint(self.documents_buffer, self.current_files)
            raise

    def load_checkpoint(self) -> Tuple[List[Document], Dict]:
        """Load checkpoint data if it exists."""
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, 'rb') as f:
                    import pickle
                    checkpoint_data = pickle.load(f)
                    logging.info(f"Loaded checkpoint with {len(checkpoint_data['documents'])} documents")
                    return checkpoint_data['documents'], checkpoint_data['processed_files']
            except Exception as e:
                logging.error(f"Error loading checkpoint: {str(e)}")
        return [], {}

    def cleanup_checkpoint(self) -> None:
        """Remove checkpoint file if it exists."""
        try:
            if os.path.exists(self.checkpoint_path):
                os.remove(self.checkpoint_path)
                logging.info("Checkpoint file cleaned up")
        except Exception as e:
            logging.error(f"Error cleaning up checkpoint: {str(e)}")

    def get_new_files_to_process(self) -> List[str]:
        """Get list of new or modified files that need processing."""
        processed_files = self.load_processed_files()
        all_files = []
        
        try:
            for root, _, files in os.walk(self.data_dir):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        rel_path = os.path.relpath(os.path.join(root, file), self.data_dir)
                        if self.should_process_file(rel_path, processed_files):
                            all_files.append(rel_path)
        except Exception as e:
            logging.error(f"Error getting new files: {str(e)}")
            raise
            
        return all_files

    def save_processed_files(self, new_processed_files: Dict) -> None:
        """Save information about processed files to disk, appending new files to existing ones."""
        try:
            os.makedirs(os.path.dirname(self.processed_files_path), exist_ok=True)
            
            # Load existing processed files
            existing_files = self.load_processed_files()
            
            # Update existing files with new ones
            existing_files.update(new_processed_files)
            
            temp_path = f"{self.processed_files_path}.tmp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(existing_files, f, indent=2)
            
            os.replace(temp_path, self.processed_files_path)
            logging.info(f"Saved {len(existing_files)} processed files to disk (including {len(new_processed_files)} new files)")
        except Exception as e:
            logging.error(f"Error saving processed files info: {str(e)}", exc_info=True)

    def load_processed_files(self) -> Dict:
        """Load information about processed files from disk."""
        if not os.path.exists(self.processed_files_path):
            logging.info(f"No processed files record found at {self.processed_files_path}")
            return {}
            
        try:
            with open(self.processed_files_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    logging.info("Processed files record is empty")
                    return {}
                return json.loads(content)
        except json.JSONDecodeError as e:
            logging.warning(f"Invalid JSON in processed files record: {str(e)}. Starting fresh.")
            return {}
        except Exception as e:
            logging.warning(f"Error loading processed files info: {str(e)}. Starting fresh.")
            return {}

    def should_process_file(self, filename: str, processed_files: Dict) -> bool:
        """Determine if a file should be processed based on its hash and previous status."""
        file_path = os.path.join(self.data_dir, filename)
        if not os.path.exists(file_path):
            return False
            
        if filename in processed_files:
            previous_status = processed_files[filename].get("status", "unknown")
            if previous_status in ["error", "timeout"]:
                logging.info(f"Retrying {filename} due to previous {previous_status}")
                return True
            
        current_hash = self._get_file_hash(file_path)
        return (filename not in processed_files or 
                processed_files[filename]["hash"] != current_hash)

    def _get_file_hash(self, file_path: str) -> str:
        """Generate a hash for a file based on its content."""
        try:
            with open(file_path, 'rb') as f:
                return md5(f.read()).hexdigest()
        except Exception as e:
            logging.error(f"Error generating hash for {file_path}: {str(e)}")
            raise

    def save_progress(self, vectorstore: Optional[FAISS] = None) -> None:
        """Save current progress (processed files and vectorstore)."""
        try:
            with threading.Lock():
                if self.current_files:
                    self.save_processed_files(self.current_files)
                
                if vectorstore and os.path.dirname(self.vectorstore_path):
                    os.makedirs(os.path.dirname(self.vectorstore_path), exist_ok=True)
                    vectorstore.save_local(self.vectorstore_path)
                    logging.info("Vectorstore saved successfully")
                
            logging.info("Progress saved successfully")
        except Exception as e:
            logging.error(f"Error saving progress: {str(e)}", exc_info=True)

    def start_auto_save_thread(self, vectorstore: Optional[FAISS] = None) -> None:
        """Start a thread that periodically saves progress."""
        def auto_save():
            while True:
                time.sleep(self.save_interval)
                if vectorstore:
                    self.save_progress(vectorstore)
                
        self.auto_save_thread = threading.Thread(target=auto_save, daemon=True)
        self.auto_save_thread.start()

    def save_checkpoint(self, documents: List[Document], processed_files: Dict) -> None:
        """Save a checkpoint of processed documents and files."""
        checkpoint_path = os.path.join(self.runtime_dir, "documents_checkpoint.pkl")
        temp_path = f"{checkpoint_path}.tmp"
        
        try:
            checkpoint_data = {
                'documents': documents,
                'processed_files': processed_files,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            with open(temp_path, 'wb') as f:
                import pickle
                pickle.dump(checkpoint_data, f)
            
            os.replace(temp_path, checkpoint_path)
            logging.info(f"Saved checkpoint with {len(documents)} documents")
            
        except Exception as e:
            logging.error(f"Error saving checkpoint: {str(e)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get statistics about processed files and GPU usage."""
        processed_files = self.load_processed_files()
        stats = {
            "total_files": len(processed_files),
            "successful": 0,
            "failed": 0,
            "timeout": 0,
            "average_processing_time": 0,
            "total_processing_time": 0,
            "gpu_stats": []
        }

        processing_times = []
        gpu_usage = {}
        
        for file_info in processed_files.values():
            status = file_info.get("status", "unknown")
            if status == "success":
                stats["successful"] += 1
            elif status == "timeout":
                stats["timeout"] += 1
            elif status == "error":
                stats["failed"] += 1
                
            if "processing_time" in file_info:
                processing_times.append(file_info["processing_time"])
            
            gpu_id = file_info.get("gpu_id")
            if gpu_id is not None:
                gpu_usage[gpu_id] = gpu_usage.get(gpu_id, 0) + 1
        
        if processing_times:
            stats["total_processing_time"] = sum(processing_times)
            stats["average_processing_time"] = sum(processing_times) / len(processing_times)
        
        if self.nvml_initialized:
            stats["gpu_stats"] = GPUManager.get_gpu_info()
            
        stats["gpu_usage_distribution"] = gpu_usage
            
        return stats

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.save_progress(None)
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")

    def __del__(self) -> None:
        """Ensure cleanup on object destruction."""
        self.cleanup()

class PDFProcessor:
    """Handles the processing of individual PDF files."""
    
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.start_time = time.time()
        self._configure_environment()

    def _configure_environment(self) -> None:
        """Configure environment for PDF processing."""
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        os.environ["OMP_NUM_THREADS"] = "4"
        os.environ["MKL_NUM_THREADS"] = "4"
        os.environ["ONNXRUNTIME_CUDA_DEVICE_ID"] = str(self.gpu_id)
        
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    def process_file(
        self,
        file_path: str,
        filename: str,
        timeout_seconds: int
    ) -> Tuple[List[Document], Dict]:
        """Process a single PDF file."""
        start_time = time.time()
        file_info = self._initialize_file_info(file_path)
        
        try:
            with timeout(timeout_seconds):
                pdf_elements = self._extract_pdf_elements(file_path)
                documents = self._process_elements(pdf_elements, file_path, filename)
                
                processing_time = time.time() - start_time
                self._update_file_info(file_info, "success", processing_time)
                logging.info(f"Successfully processed {filename} on GPU {self.gpu_id} "
                           f"in {processing_time:.2f} seconds")
                
                return documents, file_info
                
        except TimeoutException:
            self._handle_timeout(file_info, start_time, filename)
            return [], file_info
        except Exception as e:
            self._handle_error(file_info, start_time, filename, e)
            return [], file_info

    def _initialize_file_info(self, file_path: str) -> Dict:
        """Initialize file information dictionary."""
        with open(file_path, 'rb') as f:
            file_hash = md5(f.read()).hexdigest()
        
        return {
            "hash": file_hash,
            "last_processed": str(datetime.datetime.now()),
            "path": file_path,
            "status": "processing",
            "gpu_id": self.gpu_id
        }

    def _extract_pdf_elements(self, file_path: str) -> List:
        """Extract elements from PDF file."""
        return partition_pdf(
            filename=file_path,
            extract_images_in_pdf=True,
            strategy="hi_res",
            hi_res_model_name="yolox",
            infer_table_structure=True,
            detection_device="cuda",
            detection_batch_size=1,
            model_kwargs=self._get_model_kwargs()
        )

    def _get_model_kwargs(self) -> Dict:
        """Get model configuration kwargs."""
        return {
            "provider": "CUDAExecutionProvider",
            "provider_options": [{
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": int(torch.cuda.get_device_properties(0).total_memory * 0.8),
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
                "gpu_external_alloc": True,
                "gpu_external_free": True
            }]
        }

    def _process_elements(
        self,
        elements: List,
        file_path: str,
        filename: str
    ) -> List[Document]:
        """Process PDF elements into documents with enhanced structural information."""
        documents = []
        current_section = ""
        current_text = []
        file_hash = self._get_file_hash(file_path)
        
        # Track document hierarchy and relationships
        hierarchy = {
            'level': 0,  # Current hierarchy level
            'parent_sections': [],  # Stack of parent sections
            'siblings': [],  # List of sibling sections at current level
            'last_coordinates': None,  # Last processed element coordinates
        }
        
        for element in elements:
            # Extract layout and style information
            layout_info = self._extract_layout_info(element)
            if not layout_info:
                continue
            
            text = layout_info['text']
            if not text:
                continue
            
            # Determine element type and hierarchy
            element_type = self._determine_element_type(element, text, layout_info)
            if element_type.startswith('header'):
                # Process any accumulated content before starting new section
                if current_text:
                    self._add_content_document(
                        documents, current_text, file_path, filename,
                        file_hash, current_section, hierarchy, layout_info
                    )
                    current_text = []
                
                # Update hierarchy information
                header_level = int(element_type.split('_')[1]) if '_' in element_type else 1
                self._update_hierarchy(hierarchy, header_level, text)
                
                # Create header document with enhanced metadata
                current_section = text
                documents.append(self._create_document(
                    text,
                    file_path,
                    filename,
                    file_hash,
                    current_section,
                    element_type,
                    hierarchy=hierarchy,
                    layout_info=layout_info
                ))
            else:
                # Accumulate content with layout context
                current_text.append({
                    'text': text,
                    'layout': layout_info,
                    'relationship': self._determine_relationships(layout_info, hierarchy)
                })
                
                # Split content if it exceeds threshold or significant layout change
                if (len("\n".join([t['text'] for t in current_text])) >= 1000 or
                    self._should_split_content(current_text)):
                    self._add_content_document(
                        documents, current_text, file_path, filename,
                        file_hash, current_section, hierarchy, layout_info
                    )
                    current_text = []
            
            hierarchy['last_coordinates'] = layout_info.get('coordinates')
        
        # Process any remaining content
        if current_text:
            self._add_content_document(
                documents, current_text, file_path, filename,
                file_hash, current_section, hierarchy, layout_info
            )
        
        return documents

    def _extract_layout_info(self, element: Any) -> Dict:
        """Extract layout and style information from element."""
        try:
            # Get basic text content
            text = self._get_element_text(element)
            
            # Extract coordinates and style information
            if isinstance(element, dict):
                coords = element.get('coordinates', {})
                style = element.get('style', {})
            else:
                # Handle different element types (e.g., unstructured.documents elements)
                coords = getattr(element, 'coordinates', {})
                style = {
                    'font': getattr(element, 'font', ''),
                    'size': getattr(element, 'font_size', 0),
                    'weight': getattr(element, 'font_weight', ''),
                    'color': getattr(element, 'color', ''),
                    'background': getattr(element, 'background_color', '')
                }
            
            return {
                'text': text,
                'coordinates': {
                    'x1': coords.get('x1', 0),
                    'y1': coords.get('y1', 0),
                    'x2': coords.get('x2', 0),
                    'y2': coords.get('y2', 0),
                    'page': coords.get('page', 1)
                },
                'style': {
                    'font_name': style.get('font', ''),
                    'font_size': style.get('size', 0),
                    'font_weight': style.get('weight', ''),
                    'color': style.get('color', ''),
                    'background': style.get('background', ''),
                    'alignment': style.get('alignment', '')
                },
                'element_type': type(element).__name__
            }
        except Exception as e:
            logging.warning(f"Error extracting layout info: {e}")
            return None

    def _determine_element_type(self, element: Any, text: str, layout_info: Dict) -> str:
        """Determine element type based on visual characteristics."""
        style = layout_info['style']
        
        # Calculate importance score based on visual characteristics
        importance_score = 0
        if style['font_size'] > 14: importance_score += 2
        if style['font_weight'] in ['bold', '700', '800', '900']: importance_score += 2
        if text.isupper(): importance_score += 1
        if style['color'] != 'black': importance_score += 1
        
        # Check for header patterns
        if any(text.startswith(prefix) for prefix in ['Section', 'SECTION', 'Article', 'ARTICLE']):
            importance_score += 3
        
        # Determine header level based on importance
        if importance_score >= 5:
            return 'header_1'
        elif importance_score >= 3:
            return 'header_2'
        elif importance_score >= 2:
            return 'header_3'
        
        # Determine content type
        if 'Table' in layout_info['element_type']:
            return 'table'
        elif 'List' in layout_info['element_type']:
            return 'list'
        elif 'Image' in layout_info['element_type']:
            return 'image'
        
        return 'content'

    def _update_hierarchy(self, hierarchy: Dict, header_level: int, text: str) -> None:
        """Update document hierarchy information."""
        if header_level <= hierarchy['level']:
            # Moving up or sideways in hierarchy
            while hierarchy['parent_sections'] and hierarchy['level'] >= header_level:
                hierarchy['parent_sections'].pop()
                hierarchy['level'] -= 1
            hierarchy['siblings'] = []
        else:
            # Moving down in hierarchy
            if hierarchy['level'] > 0:
                hierarchy['parent_sections'].append(hierarchy['siblings'][-1] if hierarchy['siblings'] else '')
            hierarchy['siblings'] = []
        
        hierarchy['level'] = header_level
        hierarchy['siblings'].append(text)

    def _determine_relationships(self, layout_info: Dict, hierarchy: Dict) -> Dict:
        """Determine relationships between content elements."""
        relationships = {
            'spatial': [],  # Spatial relationships (above, below, next_to)
            'style': [],    # Style relationships (same_style, contrasting_style)
            'semantic': []  # Semantic relationships (continuation, reference)
        }
        
        # Check spatial relationships
        if hierarchy['last_coordinates']:
            last = hierarchy['last_coordinates']
            current = layout_info['coordinates']
            
            # Vertical relationships
            if abs(current['y1'] - last['y2']) < 20:
                relationships['spatial'].append('continuation')
            elif current['y1'] > last['y2']:
                relationships['spatial'].append('below')
            else:
                relationships['spatial'].append('above')
            
            # Horizontal relationships
            if abs(current['x1'] - last['x1']) < 10:
                relationships['spatial'].append('aligned')
            elif current['x1'] > last['x1']:
                relationships['spatial'].append('indented')
        
        # Style relationships
        if hierarchy.get('last_style'):
            last_style = hierarchy['last_style']
            current_style = layout_info['style']
            
            if (last_style['font_name'] == current_style['font_name'] and
                last_style['font_size'] == current_style['font_size']):
                relationships['style'].append('same_style')
            
            if last_style['font_size'] < current_style['font_size']:
                relationships['style'].append('emphasis')
        
        # Semantic relationships
        if hierarchy['parent_sections']:
            relationships['semantic'].append({
                'type': 'belongs_to',
                'section': hierarchy['parent_sections'][-1]
            })
        
        return relationships

    def _should_split_content(self, current_text: List[Dict]) -> bool:
        """Determine if content should be split based on layout changes."""
        if len(current_text) < 2:
            return False
        
        last = current_text[-2]['layout']
        current = current_text[-1]['layout']
        
        # Split on significant layout changes
        if (abs(current['coordinates']['x1'] - last['coordinates']['x1']) > 50 or
            abs(current['coordinates']['y1'] - last['coordinates']['y2']) > 30 or
            current['style']['font_size'] != last['style']['font_size']):
            return True
        
        return False

    def _add_content_document(
        self,
        documents: List[Document],
        content: List[Dict],
        file_path: str,
        filename: str,
        file_hash: str,
        section: str,
        hierarchy: Dict,
        layout_info: Dict
    ) -> None:
        """Add a content document with enhanced metadata."""
        # Calculate content importance based on visual characteristics
        importance_score = self._calculate_content_importance(content, hierarchy)
        
        # Create document with enhanced metadata
        documents.append(self._create_document(
            "\n".join([t['text'] for t in content]),
            file_path,
            filename,
            file_hash,
            section,
            'content',
            hierarchy=hierarchy,
            layout_info=layout_info,
            importance_score=importance_score,
            relationships=[t['relationship'] for t in content]
        ))

    def _calculate_content_importance(self, content: List[Dict], hierarchy: Dict) -> float:
        """Calculate content importance based on various factors."""
        importance = 1.0
        
        # Factor 1: Proximity to headers
        if hierarchy['level'] > 0:
            importance *= (1 + 0.2 * hierarchy['level'])
        
        # Factor 2: Visual characteristics
        for item in content:
            style = item['layout']['style']
            if style['font_weight'] in ['bold', '700', '800', '900']:
                importance *= 1.2
            if style['font_size'] > 12:
                importance *= 1.1
        
        # Factor 3: Spatial relationships
        for item in content:
            relationships = item['relationship']
            if 'indented' in relationships['spatial']:
                importance *= 0.9  # Slightly less important
            if 'emphasis' in relationships['style']:
                importance *= 1.3
        
        return importance

    def _get_element_text(self, element: Any) -> str:
        """Extract text from element."""
        if isinstance(element, dict):
            return element.get('text', '').strip()
        return element.text.strip() if hasattr(element, 'text') else str(element).strip()

    def _is_header(self, element: Any, text: str) -> bool:
        """Determine if element is a header."""
        if isinstance(element, dict):
            element_type = element.get('type', '')
        else:
            element_type = type(element).__name__
        
        element_type_str = str(element_type)
        
        return (
            any(t in element_type_str for t in ['Title', 'Header', 'Section', 'NarrativeText']) or
            text.isupper() and len(text.split()) <= 10 or
            any(text.startswith(prefix) for prefix in ['Section', 'SECTION', 'Article', 'ARTICLE'])
        )

    def _create_document(
        self,
        content: str,
        file_path: str,
        filename: str,
        file_hash: str,
        section: str,
        element_type: str,
        hierarchy: Dict = None,
        layout_info: Dict = None,
        importance_score: float = 1.0,
        relationships: List[Dict] = None
    ) -> Document:
        """Create a Document object with enhanced metadata."""
        metadata = {
            "source": file_path,
            "filename": filename,
            "file_hash": file_hash,
            "section": section,
            "processing_time": time.time() - self.start_time,
            "gpu_id": self.gpu_id,
            "element_types": element_type,
            "importance_score": importance_score
        }
        
        # Add hierarchy information
        if hierarchy:
            metadata.update({
                "hierarchy_level": hierarchy['level'],
                "parent_sections": hierarchy['parent_sections'],
                "sibling_sections": hierarchy['siblings']
            })
        
        # Add layout information
        if layout_info:
            metadata.update({
                "coordinates": layout_info['coordinates'],
                "style": layout_info['style']
            })
        
        # Add relationship information
        if relationships:
            metadata["relationships"] = relationships
        
        return Document(
            page_content=content,
            metadata=metadata
        )

    def _get_file_hash(self, file_path: str) -> str:
        """Generate MD5 hash of file."""
        with open(file_path, 'rb') as f:
            return md5(f.read()).hexdigest()

    def _update_file_info(
        self,
        file_info: Dict,
        status: str,
        processing_time: float
    ) -> None:
        """Update file information with processing results."""
        file_info.update({
            "status": status,
            "processing_time": processing_time
        })

    def _handle_timeout(
        self,
        file_info: Dict,
        start_time: float,
        filename: str
    ) -> None:
        """Handle timeout exception."""
        processing_time = time.time() - start_time
        logging.error(f"Timeout processing {filename} on GPU {self.gpu_id} "
                     f"after {processing_time:.2f} seconds")
        file_info.update({
            "status": "timeout",
            "processing_time": processing_time,
            "error": "Processing timeout"
        })

    def _handle_error(
        self,
        file_info: Dict,
        start_time: float,
        filename: str,
        error: Exception
    ) -> None:
        """Handle processing error."""
        processing_time = time.time() - start_time
        error_msg = f"Error processing {filename} on GPU {self.gpu_id}: {str(error)}"
        logging.error(error_msg)
        file_info.update({
            "status": "error",
            "processing_time": processing_time,
            "error": str(error)
        })

def main():
    """Main entry point for the script."""
    processor = None
    try:
        args = _parse_arguments()
        os.makedirs(RUNTIME_DIR, exist_ok=True)
        
        GPUManager.print_status()
        
        gpu_devices = _select_gpu_devices(args)
        processor = DocumentProcessor(
            runtime_dir=RUNTIME_DIR,
            data_dir=DATA_DIR,
            timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
            save_interval=DEFAULT_SAVE_INTERVAL,
            gpu_devices=gpu_devices
        )
        
        print("\nProcessing documents...")
        vectorstore = processor.load_or_create_vectorstore()
        
        _print_processing_statistics(processor)
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        logging.info("Process interrupted by user")
    except Exception as e:
        _handle_main_error(e)
    finally:
        if processor:
            processor.cleanup()
        try:
            print("\nFinal GPU Status:")
            GPUManager.print_status()
        except:
            pass

def _parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process PDFs with GPU acceleration')
    parser.add_argument('--gpus', type=str,
                       help='Comma-separated list of GPU indices to use (e.g., "0,1,2")')
    parser.add_argument('--num-gpus', type=int,
                       help='Number of GPUs to use (will select least utilized)')
    return parser.parse_args()

def _select_gpu_devices(args) -> Optional[str]:
    """Select GPU devices based on arguments."""
    if args.gpus:
        print(f"\nUsing specified GPUs: {args.gpus}")
        return args.gpus
    elif args.num_gpus:
        gpu_devices = GPUManager.select_best_gpus(args.num_gpus)
        print(f"\nAutomatically selected GPUs: {gpu_devices}")
        return gpu_devices
    print("\nUsing all available GPUs")
    return None

def _print_processing_statistics(processor: DocumentProcessor) -> None:
    """Print processing statistics."""
    stats = processor.get_processing_statistics()
    print("\nProcessing Statistics:")
    print(f"Total files processed: {stats['total_files']}")
    print(f"Successfully processed: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Timeouts: {stats['timeout']}")
    
    if stats['total_files'] > 0:
        print(f"Average processing time: {stats['average_processing_time']:.2f} seconds")
        print(f"Total processing time: {stats['total_processing_time']:.2f} seconds")
    
    if stats.get('gpu_usage_distribution'):
        print("\nGPU Usage Distribution:")
        for gpu_id, count in stats['gpu_usage_distribution'].items():
            print(f"GPU {gpu_id}: {count} files processed")

def _handle_main_error(error: Exception) -> None:
    """Handle main process error."""
    print("\nError occurred:")
    print("="*50)
    print("Error message:", str(error))
    print("\nFull stack trace:")
    print(traceback.format_exc())
    logging.error("Main process error:", exc_info=True)

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass  # method already set
    main()
