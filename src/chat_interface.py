"""
chat_interface.py - Interactive Web Interface for RAG System

This module implements a Streamlit-based web interface for the RAG system, providing:
- Interactive chat interface for question answering
- Real-time processing feedback and logging
- Configurable quality control settings
- Source document citations and downloads
- Debug console for system monitoring

The interface is designed to be user-friendly while providing advanced features
for both end users and system administrators.
"""

import streamlit as st
import logging
from rag_pipeline import (
    validate_config,
    create_workflow,
    process_query,
    format_answer
)

def update_log_display():
    """
    Update the log display in real-time.
    
    This function updates the Streamlit text area with the latest logs
    from the session state. It's called automatically whenever new log
    entries are added through the UILogHandler.
    """
    if st.session_state.log_container is not None:
        st.session_state.log_container.text_area(
            "Console Output",
            value="\n".join(st.session_state.logs),
            height=300,
            disabled=True
        )

class UILogHandler(logging.Handler):
    """
    Custom logging handler that stores logs in Streamlit session state.
    
    This handler maintains a rolling buffer of the last 100 log entries
    and updates the UI in real-time as new logs are added. It's particularly
    useful for monitoring system operations and debugging issues.
    """
    def emit(self, record):
        try:
            if 'logs' not in st.session_state:
                st.session_state.logs = []
            log_entry = self.format(record)
            st.session_state.logs.insert(0, log_entry)
            # Keep only the last 100 log entries
            if len(st.session_state.logs) > 100:
                st.session_state.logs = st.session_state.logs[:100]
            # Update display in real-time
            update_log_display()
        except Exception:
            pass

# Set up logging
ui_handler = UILogHandler()
ui_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(),
        ui_handler
    ]
)

def initialize_session_state():
    """
    Initialize Streamlit session state variables.
    
    Sets up the following session state variables:
    - chat_history: List of (question, answer) tuples
    - workflow: RAG pipeline workflow instance
    - enable_answer_grader: Toggle for answer quality checking
    - enable_hallucination_grader: Toggle for hallucination detection
    - show_console: Toggle for debug console visibility
    - logs: List of system log entries
    - log_container: Streamlit container for log display
    """
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'workflow' not in st.session_state:
        st.session_state.workflow = None
    if 'enable_answer_grader' not in st.session_state:
        st.session_state.enable_answer_grader = False
    if 'enable_hallucination_grader' not in st.session_state:
        st.session_state.enable_hallucination_grader = False
    if 'show_console' not in st.session_state:
        st.session_state.show_console = True
    if 'logs' not in st.session_state:
        st.session_state.logs = []
    if 'log_container' not in st.session_state:
        st.session_state.log_container = None

def setup_app():
    """
    Initialize the RAG pipeline components.
    
    This function:
    1. Validates the system configuration (API keys, model settings)
    2. Creates the RAG workflow if not already initialized
    3. Returns True if setup is successful, False otherwise
    
    Returns:
        bool: True if setup successful, False if errors occurred
    """
    try:
        validate_config()
        if st.session_state.workflow is None:
            st.session_state.workflow = create_workflow()
        return True
    except Exception as e:
        st.error(f"Error initializing app: {str(e)}")
        logging.error("Error in setup_app:", exc_info=True)
        return False

def display_chat_message(role, content):
    """
    Display a chat message with proper formatting.
    
    Args:
        role (str): Either "user" or "assistant" to determine message styling
        content (str): The message content to display
    
    Features:
    - Different styling for user and assistant messages
    - Automatic source separation and expandable source section
    - Markdown rendering for formatted text
    - Custom avatars for each role
    """
    if role == "user":
        with st.chat_message("user", avatar="🧑"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            if "SOURCES:" in content:
                answer, sources_text = content.split("SOURCES:", 1)
                st.markdown(answer.strip())
                with st.expander("📚 Sources & Downloads"):
                    st.markdown(sources_text)
            else:
                st.markdown(content)

def display_chat_history():
    """
    Display the chat history using Streamlit's chat interface.
    
    Iterates through the chat_history in session state and displays
    each message with appropriate formatting using display_chat_message().
    """
    for question, answer in st.session_state.chat_history:
        display_chat_message("user", question)
        display_chat_message("assistant", answer)

def handle_user_input(user_input):
    """
    Process user input and update chat history.
    
    Args:
        user_input (str): The user's question or command
    
    This function:
    1. Validates the input
    2. Processes the query through the RAG pipeline
    3. Formats the response with sources
    4. Updates the chat history
    5. Handles any errors that occur during processing
    """
    if not user_input:
        return

    try:
        with st.spinner("Processing..."):
            result = process_query(
                user_input,
                enable_answer_grader=st.session_state.enable_answer_grader,
                enable_hallucination_grader=st.session_state.enable_hallucination_grader
            )
            if result:
                formatted_answer = format_answer(result)
                st.session_state.chat_history.append((user_input, formatted_answer))
            else:
                st.session_state.chat_history.append(
                    (user_input, "No relevant information found. Please try rephrasing your question.")
                )
    except Exception as e:
        st.session_state.chat_history.append(
            (user_input, f"Error: {str(e)}")
        )
        logging.error("Error processing query:", exc_info=True)

def main():
    """
    Main application entry point.
    
    Sets up and runs the Streamlit web interface with:
    1. Page configuration and layout
    2. Sidebar with settings and controls
    3. Main chat interface
    4. Real-time processing and updates
    """
    st.set_page_config(
        page_title="Q&A Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    initialize_session_state()
    
    if not setup_app():
        st.error("Failed to initialize the application.")
        return
    
    with st.sidebar:
        st.title("Settings")
        
        # Grading toggles
        st.subheader("Grading Options")
        st.session_state.enable_answer_grader = st.toggle(
            "Enable Answer Grading",
            value=st.session_state.enable_answer_grader
        )
        st.session_state.enable_hallucination_grader = st.toggle(
            "Enable Hallucination Checking",
            value=st.session_state.enable_hallucination_grader
        )
        
        # Console toggle
        st.subheader("Debug Console")
        st.session_state.show_console = st.toggle(
            "Show Console",
            value=st.session_state.show_console
        )
        
        if st.session_state.show_console:
            # Create a fresh container for log updates on each render
            st.session_state.log_container = st.empty()
            update_log_display()
        
        st.title("Help")
        st.markdown("""
        - Click the "Clear Chat" button to start a new conversation.
        """)
        if st.button("Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()
    
    st.title("Q&A Assistant")
    
    display_chat_history()
    
    user_input = st.chat_input("Ask a question...")
    if user_input:
        handle_user_input(user_input)
        st.rerun()

if __name__ == "__main__":
    main()
