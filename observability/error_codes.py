from enum import Enum


class ErrorCode(str, Enum):
    MODEL_CALL_FAILED = "MODEL_CALL_FAILED"
    VECTOR_STORE_UNAVAILABLE = "VECTOR_STORE_UNAVAILABLE"
    RETRIEVAL_EMPTY = "RETRIEVAL_EMPTY"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    TOOL_TIMEOUT = "TOOL_TIMEOUT"
    INVALID_USER_ID = "INVALID_USER_ID"
    KNOWLEDGE_SYNC_FAILED = "KNOWLEDGE_SYNC_FAILED"
    TOOL_EXECUTION_FAILED = "TOOL_EXECUTION_FAILED"
    DB_WRITE_FAILED = "DB_WRITE_FAILED"


ERROR_MESSAGES = {
    ErrorCode.MODEL_CALL_FAILED: "Model call failed. Please try again later.",
    ErrorCode.VECTOR_STORE_UNAVAILABLE: "Knowledge base is temporarily unavailable.",
    ErrorCode.RETRIEVAL_EMPTY: "No relevant information found for your query.",
    ErrorCode.LOW_CONFIDENCE: "I'm not confident enough to answer this question.",
    ErrorCode.TOOL_TIMEOUT: "The requested operation timed out.",
    ErrorCode.INVALID_USER_ID: "Invalid customer ID. Please provide a valid ID.",
    ErrorCode.KNOWLEDGE_SYNC_FAILED: "Knowledge base synchronization failed.",
    ErrorCode.TOOL_EXECUTION_FAILED: "Tool execution failed.",
    ErrorCode.DB_WRITE_FAILED: "Failed to save data.",
}
