from fastapi import HTTPException, status


class RiskAnalyzerException(Exception):
    """Base exception for risk analyzer module."""
    pass


class ServiceConnectionError(RiskAnalyzerException):
    """Exception raised when a service connection fails."""
    def __init__(self, service_name: str, message: str = None):
        self.service_name = service_name
        self.message = message or f"Failed to connect to {service_name} service"
        super().__init__(self.message)


class DatabaseError(RiskAnalyzerException):
    """Exception raised for database errors."""
    def __init__(self, message: str = "Database operation failed"):
        self.message = message
        super().__init__(self.message)


class VectorOperationError(RiskAnalyzerException):
    """Exception raised for vector operation errors."""
    def __init__(self, message: str = "Vector operation failed"):
        self.message = message
        super().__init__(self.message)


class ModelError(RiskAnalyzerException):
    """Exception raised for model errors."""
    def __init__(self, message: str = "Model operation failed"):
        self.message = message
        super().__init__(self.message)


class EventError(RiskAnalyzerException):
    """Exception raised for event handling errors."""
    def __init__(self, message: str = "Event handling failed"):
        self.message = message
        super().__init__(self.message)


def http_exception_handler(exc: RiskAnalyzerException) -> HTTPException:
    """Convert RiskAnalyzerException to HTTPException."""
    if isinstance(exc, ServiceConnectionError):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=exc.message
        )
    elif isinstance(exc, (DatabaseError, VectorOperationError)):
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=exc.message
        )
    elif isinstance(exc, ModelError):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=exc.message
        )
    else:
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc) or "An unexpected error occurred"
        ) 