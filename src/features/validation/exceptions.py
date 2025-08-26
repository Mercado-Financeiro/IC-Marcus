"""Custom exceptions for feature validation."""


class ValidationError(Exception):
    """Base exception for feature validation errors."""
    pass


class ColumnMissingError(ValidationError):
    """Raised when required columns are missing from DataFrame."""
    
    def __init__(self, missing_columns, function_name=None):
        self.missing_columns = missing_columns
        self.function_name = function_name
        
        if isinstance(missing_columns, str):
            missing_columns = [missing_columns]
        
        columns_str = ', '.join(f"'{col}'" for col in missing_columns)
        
        if function_name:
            message = f"Missing required columns {columns_str} in {function_name}"
        else:
            message = f"Missing required columns: {columns_str}"
        
        super().__init__(message)


class DataInconsistencyError(ValidationError):
    """Raised when data fails consistency checks."""
    
    def __init__(self, message, inconsistent_rows=None):
        self.inconsistent_rows = inconsistent_rows
        super().__init__(message)


class InvalidDataTypeError(ValidationError):
    """Raised when columns have invalid data types."""
    
    def __init__(self, column, expected_type, actual_type):
        self.column = column
        self.expected_type = expected_type
        self.actual_type = actual_type
        
        message = f"Column '{column}' has type {actual_type}, expected {expected_type}"
        super().__init__(message)


class InsufficientDataError(ValidationError):
    """Raised when DataFrame has insufficient rows for calculation."""
    
    def __init__(self, required_rows, actual_rows, function_name=None):
        self.required_rows = required_rows
        self.actual_rows = actual_rows
        self.function_name = function_name
        
        if function_name:
            message = f"{function_name} requires at least {required_rows} rows, got {actual_rows}"
        else:
            message = f"Insufficient data: need {required_rows} rows, got {actual_rows}"
        
        super().__init__(message)


class InvalidRangeError(ValidationError):
    """Raised when data values are outside valid ranges."""
    
    def __init__(self, column, invalid_count, total_count, valid_range=None):
        self.column = column
        self.invalid_count = invalid_count
        self.total_count = total_count
        self.valid_range = valid_range
        
        if valid_range:
            message = f"Column '{column}' has {invalid_count}/{total_count} values outside valid range {valid_range}"
        else:
            message = f"Column '{column}' has {invalid_count}/{total_count} invalid values"
        
        super().__init__(message)