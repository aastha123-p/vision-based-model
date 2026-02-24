import uuid

class TokenGenerator:
    """
    Generates secure unique tokens for patients
    """

    @staticmethod
    def generate_token() -> str:
        return str(uuid.uuid4())