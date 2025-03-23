class ConversationHistory:
    def __init__(self):
        self.history = []

    def add_user_input(self, user_input):
        """Add user input to the conversation history."""
        self.history.append({"role": "user", "content": user_input})

    def add_model_response(self, model_response):
        """Add model response to the conversation history."""
        self.history.append({"role": "assistant", "content": model_response})

    def get_history(self):
        """Return the entire conversation history."""
        return self.history

    def clear_history(self):
        """Clear the conversation history."""
        self.history = []