from ollama import chat


class Ollama:
    """Thin wrapper around a locally running Ollama server, used for the
    Code-LLM baselines (e.g. Llama-3 / Llama-4) in the experiments.

    Requires the ``ollama`` Python package and a running Ollama instance with
    the requested model pulled, e.g. ``ollama pull llama3``. See
    https://ollama.com for installation instructions and available model tags.
    """

    def __init__(self, model_name='llama3'):
        self.model_name = model_name

    def recommend(self, prompt):
        """Return the model's free-text completion for ``prompt``."""
        response = chat(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}],
        )
        return response.message.content
