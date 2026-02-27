import time
from your_template_module import TemplateProvider  # Import your template provider

class LlmEngine:
    def __init__(self):
        self.providers = [OpenAIProvider(), ClaudeProvider(), TemplateProvider()]

    def run_query(self, query):
        for provider in self.providers:
            try:
                response = provider.send_query(query)
                return response
            except ApiQuotaExceededError:
                print("Quota exceeded for provider: {}. Moving to next provider.".format(provider.__class__.__name__))
                continue
            except ApiRateLimitError:
                print("Rate limit hit for provider: {}. Retrying...".format(provider.__class__.__name__))
                time.sleep(5)  # wait before retrying
                continue
            except Exception as e:
                print(f"Error with provider {provider.__class__.__name__}: {e}")
        raise Exception("All providers failed to handle the request.")  # If all providers fail

# Example usage
# engine = LlmEngine()
# response = engine.run_query("Your query here")