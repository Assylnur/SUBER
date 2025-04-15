# environment/LLM/dummy_llm.py
from environment.LLM.llm import LLM

class DummyLLM(LLM):
    def __init__(self):
        super().__init__(name="dummy")

    def request_rating_0_9(self, system_prompt, dialog):
        return "", "6.5"

    def request_rating_1_10(self, system_prompt, dialog):
        return "", "8.0"

    def request_rating_1_5(self, system_prompt, dialog):
        return "", "4"

    def request_rating_text(self, system_prompt, dialog):
        return "", "seven"

    def request_explanation(self, system_prompt, dialog):
        return "", "This is a dummy explanation."
