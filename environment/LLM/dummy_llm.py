# environment/LLM/dummy_llm.py
from environment.LLM.llm import LLM
import numpy as np
class DummyLLM(LLM):
    def __init__(self):
        super().__init__(name="dummy")
        self.previous_items_features_list = None
    def request_rating_0_9(self, system_prompt, dialog):
        return "", f"{np.random.randint(0, 10)}"

    def request_rating_1_10(self, system_prompt, dialog):
        return "", f"{np.random.randint(1, 11)}"

    def request_rating_1_5(self, system_prompt, dialog):
        return "", f"{np.random.randint(1, 6)}"

    def request_rating_text(self, system_prompt, dialog):
        return "", "seven"

    def request_explanation(self, system_prompt, dialog):
        return "", "This is a dummy explanation."
