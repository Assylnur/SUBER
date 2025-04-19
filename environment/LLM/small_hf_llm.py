# # environment/LLM/small_hf_llm.py

# from environment.LLM.llm import LLM
# from transformers import AutoTokenizer, AutoModelForCausalLM

# class SmallHuggingfaceLLM(LLM):
#     def __init__(self, model_name="sshleifer/tiny-gpt2", device="cpu"):
#         super().__init__(model_name)
#         self.device = device
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
#         self.model.eval()

#     def _generate(self, prompt):
#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
#         # outputs = self.model.generate(**inputs, max_new_tokens=10)
#         outputs = self.model.generate(
#                                         **inputs,
#                                         pad_token_id=self.tokenizer.eos_token_id,
#                                         max_new_tokens=10
#                                     )
#         text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return text

#     def request_rating_0_9(self, system_prompt, dialog):
#         prompt = self._dialog_to_text(dialog)
#         response = self._generate(prompt)
#         return prompt, self._extract_number(response, 0, 9)

#     def request_rating_1_10(self, system_prompt, dialog):
#         prompt = self._dialog_to_text(dialog)
#         response = self._generate(prompt)
#         return prompt, self._extract_number(response, 1, 10)

#     def request_rating_text(self, system_prompt, dialog):
#         prompt = self._dialog_to_text(dialog)
#         response = self._generate(prompt)
#         return prompt, response

#     def request_explanation(self, system_prompt, dialog):
#         prompt = self._dialog_to_text(dialog)
#         response = self._generate(prompt)
#         return prompt, response

#     def _dialog_to_text(self, dialog):
#         return "\n".join([f"{d['role'].capitalize()}: {d['content']}" for d in dialog])

#     def _extract_number(self, text, min_val, max_val):
#         import re
#         numbers = re.findall(r'\d+', text)
#         for num in numbers:
#             n = int(num)
#             if min_val <= n <= max_val:
#                 return str(n)
#         return str((min_val + max_val) // 2)  # fallback: mid value
