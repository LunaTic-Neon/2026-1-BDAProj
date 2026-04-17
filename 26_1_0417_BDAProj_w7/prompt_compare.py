import ollama

# 나쁜 프롬프트: 모호하고 불명확
bad_prompt = "인간과 AI 차이점이 뭘까?"

# 좋은 프롬프트: 구체적이고 명확
good_prompt = """당신은 인간과 AI에 대해 연구하는 학자입니다.
AI에 관점에서 본 인간과 AI의 차이점을 알려주세요.

조건:
- 3가지 차이점을 각각 한 문장으로 설명
- 비유를 활용하여 쉽게 설명
- 마지막에 한 줄 요약 추가
"""

print("=" * 20)
print("나쁜 프롬프트")
print("=" * 20)
response = ollama.generate(model="gemma3:4b", prompt=bad_prompt)
print(response["response"][:300])  # 앞 300자만

print("\n" + "=" * 20)
print("좋은 프롬프트")
print("=" * 20)
response = ollama.generate(model="gemma3:4b", prompt=good_prompt)
print(response["response"])
