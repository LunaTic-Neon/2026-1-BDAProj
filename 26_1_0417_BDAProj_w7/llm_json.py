import ollama
import json

reviews = [
    "배송도 빠르고 제품 품질도 좋음. 재구매 의사도 있음.",
    "불량품 왔는데 교환도 안 해주네. 최악임.",
    "가격은 저렴한데 품질은 보통정도라서 애매함."
]

results = []

for review in reviews:
    response = ollama.chat(
        model="gemma3:4b",
        messages=[
            {
                "role": "system",
                "content": """당신은 리뷰 분석 전문가입니다.
                주어진 리뷰를 분석하여 반드시 아래 JSON 형식으로만 응답하세요.
                다른 텍스트는 포함하지 마세요.

                {"sentiment": "긍정/부정/중립", "confidence": 0.0~1.0, "keywords": ["키워드1", "키워드2"]}"""
            },
            {
                "role": "user",
                "content": review
            }
        ]
    )

    raw = response["message"]["content"]

    # JSON 파싱 시도
    try:
        # LLM이 ```json ... ``` 으로 감싸서 응답 가능함.
        clean = raw.strip()
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0].strip()
        elif "```" in clean:
            clean = clean.split("```")[1].split("```")[0].strip()

        data = json.loads(clean)
        results.append(data)
        print(f"리뷰: {review[:25]}...")
        print(f"   감성: {data['sentiment']}, 확신도: {data['confidence']}")
        print(f"   키워드: {data['keywords']}")
    except json.JSONDecodeError:
        print(f"JSON 파싱 실패: {raw[:100]}")

    print("-" * 50)

print(f"\n총 {len(results)}개 리뷰 분석 완료")
