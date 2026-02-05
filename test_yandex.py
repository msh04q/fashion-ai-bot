import asyncio
import os
from dotenv import load_dotenv
import openai

load_dotenv()

async def test_yandex():
    print("🔍 Тестируем Yandex GPT...")
    
    # Настройки как в AIConfig
    api_key = os.getenv("YANDEX_GPT_API_KEY")
    folder_id = os.getenv("YANDEX_FOLDER_ID")
    
    if not api_key or not folder_id:
        print("❌ Ключи не найдены!")
        return
    
    print(f"✅ Ключ: {api_key[:10]}...")
    print(f"✅ Папка: {folder_id}")
    
    client = openai.AsyncOpenAI(
        api_key=api_key,
        base_url="https://llm.api.cloud.yandex.net/v1",
        project=folder_id
    )
    
    model_path = f"gpt://{folder_id}/yandexgpt/latest"
    
    try:
        print("📞 Делаем запрос к Yandex GPT...")
        response = await client.chat.completions.create(
            model=model_path,
            messages=[
                {"role": "user", "content": "Привет! Как дела?"}
            ],
            max_tokens=100,
            temperature=0.7
        )
        
        print(f"✅ Ответ получен!")
        print(f"Текст: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")

asyncio.run(test_yandex())