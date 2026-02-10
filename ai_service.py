import asyncio
import aiohttp
import json
import os
import logging
from typing import Dict, List, Optional
from enum import Enum
from dotenv import load_dotenv
import openai

load_dotenv()

logger = logging.getLogger(__name__)

class AIProvider(Enum):
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"
    YANDEX_GPT = "yandex_gpt"
    FALLBACK = "fallback"
    LOCAL = "local"

class AIConfig:
    """Конфигурация AI провайдеров"""
    
    PROVIDERS = {
        AIProvider.OPENAI: {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": "https://api.openai.com/v1/chat/completions",
            "model": "gpt-3.5-turbo",
            "max_tokens": 1500,
            "temperature": 0.7,
            "headers": lambda api_key: {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        },
        AIProvider.DEEPSEEK: {
            "api_key": os.getenv("DEEPSEEK_API_KEY"),
            "base_url": "https://api.deepseek.com/v1/chat/completions",
            "model": "deepseek-chat",
            "max_tokens": 2000,
            "temperature": 0.7,
            "headers": lambda api_key: {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        },
        AIProvider.GEMINI: {
            "api_key": os.getenv("GEMINI_API_KEY"),
            "base_url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
            "model": "gemini-pro",
            "max_tokens": 1000,
            "temperature": 0.7,
            "headers": lambda api_key: {}
        },
        AIProvider.YANDEX_GPT: {
            "api_key": os.getenv("YANDEX_GPT_API_KEY"),
            "folder_id": os.getenv("YANDEX_FOLDER_ID"),
            "base_url": "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
            "model": "yandexgpt",
            "max_tokens": 1000,
            "temperature": 0.7,
            "headers": lambda api_key: {
                "Authorization": f"Api-Key {api_key}",
                "Content-Type": "application/json"
            }
        }
    }
    
    @classmethod
    def get_available_providers(cls):
        """Получить доступные провайдеры"""
        available = []
        for provider, config in cls.PROVIDERS.items():
            if provider == AIProvider.YANDEX_GPT:
                if config["api_key"] and config["folder_id"]:
                    available.append(provider)
            elif config["api_key"]:
                available.append(provider)
        
        available.append(AIProvider.LOCAL)
        available.append(AIProvider.FALLBACK)
        return available

class AIService:
    """Универсальный сервис для работы с разными AI API"""
    
    def __init__(self, local_data_service=None):
        self.available_providers = AIConfig.get_available_providers()
        self.local_data_service = local_data_service
        logger.info(f"Доступные AI провайдеры: {[p.value for p in self.available_providers]}")
    
    async def try_all_providers(self, messages: List[Dict], system_prompt: str = None) -> Optional[str]:
        """Попробовать все доступные провайдеры по очереди"""

        for provider in self.available_providers:
            if provider in [AIProvider.LOCAL, AIProvider.FALLBACK]:
                continue

            logger.info(f"Пробуем {provider.value.upper()}...")
            result = await self._make_request(provider, messages, system_prompt)
            if result and result.strip():
                logger.info(f"✅ {provider.value.upper()} успешно ответил")
                return result
            else:
                logger.warning(f"❌ {provider.value.upper()} не ответил")

        logger.info("Все провайдеры не ответили, используем локальные шаблоны...")
        return None
    
    async def _make_request(self, provider: AIProvider, messages: List[Dict], system_prompt: str = None) -> Optional[str]:
        """Общий метод для запросов к AI API"""
        
        if provider in [AIProvider.FALLBACK, AIProvider.LOCAL]:
            return None
        
        config = AIConfig.PROVIDERS[provider]
        
        try:
            if provider == AIProvider.GEMINI:
                return await self._make_gemini_request(config, messages)
            elif provider == AIProvider.YANDEX_GPT:
                return await self._make_yandex_request(config, messages)
            else:
                return await self._make_openai_compatible_request(config, messages, system_prompt)
                
        except Exception as e:
            logger.error(f"Ошибка при запросе к {provider.value}: {e}")
            return None
    
    async def _make_openai_compatible_request(self, config: Dict, messages: List[Dict], system_prompt: str) -> Optional[str]:
        """Запрос к OpenAI-совместимым API"""
        all_messages = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        all_messages.extend(messages)
        
        data = {
            "model": config["model"],
            "messages": all_messages,
            "temperature": config["temperature"],
            "max_tokens": config["max_tokens"]
        }
        
        timeout = aiohttp.ClientTimeout(total=15)
        headers = config["headers"](config["api_key"])
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                config["base_url"], 
                headers=headers, 
                json=data
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    return result['choices'][0]['message']['content']
                else:
                    error_text = await response.text()
                    logger.error(f"{config['model']} API Error {response.status}: {error_text[:200]}")
                    return None
    
    async def _make_gemini_request(self, config: Dict, messages: List[Dict]) -> Optional[str]:
        """Запрос к Google Gemini API"""
        url = f"{config['base_url']}?key={config['api_key']}"
        
        gemini_messages = []
        for msg in messages:
            if msg["role"] == "user":
                gemini_messages.append({"parts": [{"text": msg["content"]}]})
        
        data = {
            "contents": gemini_messages,
            "generationConfig": {
                "temperature": config["temperature"],
                "maxOutputTokens": config["max_tokens"]
            }
        }
        
        timeout = aiohttp.ClientTimeout(total=15)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    if "candidates" in result and len(result["candidates"]) > 0:
                        return result["candidates"][0]["content"]["parts"][0]["text"]
                return None
    
    async def _make_yandex_request(self, config: Dict, messages: List[Dict]) -> Optional[str]:
        """Запрос к Yandex GPT"""
        try:
            # Сначала пробуем через OpenAI-совместимый API
            result = await self._make_yandex_openai_request(config, messages)
            if result:
                return result
        except Exception as e:
            logger.warning(f"OpenAI способ не сработал: {e}")
        
        # Если не сработало, пробуем нативный API
        return await self._make_yandex_native_request(config, messages)

    async def _make_yandex_openai_request(self, config: Dict, messages: List[Dict]) -> Optional[str]:
        """Используем OpenAI-совместимый API Яндекса - ОБХОДИМ ОШИБКУ PROXIES"""
        
        api_key = config.get("api_key")
        folder_id = config.get("folder_id")
        
        if not api_key or not folder_id:
            logger.warning("Нет ключа или folder_id для Yandex GPT")
            return None
        
        try:
            # СПЕЦИАЛЬНЫЙ ПАТЧ: создаём клиент БЕЗ лишних параметров
            import httpx
            from openai import AsyncOpenAI
            
            # Создаём чистый HTTP-клиент
            http_client = httpx.AsyncClient(
                timeout=15.0,
                limits=httpx.Limits(max_connections=5, max_keepalive_connections=5)
            )
            
            # Создаём OpenAI клиент с нашим HTTP-клиентом
            yandex_client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://llm.api.cloud.yandex.net/v1",
                http_client=http_client  # Передаём свой клиент, чтобы контролировать параметры
            )

            # Папка передаётся в model
            model_path = f"gpt://{folder_id}/yandexgpt/latest"
            
            logger.info(f"🔍 [YANDEX] Отправляем запрос с model={model_path}")
            
            response = await yandex_client.chat.completions.create(
                model=model_path,
                messages=messages,
                max_tokens=config.get("max_tokens", 1000),
                temperature=config.get("temperature", 0.7)
            )

            if response and response.choices:
                content = response.choices[0].message.content
                logger.info(f"✅ [YANDEX] Успешный ответ! Длина: {len(content)}")
                return content
            else:
                logger.warning("🔍 [YANDEX] Пустой ответ от API")
                return None
                
        except Exception as e:
            logger.error(f"❌ [YANDEX] Ошибка: {e}")
            return None
        
    async def _make_yandex_native_request(self, config: Dict, messages: List[Dict]) -> Optional[str]:
        """Старый способ через native API (резервный)"""
        formatted_messages = []
        for msg in messages:
            role_map = {
                "system": "system",
                "user": "user",
                "assistant": "assistant"
            }
            formatted_messages.append({
                "role": role_map.get(msg["role"], "user"),
                "text": msg["content"]
            })

        data = {
            "modelUri": f"gpt://{config['folder_id']}/{config['model']}/latest",
            "completionOptions": {
                "stream": False,
                "temperature": config["temperature"],
                "maxTokens": config["max_tokens"]
            },
            "messages": formatted_messages
        }

        timeout = aiohttp.ClientTimeout(total=15)
        headers = config["headers"](config["api_key"])

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    config["base_url"],
                    headers=headers, 
                    json=data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "result" in result and "alternatives" in result["result"]:
                            return result["result"]["alternatives"][0]["message"]["text"]
                    else:
                        error_text = await response.text()
                        logger.error(f"Yandex GPT API Error {response.status}: {error_text[:200]}")
                    return None
        except Exception as e:
            logger.error(f"Ошибка соединения с Yandex GPT: {e}")
            return None

    async def generate_outfit(self, style: str, season: str, occasion: str, gender: str, budget: str = None) -> Dict:
        """Генерация образа одежды"""
        
        # Форматируем промпт для лучшего ответа
        prompt = f"""Ты профессиональный стилист. Создай {style} образ для {occasion} в сезон {season}.
Для: {gender}.
Бюджет: {budget or 'не важен'}.

ОБЯЗАТЕЛЬНО укажи КОНКРЕТНЫЕ названия вещей, а не общие фразы.

ПРИМЕР ПРАВИЛЬНОГО ОТВЕТА:
Название образа: Вечерний кэжуал

Описание: Уютный образ для осеннего ужина...

Состав:
- Приталенная рубашка из хлопка
- Темные джинсы скинни
- Кожаные кеды
- Кожаная куртка-бомбер
- Серебряный браслет

ТВОЙ ОТВЕТ ДОЛЖЕН БЫТЬ ТОЛЬКО В ЭТОМ ФОРМАТЕ:

Название образа: [конкретное название]

Описание: [2-3 предложения]

Состав:
- [конкретная вещь 1]
- [конкретная вещь 2] 
- [конкретная вещь 3]
- [конкретная вещь 4]
- [конкретная вещь 5]

Цвета: [конкретные цвета]

Советы:
1. [конкретный совет 1]
2. [конкретный совет 2]
3. [конкретный совет 3]"""
        
        messages = [{"role": "user", "content": prompt}]
        
        logger.info(f"🤖 Генерируем образ: {style}, {season}, {occasion}, {gender}")
        
        # Пробуем получить ответ от AI
        ai_response = await self.try_all_providers(messages)
        
        if ai_response and ai_response.strip():
            logger.info(f"✅ AI ответ получен, длина: {len(ai_response)}")
            return self._parse_outfit_response(
                response=ai_response,
                style=style,
                season=season,
                occasion=occasion,
                has_ai=True
            )
        else:
            logger.info("⚠️ AI не ответил, используем локальные шаблоны")
            # Локальные шаблоны (запасной вариант)
            return self._get_fallback_outfit(style, season, occasion, gender)
    
    def _get_fallback_outfit(self, style: str, season: str, occasion: str, gender: str) -> Dict:
        """Запасной вариант образа (локальные шаблоны)"""
        return {
            "name": f"{style} образ для {occasion}",
            "description": f"Стильный {style} образ на {season} для {occasion}. Подходит для {gender}.",
            "items": [
                f"👕 Верхняя одежда в стиле {style}",
                f"👖 Нижняя часть для {season}",
                f"👟 Обувь для {occasion}",
                "🧥 Утепленный верх" if season in ["Зима", "Осень"] else "🕶️ Солнцезащитные аксессуары",
                "👜 Стильные аксессуары",
                "⌚ Часы или браслет"
            ],
            "colors": "Нейтральные тона с акцентами",
            "tips": [
                f"Учитывайте погоду: {season}",
                f"Соответствуйте поводу: {occasion}",
                "Добавляйте индивидуальные детали"
            ],
            "has_ai": False
        }

    async def get_color_advice(self, color: str) -> str:
        """Получение советов по цветам"""
        prompt = f"""Дай советы по сочетанию цвета {color} в одежде.
        Включи:
        1. Какие цвета сочетаются с {color}
        2. Какие стили подходят
        3. Практические советы
        4. Чего избегать
        
        Ответь в формате Markdown."""
        
        messages = [{"role": "user", "content": prompt}]
        
        ai_response = await self.try_all_providers(messages)
        
        if ai_response:
            return ai_response
        else:
            return f"""🎨 **Советы по цвету {color}:**

**Сочетания:**
• {color} + белый — свежо и чисто
• {color} + черный — элегантно и контрастно
• {color} + бежевый/коричневый — натурально
• {color} + нейтральные оттенки — безопасно

**Стили:**
• Повседневный стиль
• Деловой образ
• Вечерний наряд

**Советы:**
1. Начните с аксессуаров цвета {color}
2. Используйте {color} как акцентный цвет
3. Сочетайте с принтами, содержащими {color}

**Избегайте:**
• Слишком ярких контрастов
• Более 3 цветов в одном образе"""

    async def get_trends(self) -> str:
        """Получение модных трендов"""
        prompt = """Расскажи о текущих модных трендах в одежде.
        Включи:
        1. Популярные стили
        2. Актуальные цвета
        3. Модные аксессуары
        4. Советы по внедрению трендов
        
        Ответь в формате Markdown."""
        
        messages = [{"role": "user", "content": prompt}]
        
        ai_response = await self.try_all_providers(messages)
        
        if ai_response:
            return ai_response
        else:
            return """🔥 **Актуальные модные тренды:**

**Популярные стили:**
• Athleisure (спортивный шик)
• Минимализм
• Устойчивая мода
• Винтажные элементы

**Актуальные цвета:**
• Земляные тона
• Пастельные оттенки
• Яркие акценты
• Металлики

**Модные аксессуары:**
• Объемные сумки
• Широкие ремни
• Стильные солнцезащитные очки
• Слоёные украшения

**Советы:**
1. Начните с одного трендового элемента
2. Сочетайте тренды с базовыми вещами
3. Выбирайте то, что подходит вашему стилю"""

    async def analyze_wardrobe(self, wardrobe_description: str) -> Dict:
        """Анализ гардероба"""
        
        prompt = f"""Проанализируй гардероб и дай рекомендации:
        {wardrobe_description}
        
        Включи в анализ:
        1. Сильные стороны гардероба
        2. Чего не хватает
        3. Какие вещи можно комбинировать
        4. Практические советы по обновлению
        
        Ответь в формате Markdown."""
        
        messages = [{"role": "user", "content": prompt}]
        
        ai_response = await self.try_all_providers(messages)
        
        if ai_response:
            return {
                "analysis": ai_response,
                "has_ai": True
            }
        else:
            if self.local_data_service:
                try:
                    analysis = await self.local_data_service.get_local_wardrobe_advice(wardrobe_description)
                    return {
                        "analysis": analysis,
                        "has_ai": False
                    }
                except:
                    pass
            
            # Фолбэк вариант
            fallback_advice = f"""🧳 **Анализ вашего гардероба:**

**Основные вещи:** {wardrobe_description[:100]}...

**Рекомендации:**
1. Добавьте базовые вещи (белая футболка, джинсы)
2. Включите аксессуары для разнообразия
3. Создайте капсульный гардероб

**Что докупить:**
• Универсальную верхнюю одежду
• Обувь на разные случаи
• Аксессуары для акцентов

💡 **Совет:** Сочетайте вещи по принципу "1 вещь = 3 образа"."""
            
            return {
                "analysis": fallback_advice,
                "has_ai": False
            }

    def _parse_outfit_response(self, response: str, style: str, season: str, occasion: str, has_ai: bool = True) -> Dict:
        """Парсинг ответа от AI"""
        try:
            # Пытаемся извлечь структурированные данные из ответа AI
            lines = response.strip().split('\n')
        
            # Ищем название образа
            name = f"{style} образ для {occasion}"
            for i, line in enumerate(lines):
                if 'название' in line.lower() or 'образ:' in line.lower():
                    name = line.split(':')[-1].strip().strip('«»""')
                    break
        
            # Ищем описание
            description = ""
            for i, line in enumerate(lines):
                if 'описание' in line.lower() or 'краткое' in line.lower():
                    if ':' in line:
                        description = line.split(':')[-1].strip()
                    elif i + 1 < len(lines):
                        description = lines[i + 1].strip()
                    break
        
            if not description:
                description = response[:200] + "..." if len(response) > 200 else response
        
            # Ищем список вещей
            items = []
            in_items_section = False
            for line in lines:
                if 'список' in line.lower() or 'вещи' in line.lower() or 'состав' in line.lower():
                    in_items_section = True
                    continue
            
                if in_items_section:
                    if line.strip() and (line.strip().startswith('-') or line.strip().startswith('•')):
                        item = line.strip().lstrip('-• ').strip()
                        if item:
                            items.append(f"👕 {item}")
                    elif 'цвет' in line.lower() or 'совет' in line.lower():
                        break
        
            # Если не нашли структурированный список, создаем базовый
            if not items:
                items = [
                    f"👕 Верхняя одежда в стиле {style}",
                    f"👖 Нижняя часть для {season}",
                    f"👟 Соответствующая обувь",
                    "🧥 Утепленный верх" if season in ["Зима", "Осень"] else "🕶️ Солнцезащитные аксессуары",
                    "👜 Стильные аксессуары"
                ]
        
            # Ищем цветовую палитру
            colors = "Гармонирующая палитра"
            for line in lines:
                if 'цвет' in line.lower() and ('палитр' in line.lower() or 'гамм' in line.lower()):
                    if ':' in line:
                        colors = line.split(':')[-1].strip()
                    else:
                        colors = line.strip()
                    break
        
            # Ищем советы
            tips = []
            in_tips_section = False
            for line in lines:
                if 'совет' in line.lower() or 'рекомендац' in line.lower():
                    in_tips_section = True
                    continue
                    
                if in_tips_section:
                    if line.strip() and (line.strip().startswith('-') or line.strip().startswith('•') or line.strip().startswith('1.') or line.strip().startswith('2.')):
                        tip = line.strip().lstrip('-•123456789. ').strip()
                        if tip and len(tips) < 3:
                            tips.append(tip)
                    elif line.strip() and len(tips) < 3 and len(line.strip()) < 100:
                        tips.append(line.strip())
        
            if not tips:
                tips = [
                    "Сочетайте комфорт и стиль",
                    f"Учитывайте погодные условия: {season}",
                    "Добавляйте индивидуальные акценты"
                ]
        
            return {
                "name": name,
                "description": description,
                "items": items[:8],  # Ограничиваем количество элементов
                "colors": colors,
                "tips": tips[:3],    # Ограничиваем количество советов
                "has_ai": has_ai
            }
        
        except Exception as e:
            logger.error(f"Ошибка парсинга ответа AI: {e}")
            # Fallback на упрощенный вариант
            return {
                "name": f"{style} образ для {occasion}",
                "description": response[:200] + "..." if len(response) > 200 else response,
                "items": [
                    f"👕 Верхняя одежда в стиле {style}",
                    f"👖 Нижняя часть для {season}",
                    "👟 Соответствующая обувь",
                    "🧥 Верхний слой",
                    "👜 Аксессуары"
                ],
                "colors": "Гармонирующая палитра",
                "tips": [
                    "Сочетайте комфорт и стиль",
                    "Учитывайте погодные условия"
                ],
                "has_ai": has_ai
            }

# Простой тест
if __name__ == "__main__":
    async def test():
        service = AIService()
        
        print("🧪 Тестируем AIService...")
        
        # Тест генерации образа
        outfit = await service.generate_outfit(
            style="Кэжуал",
            season="Осень",
            occasion="Вечеринка",
            gender="Унисекс"
        )
        
        print(f"✅ Образ создан!")
        print(f"Название: {outfit['name']}")
        print(f"has_ai: {outfit['has_ai']}")
        print(f"Описание: {outfit['description'][:100]}...")
        
        # Тест цветовых советов
        colors = await service.get_color_advice("синий")
        print(f"\n🎨 Цветовые советы получены (длина: {len(colors)})")
        
        # Тест трендов
        trends = await service.get_trends()
        print(f"\n🔥 Тренды получены (длина: {len(trends)})")
    
    asyncio.run(test())