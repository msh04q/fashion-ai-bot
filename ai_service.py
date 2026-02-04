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
            },
            "openai_config": {
                "api_key": os.getenv("YANDEX_GPT_API_KEY"),
                "base_url": "https://llm.api.cloud.yandex.net/v1",
                "project": os.getenv("YANDEX_FOLDER_ID"),
                "model": lambda folder_id: f"gpt://{folder_id}/yandexgpt/latest"
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
        """Запрос к Yandex GPT через OpenAI-совместимый API"""
        try:
            # Добавляем форматирование для Yandex GPT
            if messages and messages[0].get("role") == "user":
                original_prompt = messages[0]["content"]
                formatted_prompt = f"""{original_prompt}

    Пожалуйста, ответь в четком формате:

    Название образа: [название]

    Описание: [2-3 предложения]

    Состав:
    - [элемент 1]
    - [элемент 2]
    - [элемент 3]
    - [элемент 4]
    - [элемент 5]

    Цвета: [цветовая схема]

    Советы:
    1. [совет 1]
    2. [совет 2]
    3. [совет 3]"""

                messages = [{"role": "user", "content": formatted_prompt}]

            return await self._make_yandex_openai_request(config, messages)
        except Exception as e:
            logger.warning(f"OpenAI способ не сработал: {e}")
            return await self._make_yandex_native_request(config, messages)

        async def _make_yandex_openai_request(self, config: Dict, messages: List[Dict]) -> Optional[str]:
            """Используем OpenAI-совместимый API Яндекса"""
            if not config.get("openai_config"):
                return None

            openai_config = config["openai_config"]

            yandex_client = openai.AsyncOpenAI(
                api_key=openai_config["api_key"],
                base_url=openai_config["base_url"],
                project=openai_config["project"]
            )

            model_path = openai_config["model"](config["folder_id"])

            try:
                response = await yandex_client.chat.completions.create(
                    model=model_path,
                    messages=messages,
                    max_tokens=config["max_tokens"],
                    temperature=config["temperature"]
                )

                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"Ошибка Yandex GPT (OpenAI API): {e}")
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
    
    async def try_all_providers(self, messages: List[Dict], system_prompt: str = None) -> Optional[str]:
        """Попробовать все доступные провайдеры по очереди"""
        
        for provider in self.available_providers:
            if provider in [AIProvider.LOCAL, AIProvider.FALLBACK]:
                continue
                
            logger.info(f"Пробуем {provider.value.upper()}...")
            result = await self._make_request(provider, messages, system_prompt)
            if result:
                logger.info(f"✅ {provider.value.upper()} успешно ответил")
                return result
            else:
                logger.warning(f"❌ {provider.value.upper()} не ответил")
        
        logger.info("Используем локальные шаблоны...")
        return None
    
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
                    # Берем следующую строку или часть после двоеточия
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
                    "🧥 Утепленный верх",
                    "👜 Стильные аксессуары",
                    "⌚ Дополнительные детали"
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

    async def analyze_wardrobe(self, wardrobe_description: str) -> Dict:
        """Анализ гардероба"""
        
        prompt = f"""
        Проанализируй гардероб и дай рекомендации:
        {wardrobe_description}
        """
        
        messages = [{"role": "user", "content": prompt}]
        
        ai_response = await self.try_all_providers(messages)
        
        if ai_response:
            return {
                "analysis": ai_response,
                "has_ai": True
            }
        else:
            if self.local_data_service:
                analysis = await self.local_data_service.get_local_wardrobe_advice(wardrobe_description)
                return {
                    "analysis": analysis,
                    "has_ai": False
                }
            return {
                "analysis": self._get_fallback_wardrobe_advice(wardrobe_description),
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
                    # Берем следующую строку или часть после двоеточия
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
                    "🧥 Утепленный верх",
                    "👜 Стильные аксессуары",
                    "⌚ Дополнительные детали"
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