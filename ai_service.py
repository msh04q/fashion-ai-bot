import asyncio
import aiohttp
import json
import os
import logging
from typing import Dict, List, Optional
from enum import Enum
from dotenv import load_dotenv
import random

load_dotenv()

logger = logging.getLogger(__name__)

class AIProvider(Enum):
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"  # Google Gemini (бесплатный)
    YANDEX_GPT = "yandex_gpt"  # Yandex GPT (бесплатные запросы)
    FALLBACK = "fallback"
    LOCAL = "local"  # Локальные шаблоны

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
            if config["api_key"]:
                available.append(provider)
        # Всегда добавляем локальный провайдер как запасной
        available.append(AIProvider.LOCAL)
        available.append(AIProvider.FALLBACK)
        return available

class AIService:
    """Универсальный сервис для работы с разными AI API"""
    
    def __init__(self):
        self.available_providers = AIConfig.get_available_providers()
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
        # Gemini имеет другую структуру
        url = f"{config['base_url']}?key={config['api_key']}"
        
        # Преобразуем сообщения в формат Gemini
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
        """Запрос к Yandex GPT API"""
        # Yandex GPT также имеет другую структуру
        last_message = messages[-1]["content"] if messages else ""
        
        data = {
            "modelUri": f"gpt://{config['api_key']}/yandexgpt/latest",
            "completionOptions": {
                "stream": False,
                "temperature": config["temperature"],
                "maxTokens": config["max_tokens"]
            },
            "messages": [
                {
                    "role": "user",
                    "text": last_message
                }
            ]
        }
        
        timeout = aiohttp.ClientTimeout(total=15)
        headers = config["headers"](config["api_key"])
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(config["base_url"], headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    if "result" in result and "alternatives" in result["result"]:
                        return result["result"]["alternatives"][0]["message"]["text"]
                return None
    
    async def try_all_providers(self, messages: List[Dict], system_prompt: str = None) -> Optional[str]:
        """Попробовать все доступные провайдеры по очереди"""
        
        # Сначала пробуем платные провайдеры
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
        
        # Если ни один не ответил, используем локальные шаблоны
        logger.info("Используем локальные шаблоны...")
        return None
    
    async def generate_outfit(self, style: str, season: str, occasion: str, 
                             gender: str = "унисекс", budget: str = "средний") -> Dict:
        """Генерация образа одежды"""
        
        system_prompt = """Ты профессиональный стилист. Создавай практичные и модные образы."""
        
        prompt = f"""
        Создай модный образ:
        - Стиль: {style}
        - Сезон: {season}
        - Повод: {occasion}
        - Для: {gender}
        - Бюджет: {budget}
        
        Ответ должен включать:
        1. Название образа
        2. Краткое описание
        3. Список вещей
        4. Цветовую палитру
        5. 2-3 совета
        """
        
        messages = [{"role": "user", "content": prompt}]
        
        # Пробуем получить ответ от AI
        ai_response = await self.try_all_providers(messages, system_prompt)
        
        if ai_response:
            return self._parse_outfit_response(ai_response, style, season, occasion, has_ai=True)
        else:
            return self._get_local_outfit(style, season, occasion, gender, budget)
    
    async def get_color_advice(self, base_color: str, style: str = "повседневный") -> str:
        """Советы по сочетанию цветов"""
        
        prompt = f"""
        Дай рекомендации по сочетанию цвета: {base_color}
        Стиль: {style}
        """
        
        messages = [{"role": "user", "content": prompt}]
        
        ai_response = await self.try_all_providers(messages)
        
        if ai_response:
            return f"🎨 *AI анализ цвета {base_color}:*\n\n{ai_response}\n\n✨ *Совет:* Экспериментируйте!"
        else:
            return self._get_local_color_advice(base_color)
    
    async def get_trends(self) -> str:
        """Получение актуальных трендов"""
        from datetime import datetime
        current_date = datetime.now().strftime("%d %B %Y")
        
        prompt = f"""
        Предоставь актуальные модные тренды на {current_date}.
        Включи 5 основных трендов и советы.
        """
        
        messages = [{"role": "user", "content": prompt}]
        
        ai_response = await self.try_all_providers(messages)
        
        if ai_response:
            return f"🔥 *AI анализ трендов:*\n\n{ai_response}"
        else:
            return self._get_local_trends()
    
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
            return {
                "analysis": self._get_local_wardrobe_advice(wardrobe_description),
                "has_ai": False
            }
    
    def _get_local_outfit(self, style: str, season: str, occasion: str, gender: str, budget: str) -> Dict:
        """Локальные шаблоны образов"""
        
        # База данных образов
        outfits_database = [
            {
                "style": "кэжуал", "season": "зима", "occasion": "учёба", "gender": "мужской",
                "name": "Зимний академический кэжуал",
                "description": "Тёплый и удобный образ для учёбы в холодное время года",
                "items": [
                    "🧥 Утепленная куртка или пуховик",
                    "👕 Термобелье или футболка",
                    "🧥 Теплый свитер или худи",
                    "👖 Утепленные джинсы",
                    "👟 Зимние кроссовки или ботинки",
                    "🧣 Шарф и перчатки",
                    "🎒 Вместительный рюкзак",
                    "⌚ Умные часы"
                ],
                "colors": "⚫ Черный + ⚪ Серый + 🔵 Синий",
                "tips": [
                    "Слоёность — ключ к теплу и стилю",
                    "Выбирайте водонепроницаемую обувь"
                ],
                "budget_tips": {
                    "low": "Ищите скидки в масс-маркете",
                    "medium": "Инвестируйте в качественную обувь",
                    "high": "Рассмотрите премиальные бренды"
                }
            },
            {
                "style": "офисный", "season": "зима", "occasion": "работа", "gender": "унисекс",
                "name": "Зимний офисный шик",
                "description": "Элегантный и теплый образ для деловой среды",
                "items": [
                    "🧥 Шерстяное пальто или тренч",
                    "👔 Рубашка или блузка",
                    "🧥 Кашемировый джемпер",
                    "👖 Классические брюки или юбка",
                    "👞 Кожаные туфли или ботильоны",
                    "⌚ Классические часы",
                    "💼 Кожаный портфель",
                    "🧣 Кашемировый шарф"
                ],
                "colors": "⚫ Черный + ⚪ Белый + 🟤 Бежевый",
                "tips": [
                    "Инвестируйте в качественное пальто",
                    "Нейтральные цвета всегда в тренде"
                ]
            },
            {
                "style": "спортивный", "season": "зима", "occasion": "прогулка", "gender": "унисекс",
                "name": "Активная зимняя прогулка",
                "description": "Спортивный и функциональный образ для активного отдыха",
                "items": [
                    "🧥 Спортивная куртка с мембраной",
                    "👕 Термофутболка",
                    "🧥 Флисовая кофта",
                    "👖 Спортивные штаны или леггинсы",
                    "👟 Трекинговые ботинки",
                    "🧤 Термоперчатки",
                    "🎒 Рюкзак для гидратации",
                    "🧢 Шапка из флиса"
                ],
                "colors": "🟢 Темно-зеленый + ⚫ Черный + 🟠 Оранжевый",
                "tips": [
                    "Выбирайте одежду с мембраной",
                    "Яркие цвета для безопасности"
                ]
            },
            {
                "style": "вечерний", "season": "зима", "occasion": "вечеринка", "gender": "унисекс",
                "name": "Зимний вечерний гламур",
                "description": "Элегантный и теплый образ для вечерних мероприятий",
                "items": [
                    "🧥 Меховое пальто или накидка",
                    "👗 Вечернее платье или 👔 Костюм",
                    "✨ Блестящие аксессуары",
                    "👠 Каблуки или 👞 Лоферы",
                    "💎 Минималистичная бижутерия",
                    "👜 Клатч на цепочке",
                    "🧣 Шелковый шарф",
                    "💍 Кольца"
                ],
                "colors": "⚫ Черный + ✨ Золотой + 🔴 Бордовый",
                "tips": [
                    "Сочетайте роскошь и комфорт",
                    "Утепляйтесь стильно"
                ]
            }
        ]
        
        # Поиск подходящего образа
        for outfit in outfits_database:
            if (outfit["style"] in style.lower() and 
                outfit["season"] in season.lower() and 
                outfit["occasion"] in occasion.lower() and
                (outfit["gender"] == "унисекс" or outfit["gender"] in gender.lower())):
                
                # Адаптируем под бюджет
                budget_tip = outfit.get("budget_tips", {}).get(budget.lower(), "")
                if budget_tip:
                    outfit["tips"].append(f"Бюджет: {budget_tip}")
                
                outfit["has_ai"] = False
                return outfit
        
        # Если не нашли, создаем общий образ
        return {
            "name": f"{style} образ для {occasion}",
            "description": f"Стильный {style.lower()} образ для {occasion.lower()} в {season.lower()} время года",
            "items": [
                f"👕 Верхняя одежда в стиле {style}",
                f"👖 Нижняя часть для {season}",
                "👟 Соответствующая обувь",
                "🧥 Утепленный верх (если холодно)",
                "👜 Практичные аксессуары",
                "🧣 Сезонные аксессуары"
            ],
            "colors": "🌈 Гармонирующие цвета по сезону",
            "tips": [
                "Учитывайте погодные условия",
                "Сочетайте комфорт и стиль",
                f"Бюджет: {budget}"
            ],
            "has_ai": False
        }
    
    def _get_local_color_advice(self, base_color: str) -> str:
        """Локальные советы по цветам"""
        
        color_database = {
            "синий": {
                "psychology": "Спокойствие, уверенность, надежность",
                "combinations": [
                    "🔵 + ⚪ Белый — свежо и чисто",
                    "🔵 + 🟡 Бежевый — элегантно",
                    "🔵 + 🔴 Красный — смело и стильно",
                    "🔵 + 🟡 Желтый — ярко и оптимистично",
                    "🔵 + ⚫ Черный — классически"
                ],
                "materials": "Хлопок, деним, шерсть, шелк",
                "seasons": "Всесезонный, особенно хорош осенью и зимой"
            },
            "черный": {
                "psychology": "Элегантность, сила, таинственность",
                "combinations": [
                    "⚫ + ⚪ Белый — вечная классика",
                    "⚫ + 🔴 Красный — драматично",
                    "⚫ + 🟤 Коричневый — стильно",
                    "⚫ + ✨ Золотой — роскошно",
                    "⚫ + любой цвет — безопасно"
                ],
                "materials": "Кожа, шерсть, хлопок, шифон",
                "seasons": "Всесезонный, базовый цвет"
            },
            "белый": {
                "psychology": "Чистота, свежесть, минимализм",
                "combinations": [
                    "⚪ + ⚫ Черный — контрастно",
                    "⚪ + 🔵 Синий — морская тема",
                    "⚪ + 🟢 Зеленый — натурально",
                    "⚪ + любой яркий цвет — летне",
                    "⚪ + ⚪ Белый — total look"
                ],
                "materials": "Лен, хлопок, шифон, кружево",
                "seasons": "Лето, весна, свадьбы"
            },
            "красный": {
                "psychology": "Страсть, энергия, уверенность",
                "combinations": [
                    "🔴 + ⚫ Черный — дерзко",
                    "🔴 + ⚪ Белый — свежо",
                    "🔴 + 🔵 Деним — повседневно",
                    "🔴 + 🟤 Коричневый — винтажно",
                    "🔴 + ✨ Золотой — празднично"
                ],
                "materials": "Шелк, бархат, шерсть",
                "seasons": "Зима (новый год), всегда для акцентов"
            },
            "зеленый": {
                "psychology": "Гармония, природа, рост",
                "combinations": [
                    "🟢 + 🟤 Коричневый — земляные тона",
                    "🟢 + ⚪ Белый — свежо",
                    "🟢 + 🟡 Бежевый — спокойно",
                    "🟢 + 🔵 Синий — контрастно",
                    "🟢 + 🌸 Розовый — модно"
                ],
                "materials": "Хлопок, лен, вельвет",
                "seasons": "Весна, осень, лето"
            }
        }
        
        color_info = color_database.get(base_color.lower())
        
        if color_info:
            response = f"""
🎨 *Цвет {base_color.capitalize()}*

*Психология:* {color_info['psychology']}

*Лучшие сочетания:*
{chr(10).join(color_info['combinations'])}

*Подходящие материалы:* {color_info['materials']}

*Сезонность:* {color_info['seasons']}

💡 *Совет:* Используйте {base_color} как базовый или акцентный цвет!
            """
        else:
            response = f"""
🎨 *Цвет {base_color.capitalize()}*

*Основные сочетания:*
• {base_color} + Белый — свежо
• {base_color} + Черный — контрастно  
• {base_color} + Нейтральные — безопасно
• {base_color} + Контрастный — смело

💡 *Совет:* Экспериментируйте с оттенками {base_color}!
            """
        
        return response
    
    def _get_local_trends(self) -> str:
        """Локальные тренды"""
        from datetime import datetime
        
        trends = [
            {
                "name": "Oversize Silhouettes",
                "description": "Свободный крой во всем — от пиджаков до брюк",
                "tip": "Сочетайте с облегающими вещами для баланса"
            },
            {
                "name": "Colorful Accessories",
                "description": "Яркие сумки, обувь и украшения как акцент",
                "tip": "Один яркий аксессуар на образ достаточно"
            },
            {
                "name": "Sporty Chic",
                "description": "Кроссовки с классикой, спортивные элементы",
                "tip": "Качественные кроссовки — must-have"
            },
            {
                "name": "Sustainable Fashion",
                "description": "Экологичные материалы и винтаж",
                "tip": "Покупайте качественное, а не много"
            },
            {
                "name": "Layering",
                "description": "Искусство многослойности в одежде",
                "tip": "Сочетайте разные текстуры и длины"
            }
        ]
        
        trend_list = "\n".join([
            f"🔥 *{t['name']}*\n{t['description']}\n💡 {t['tip']}\n"
            for t in trends
        ])
        
        return f"""
🔥 *Актуальные тренды {datetime.now().strftime('%d.%m.%Y')}*

{trend_list}

🎯 *Образ дня:*
• Oversize блейзер
• Базовая футболка
• Прямые джинсы
• Белые кроссовки
• Цветная сумка

✨ *Совет:* Выбирайте тренды, которые подходят вашему стилю!
        """
    
    def _get_local_wardrobe_advice(self, description: str) -> str:
        """Локальные советы по гардеробу"""
        
        advice = """
🧳 *Анализ вашего гардероба*

*Общие рекомендации:*

1. *Базовые вещи:*
   • Белая и черная футболки
   • Классические джинсы
   • Универсальное платье/рубашка
   • Удобные кроссовки

2. *Цветовая палитра:*
   • 70% нейтральные цвета
   • 20% основные цвета гардероба  
   • 10% яркие акценты

3. *Сочетания:*
   • Экспериментируйте с layering
   • Сочетайте дорогое и доступное
   • Не бойтесь микса стилей

4. *Что докупить:*
   • Качественное пальто/куртку
   • Универсальную сумку
   • Аксессуары для деталей

💎 *Главный совет:* Создавайте капсульный гардероб!
        """
        
        return advice
    
    def _parse_outfit_response(self, response: str, style: str, season: str, occasion: str, has_ai: bool = True) -> Dict:
        """Парсинг ответа от AI (упрощенный)"""
        return {
            "name": f"{style} образ для {occasion}",
            "description": response[:200] + "..." if len(response) > 200 else response,
            "items": [
                "👕 Верхняя одежда",
                "👖 Нижняя часть",
                "👟 Обувь",
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

# Создаем глобальный экземпляр
ai_service = AIService()