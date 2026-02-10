import asyncio
import logging
import os
from datetime import datetime
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    ReplyKeyboardMarkup, 
    KeyboardButton, 
    InlineKeyboardMarkup, 
    InlineKeyboardButton,
    CallbackQuery
)
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from dotenv import load_dotenv

# Импортируем наши сервисы
from ai_service import AIService
from local_data_service import LocalDataService

# Загружаем переменные окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Получаем токен
TOKEN = os.getenv("TELEGRAM_TOKEN")

if not TOKEN:
    logger.error("❌ TELEGRAM_TOKEN не найден в .env файле!")
    exit(1)

logger.info("✅ Токен загружен")

# Инициализация AI сервиса с локальными данными
local_service = LocalDataService()
ai_service = AIService(local_data_service=local_service)

# Инициализация бота
bot = Bot(token=TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

# Состояния FSM
class FashionStates(StatesGroup):
    waiting_style = State()
    waiting_season = State()
    waiting_occasion = State()
    waiting_gender = State()
    waiting_budget = State()
    waiting_color = State()
    waiting_wardrobe = State()

# ========== КЛАВИАТУРЫ ==========
def get_main_keyboard():
    """Основная клавиатура"""
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🤖 AI Образ"), KeyboardButton(text="🎨 AI Цвета")],
            [KeyboardButton(text="💡 AI Тренды"), KeyboardButton(text="🧳 AI Гардероб")],
            [KeyboardButton(text="🛍️ Магазины"), KeyboardButton(text="ℹ️ Помощь")],
            [KeyboardButton(text="🔧 AI Статус")]
        ],
        resize_keyboard=True,
        input_field_placeholder="Выберите действие..."
    )

def get_style_keyboard():
    """Клавиатура выбора стиля"""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="👔 Кэжуал", callback_data="style_casual"),
                InlineKeyboardButton(text="💼 Офисный", callback_data="style_office")
            ],
            [
                InlineKeyboardButton(text="🌟 Вечерний", callback_data="style_evening"),
                InlineKeyboardButton(text="🏃 Спортивный", callback_data="style_sport")
            ],
            [
                InlineKeyboardButton(text="🌊 Уличный", callback_data="style_street"),
                InlineKeyboardButton(text="🎨 Креативный", callback_data="style_creative")
            ]
        ]
    )

def get_season_keyboard():
    """Клавиатура выбора сезона"""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="☀️ Лето", callback_data="season_summer"),
                InlineKeyboardButton(text="🍂 Осень", callback_data="season_autumn")
            ],
            [
                InlineKeyboardButton(text="❄️ Зима", callback_data="season_winter"),
                InlineKeyboardButton(text="🌱 Весна", callback_data="season_spring")
            ]
        ]
    )

def get_occasion_keyboard():
    """Клавиатура выбора повода"""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="🏢 Работа", callback_data="occasion_work"),
                InlineKeyboardButton(text="🎓 Учёба", callback_data="occasion_study")
            ],
            [
                InlineKeyboardButton(text="🎉 Вечеринка", callback_data="occasion_party"),
                InlineKeyboardButton(text="🍽️ Ужин", callback_data="occasion_dinner")
            ],
            [
                InlineKeyboardButton(text="🚶 Прогулка", callback_data="occasion_walk"),
                InlineKeyboardButton(text="🛍️ Шопинг", callback_data="occasion_shopping")
            ]
        ]
    )

def get_gender_keyboard():
    """Клавиатура выбора пола"""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="👩 Женский", callback_data="gender_female"),
                InlineKeyboardButton(text="👨 Мужской", callback_data="gender_male")
            ],
            [
                InlineKeyboardButton(text="👥 Унисекс", callback_data="gender_unisex"),
                InlineKeyboardButton(text="🚫 Не важно", callback_data="gender_any")
            ]
        ]
    )

def get_budget_keyboard():
    """Клавиатура выбора бюджета"""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="💰 Бюджетный", callback_data="budget_low"),
                InlineKeyboardButton(text="💵 Средний", callback_data="budget_medium")
            ],
            [
                InlineKeyboardButton(text="💎 Премиум", callback_data="budget_high"),
                InlineKeyboardButton(text="🎯 Не важно", callback_data="budget_any")
            ]
        ]
    )

# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========
async def show_provider_info(message: types.Message):
    """Показать информацию о доступных провайдерах"""
    providers_info = {
        "openai": "OpenAI GPT — мощный, но платный",
        "deepseek": "DeepSeek — хорошая альтернатива OpenAI", 
        "gemini": "Google Gemini — бесплатные запросы",
        "yandex_gpt": "Yandex GPT — бесплатные запросы для РФ",
        "local": "Локальные шаблоны — всегда доступны"
    }
    
    available = [p.value for p in ai_service.available_providers if p.value not in ["fallback", "local"]]
    
    info_text = "🤖 <b>Доступные AI провайдеры:</b>\n\n"
    
    # AI провайдеры
    if available:
        for provider in available:
            info_text += f"• <b>{provider.upper()}</b> — {providers_info.get(provider, 'Неизвестный')}\n"
    else:
        info_text += "• <i>Нет доступных AI провайдеров</i>\n"
    
    # Локальные шаблоны
    info_text += f"• <b>LOCAL</b> — {providers_info.get('local', 'Локальные шаблоны')} (✅ Всегда доступны)\n"
    
    info_text += "\n💡 <i>Бот автоматически выбирает рабочий провайдер, если нет — использует локальные шаблоны</i>"
    
    await message.answer(info_text, parse_mode="HTML")

# ========== ОБРАБОТЧИКИ КОМАНД ==========
@dp.message(Command("providers"))
async def cmd_providers(message: types.Message):
    """Команда для показа информации о провайдерах"""
    await show_provider_info(message)

@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    """Обработчик команды /start"""
    # Показываем доступные AI провайдеры
    providers = [p.value for p in ai_service.available_providers if p.value not in ["fallback", "local"]]
    
    if providers:
        ai_status = f"✅ AI доступен ({', '.join(providers)})"
    else:
        ai_status = "⚠️ AI провайдеры не настроены, используются локальные шаблоны"
    
    welcome_text = f"""
✨ <b>Добро пожаловать в Fashion AI X, {message.from_user.first_name}!</b> ✨

🤖 <i>Ваш персональный AI-стилист</i>
{ai_status}

🚀 <b>Основные функции:</b>
• <b>🤖 AI Образ</b> — создание уникальных образов
• <b>🎨 AI Цвета</b> — цветовые рекомендации  
• <b>💡 AI Тренды</b> — актуальные тренды
• <b>🧳 AI Гардероб</b> — анализ гардероба

👇 <b>Выберите действие:</b>
    """
    
    await message.answer(welcome_text, parse_mode="HTML", reply_markup=get_main_keyboard())
    logger.info(f"Пользователь {message.from_user.id} запустил бота")

@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    """Помощь"""
    help_text = """
🆘 <b>Помощь по использованию Fashion AI X</b>

<b>Основные команды:</b>
/start — Начать работу
/help — Эта справка
/providers — Показать доступные AI провайдеры

<b>AI функции:</b>
• 🤖 AI Образ — создание образов с AI
• 🎨 AI Цвета — советы по сочетанию цветов
• 💡 AI Тренды — модные тренды от AI
• 🧳 AI Гардероб — анализ вашего гардероба

<b>Стандартные функции:</b>
• 🛍️ Магазины — где купить одежду
• ℹ️ Помощь — эта справка
• 🔧 AI Статус — информация о провайдерах

💡 <b>Совет:</b> Используйте кнопки меню для удобной навигации!
    """
    await message.answer(help_text, parse_mode="HTML")

@dp.message(F.text == "🔧 AI Статус")
async def ai_status(message: types.Message):
    """Показать статус AI провайдеров"""
    await show_provider_info(message)
    
# ========== AI ОБРАЗ ==========
@dp.message(F.text == "🤖 AI Образ")
async def start_ai_outfit(message: types.Message, state: FSMContext):
    """Начало создания AI образа"""
    await message.answer(
        "🤖 <b>Создание AI образа</b>\n\n"
        "Я создам для вас стильный образ!\n\n"
        "👇 <b>Выберите стиль:</b>",
        parse_mode="HTML",
        reply_markup=get_style_keyboard()
    )
    await state.set_state(FashionStates.waiting_style)

@dp.callback_query(F.data.startswith("style_"))
async def process_style(callback: CallbackQuery, state: FSMContext):
    """Обработка выбора стиля"""
    style_map = {
        "style_casual": "Кэжуал",
        "style_office": "Офисный",
        "style_evening": "Вечерний",
        "style_sport": "Спортивный",
        "style_street": "Уличный",
        "style_creative": "Креативный"
    }
    
    style = style_map.get(callback.data, "Кэжуал")
    await state.update_data(style=style)
    
    await callback.message.edit_text(
        f"✅ <b>Стиль:</b> {style}\n\n"
        "👇 <b>Выберите сезон:</b>",
        parse_mode="HTML",
        reply_markup=get_season_keyboard()
    )
    await state.set_state(FashionStates.waiting_season)

@dp.callback_query(F.data.startswith("season_"))
async def process_season(callback: CallbackQuery, state: FSMContext):
    """Обработка выбора сезона"""
    season_map = {
        "season_summer": "Лето",
        "season_autumn": "Осень",
        "season_winter": "Зима",
        "season_spring": "Весна"
    }
    
    season = season_map.get(callback.data, "Лето")
    await state.update_data(season=season)
    
    await callback.message.edit_text(
        f"✅ <b>Стиль:</b> {(await state.get_data()).get('style')}\n"
        f"✅ <b>Сезон:</b> {season}\n\n"
        "👇 <b>Выберите повод:</b>",
        parse_mode="HTML",
        reply_markup=get_occasion_keyboard()
    )
    await state.set_state(FashionStates.waiting_occasion)

@dp.callback_query(F.data.startswith("occasion_"))
async def process_occasion(callback: CallbackQuery, state: FSMContext):
    """Обработка выбора повода"""
    occasion_map = {
        "occasion_work": "Работа",
        "occasion_study": "Учёба",
        "occasion_party": "Вечеринка",
        "occasion_dinner": "Ужин",
        "occasion_walk": "Прогулка",
        "occasion_shopping": "Шопинг"
    }
    
    occasion = occasion_map.get(callback.data, "Прогулка")
    await state.update_data(occasion=occasion)
    
    await callback.message.edit_text(
        f"✅ <b>Стиль:</b> {(await state.get_data()).get('style')}\n"
        f"✅ <b>Сезон:</b> {(await state.get_data()).get('season')}\n"
        f"✅ <b>Повод:</b> {occasion}\n\n"
        "👇 <b>Для кого образ?</b>",
        parse_mode="HTML",
        reply_markup=get_gender_keyboard()
    )
    await state.set_state(FashionStates.waiting_gender)

@dp.callback_query(F.data.startswith("gender_"))
async def process_gender(callback: CallbackQuery, state: FSMContext):
    """Обработка выбора пола"""
    gender_map = {
        "gender_female": "Женский",
        "gender_male": "Мужской",
        "gender_unisex": "Унисекс",
        "gender_any": "Не важно"
    }
    
    gender = gender_map.get(callback.data, "Унисекс")
    await state.update_data(gender=gender)
    
    await callback.message.edit_text(
        f"✅ <b>Стиль:</b> {(await state.get_data()).get('style')}\n"
        f"✅ <b>Сезон:</b> {(await state.get_data()).get('season')}\n"
        f"✅ <b>Повод:</b> {(await state.get_data()).get('occasion')}\n"
        f"✅ <b>Для:</b> {gender}\n\n"
        "👇 <b>Выберите бюджет:</b>",
        parse_mode="HTML",
        reply_markup=get_budget_keyboard()
    )
    await state.set_state(FashionStates.waiting_budget)

@dp.callback_query(F.data.startswith("budget_"))
async def process_budget_and_generate(callback: CallbackQuery, state: FSMContext):
    """Обработка бюджета и генерация образа"""
    await callback.answer()

    budget_map = {
        "budget_low": "Бюджетный",
        "budget_medium": "Средний", 
        "budget_high": "Премиум",
        "budget_any": "Не важно"
    }

    budget = budget_map.get(callback.data, "Средний")

    # Получаем все данные
    data = await state.get_data()
    style = data.get("style", "Кэжуал")
    season = data.get("season", "Лето")
    occasion = data.get("occasion", "Прогулка")
    gender = data.get("gender", "Унисекс")

    # Показываем сообщение о генерации
    processing_msg = await callback.message.answer(
        "🔄 <b>AI генерирует ваш образ...</b>\n\n"
        "<i>Это займет несколько секунд</i>",
        parse_mode="HTML"
    )

    try:
        # Генерируем образ через AI сервис
        outfit = await ai_service.generate_outfit(
            style=style,
            season=season,
            occasion=occasion,
            gender=gender,
            budget=budget
        )

        # ✅ ДОБАВЬ ОТЛАДКУ
        print(f"🔍 [BOT DEBUG] ===========================")
        print(f"🔍 [BOT DEBUG] Получен образ от AI сервиса")
        print(f"🔍 [BOT DEBUG] Название: {outfit.get('name')}")
        print(f"🔍 [BOT DEBUG] has_ai значение: {outfit.get('has_ai')}")
        print(f"🔍 [BOT DEBUG] Тип has_ai: {type(outfit.get('has_ai'))}")
        print(f"🔍 [BOT DEBUG] Все ключи: {list(outfit.keys())}")
        print(f"🔍 [BOT DEBUG] ===========================")

        # ✅ ИСПРАВЛЕННАЯ ПРОВЕРКА
        has_ai_value = outfit.get("has_ai")
        
        # Проверяем разные варианты
        if has_ai_value is True:
            ai_indicator = "✨ <b>СГЕНЕРИРОВАНО AI</b> ✨\n\n"
            print(f"🔍 [BOT DEBUG] Показываем AI индикатор (has_ai=True)")
        elif has_ai_value is False:
            ai_indicator = "📚 <b>Локальный шаблон</b>\n\n"
            print(f"🔍 [BOT DEBUG] Показываем локальный шаблон (has_ai=False)")
        else:
            # Если ключа нет или значение None, считаем что это AI
            ai_indicator = "✨ <b>СГЕНЕРИРОВАНО AI</b> ✨\n\n"
            print(f"🔍 [BOT DEBUG] Показываем AI (has_ai={has_ai_value})")

        # Формируем ответ
        response = f"""
{ai_indicator}
🎉 <b>Ваш образ готов!</b>

<b>📌 Детали:</b>
• Стиль: {style}
• Сезон: {season}
• Повод: {occasion}
• Для: {gender}
• Бюджет: {budget}

<b>👕 {outfit['name']}</b>
<i>{outfit['description']}</i>

<b>🧺 Состав образа:</b>
{chr(10).join([f'• {item}' for item in outfit.get('items', [])[:6]])}

<b>🎨 Цветовая гамма:</b>
{outfit.get('colors', 'Гармоничные сочетания')}

<b>💡 Советы:</b>
{chr(10).join([f'• {tip}' for tip in outfit.get('tips', ['Экспериментируйте!'])[:3]])}
        """

        await callback.message.answer(response, parse_mode="HTML", reply_markup=get_main_keyboard())

    except Exception as e:
        logger.error(f"Ошибка генерации образа: {e}")
        await callback.message.answer(
            "⚠️ <b>Произошла ошибка при генерации образа</b>\n\n"
            "Попробуйте ещё раз позже или обратитесь в поддержку.",
            parse_mode="HTML",
            reply_markup=get_main_keyboard()
        )

    finally:
        await state.clear()
        try:
            await processing_msg.delete()
        except:
            pass
# ========== AI ЦВЕТА ==========
@dp.message(F.text == "🎨 AI Цвета")
async def start_ai_colors(message: types.Message, state: FSMContext):
    """Начало работы с цветами"""
    await message.answer(
        "🎨 <b>AI рекомендации по цветам</b>\n\n"
        "Напишите название цвета, и я дам детальные советы по сочетаниям!\n\n"
        "<i>Пример: синий, красный, фиолетовый, хаки...</i>",
        parse_mode="HTML"
    )
    await state.set_state(FashionStates.waiting_color)

@dp.message(FashionStates.waiting_color)
async def process_color_request(message: types.Message, state: FSMContext):
    """Обработка запроса цвета"""
    color = message.text.strip()
    
    processing_msg = await message.answer(
        f"🔄 <b>Анализирую цвет {color}...</b>",
        parse_mode="HTML"
    )
    
    try:
        advice = await ai_service.get_color_advice(color)
        # Заменяем Markdown на HTML для правильного отображения
        advice_html = advice.replace("*", "").replace("🎨", "<b>🎨").replace("💡", "</b>\n\n💡")
        await message.answer(advice_html, parse_mode="HTML", reply_markup=get_main_keyboard())
        
    except Exception as e:
        logger.error(f"Ошибка получения цветовых советов: {e}")
        await message.answer(
            f"🎨 <b>Базовые сочетания для {color}:</b>\n\n"
            f"• {color} + белый — свежо\n"
            f"• {color} + черный — контрастно\n"
            f"• {color} + нейтральные — безопасно\n"
            f"• {color} + контрастный — смело",
            parse_mode="HTML",
            reply_markup=get_main_keyboard()
        )
    
    finally:
        await state.clear()
        try:
            await processing_msg.delete()
        except:
            pass

# ========== AI ТРЕНДЫ ==========
@dp.message(F.text == "💡 AI Тренды")
async def get_ai_trends(message: types.Message):
    """Получение AI трендов"""
    processing_msg = await message.answer(
        "🔄 <b>AI анализирует тренды...</b>",
        parse_mode="HTML"
    )
    
    try:
        trends = await ai_service.get_trends()
        # Заменяем Markdown на HTML
        trends_html = trends.replace("*", "").replace("🔥", "<b>🔥").replace("✨", "</b>\n\n✨")
        await message.answer(trends_html, parse_mode="HTML", reply_markup=get_main_keyboard())
        
    except Exception as e:
        logger.error(f"Ошибка получения трендов: {e}")
        await message.answer(
            "🔥 <b>Актуальные тренды:</b>\n\n"
            "• Oversize силуэты\n"
            "• Яркие аксессуары\n"
            "• Спортивный шик\n"
            "• Устойчивая мода\n\n"
            "💡 <i>Совет:</i> Сочетайте тренды с классикой!",
            parse_mode="HTML",
            reply_markup=get_main_keyboard()
        )
    
    finally:
        try:
            await processing_msg.delete()
        except:
            pass

# ========== AI ГАРДЕРОБ ==========
@dp.message(F.text == "🧳 AI Гардероб")
async def start_wardrobe_analysis(message: types.Message, state: FSMContext):
    """Начало анализа гардероба"""
    await message.answer(
        "🧳 <b>AI анализ гардероба</b>\n\n"
        "Опишите основные вещи вашего гардероба:\n\n"
        "<i>Пример: 'У меня есть 5 футболок, 3 джинсы, 2 пиджака, 1 платье...'</i>",
        parse_mode="HTML"
    )
    await state.set_state(FashionStates.waiting_wardrobe)

@dp.message(FashionStates.waiting_wardrobe)
async def process_wardrobe(message: types.Message, state: FSMContext):
    """Обработка описания гардероба"""
    description = message.text
    
    processing_msg = await message.answer(
        "🔄 <b>AI анализирует гардероб...</b>",
        parse_mode="HTML"
    )
    
    try:
        analysis = await ai_service.analyze_wardrobe(description)
        
        ai_indicator = "✨ <b>АНАЛИЗ AI</b> ✨\n\n" if analysis.get("has_ai", False) else "📚 <b>Локальные рекомендации</b>\n\n"
        
        response = f"""
{ai_indicator}
🧳 <b>Анализ вашего гардероба</b>

{analysis.get('analysis', 'Не удалось проанализировать.')}

<b>💎 Главный совет:</b>
Создавайте капсульный гардероб из сочетаемых вещей!
        """
        
        await message.answer(response, parse_mode="HTML", reply_markup=get_main_keyboard())
        
    except Exception as e:
        logger.error(f"Ошибка анализа гардероба: {e}")
        await message.answer(
            "⚠️ <b>Не удалось проанализировать гардероб</b>\n\n"
            "Попробуйте описать более подробно.",
            parse_mode="HTML",
            reply_markup=get_main_keyboard()
        )
    
    finally:
        await state.clear()
        try:
            await processing_msg.delete()
        except:
            pass

# ========== ДОПОЛНИТЕЛЬНЫЕ ФУНКЦИИ ==========
@dp.message(F.text == "🛍️ Магазины")
async def shops(message: types.Message):
    """Рекомендации магазинов"""
    shops_text = """
🛍️ <b>Рекомендации магазинов</b>

<b>💎 Премиум:</b>
• ZARA — трендовые вещи
• Mango — качественные ткани
• Massimo Dutti — элегантность

<b>💰 Бюджетные:</b>
• H&M — доступная мода
• Bershka — молодёжный стиль
• Pull&Bear — повседневная одежда

<b>🛒 Онлайн:</b>
• Wildberries — большой выбор
• Lamoda — быстрая доставка
• ASOS — международные бренды

💡 <b>Совет:</b> Проверяйте качество перед покупкой!
    """
    await message.answer(shops_text, parse_mode="HTML")

@dp.message()
async def handle_other_messages(message: types.Message):
    """Обработка остальных сообщений"""
    responses = [
        "Используйте кнопки меню для навигации 👇",
        "Выберите действие из меню!",
        "Хотите создать образ? Нажмите \"🤖 AI Образ\"!",
        "Нужны цветовые советы? Выберите \"🎨 AI Цвета\"!"
    ]
    import random
    await message.answer(random.choice(responses), reply_markup=get_main_keyboard())

# ========== ЗАПУСК БОТА ==========
async def main():
    """Главная функция"""
    logger.info("🚀 Запуск Fashion AI X бота...")
    
    # Проверяем AI провайдеры
    providers = [p.value for p in ai_service.available_providers if p.value not in ["fallback", "local"]]
    
    if providers:
        logger.info(f"✅ Доступные AI провайдеры: {', '.join(providers)}")
    else:
        logger.info("ℹ️ AI провайдеры не настроены, будут использоваться локальные шаблоны")
    
    try:
        bot_info = await bot.get_me()
        logger.info(f"🤖 Бот: @{bot_info.username}")
        logger.info("✅ Бот готов к работе!")
        
        await dp.start_polling(bot)
        
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}", exc_info=True)
    finally:
        await bot.session.close()
        logger.info("🛑 Бот остановлен")

if __name__ == "__main__":
    asyncio.run(main())