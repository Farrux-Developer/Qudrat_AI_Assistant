import asyncio
import glob
import io
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from typing import Any

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from PIL import Image

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    BotCommand,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from google import genai
from google.genai import types as genai_types

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_DEFAULT = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN))
dp = Dispatcher(storage=MemoryStorage())
gemini_client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None

# --- ФАЙЛОВОЕ ХРАНИЛИЩЕ НАСТРОЕК И ПАМЯТИ ---
SETTINGS_FILE = "user_settings.json"
MEMORIES_FILE = "user_memories.json"
KB_DIR = "knowledge_base"
os.makedirs(KB_DIR, exist_ok=True)


def load_json(filepath: str) -> dict:
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_json(filepath: str, data: dict):
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Ошибка сохранения {filepath}: {e}")


def get_user_setting(user_id: int, key: str, default: Any = None) -> Any:
    settings = load_json(SETTINGS_FILE)
    user_str = str(user_id)
    return settings.get(user_str, {}).get(key, default)


def update_user_settings(user_id: int, **kwargs):
    settings = load_json(SETTINGS_FILE)
    user_str = str(user_id)
    if user_str not in settings:
        settings[user_str] = {}
    settings[user_str].update(kwargs)
    save_json(SETTINGS_FILE, settings)


def get_user_memories(user_id: int) -> list[str]:
    memories = load_json(MEMORIES_FILE)
    return memories.get(str(user_id), [])


def add_user_memory(user_id: int, fact: str):
    memories = load_json(MEMORIES_FILE)
    user_str = str(user_id)
    user_list = memories.setdefault(user_str, [])
    fact_clean = fact.strip()
    if fact_clean and fact_clean not in user_list:
        user_list.append(fact_clean)
        # Храним до 30 ключевых фактов
        memories[user_str] = user_list[-30:]
        save_json(MEMORIES_FILE, memories)


def clear_user_memories(user_id: int):
    memories = load_json(MEMORIES_FILE)
    user_str = str(user_id)
    if user_str in memories:
        memories[user_str] = []
        save_json(MEMORIES_FILE, memories)


# --- FSM СОСТОЯНИЯ ---
class BotStates(StatesGroup):
    choosing_language = State()
    waiting_oferta = State()
    active_chat = State()


# --- РЕЖИМЫ И СИСТЕМНЫЕ ПРОМПТЫ ---
PROMPTS = {
    "agent": {
        "ru": (
            "Ты — Qudrat Agent, высокоэффективный интеллектуальный помощник. "
            "Твоя задача — не просто давать ответы, а решать задачи пользователя максимально быстро и точно.\n\n"
            "Алгоритм работы:\n"
            "1. Анализ: Если запрос сложный, сначала декомпозируй его на подзадачи.\n"
            "2. Приоритет: Используй доступные инструменты (поиск в сети search_web, вычисления execute_python_code, "
            "базу знаний read_knowledge_base, сохранение предпочтений save_memory) для дополнения своих знаний.\n"
            "3. Форматирование: Всегда используй Markdown (жирный шрифт, списки, блоки кода), чтобы сделать ответ читабельным.\n"
            "4. Краткость: Не пиши «воду». Сначала суть, затем детали.\n"
            "5. Персонализация: Если в профиле пользователя или истории есть данные о предпочтениях, используй их для адаптации ответа.\n\n"
            "Если пользователь неправ, вежливо укажи на ошибку, опираясь на факты. "
            "Если вопрос выходит за рамки твоих возможностей — честно признайся."
        ),
        "uz": (
            "Siz — Qudrat Agent yuqori samarali intellektual yordamchisiz. "
            "Vazifangiz — shunchaki javob berish emas, foydalanuvchi muammolarini imkon qadar tez va aniq hal qilishdir.\n\n"
            "Ishlash algoritmi:\n"
            "1. Tahlil: Agar so'rov murakkab bo'lsa, uni kichik vazifalarga ajrating.\n"
            "2. Asboblar: Mavjud vositalardan (search_web, execute_python_code, read_knowledge_base, save_memory) faol foydalaning.\n"
            "3. Formatlash: Har doim javoblarni chiroyli Markdown shaklida taqdim eting.\n"
            "4. Qisqalik: Ortiqcha so'zlarsiz, avval asosiy mohiyat, so'ng tafsilotlar.\n"
            "5. Shaxsiylashtirish: Foydalanuvchi afzalliklari va xotirasidagi ma'lumotlarni hisobga oling."
        ),
    },
    "friend": {
        "ru": (
            "Ты — Qudrat, экспертный ИИ-ассистент с человеческим лицом. "
            "Ты общаешься с пользователем как с близким коллегой или умным другом.\n\n"
            "Твой стиль общения:\n"
            "- Ты всегда позитивен, эмпатичен и заинтересован в успехе пользователя.\n"
            "- Твои ответы глубокие, но понятные. Ты объясняешь сложные вещи простыми аналогиями.\n"
            "- Твоя цель: стать незаменимым инструментом, который экономит время пользователя.\n"
            "- Избегай роботизированных конструкций. Старайся, чтобы каждый ответ был полезен и содержал призыв к действию или уместный вопрос.\n"
            "- При необходимости используй инструменты (поиск, код, база знаний, сохранение памяти)."
        ),
        "uz": (
            "Siz — Qudrat, insoniy yondashuvga ega ekspert AI-yordamchisiz. "
            "Foydalanuvchi bilan yaqin hamkasb yoki aqlli do'st kabi muloqot qilasiz.\n\n"
            "Muloqot uslubingiz:\n"
            "- Har doim ijobiy, samimiy va foydalanuvchi muvaffaqiyatiga qiziqqansiz.\n"
            "- Murakkab narsalarni oddiy hayotiy misollar va o'xshatishlar bilan tushuntirasiz.\n"
            "- Qotib qolgan robot iboralaridan qoching. Har bir javob foydali bo'lsin va keyingi qadamni taklif qilsin.\n"
            "- Zarur bo'lganda asboblardan (qidiruv, hisob-kitob, bilimlar bazasi, xotira) foydalaning."
        ),
    },
    "analyst": {
        "ru": (
            "Ты — ИИ-аналитик Qudrat. Твоя специализация — обработка информации, поиск связей и генерация решений на основе данных.\n\n"
            "Твои правила:\n"
            "1. Опора на факты: При ответе на вопросы используй базу знаний (read_knowledge_base) и поиск (search_web). "
            "Если информации недостаточно, ответь: «У меня нет точных данных по этому вопросу, но я могу предположить...».\n"
            "2. Критическое мышление: Подвергай сомнению неточные формулировки пользователя и задавай уточняющие вопросы для достижения результата.\n"
            "3. Структура: Ответ всегда должен строиться по схеме: Краткий итог -> Детальный разбор -> Рекомендация/Вывод.\n"
            "4. Инструментарий: Активно используй функцию исполнения Python-кода (execute_python_code) для расчетов, чтобы исключить ошибки в вычислениях."
        ),
        "uz": (
            "Siz — Qudrat tahliliy AI-ekspertisiz. Mutaxassisligingiz — ma'lumotlarni qayta ishlash, mantiqiy bog'liqliklarni topish va ma'lumotlar asosida yechimlar yaratish.\n\n"
            "Qoidalaringiz:\n"
            "1. Faktlarga tayanish: Bilimlar bazasi (read_knowledge_base) va qidiruvdan foydalaning. Agar ma'lumot kam bo'lsa, buni ochiq bildiring.\n"
            "2. Tanqidiy fikrlash: Aniq natijaga erishish uchun aniqlashtiruvchi savollar bering.\n"
            "3. Tuzilma: Har doim: Qisqa xulosa -> Batafsil tahlil -> Tavsiya/Xulosa tartibida javob bering.\n"
            "4. Asboblar: Hisob-kitoblar mutlaq aniq bo'lishi uchun execute_python_code funksiyasidan foydalaning."
        ),
    },
}

MODE_NAMES = {
    "uz": {
        "agent": "⚡ Samarali Agent",
        "friend": "🤝 Do'st / Hamkasb",
        "analyst": "📊 Tahliliy Ekspert",
    },
    "ru": {
        "agent": "⚡ Эффективный Агент",
        "friend": "🤝 Коллега / Друг",
        "analyst": "📊 Аналитический Эксперт",
    },
}

# --- ТЕКСТЫ ИНТЕРФЕЙСА ---
TEXTS = {
    "uz": {
        "oferta": (
            "📋 *OMMAVIY OFERTA (FOYDALANISH SHARTLARI)*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "1. *Predmet:* «Qudrat AI» boti Google Gemini sun’iy intellektiga vositachi kirishni «boricha» (AS IS) taqdim etadi.\n\n"
            "2. *Mas’uliyatdan voz kechish:* Javoblar neyrotarmoq tomonidan avtomatik yaratiladi. "
            "Ma’muriyat javoblarning mutlaq to‘g‘riligiga kafolat bermaydi. Foydalanish uchun to‘liq mas’uliyat foydalanuvchining zimmasida.\n\n"
            "3. *Qoidalar:* Qonunga zid maqsadlarda, spamlash va buzg‘unchilik uchun foydalanish taqiqlanadi.\n\n"
            "4. *Aksept:* Quyidagi tugmani bosish shartlarga to‘liq rozilik bildirish hisoblanadi."
        ),
        "btn_accept": "✅ Roziman / Qabul qilish",
        "ready": (
            "✅ *QUDRAT AI ISHGA TAYYOR!*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Hozirgi rejim: *{mode}*\n\n"
            "💡 *Mavjud buyruqlar:*\n"
            "• /mode — Ishlash rejimini o‘zgartirish\n"
            "• /memory — Saqlangan profil ma’lumotlari\n"
            "• /clear — Suhbat tarixini tozalash\n"
            "• /help — Yordam va asboblar haqida\n\n"
            "💬 *Savolingizni yozing yoki rasm yuboring:*"
        ),
        "mode_select": "🎭 *Qudrat AI ish rejimini tanlang:*",
        "mode_updated": "✅ Rejim o‘zgartirildi: *{mode}*",
        "cleared": "🧹 Suhbat tarixi muvaffaqiyatli tozalandi.",
        "memory_info": "🧠 *Siz haqingizda saqlangan xotiralar:*\n\n{facts}",
        "memory_empty": "🧠 Hozircha sizning afzalliklaringiz haqida xotiralar saqlanmagan. Suhbat davomida men ularni avtomatik qayd etib boraman.",
        "memory_cleared": "🗑 Shaxsiy xotiralar tozalandi.",
        "btn_clear_mem": "🗑 Xotirani tozalash",
        "help": (
            "🤖 *QUDRAT AI YORDAM QO‘LLANMASI*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "✨ *Rejimlar:* \n"
            "• ⚡ *Agent* — tezkor yechimlar, qisqa va aniq harakatlar.\n"
            "• 🤝 *Hamkasb* — do‘stona, tushunarli va batafsil suhbat.\n"
            "• 📊 *Tahlilchi* — faktlar, bilimlar bazasi va hisob-kitoblar.\n\n"
            "🛠 *Avtomatik asboblar:*\n"
            "• Internetdan qidirish (search_web)\n"
            "• Python hisoblashlari (execute_python_code)\n"
            "• Bilimlar bazasi (read_knowledge_base)\n"
            "• Shaxsiy xotira (save_memory)\n\n"
            "Buyruqlar: /mode, /memory, /clear, /start"
        ),
        "not_accepted": (
            "⚠️ *Botdan foydalanish uchun avval ommaviy ofertani qabul qiling!*\n"
            "Qayta boshlash uchun /start buyrug‘ini bosing."
        ),
        "error": "⚠️ Javob olishda xatolik yuz berdi. Iltimos, keyinroq urinib ko‘ring.",
    },
    "ru": {
        "oferta": (
            "📋 *ПУБЛИЧНАЯ ОФЕРТА (УСЛОВИЯ ИСПОЛЬЗОВАНИЯ)*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "1. *Предмет:* Бот «Qudrat AI» предоставляет доступ к интерфейсу ИИ Google Gemini на условиях «КАК ЕСТЬ» (AS IS).\n\n"
            "2. *Отказ от ответственности:* Ответы формируются нейросетью автоматически без предварительной модерации. "
            "Администрация не гарантирует абсолютную точность ответов. Пользователь несёт личную ответственность за использование информации.\n\n"
            "3. *Правила:* Запрещено использовать сервис в противоправных целях и для спама.\n\n"
            "4. *Акцепт:* Нажатие кнопки означает полное и безоговорочное согласие с условиями."
        ),
        "btn_accept": "✅ Принять оферту",
        "ready": (
            "✅ *QUDRAT AI ГОТОВ К РАБОТЕ!*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Текущий режим: *{mode}*\n\n"
            "💡 *Доступные команды:*\n"
            "• /mode — Сменить режим работы\n"
            "• /memory — Профиль и сохранённая память\n"
            "• /clear — Очистить контекст диалога\n"
            "• /help — Инструкция и возможности\n\n"
            "💬 *Напишите ваш вопрос или отправьте фото:*"
        ),
        "mode_select": "🎭 *Выберите режим работы Qudrat AI:*",
        "mode_updated": "✅ Режим успешно изменён на: *{mode}*",
        "cleared": "🧹 Контекст диалога успешно очищен.",
        "memory_info": "🧠 *Сохранённые факты о вас:*\n\n{facts}",
        "memory_empty": "🧠 Память пока пуста. В ходе общения бот автоматически запоминает ваши интересы и предпочтения.",
        "memory_cleared": "🗑 Память профиля очищена.",
        "btn_clear_mem": "🗑 Очистить память",
        "help": (
            "🤖 *СПРАВКА И ВОЗМОЖНОСТИ QUDRAT AI*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "✨ *Режимы:*\n"
            "• ⚡ *Эффективный Агент* — быстрое решение задач, декомпозиция, суть без воды.\n"
            "• 🤝 *Коллега / Друг* — живой, эмпатичный стиль, простые аналогии.\n"
            "• 📊 *Аналитический Эксперт* — строгая опора на факты, RAG, структура и расчеты.\n\n"
            "🛠 *Встроенные инструменты (Function Calling):*\n"
            "• Поиск в интернете (search_web)\n"
            "• Вычисления Python (execute_python_code)\n"
            "• База знаний из папки ./knowledge_base (read_knowledge_base)\n"
            "• Долгосрочная память (save_memory)\n\n"
            "Команды: /mode, /memory, /clear, /help, /start"
        ),
        "not_accepted": (
            "⚠️ *Для работы с ботом необходимо принять оферту!*\n"
            "Нажмите /start для начала работы."
        ),
        "error": "⚠️ Произошла ошибка при получении ответа от ИИ. Попробуйте ещё раз.",
    },
}


# --- РЕАЛИЗАЦИЯ ИНСТРУМЕНТОВ ДЛЯ GEMINI ---
def search_web(query: str) -> str:
    """Searches the internet for up-to-date information, news, or factual answers.
    Args:
        query: The search query string.
    """
    query_clean = query.strip()
    if not query_clean:
        return "Empty search query."

    # 1. DuckDuckGo Instant Answer API
    try:
        url = "https://api.duckduckgo.com/?q=" + urllib.parse.quote(query_clean) + "&format=json&no_html=1&skip_disambig=1"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read().decode())
            abstract = data.get("AbstractText", "")
            if abstract:
                return abstract
            topics = data.get("RelatedTopics", [])
            snippets = [t.get("Text") for t in topics if isinstance(t, dict) and t.get("Text")]
            if snippets:
                return "\n".join(snippets[:3])
    except Exception:
        pass

    # 2. DuckDuckGo HTML Search
    try:
        url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query_clean)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=6) as r:
            soup = BeautifulSoup(r.read(), "html.parser")
            snippets = [el.get_text(strip=True) for el in soup.select(".result__snippet")[:4]]
            if snippets:
                return "\n---\n".join(snippets)
    except Exception as e:
        return f"Web search error: {e}"

    return "No web results found for this query."


def read_knowledge_base(query: str) -> str:
    """Searches and reads relevant documentation and text from the local knowledge base folder.
    Args:
        query: Search keywords or topic to look up in knowledge base documents.
    """
    if not os.path.exists(KB_DIR):
        return "Knowledge base folder is empty."

    files = glob.glob(os.path.join(KB_DIR, "*.*"))
    if not files:
        return "База знаний пока пуста. Поместите текстовые файлы (.txt, .md, .json) в папку knowledge_base."

    query_words = set(query.lower().split())
    matches = []

    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                lines = content.split("\n")
                relevant = [line.strip() for line in lines if any(w in line.lower() for w in query_words) and line.strip()]
                if relevant:
                    matches.append(f"[{fname}]:\n" + "\n".join(relevant[:5]))
                elif len(content) < 1500:
                    matches.append(f"[{fname}]:\n" + content[:600])
        except Exception:
            continue

    if matches:
        return "\n\n".join(matches[:4])
    return "В базе знаний нет точных совпадений по данному запросу."


def execute_python_code(code: str) -> str:
    """Safely executes Python code or mathematical expressions for precise calculations and data analysis.
    Args:
        code: Valid Python code script or expression to execute.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=7,
        )
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        if stderr:
            return f"Output:\n{stdout}\nError:\n{stderr}" if stdout else f"Error: {stderr}"
        return stdout if stdout else "Code executed successfully (no print output)."
    except subprocess.TimeoutExpired:
        return "Execution timed out (7s limit)."
    except Exception as e:
        return f"Execution error: {e}"


def build_user_tools(user_id: int):
    """Creates tool declarations, binding user-specific memory saving."""

    def save_memory(fact: str) -> str:
        """Saves an important fact, interest, or preference about the user into their profile for future personalized interactions.
        Args:
            fact: The specific user fact or preference to remember.
        """
        add_user_memory(user_id, fact)
        return f"Fact successfully remembered: {fact}"

    return [search_web, read_knowledge_base, execute_python_code, save_memory]


# --- УПРАВЛЕНИЕ СЕССИЯМИ ЧАТА GEMINI ---
# user_id -> genai.chats.Chat
chat_sessions: dict[int, Any] = {}


def get_full_system_instruction(user_id: int, lang: str, mode: str) -> str:
    base_prompt = PROMPTS.get(mode, PROMPTS["agent"]).get(lang, PROMPTS["agent"]["ru"])
    memories = get_user_memories(user_id)
    if memories:
        facts_str = "\n".join(f"- {f}" for f in memories)
        if lang == "uz":
            base_prompt += f"\n\n[Foydalanuvchi profili va shaxsiy xotiralari]:\n{facts_str}"
        else:
            base_prompt += f"\n\n[Профиль и сохраненные предпочтения пользователя]:\n{facts_str}"
    return base_prompt


def get_or_create_chat(user_id: int, lang: str, mode: str, model_name: str = GEMINI_MODEL_DEFAULT):
    session = chat_sessions.get(user_id)
    if session is not None:
        return session

    tools = build_user_tools(user_id)
    sys_instruction = get_full_system_instruction(user_id, lang, mode)

    chat = gemini_client.chats.create(
        model=model_name,
        config=genai_types.GenerateContentConfig(
            system_instruction=sys_instruction,
            tools=tools,
            max_output_tokens=2500,
        ),
    )
    chat_sessions[user_id] = chat
    return chat


def reset_user_chat(user_id: int):
    if user_id in chat_sessions:
        del chat_sessions[user_id]


# --- КЛАВИАТУРЫ ---
def language_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="🇺🇿 O'zbekcha", callback_data="lang:uz"),
            InlineKeyboardButton(text="🇷🇺 Русский", callback_data="lang:ru"),
        ]
    ])


def oferta_keyboard(lang: str) -> InlineKeyboardMarkup:
    btn_text = TEXTS[lang]["btn_accept"]
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=btn_text, callback_data="accept_oferta")]
    ])


def mode_keyboard(lang: str) -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton(text=MODE_NAMES[lang]["agent"], callback_data="set_mode:agent")],
        [InlineKeyboardButton(text=MODE_NAMES[lang]["friend"], callback_data="set_mode:friend")],
        [InlineKeyboardButton(text=MODE_NAMES[lang]["analyst"], callback_data="set_mode:analyst")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def memory_keyboard(lang: str) -> InlineKeyboardMarkup:
    btn_text = TEXTS[lang]["btn_clear_mem"]
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=btn_text, callback_data="clear_memory")]
    ])


# --- 1. СТАРТ: ВЫБОР ЯЗЫКА ---
@dp.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()
    await state.set_state(BotStates.choosing_language)

    await message.answer(
        "✨ *━━━━━━━━━━━━━━━━━━━━*\n"
        "🤖 *QUDRAT AI AGENT*\n"
        "✨ *━━━━━━━━━━━━━━━━━━━━*\n\n"
        "🌐 *Muloqot tilini tanlang / Выберите язык общения:*",
        reply_markup=language_keyboard(),
    )


# --- 2. ВЫБОР ЯЗЫКА -> ОФЕРТА ---
@dp.callback_query(F.data.startswith("lang:"), BotStates.choosing_language)
async def process_language_choice(query: CallbackQuery, state: FSMContext):
    await query.answer()
    lang = query.data.split(":")[1]

    user_id = query.from_user.id
    update_user_settings(user_id, lang=lang)
    await state.update_data(lang=lang)

    # Проверяем, была ли оферта уже принята ранее
    if get_user_setting(user_id, "accepted_oferta", False):
        await state.set_state(BotStates.active_chat)
        mode = get_user_setting(user_id, "mode", "agent")
        mode_label = MODE_NAMES[lang].get(mode, mode)
        try:
            await query.message.delete()
        except Exception:
            pass
        await query.message.answer(TEXTS[lang]["ready"].format(mode=mode_label))
        return

    await state.set_state(BotStates.waiting_oferta)
    try:
        await query.message.delete()
    except Exception:
        pass

    await query.message.answer(
        TEXTS[lang]["oferta"],
        reply_markup=oferta_keyboard(lang),
    )


# --- 3. ПРИНЯТИЕ ОФЕРТЫ -> АКТИВНЫЙ РЕЖИМ ---
@dp.callback_query(F.data == "accept_oferta", BotStates.waiting_oferta)
async def process_accept_oferta(query: CallbackQuery, state: FSMContext):
    await query.answer()
    user_id = query.from_user.id
    update_user_settings(user_id, accepted_oferta=True)

    data = await state.get_data()
    lang = data.get("lang") or get_user_setting(user_id, "lang", "uz")
    mode = get_user_setting(user_id, "mode", "agent")
    mode_label = MODE_NAMES[lang].get(mode, mode)

    await state.set_state(BotStates.active_chat)
    try:
        await query.message.delete()
    except Exception:
        pass

    await query.message.answer(TEXTS[lang]["ready"].format(mode=mode_label))


# --- 4. КОМАНДА /mode: ВЫБОР РЕЖИМА ---
@dp.message(Command("mode"))
async def cmd_mode(message: Message):
    user_id = message.from_user.id
    lang = get_user_setting(user_id, "lang", "ru")
    await message.answer(TEXTS[lang]["mode_select"], reply_markup=mode_keyboard(lang))


@dp.callback_query(F.data.startswith("set_mode:"))
async def process_mode_change(query: CallbackQuery):
    await query.answer()
    user_id = query.from_user.id
    new_mode = query.data.split(":")[1]
    lang = get_user_setting(user_id, "lang", "ru")

    update_user_settings(user_id, mode=new_mode)
    reset_user_chat(user_id)  # Пересоздаем сессию с новым системным промптом

    mode_label = MODE_NAMES[lang].get(new_mode, new_mode)
    try:
        await query.message.edit_text(TEXTS[lang]["mode_updated"].format(mode=mode_label))
    except Exception:
        await query.message.answer(TEXTS[lang]["mode_updated"].format(mode=mode_label))


# --- 5. КОМАНДА /clear: ОЧИСТКА ИСТОРИИ ---
@dp.message(Command("clear"))
async def cmd_clear(message: Message):
    user_id = message.from_user.id
    lang = get_user_setting(user_id, "lang", "ru")
    reset_user_chat(user_id)
    await message.answer(TEXTS[lang]["cleared"])


# --- 6. КОМАНДА /memory: УПРАВЛЕНИЕ ПАМЯТЬЮ ---
@dp.message(Command("memory"))
async def cmd_memory(message: Message):
    user_id = message.from_user.id
    lang = get_user_setting(user_id, "lang", "ru")
    memories = get_user_memories(user_id)

    if memories:
        facts_text = "\n".join(f"• {f}" for f in memories)
        await message.answer(
            TEXTS[lang]["memory_info"].format(facts=facts_text),
            reply_markup=memory_keyboard(lang),
        )
    else:
        await message.answer(TEXTS[lang]["memory_empty"])


@dp.callback_query(F.data == "clear_memory")
async def process_clear_memory(query: CallbackQuery):
    await query.answer()
    user_id = query.from_user.id
    lang = get_user_setting(user_id, "lang", "ru")
    clear_user_memories(user_id)
    reset_user_chat(user_id)
    try:
        await query.message.edit_text(TEXTS[lang]["memory_cleared"])
    except Exception:
        await query.message.answer(TEXTS[lang]["memory_cleared"])


# --- 7. КОМАНДА /help: СПРАВКА ---
@dp.message(Command("help"))
async def cmd_help(message: Message):
    user_id = message.from_user.id
    lang = get_user_setting(user_id, "lang", "ru")
    await message.answer(TEXTS[lang]["help"])


# --- 8. ОБРАБОТКА СООБЩЕНИЙ И ФОТО С GEMINI AFC И FALLBACK ---
@dp.message(F.text | F.photo)
async def handle_ai_message(message: Message, state: FSMContext):
    user_id = message.from_user.id

    # Проверка оферты
    accepted = get_user_setting(user_id, "accepted_oferta", False)
    current_state = await state.get_state()
    if not accepted and current_state != BotStates.active_chat:
        lang = get_user_setting(user_id, "lang", "ru")
        await message.answer(TEXTS[lang]["not_accepted"])
        return

    lang = get_user_setting(user_id, "lang", "ru")
    mode = get_user_setting(user_id, "mode", "agent")

    await bot.send_chat_action(chat_id=message.chat.id, action="typing")

    text = message.text or message.caption or ""
    image_bytes = None

    # Обработка фото
    if message.photo:
        photo_io = io.BytesIO()
        await bot.download(message.photo[-1], destination=photo_io)
        image = Image.open(photo_io).convert("RGB")
        out_buf = io.BytesIO()
        image.save(out_buf, format="JPEG", quality=90)
        image_bytes = out_buf.getvalue()

    if image_bytes and not text:
        text = "Ushbu rasmni tasvirlab bering yoki tahlil qiling." if lang == "uz" else "Опишите или проанализируйте это изображение."

    # Формируем payload для chat.send_message
    payload: Any = []
    if image_bytes:
        payload.append(genai_types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))
    if text:
        payload.append(text)

    if len(payload) == 1 and isinstance(payload[0], str):
        payload = payload[0]

    # Модели с каскадным переключением при 503 / недоступности
    candidate_models = [
        GEMINI_MODEL_DEFAULT,
        "gemini-3.8-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
    ]
    seen_models = set()
    candidate_models = [m for m in candidate_models if not (m in seen_models or seen_models.add(m))]

    answer = None
    last_error = None

    chat = get_or_create_chat(user_id, lang, mode, candidate_models[0])

    for model_name in candidate_models:
        try:
            # Ограничиваем историю чата последними 16 сообщениями для скорости и стабильности
            if hasattr(chat, "_history") and len(chat._history) > 16:
                chat._history = chat._history[-16:]

            response = await asyncio.to_thread(chat.send_message, payload)
            answer = response.text or ""
            break
        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            if any(x in err_str for x in ("503", "unavailable", "404", "high demand", "not_found")):
                # Сохраняем текущую историю и пересоздаем Chat на следующей рабочей модели
                current_hist = chat.get_history() if hasattr(chat, "get_history") else []
                tools = build_user_tools(user_id)
                sys_instruction = get_full_system_instruction(user_id, lang, mode)
                try:
                    chat = gemini_client.chats.create(
                        model=model_name,
                        history=current_hist,
                        config=genai_types.GenerateContentConfig(
                            system_instruction=sys_instruction,
                            tools=tools,
                            max_output_tokens=2500,
                        ),
                    )
                    chat_sessions[user_id] = chat
                except Exception:
                    pass
                continue
            break

    if not answer:
        print(f"Ошибка Gemini ({user_id}): {last_error}")
        await message.answer(TEXTS[lang]["error"])
        return

    # Отправка частями (с безопасным парсингом Markdown)
    for i in range(0, len(answer), 4000):
        chunk = answer[i:i + 4000]
        try:
            await message.answer(chunk, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            await message.answer(chunk, parse_mode=None)


# --- ЗАПУСК БОТА ---
async def main():
    if not BOT_TOKEN:
        print("❌ ОШИБКА: TELEGRAM_BOT_TOKEN не настроен в .env!")
        return

    # Регистрируем меню команд бота
    await bot.set_my_commands([
        BotCommand(command="start", description="Boshlash / Перезапуск"),
        BotCommand(command="mode", description="Rejimni tanlash / Выбор режима"),
        BotCommand(command="memory", description="Mening profilim / Мой профиль"),
        BotCommand(command="clear", description="Tarixni tozalash / Очистить контекст"),
        BotCommand(command="help", description="Yordam / Справка"),
    ])

    print("🚀 Qudrat AI Bot (aiogram + Gemini Tools AFC) muvaffaqiyatli ishga tushdi...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
