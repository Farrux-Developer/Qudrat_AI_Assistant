import asyncio
import base64
import io
import os
import re

from dotenv import load_dotenv
from PIL import Image

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
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
GEMINI_MODEL_DEFAULT = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN))
dp = Dispatcher(storage=MemoryStorage())
gemini_client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None

# Хранилище контекста диалогов (user_id -> list[genai_types.Content])
histories: dict[int, list] = {}


# --- FSM СОСТОЯНИЯ ---
class BotStates(StatesGroup):
    choosing_language = State()
    waiting_oferta = State()
    active_chat = State()


# --- ТЕКСТЫ ОФЕРТЫ И СООБЩЕНИЙ ---
TEXTS = {
    "uz": {
        "oferta": (
            "📋 *OMMAVIY OFERTA (FOYDALANISH SHARTLARI)*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "1. *Predmet:* «Qudrat AI» boti Google Gemini sun’iy intellektiga vositachi kirishni «boricha» (AS IS) taqdim etadi.\n\n"
            "2. *Mas’uliyatdan voz kechish:* Javoblar neyrotarmoq tomonidan inson aralashuvisiz avtomatik yaratiladi. "
            "Ma’muriyat javoblarning mutlaq to‘g‘riligiga kafolat bermaydi. Bot javoblari rasmiy yuridik, tibbiy yoki "
            "moliyaviy maslahat hisoblanmaydi. Foydalanish uchun to‘liq mas’uliyat foydalanuvchining zimmasida.\n\n"
            "3. *Qoidalar va cheklovlar:* Botdan noqonuniy maqsadlarda, spamlash yoki buzg‘unchilik uchun foydalanish taqiqlanadi.\n\n"
            "4. *Aksept:* «✅ Roziman / Qabul qilish» tugmasini bosish ushbu shartlarga to‘liq va so‘zsiz rozilik bildirish hisoblanadi."
        ),
        "btn_accept": "✅ Roziman / Qabul qilish",
        "ready": (
            "✅ *AI-YORDAMCHI ISHGA TAYYOR*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "💬 *Savolingizni yozing yoki rasm yuboring:*"
        ),
        "not_accepted": (
            "⚠️ *Botdan foydalanish uchun avval ommaviy ofertani qabul qiling!*\n"
            "Qayta boshlash uchun /start buyrug‘ini bosing."
        ),
        "system_prompt": (
            "Siz — Qudrat AI yuqori toifali intellektual yordamchisisiz. "
            "O'zbek tilida aniq, teran, xushmuomala va professional darajada javob bering. "
            "Javoblarni chiroyli Markdown formatida (sarlavhalar, ro'yxatlar, qalin shrift) tuzing."
        ),
        "empty_answer": "Sun’iy intellekt bo‘sh javob qaytardi. Qayta urinib ko‘ring.",
        "error": "⚠️ Javob olishda xatolik yuz berdi. Iltimos, keyinroq urinib ko‘ring.",
    },
    "ru": {
        "oferta": (
            "📋 *ПУБЛИЧНАЯ ОФЕРТА (УСЛОВИЯ ИСПОЛЬЗОВАНИЯ)*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "1. *Предмет:* Бот «Qudrat AI» предоставляет доступ к интерфейсу ИИ Google Gemini на условиях «КАК ЕСТЬ» (AS IS).\n\n"
            "2. *Отказ от ответственности:* Ответы формируются нейросетью автоматически без предварительной модерации человеком. "
            "Администрация не гарантирует абсолютную точность ответов. Ответы не являются официальной юридической, медицинской или "
            "финансовой консультацией. Пользователь несёт личную ответственность за использование информации.\n\n"
            "3. *Правила и ограничения:* Запрещено использовать сервис в противоправных целях, для спама и создания вредоносного ПО.\n\n"
            "4. *Акцепт:* Нажатие кнопки «✅ Принять оферту» означает полное и безоговорочное согласие с условиями."
        ),
        "btn_accept": "✅ Принять оферту",
        "ready": (
            "✅ *AI-АССИСТЕНТ ГОТОВ К РАБОТЕ*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "💬 *Напишите ваш вопрос или отправьте фото для анализа:*"
        ),
        "not_accepted": (
            "⚠️ *Для работы с ботом необходимо принять оферту!*\n"
            "Нажмите /start для начала работы."
        ),
        "system_prompt": (
            "Ты — высококлассный интеллектуальный AI-ассистент Qudrat AI. "
            "Отвечай на грамотном русском языке точно, вежливо, структурированно и профессионально. "
            "Используй форматирование Markdown (заголовки, списки, выделения) для наглядности."
        ),
        "empty_answer": "ИИ вернул пустой ответ. Попробуйте переформулировать вопрос.",
        "error": "⚠️ Произошла ошибка при получении ответа от ИИ. Попробуйте ещё раз.",
    },
}


# --- КЛАВИАТУРЫ ---
def language_keyboard() -> InlineKeyboardMarkup:
    """1. Выбор языка: O'zbekcha / Русский."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="🇺🇿 O'zbekcha", callback_data="lang:uz"),
            InlineKeyboardButton(text="🇷🇺 Русский", callback_data="lang:ru"),
        ]
    ])


def oferta_keyboard(lang: str) -> InlineKeyboardMarkup:
    """2. Единственная кнопка под текстом оферты."""
    btn_text = TEXTS[lang]["btn_accept"]
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=btn_text, callback_data="accept_oferta")]
    ])


# --- 1. СТАРТ: ВЫБОР ЯЗЫКА ---
@dp.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()
    await state.set_state(BotStates.choosing_language)

    await message.answer(
        "✨ *━━━━━━━━━━━━━━━━━━━━*\n"
        "🤖 *QUDRAT AI ASSISTANT*\n"
        "✨ *━━━━━━━━━━━━━━━━━━━━*\n\n"
        "🌐 *Muloqot tilini tanlang / Выберите язык общения:*",
        reply_markup=language_keyboard(),
    )


# --- 2. ВЫБОР ЯЗЫКА -> ОТПРАВКА ОФЕРТЫ С ОДНОЙ КНОПКОЙ ---
@dp.callback_query(F.data.startswith("lang:"), BotStates.choosing_language)
async def process_language_choice(query: CallbackQuery, state: FSMContext):
    await query.answer()
    lang = query.data.split(":")[1]

    # Сохраняем выбранный язык в FSM
    await state.update_data(lang=lang)
    await state.set_state(BotStates.waiting_oferta)

    # Динамично удаляем сообщение с выбором языка
    try:
        await query.message.delete()
    except Exception:
        pass

    # Отправляем оферту с единственной кнопкой принятия
    await query.message.answer(
        TEXTS[lang]["oferta"],
        reply_markup=oferta_keyboard(lang),
    )


# --- 3. ПРИНЯТИЕ ОФЕРТЫ -> ЗАПУСК ИИ («AI-YORDAMCHI ISHGA TAYYOR») ---
@dp.callback_query(F.data == "accept_oferta", BotStates.waiting_oferta)
async def process_accept_oferta(query: CallbackQuery, state: FSMContext):
    await query.answer()
    data = await state.get_data()
    lang = data.get("lang", "uz")

    # Переводим пользователя в активный режим общения с ИИ
    await state.set_state(BotStates.active_chat)

    # Динамично удаляем сообщение оферты
    try:
        await query.message.delete()
    except Exception:
        pass

    # Выводим подтверждение готовности ИИ
    await query.message.answer(TEXTS[lang]["ready"])


# --- 4. ДИАЛОГ С ИИ: ОБРАБОТКА ТЕКСТА И ФОТО (АКТИВНЫЙ РЕЖИМ) ---
@dp.message(BotStates.active_chat, F.text | F.photo)
async def handle_ai_message(message: Message, state: FSMContext):
    data = await state.get_data()
    lang = data.get("lang", "uz")

    # Индикатор набора текста
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
        text = "Ushbu rasmni tasvirlab bering." if lang == "uz" else "Опишите это изображение."

    # История контекста пользователя
    user_id = message.from_user.id
    history = histories.setdefault(user_id, [])

    parts = []
    if text:
        parts.append(genai_types.Part(text=text))
    if image_bytes:
        parts.append(genai_types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))

    # Добавляем реплику пользователя
    history.append(genai_types.Content(role="user", parts=parts))
    # Ограничиваем последние 12 реплик для скорости
    history[:] = history[-12:]

    # Запрос к Gemini
    try:
        candidate_models = [
            GEMINI_MODEL_DEFAULT,
            "gemini-3.1-flash-lite",
            "gemini-3.5-flash-lite",
            "gemini-3.8-flash",
        ]
        answer = None
        last_error = None

        for model_name in candidate_models:
            try:
                response = await asyncio.to_thread(
                    gemini_client.models.generate_content,
                    model=model_name,
                    contents=history,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=TEXTS[lang]["system_prompt"],
                        max_output_tokens=1500,
                    ),
                )
                answer = response.text or ""
                break
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                if any(x in err_str for x in ("503", "unavailable", "404", "high demand")):
                    continue
                raise e

        if not answer:
            if last_error:
                raise last_error
            answer = TEXTS[lang]["empty_answer"]

        # Сохраняем ответ модели в историю
        history.append(genai_types.Content(role="model", parts=[genai_types.Part(text=answer)]))
        history[:] = history[-12:]

        # Отправляем ответ частями (лимит Telegram 4096 символов)
        for i in range(0, len(answer), 4000):
            chunk = answer[i:i + 4000]
            try:
                await message.answer(chunk, parse_mode=ParseMode.MARKDOWN)
            except Exception:
                await message.answer(chunk, parse_mode=None)

    except Exception as error:
        if history and history[-1].role == "user":
            history.pop()

        print(f"Ошибка Gemini: {error}")
        await message.answer(TEXTS[lang]["error"], parse_mode=None)


# --- 5. ЗАЩИТНЫЙ ХЭНДЛЕР: ЕСЛИ ОФЕРТА ЕЩЁ НЕ ПРИНЯТА ---
@dp.message()
async def block_unaccepted(message: Message, state: FSMContext):
    data = await state.get_data()
    lang = data.get("lang", "uz")
    await message.answer(TEXTS[lang]["not_accepted"])


# --- ЗАПУСК БОТА ---
async def main():
    if not BOT_TOKEN:
        print("❌ ОШИБКА: TELEGRAM_BOT_TOKEN не настроен в .env!")
        return

    # Регистрируем команду /start в меню Telegram
    await bot.set_my_commands([
        BotCommand(command="start", description="Boshlash / Запуск бота"),
    ])

    print("Qudrat AI Bot (aiogram) muvaffaqiyatli ishga tushdi...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
