# ⚡ Рабочая Демо-Версия LLM модели Falcon-7b-instruct
🔥 Обещанная демо-версия из Telegram-а: **"Эйнштейний | Про IT"** [Telegram](https://t.me/einsteinit)

## Как использовать Falcon? 🤔
1. Установите необходимые зависимости: `pip install huggingface_hub langchain streamlit`
2. Вставьте свой HuggingFace Token в нужное место
3. Запустите скрипт Falcon на выбранной вами среде, используя команду `streamlit run app.py`
4. Взаимодействуйте с Falcon, вводя запросы или указания на естественном языке (Желательно английский!)
5. Исследуйте ответы Falcon и наслаждайтесь ее помощью! 🚀🔍💡

## Работа модели 🚀
Самая загвоздка хорошей работы модели - Template(шаблон). Без шаблона модель будет работать не так, как Вы хотели бы.
В шаблоне можно задать работу модели. ЛЮБУЮ! Например: `Представь, что ты феечка, которая рассказывает интересные сказки на ночь` 🧚

**Вот Template:**

```
# Template
template = """
You are smart and polite artificial intelligence assistant.
You should give answer to any question user could ask you. Your answers should be truthful, detailed and consice.

-----------------------

Question: {question}

Answer:
"""
```
