# Neural Networks & Data Science · Учебное портфолио

**Алексей Сафонов · Python · Computer Vision · NLP · Neural Networks · Data Analysis**

Вместо стандартной страницы GitHub здесь теперь собран навигатор по историческим лабораторным и учебным Jupyter-ноутбукам. Это *архив практики*, а не утверждение, что все модели актуальны, воспроизводимы или пригодны для продакшена.

[**24 работы по нейросетям →**](docs/NEURAL_NETWORK_NOTEBOOKS.md) · [**Новый воспроизводимый MLP-пример →**](ML/reproducible_mlp_xor_numpy.ipynb) · [**AI PATH — программы обучения →**](https://safal207.github.io/test-it-python/ai-path/)

---

## Что в репозитории

**Историческая коллекция:** 65 Jupyter-ноутбуков в снимке до обновления (2026-10-18). **Новый пример:** плюс один контролируемый notebook, поэтому после обновления ожидается **66 файлов .ipynb**.

| Практика | Каталог | Исторических ноутбуков |
| --- | --- | ---: |
| Компьютерное зрение, CIFAR, MNIST, сегментация | [CV](CV) | 7 |
| Обработка языка, тональность, авторство, RNN/LSTM/GRU | [NLP](NLP) | 10 |
| Аудио и классификация птиц | [AudioNet](AudioNet) | 2 |
| Классификация и эксперименты на изображениях | [Classification](Classification) | 3 |
| Настройка архитектур и Keras Tuner | [SeletNetAI](SeletNetAI) | 2 |
| Исторические медицинские исследовательские задания | [medicina](medicina) | 10 |
| Python, NumPy, Pandas, визуализация, статистика | [python](python), [Numpy](Numpy), [Pandas](Pandas), [ML](ML) | см. каталоги |
| Временные ряды, акции, прикладной анализ | [Stock](Stock), [Trafic](Trafic) | см. каталоги |

**Навигация:** [каталог 24 нейросетевых ноутбуков с прямыми ссылками](docs/NEURAL_NETWORK_NOTEBOOKS.md).

## С чего начать новичку

1. Установить Python 3.10+ и NumPy: `python -m pip install "numpy>=1.24"`.
2. Открыть [`ML/reproducible_mlp_xor_numpy.ipynb`](ML/reproducible_mlp_xor_numpy.ipynb) в Jupyter или Colab.
3. Запустить по порядку: генерация **синтетических** данных, непересекающиеся train/validation/test, нормализация только на train, нейросеть с `tanh`, итоговая проверка на test.
4. Убедиться, что код понятен без ИИ, изменить один параметр и сохранить отдельный отчёт об эксперименте.
5. Затем выбрать **один** исторический ноутбук и воспроизвести его с указанием ограничений.

### Контроль воспроизводимости

В репозиторий добавлены:

- [`tools/check_notebooks.py`](tools/check_notebooks.py) — **безопасный структурный аудит** формата notebook, числа ячеек и потенциально устаревших outputs; не запускает архивный код.
- [`ML/reproducible_mlp_xor_numpy.ipynb`](ML/reproducible_mlp_xor_numpy.ipynb) — один **новый** CPU-пример без скачивания данных, ключей и облачных сервисов.
- [`.github/workflows/notebook-audit.yml`](.github/workflows/notebook-audit.yml) — проверки структуры Jupyter и выполнение **только** нового примера с NumPy.

Локальная проверка после установки NumPy:

```bash
python tools/check_notebooks.py --run-example
```

Если архивный notebook требует старый TensorFlow/Keras, несовместимую версию Python или внешний датасет — сначала выделите отдельное окружение, зафиксируйте зависимости и перепроверьте данные. Общая зелёная проверка структуры **не означает** успешного воспроизведения всех старых экспериментов.

## Правила достоверной презентации

- Файлы в `medicina/` — исторические задания по классификации медицинских изображений, **не клинические диагностические изделия**.
- Цифры точности внутри *имён старых файлов* не считаются независимо подтверждёнными метриками.
- Не публикуем личные данные, медицинские снимки без прав, токены или закрытые датасеты.
- При обмене кодом/данными соблюдаем соответствующие лицензии. Отсутствие LICENSE означает, что автоматически выдавать разрешение на любое использование нельзя.
- При подготовке материалов для OpenAI, Anthropic и других лабораторий используем только воспроизводимые результаты, явно отмечаем предыдущие проекты и ограничения.

## English summary

A curated educational archive of 65 historical Jupyter notebooks plus one new, deterministic NumPy MLP example. Topics include computer vision, NLP, audio, classification, data science and time series. The [24-notebook ML index](docs/NEURAL_NETWORK_NOTEBOOKS.md) links to original files; the structural audit intentionally **does not execute legacy code**. This repository does not claim state-of-the-art performance or clinical validity.

**Author:** [Aleksei Safonov](https://github.com/safal207) · [AI PATH](https://safal207.github.io/test-it-python/ai-path/)
