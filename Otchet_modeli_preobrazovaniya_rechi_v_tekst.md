# Сравнительный анализ доступных On-Premise ASR-систем и STT-моделей

ASR-система обычно поставляется как готовый сервис/API с продуманным пайплайном (препроцессинг → извлечение признаков → декодер → пост-обработка), её развёртывание и подключение сводятся к конфигурации сервера и обращению по HTTP/gRPC.

В свою очередь, STT-модель — это просто файл весов и SDK, который придётся встроить в код и окружение, а при необходимости реализовать на базе нее собственный сервис (управление входными потоками, очередями, масштабирование).

## Локальные (On-Premise), бесплатные ASR-системы с поддержкой русского языка:

| Технология   | Релиз системы          | Лицензия           | Описание (факты)                                                                                             | ASR-модель              | Версия модели    | Вес модели  |
| ------------ | ---------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------ | ----------------------- | ---------------- | ----------- |
| Kaldi        | v5.5.636 (26 Oct 2020) | Apache License 2.0 | C++-toolkit для построения ASR-пайплайнов на HMM/DNN, кроссплатформенный (Linux/BSD/macOS/Win через Cygwin)  | Librispeech Chain Model | tdnn\_1d\_sp     | 214 MB      |
|              |                        |                    |                                                                                                              | Librispeech 3-gram LM   | pruned           | 197 MB      |
|              |                        |                    |                                                                                                              | I-vector extractor      | —                | 19 MB       |
| Vosk         | v0.3.50 (Apr 2024)     | Apache License 2.0 | API на Python/Java/C#/…, офлайн-распознавание с кастомизацией словаря без перезапуска сервиса                | vosk-model-small-ru     | 0.22             | 45 MB       |
|              |                        |                    |                                                                                                              | vosk-model-ru           | 0.42             | 1.8 GB      |
| Julius       | v4.6 (02 Sep 2020)     | BSD-3-Clause       | Двухпроходный LVCSR-декодер с контекст-зависимыми HMM и 60k-словной 3-граммной LM                            | Japanese Dictation Kit  | GMM/DNN triphone | 150 MB      |
| DeepSpeech   | v0.9.3 (10 Dec 2020)   | MPL 2.0            | TensorFlow-движок; поддерживает форматы `.pbmm` и `.tflite` + внешняя языковая модель `.scorer`              | deepspeech-0.9.3        | pbmm             | 188 MB      |
|              |                        |                    |                                                                                                              | deepspeech-0.9.3        | tflite           | 57 MB       |
| PocketSphinx | v5.0.4 (10 Jan 2025)   | BSD-2-Clause       | C-движок для встроенных систем; минимальные зависимости; офлайн-распознавание                                | en-US acoustic + LM     | —                | 55 MB       |
| Coqui STT    | v1.4.0 (03 Sep 2022)   | MPL 2.0            | Форк DeepSpeech; поддержка GPU/CPU и TensorFlow Lite, совместимость с оригинальным API                       | coqui-stt-1.4.0         | pbmm             | 185 MB      |
|              |                        |                    |                                                                                                              | coqui-stt-1.4.0         | scorer           | 108 MB      |
| OpenSeq2Seq  | v0.17.0 (Apr 2020)     | Apache License 2.0 | NVIDIA-toolkit на TensorFlow для seq2seq-моделей (ASR/TTS), Conformer/Transformer-LM                         | librispeech-asr-chain   | —                | 214 MB      |
| SpeechBrain  | v0.5.11 (Nov 2023)     | Apache License 2.0 | PyTorch-фреймворк для ASR/TTS/NLP с модульным API; рецепты на Hugging Face                                   |                         |                  |             |

Kaldi: https://github.com/kaldi-asr/kaldi 

Vosk: https://alphacephei.com/vosk/ 

Julius: https://github.com/julius-speech/julius 

DeepSpeech: https://github.com/mozilla/DeepSpeech 

PocketSphinx: https://github.com/cmusphinx/pocketsphinx

Coqui STT: https://github.com/coqui-ai/STT

Wav2Letter++: https://github.com/facebookresearch/wav2letter 

OpenSeq2Seq: https://github.com/NVIDIA/OpenSeq2Seq 

SpeechBrain: https://github.com/speechbrain/speechbrain

Whisper.cpp: https://github.com/ggerganov/whisper.cpp

## Открытые (Open-Source), бесплатные STT-модели с поддержкой русского языка:

| Технология          | Модель                     | Версия | Вес          | Лицензия           | Описание (факты)                                                                      |
| ------------------- | -------------------------- | ------ | ------------ | ------------------ | ------------------------------------------------------------------------------------- |
| GigaAM              | ctc                        | v2.0   | 233 M params | MIT License        | Conformer-SSL семейство моделей для Russian ASR; предтренировано на 50 000+ ч аудио   |
|                     | rnnt                       | v2.0   | 234 M params |                    | RNNT-вариант семейства                                                                |
|                     | emo                        | v1.0   | 240 M params |                    | Модель для распознавания эмоций в речи                                                |
| Whisper             | tiny-ru                    | —      | 39 MB        | MIT License        | Multilingual Transformer-STT от OpenAI, адаптация tiny для русского                   |
|                     | base-ru                    | —      | 74 MB        |                    | Адаптация base                                                                        |
|                     | small-ru                   | —      | 244 MB       |                    | Адаптация small                                                                       |
|                     | medium-ru                  | —      | 769 MB       |                    | Адаптация medium                                                                      |
|                     | large-ru                   | —      | 2.9 GB       |                    | Адаптация large                                                                       |
| NeMo Conformer-RNNT | conformer-rnnt-ru          | —      | 387 M params | Apache License 2.0 | NVIDIA Conformer-RNNT для ASR, поддерживает русский; модели доступны на Hugging Face  |
| Silero-models       | silero\_stt\_small\_ru     | —      | 47 MB        | MIT License        | Zipformer-подобная архитектура для Russian STT; поддержка PyTorch/TorchScript/ONNX    |
|                     | silero\_stt\_large\_ru     | —      | 146 MB       |                    | Большая версия Zipformer-архитектуры                                                  |
| Coqui STT (“ru”)    | coqui-ru-1.3.0             | —      | 180 MB       | MPL 2.0            | Fork DeepSpeech с поддержкой русского, TensorFlow Lite форматы                        |
|                     | coqui-ru-1.4.0             | —      | 185 MB       |                    | Обновлённый fork DeepSpeech                                                           |
| DeepSpeech (“ru”)   | deepspeech-0.9.3-ru-models | —      | 188 MB       | MPL 2.0            | Community-модель для русского, форматы PBMM/TFLite                                    |
| Whisper.cpp (“ru”)  | tiny-ru                    | —      | 39 MB        | MIT License        | C++-порт Whisper для русского                                                         |
|                     | base-ru                    | —      | 74 MB        |                    | —                                                                                     |
|                     | small-ru                   | —      | 244 MB       |                    | —                                                                                     |
|                     | medium-ru                  | —      | 769 MB       |                    | —                                                                                     |
|                     | large-ru                   | —      | 2.9 GB       |                    | —                                                                                     |
| Vosk-model-ru       | ru-small-0.22              | —      | 45 MB        | Apache License 2.0 | Модель малой емкости для мобильных задач                                              |
|                     | ru-0.42                    | —      | 1.8 GB       |                    | Серверная модель                                                                      |
| Wav2Vec 2.0 XLSR-53 | large-xlsr-53              | —      | 990 M params | Apache License 2.0 | Self-supervised модель Facebook для 53 языков (включая русский)                       |
| Wav2Letter++ (“ru”) | v0.2                       | —      | 1.0 GB       | Apache License 2.0 | C++-реализация ASR от Facebook, адаптированная к русским данным                       |

GigaAM: https://github.com/salute-developers/GigaAM

Whisper: https://github.com/openai/whisper

NeMo Conformer-RNNT: https://github.com/NVIDIA/NeMo 

Silero: https://github.com/snakers4/silero-models

Coqui STT: https://github.com/coqui-ai/STT

DeepSpeech (“ru”): https://github.com/mozilla/DeepSpeech

Whisper.cpp: https://github.com/ggerganov/whisper.cpp

Vosk-model-ru: https://huggingface.co/alphacep/vosk-model-small-ru

Wav2Vec 2.0 XLSR-53: https://huggingface.co/facebook/wav2vec2-large-xlsr-53

Wav2Letter++: https://github.com/facebookresearch/wav2letter

## Метрики качества распознавания речи для оценки speech-to-text (STT) моделей:

### Word Error Rate (WER) 

Основной метрикой качества Automatic Speech Recognition (ASR) является Word Error Rate (WER), которая рассчитывается между гипотезой модели и истинной транскрипцией с помощью расстояния Левенштейна и может быть интерпретирована как доля неправильно распознанных слов. 

Чем ниже WER, тем точнее модель.

WER рассчитывается по формуле:

$$
WER = \frac{\text{подстановки} + \text{вставки} + \text{удаления}}{\text{общее число слов в эталонном тексте}}
$$ 

### Character Error Rate (CER)

Character Error Rate (CER) - дополнительная метрика, измеряющая точность распознавания на уровне отдельных символов, а не слов. 

CER обеспечивает более детальный взгляд на точность распознавания, особенно в случаях, когда модель может правильно распознать большую часть слова, но ошибиться в нескольких символах. 

$$
CER = \frac{\text{подстановки} + \text{вставки} + \text{удаления (на уровне символов)}}{\text{общее число символов в эталонном тексте}} 
$$

### GigaAM

GigaAM (Giga Acoustic Model) - семейство акустических моделей для обработки звучащей речи на русском языке. 

Среди решаемых задач - задачи распознавания речи, распознавания эмоций и извлечения эмбеддингов из аудио. 

Модели построены на основе архитектуры Conformer с использованием методов self-supervised learning (wav2vec2-подход для GigaAM-v1 и HuBERT-подход для GigaAM-v2).

GigaAM (Giga Acoustic Model) — фундаментальная акустическая модель, основанная на Conformer-энкодере (около 240M параметров) и обученная на 50 тысячах часов разнообразных русскоязычных данных.

Доступны 2 версии модели, отличающиеся алгоритмом предобучения:

GigaAM-v1 была обучена на основе подхода wav2vec2. Версия модели для использования - v1_ssl.

GigaAM-v2 была обучена на основе подхода HuBERT и позволила улучшить качество распознавания речи. Версия модели для использования - v2_ssl или ssl.

GigaAM энкодер был дообучен для задачи распознавания речи с двумя разными декодерами:

Модели GigaAM-CTC были дообучены с CTC функцией потерь.

Модели GigaAM-RNNT была дообучена с RNN-T функцией потерь.

[Официальные результаты бенчмарка (Метрика Word Error Rate), подготовленного командой разработчиков GigaAM (Giga Acoustic Model) - акустической модели для обработки звучащей речи на русском языке:](https://github.com/salute-developers/GigaAM/blob/main/README_ru.md#метрики-качества-word-error-rate)

| Model              | Parameters | Golos Crowd | Golos Farfield | OpenSTT YouTube | OpenSTT Phone Calls | OpenSTT Audiobooks | Mozilla Common Voice 12 | Mozilla Common Voice 19 | Russian LibriSpeech |
|--------------------|------------|-------------|----------------|-----------------|----------------------|--------------------|-------|-------|---------------------|
| Whisper-large-v3   | 1.5B       | 13.9        | 16.6           | 18.0            | 28.0                 | 14.4               | 5.7   | 5.5   | 9.5                 |
| NeMo Conformer‑RNNT  | 120M     | 2.6         | 7.2            | 24.0            | 33.8                 | 17.0               | 2.8   | 2.8   | 13.5                |
| **GigaAM-CTC-v1**  | 242M       | 3.0         | 5.7            | 16.0            | 23.2                 | 12.5               | 2.0   | 10.5  | 7.5                 |
| **GigaAM-RNNT-v1** | 243M       | 2.3         | 5.0            | 14.0            | 21.7                 | 11.7               | 1.9   | 9.9   | 7.7                 |
| **GigaAM-CTC-v2**  | 242M       | 2.5         | 4.3            | 14.1            | 21.1                 | 10.7               | 2.1   | 3.1   | 5.5                 |
| **GigaAM-RNNT-v2** | 243M       | **<span style="color:green">2.2</span>**         | **<span style="color:green">3.9</span>**            | **<span style="color:green">13.3</span>**            | **<span style="color:green">20.0</span>**                | **<span style="color:green">10.2</span>**               | **<span style="color:green">1.8</span>**   | **<span style="color:green">2.7</span>**   | **<span style="color:green">5.5</span>**  

## Whisper 

<!-- There are six model sizes, four with English-only versions, offering speed and accuracy tradeoffs.
Below are the names of the available models and their approximate memory requirements and inference speed relative to the large model.
The relative speeds below are measured by transcribing English speech on a A100, and the real-world speed may vary significantly depending on many factors including the language, the speaking speed, and the available hardware. -->

Существует шесть размеров моделей, четыре из которых имеют версии только на английском языке, предлагая компромиссы между скоростью и точностью. 

Ниже приведены названия доступных моделей, их приблизительные требования к памяти и скорость вывода относительно большой модели. 

Относительные скорости, указанные ниже, измеряются путем транскрибирования английской речи на A100, а реальная скорость может значительно отличаться в зависимости от многих факторов, включая язык, скорость речи и доступное оборудование.

|  Size  | Parameters | English-only model | Multilingual model | Required VRAM | Relative speed |
|:------:|:----------:|:------------------:|:------------------:|:-------------:|:--------------:|
|  tiny  |    39 M    |     `tiny.en`      |       `tiny`       |     ~1 GB     |      ~10x      |
|  base  |    74 M    |     `base.en`      |       `base`       |     ~1 GB     |      ~7x       |
| small  |   244 M    |     `small.en`     |      `small`       |     ~2 GB     |      ~4x       |
| medium |   769 M    |    `medium.en`     |      `medium`      |     ~5 GB     |      ~2x       |
| large  |   1550 M   |        N/A         |      `large`       |    ~10 GB     |       1x       |
| turbo  |   809 M    |        N/A         |      `turbo`       |     ~6 GB     |      ~8x       |

<!-- Whisper's performance varies widely depending on the language. The figure below shows a performance breakdown of `large-v3` and `large-v2` models by language, using WERs (word error rates) or CER (character error rates, shown in *Italic*) evaluated on the Common Voice 15 and Fleurs datasets. Additional WER/CER metrics corresponding to the other models and datasets can be found in Appendix D.1, D.2, and D.4 of [the paper](https://arxiv.org/abs/2212.04356), as well as the BLEU (Bilingual Evaluation Understudy) scores for translation in Appendix D.3. -->

Производительность Whisper сильно различается в зависимости от языка. 

На рисунке ниже показана разбивка производительности large-v3 и large-v2 моделей по языкам с использованием WER (коэффициенты ошибок в словах) или CER (коэффициенты ошибок в символах, выделенные курсивом ), оцененных на наборах данных Common Voice 15 и Fleurs. 

Дополнительные метрики WER/CER, соответствующие другим моделям и наборам данных, можно найти в Приложении D.1, D.2 и D.4 статьи , а также оценки BLEU (Bilingual Evaluation Understudy) для перевода в Приложении D.3.

![WER, CER, large-v3, large-v2 моделей](image-1.png)

### Метрики качества (Word Error Rate, Character Error Rate, Speed)

Оценка качества автоматической транскрипции голосовых сообщений (Сравнительный анализ на аудиоданных, собранных при взаимодействии студентов с тестовой версией чат-бота «Вопрошалыч»):

|Model								|Parameters	| Word Error Rate (WER)	| Character Er-ror Rate (CER)	|Speed (x=1 second) |
|-----------------------------------|-----------|-----------------------|-------------------------------|-------------------|
|Whisper-large-v3					|1.5B  		|6.86 					| 3.25							| 0.83x				|
|vosk-model-small-ru-0.22			|45M		|5.67					| 1.71							| 0.231x			|
|h2oai/faster-whisper-large-v3-turbo|809M		|2.98					| 0.76							| 0.44x				|
|GigaAM-CTC-v1						|242M		|6.53					| 2.63							| 0.566x			|
|GigaAM-RNNT-v1						|243M		|3.03					| 0.75							| 0.476x			|
|GigaAM-CTC-v2						|242M		|3.1					| 0.71							| 0.406x			|
|GigaAM-RNNT-v2						|243M		|2.94					| 0.71							| 0.383x			|

Обработка и транскрипция аудиозаписей осуществлялись локально на вычислительной платформе (Процессор AMD Ryzen 7 3700X, 16 ГБ оперативной памяти стандарта DDR4).

Для оценки качества распознавания речи использовалась библиотека jiwer.
