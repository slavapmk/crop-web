# clipweb

CLI для обработки записей стримов: анализирует заданную область веб-камеры, удаляет речь стримера, длительные интервалы отсутствия лица и вероятные эпизоды курения. Итоговое видео кадрируется по прямоугольнику веб-камеры и сохраняет исходное аудио оставленных участков.

## Запуск

```bash
uv sync
uv run clipweb
```

Положите видео в `input/`; результат появится в `output/`. Для первого запуска скопируйте `config.yaml` и задайте координаты области камеры в пикселях относительно исходного кадра. Можно обработать файл напрямую: `uv run clipweb --source /path/to/stream.mp4`.

Результат анализа автоматически сохраняется в `output/.clipweb_cache/` до начала сборки. Если сборка прервётся, повторный запуск возьмёт интервалы из кэша; `uv run clipweb --reanalyze` принудительно повторит анализ. Изменение исходного видео, координат или параметров анализа автоматически делает кэш недействительным. Сборка использует FFmpeg concat demuxer с переходами к выбранным точкам, чтобы не прогонять всё длинное видео через отдельную ветвь фильтра для каждого фрагмента.

Все параметры хранятся в `config.yaml`. `analysis.invert_speech: true` оставляет только речевые фрагменты с обнаруженным лицом; короткие паузы и края слов сглаживаются параметрами окна и padding. `speech_detection: both` — рекомендуемый режим: WebRTC VAD ищет голос, а движение рта в вебке связывает его со стримером. Так игровой диалог без движения рта обычно не считается речью стримера. `face_missing_tolerance_seconds` задаёт допустимые короткие пропуски детектора лица.

Выходной кадр вырезается по `webcam.x/y/width/height`. По умолчанию `smoking.enabled: true`: CPU ONNX-модель проверяет только фрагменты, уже прошедшие фильтр лица и речи, и удаляет вероятные эпизоды курения с заданным запасом до и после обнаружения. Это экспериментальный одно-классовый детектор, а не надёжная классификация: маленькие сигареты, электронки и рука у лица могут быть пропущены или ошибочно распознаны. Порог, частота выборки и сглаживание меняются в `smoking`; фильтр можно отключить через `smoking.enabled: false`. Модель взята из [Smoking-detection-yolo11](https://github.com/alihassanml/Smoking-detection-yolo11), опубликована там под MIT; в проекте используется ONNX через OpenCV DNN без PyTorch/Ultralytics.

Нужно установить системный FFmpeg; см. [REQUERMENTS.md](REQUERMENTS.md). Python-зависимости устанавливает `uv sync`. Никаких ручных команд создания/активации окружения не требуется: `uv` использует `.venv` проекта автоматически.

## Ограничения

Детектор лица и оценка движения рта эвристические. Жевание, повороты головы, слабый свет, маленькое лицо или играющая речь могут давать ошибки; пороги можно настроить. Это не распознавание личности и не точная фонемная синхронизация. Проверяйте выбранные режимы на нескольких коротких отрезках исходной записи перед обработкой длинного стрима. Целевая скорость зависит от исходного кодека и числа ядер; замер на рабочем CPU обязателен.

## Diagnostic coverage

The analysis log reports the percentage of samples with a detected face, voice activity, mouth motion and speech cuts, followed by the duration retained before and after smoking removal. Low face coverage may indicate an incorrect webcam rectangle or an unreliable face detector. The analysis cache is invalidated automatically after the sampling fix.

`analysis.speech_detection: both` combines mouth motion with voice activity; `mouth` uses motion alone. Neither reliably distinguishes speech from eating, drinking, yawning or background voices. Smoking detection is experimental and can also produce false positives. Validate these settings on representative footage.

[Русская версия](README.ru.md)
