# Этап 0 — результаты технической проверки

**Дата:** 2026-11-07

**Среда:** Linux, Python 3.12.3, раздельные `.venvs/era` и `.venvs/aide`

**Live-статус:** не запускался — требуется VPN

## Зафиксированные версии

| Компонент | Версия |
|---|---|
| ERA | `012c2d3957a3a6a249af2bbeeae53bbbd2182558` |
| AIDE ML | `ad73d227af481bea3de06f884f41b2993dd6453a` |
| Модель для live-проверки | `qwen3.8-27b` |
| OpenAI Python SDK | `3.46.0` |
| pytest | `9.0.3` |

## Матрица проверок

| Проверка | Статус | Результат |
|---|---|---|
| Подмодули и окружения | PASS | Оба репозитория закреплены на ожидаемых SHA; bootstrap повторяемо создаёт два venv. |
| FUTS без LLM | PASS | Проверены 12 раскрытий, журнал рёбер, лучший score, rank score, PUCT и backpropagation посещений. |
| AIDE Journal | PASS | Проверены метрики, `parent → child`, `children` и два последовательных mock-шага. |
| Подмена `search_policy()` | PASS | Локальный адаптер выбирает произвольный good-node, не изменяя генератор, feedback и upstream AIDE. |
| OpenAI-compatible запросы | PASS | MockTransport подтвердил Bearer-заголовок, модель, messages, `max_tokens`, temperature и forced tool call. |
| Docker sandbox | PASS | Проверены успешный код, ненулевой exit code, таймаут, отсутствие токена и недоступность сети. |
| Live chat completion | PASS | `qwen3.8-27b` вернула непустой ответ через VPN. |
| Live tool calling | PASS | Forced tool call вернул валидный аргумент. |
| AIDE с реальной LLM | PARTIAL | Первый draft-запрос был отклонён hosted vLLM из-за отсутствия user-message; затем повторный запуск остановился на reuse частичного workspace. Добавлены user-message adapter и уникальный workspace на попытку; требуется повторный прогон. |

## Выполненные команды

```bash
./scripts/stage0/bootstrap.sh
./scripts/stage0/run_offline.sh
```

Итог offline-набора: **18 passed**. Конфигурация из `../.env` отдельно провалидирована без сетевого запроса и без вывода адреса или токена.

## Подготовленный live-прогон

```bash
./scripts/stage0/run_live.sh --env-file ../.env
```

Порядок: обычный Chat Completions → forced tool call → два шага AIDE. Для API-smoke используются `max_tokens=512` и `temperature=0.7`; для code/feedback AIDE — 4096 токенов. Результат пишется в `artifacts/stage0/live/result.json`, а журнал, дерево и лучшее решение — в каталог логов AIDE внутри live-артефактов.

Для hosted vLLM локальный адаптер добавляет нейтральное user-сообщение к system-only запросам upstream AIDE. Он действует только во время live-прогона и не изменяет код подмодуля.

## Ограничения текущего результата

- Совместимость реального endpoint с chat completion и tool calling подтверждена.
- Полный цикл AIDE с реальной LLM нужно повторить после user-message compatibility-патча.
- Качество генерации модели не оценивалось: Этап 0 проверяет технический контракт и целостность контура.
- Host-окружение AIDE намеренно содержит только зависимости управляющего контура; библиотеки исполняемых моделей находятся в Docker-образе.
- FUTS ещё не интегрирован в AIDE: подтверждена только возможность заменить выбор родителя.
