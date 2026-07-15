# gap (go-ai-proxy) — руководство для разработчика клиента

Гайд по использованию **`gap`** — универсального OpenAI-совместимого прокси к разным
LLM-провайдерам. Один base_url, один формат запроса, любой провайдер за кулисами
(Anthropic, OpenAI, vLLM, LM Studio, LiteLLM и т.д.).

> TL;DR: точка входа — `http://gap.local:8090/v1` (порт зависит от деплоя), формат
> запросов **OpenAI Chat Completions**. Модель выбираете либо по ID, либо по возможностям
> через `auto:<capabilities>`. `reasoning_effort`, стриминг, tool calling, multimodal —
> поддерживаются прозрачно. Сам прокси по умолчанию **без авторизации** (закладывается
> на сетевую изоляцию или внешний reverse proxy); ключ в `api_key` передаётся
> апстриму, не валидируется самим прокси.

---

## 1. Как подключиться

Любой OpenAI-совместимый SDK достаточно перенацелить на base_url прокси:

| Параметр | Значение |
|---|---|
| Base URL | `http://<gap-host>:<port>/v1` (`/v1` обязателен) |
| `api_key` | передавайте любую непустую строку — прокси сам по умолчанию её не проверяет (см. §10). Значение прозрачно отправляется апстриму как `Authorization: Bearer <key>` для тех провайдеров, что требуют ключ. |
| `model` | конкретный ID (например `claude-sonnet-4-6`, `gpt-4o`, `gemma-4-31b-it`) **или** capability-селектор `auto:reasoning,vision` |

```bash
curl http://gap.local:8090/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "claude-sonnet-4-6",
  "messages": [{"role":"user","content":"Привет!"}]
}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://gap.local:8090/v1", api_key="any-string")
resp = client.chat.completions.create(
    model="auto:reasoning",
    messages=[{"role":"user","content":"Объясни шаг за шагом"}],
    reasoning_effort="high",
    stream=True,
)
```

---

## 2. Список моделей и их возможности

```bash
curl http://gap.local:8090/v1/models
```

Ответ — стандартный OpenAI-формат `{"object":"list","data":[...]}`, дополненный полем
`capabilities` (если апстрим/конфиг его сообщает) и опционально pricing:

```json
{
  "id": "claude-sonnet-4-6",
  "object": "model",
  "owned_by": "anthropic",
  "capabilities": ["vision", "reasoning", "tools", "prompt_caching"],
  "input_cost_per_token": 3e-06,
  "output_cost_per_token": 15e-06,
  "max_model_len": 200000
}
```

Возможные значения `capabilities`: `vision`, `reasoning`, `tools`, `pdf`,
`prompt_caching`, `structured_output`, `web_search`, `audio_input`,
`audio_output`, `computer_use`, `url_context`. Для генеративных моделей без
эмбеддингов список не содержит `embeddings` — отдельная подсистема.

Поле `max_model_len` (когда оно есть) — суммарный лимит на `prompt + completion`
в токенах. Прокси отдаёт его прозрачно, если апстрим сообщил (vLLM, например);
для провайдеров, которые лимит не репортят, поле отсутствует. Прокси сам **не**
валидирует размер запроса — превышение даст HTTP 400 от апстрима.

Список кэшируется прокси и периодически обновляется (обычно раз в час). Если
вы только что подняли новую модель в апстриме и она не появилась — отправьте
любой запрос с её ID, прокси триггернёт on-demand refresh и доразрешит её.

---

## 3. Capability-based routing (`auto:*`)

Вместо конкретной модели можно запрашивать любую с нужными возможностями:

```json
{"model": "auto:vision", "messages":[...]}             // любая с vision
{"model": "auto:reasoning,tools", "messages":[...]}    // c reasoning И tools
```

Прокси резолвит селектор перед маршрутизацией: среди подходящих моделей
выбирается **наименее загруженная** (по числу активных запросов). Если ни одна
модель не покрывает заявленные capabilities — `400 model not found`.

Полезно, когда:
- клиенту всё равно, какой именно поставщик отвечает («лишь бы умел картинки»);
- хотите автоматический failover между Claude / GPT / локалью без хардкода ID;
- балансируете нагрузку на однотипные модели у нескольких апстримов.

---

## 4. Reasoning (Chain-of-Thought)

### Включение

Стандартный OpenAI-параметр `reasoning_effort` (значения `none`, `minimal`,
`low`, `medium`, `high`) — единый интерфейс для всех провайдеров:

```json
{"model":"auto:reasoning","reasoning_effort":"high","messages":[...]}
```

Дополнительно для Anthropic-семейства можно явно задать бюджет токенов через
**`budget_tokens`** (наш канонический extension, аналог Anthropic
`thinking.budget_tokens`):

```json
{"model":"claude-sonnet-4-6","budget_tokens":8000,"messages":[...]}
```

Прокси сам подбирает upstream-механизм:
- Anthropic API → `thinking: {type:"enabled", budget_tokens}` (требует `temperature=1`);
- DeepSeek API → нативное поле `reasoning_content` в ответе;
- vLLM/Qwen3/Gemma-4 / Granite / DeepSeek-V3.1 → `chat_template_kwargs.{enable_thinking|thinking}`;
- локальные R1/QwQ модели через `<think>...</think>` теги — теги распарсиваются на стороне прокси.

### Чтение

Во всех случаях reasoning возвращается клиенту в **одном** поле
`choices[0].message.reasoning_content` (для нестрима) или
`choices[0].delta.reasoning_content` (для стрима). Имя поля совместимо с
OpenAI o1 и DeepSeek-клиентами.

```jsonc
{
  "choices":[{
    "message":{
      "content": "Финальный ответ.",
      "reasoning_content": "Сначала разберу задачу..."
    },
    "finish_reason":"stop"
  }]
}
```

⚠️ Если вы выставили `reasoning_effort` или `budget_tokens` без достаточного
`max_tokens`, генерация может оборваться внутри размышлений — придёт пустой
`content` и `finish_reason:"length"`. Закладывайте запас.

### vLLM-сервируемые модели

Когда модель отдаётся через провайдер `type: vllm`, прокси прозрачно
транслирует ваш стандартный OpenAI-параметр `reasoning_effort` в
механизм `chat_template_kwargs` vLLM:

| Клиент шлёт                     | Прокси пробрасывает в vLLM-апстрим                            |
| ------------------------------- | ------------------------------------------------------------- |
| `"reasoning_effort": "low"`     | `chat_template_kwargs: {"enable_thinking": true}`             |
| `"reasoning_effort": "medium"`  | `chat_template_kwargs: {"enable_thinking": true}`             |
| `"reasoning_effort": "high"`    | `chat_template_kwargs: {"enable_thinking": true}`             |
| `"reasoning_effort": "minimal"` | `chat_template_kwargs: {"enable_thinking": false}`            |
| `"reasoning_effort": "none"`    | `chat_template_kwargs: {"enable_thinking": false}`            |
| поле отсутствует                | без инъекции — действует server-side default vLLM             |

Для семейств Granite или DeepSeek-V3.1 оператор настраивает прокси с
`vllm.thinking_key: "thinking"`; маппинг выше остаётся тем же.

Апстрим vLLM возвращает reasoning-контент в поле `reasoning`; прокси
нормализует его в стандартное поле `reasoning_content` ответа — ваш
OpenAI-совместимый клиент видит ту же форму, что и для DeepSeek или
OpenAI o1 моделей. Попадание в prefix-кэш отражается через
`usage.prompt_tokens_details.cached_tokens`.

---

## 5. Стриминг

Стандартный OpenAI SSE: `"stream": true` в запросе, ответ кусками
`data: {...}\n\n` плюс терминатор `data: [DONE]\n\n`. Каждый чанк может
содержать:
- `delta.content` — обычный текст;
- `delta.reasoning_content` — размышления (если включены);
- `delta.tool_calls` — фрагменты аргументов tool-вызова (стримятся инкрементально);
- `finish_reason` в последнем чанке.

Чтобы получить итоговое `usage` в стрим-режиме:
```json
{"stream": true, "stream_options": {"include_usage": true}, ...}
```

Прокси корректно обрабатывает отмену клиента: при закрытии HTTP-соединения
upstream-запрос тоже отменяется (контекст пробрасывается). Утечек goroutine
не должно быть, но если заметили медленные стримы — `request_timeout` на стороне
прокси может пристрелить первую токен-задержку (см. §10).

---

## 6. Tool calling (функции)

Стандартный OpenAI tool-calling без изменений:

```json
{
  "model":"auto:tools",
  "messages":[{"role":"user","content":"Погода в Париже?"}],
  "tools":[{"type":"function","function":{
     "name":"get_weather",
     "description":"Current weather for a city",
     "parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}
  }}],
  "tool_choice":"auto"
}
```

Поддерживается:
- множественные параллельные tool-calls в одном ответе;
- многошаговый цикл (assistant → tool result → assistant);
- стриминг tool_calls (инкрементальные `function.arguments` дельты);
- комбинация tools + reasoning (модель сначала «думает», потом эмитит tool_calls).

Поведение прокси одинаковое независимо от того, что под капотом — Anthropic
с `tool_use` блоками транслируются в стандартный OpenAI `tool_calls` массив.

---

## 7. Structured output (`response_format`)

Стандартный OpenAI-параметр `response_format` для получения ответа в заданной JSON-схеме:

```json
{
  "model": "claude-sonnet-4-6",
  "messages": [{"role": "user", "content": "Extract the name."}],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "person",
      "strict": true,
      "schema": {"type": "object", "properties": {"name": {"type": "string"}}, "additionalProperties": false}
    }
  }
}
```

Учитывается только `type: "json_schema"` — `text` и `json_object` принимаются на входе,
но ни на что не влияют (no-op).

Куда попадает:
- OpenAI-совместимые апстримы (LM Studio, vLLM, LiteLLM, OpenAI) получают поле как
  есть, passthrough;
- Anthropic — маппится в нативный `output_config.format`; поля `name`/`strict` не
  имеют аналога в Anthropic API и отбрасываются, `schema` передаётся как есть.

Прокси **не валидирует и не гейтит** запрос по возможностям модели: если модель не
поддерживает structured output, ошибку (обычно `400`) вернёт апстрим. При этом на
этапе роутинга, если резолвленная модель не заявляет capability `structured_output`
(или capabilities вообще неизвестны), прокси пишет `WARN` в свой лог — это
диагностика для оператора, а не блокировка запроса.

### llama.cpp-сервируемые модели

llama.cpp принимает `response_format` со schema-путём (`json_schema`), но на
некоторых моделях (например, Gemma-3) этот путь падает у сэмплера — надёжно
работает только нативное поле **`grammar`** (GBNF). Поэтому для провайдера
`type: llama-cpp` клиенту ничего менять не нужно: он шлёт обычный OpenAI
`response_format`, а прокси прозрачно конвертирует `json_schema` в GBNF-грамматику
и подставляет её в `grammar`, вырезая `response_format` из тела перед отправкой
апстриму.

Конвертер поддерживает практическое подмножество (v1) JSON Schema:
`object`/`array`/`string`/`integer`/`number`/`boolean`, `enum`, `anyOf`/`oneOf`,
вложенные `properties` (все считаются required) и `items`. Не поддерживаются
`$ref`/`$defs`/`definitions`, `allOf`, `pattern`, `format`, числовые
ограничения (`minimum`/`maximum`/`multipleOf`/...) и ограничения длины
(`minLength`/`maxLength`/`minItems`/`maxItems`). Если схема использует что-то из
неподдерживаемого списка, прокси возвращает ошибку с названием проблемного
ключевого слова и советом задать грамматику явно.

Escape hatch: клиент может прислать GBNF-грамматику напрямую — канонический
extension-параметр **`grammar`** (строка) в теле запроса. Если `grammar`
присутствует, прокси форвардит её как есть, а `response_format` (если он тоже
присутствует в теле) вырезается — `grammar` имеет приоритет.

---

## 8. Multimodal (картинки / PDF)

OpenAI vision-формат:
```json
{"role":"user","content":[
  {"type":"text","text":"Что на картинке?"},
  {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}
]}
```

PDF (поддерживают только модели с `pdf` в capabilities):
```json
{"role":"user","content":[
  {"type":"text","text":"Резюмируй документ"},
  {"type":"file","file":{"file_data":"data:application/pdf;base64,..."}}
]}
```

Запрашивайте через `auto:vision` или `auto:pdf`, чтобы прокси сам выбрал
подходящую модель.

---

## 9. `usage` и оптимизация затрат

Стандартное OpenAI-поле `usage` в ответе плюс расширения:

```jsonc
{
  "usage":{
    "prompt_tokens": 1024,
    "completion_tokens": 150,
    "total_tokens": 1174,
    "prompt_tokens_details": {
      "cached_tokens": 900     // сколько входных токенов попало в prefix-cache
    }
  }
}
```

`cached_tokens` — **OpenAI-стандарт**, на стороне прокси заполняется для всех
апстримов, которые это сообщают (vLLM prefix-cache, Anthropic prompt caching,
OpenAI prompt cache). Используйте, чтобы оценивать эффективность стабильных
системных промптов.

**Best practice:** держите системный промпт **в начале** и стабильным между
запросами. Prefix-cache работает по префиксу — любое изменение в начале
обнуляет кэш для всего хвоста.

---

## 10. Лимиты, ошибки, ретраи

### Коды ошибок

| HTTP | Когда | Что делать |
|---|---|---|
| `400` | bad request, превышен `max_tokens`-budget провайдера, неизвестная модель | проверьте тело и `model` |
| `401` | прокси настроен с `proxy_auth` и токен не подошёл (см. § proxy_auth) | проверьте `Authorization` |
| `404` | `/v1/models/...` для неизвестной модели | дождитесь refresh или запросите модель явно |
| `413` | `max_request_body_bytes` превышен | уменьшите промпт |
| `429` | rate-limit (либо прокси, либо все апстримы зарейтлимичены) | смотрите `Retry-After` |
| `500` | внутренняя ошибка прокси | проверяйте логи прокси |
| `502/503/504` | апстрим недоступен / таймаут | прокси уже пробовал fallback-провайдеров |

### Rate limits

Если прокси настроен с `server.rate_limit`, превышение → `429` с заголовком
`Retry-After: <seconds>`. По умолчанию лимиты «на инсталляцию», но можно
включить per-caller (ключуется по `Authorization`-токену или IP).

Если **апстрим** (а не прокси) вернул 429 с `Retry-After`, провайдер
автоматически помечается «cooling-down» — прокси не будет к нему обращаться
до конца окна и сразу попробует следующего кандидата. Клиент получит `429`
с `Retry-After` только если **все** кандидаты для модели зарейтлимичены.

### Fallback и таймауты

Для одной модели может быть зарегистрировано несколько провайдеров (для
надёжности). Если первый ответил 5xx/network error до отправки заголовков
ответа клиенту — прокси прозрачно пробует следующего. Для стриминга fallback
работает **только до первого токена**: если стрим начался и оборвался —
ошибка пробрасывается клиенту как есть.

Таймауты могут стрелять на трёх уровнях:
- `provider.timeout` — на получение HTTP-заголовков от апстрима;
- `provider.request_timeout` — на первый токен в стриме / на полный ответ в нестриме;
- ваш HTTP-клиент — самый верхний.

**Совет:** для медленных моделей (длинный контекст, локальный GPU) ставьте
клиентский таймаут **минутами**, не секундами.

---

## 11. Health, metrics, audit

| Endpoint | Назначение |
|---|---|
| `GET /healthz` | health-check, всегда `{"status":"ok"}` если процесс жив |
| `GET /metrics` | Prometheus text format (если `server.metrics: true` в конфиге) |

Метрики (если включены): счётчики запросов и токенов по модели/провайдеру,
гистограммы латентности, состояние rate-limit'еров, число активных запросов
на провайдере, cooling-down состояния.

Аудит: если оператор включил `server.audit_log: true`, каждый чат-запрос
логируется как структурированная запись `msg:"audit"` (модель, провайдер,
число токенов, статус, latency). Это **внутренний лог прокси** — клиенту он
не виден.

---

## 12. Авторизация прокси

По умолчанию прокси **сам по себе не валидирует** `api_key` — он рассчитан
на работу за периметром сети или за reverse proxy с авторизацией. В таком
сетапе любая строка в `api_key` подходит.

Если оператор включил `server.proxy_auth`, прокси проверяет `Bearer` токен
против сконфигурированного списка и возвращает `401` на неизвестные. В этом
сетапе `api_key` в клиенте — реальный секрет. `GET /healthz` остаётся
открытым в любом случае.

При наличии OIDC (Phase 7 roadmap) добавится self-service выпуск
Personal Access Tokens через веб-UI — детали см. в `docs/roadmap.md`.

---

## 13. Отладка: пустой ответ или «перемешанный» текст

Если клиент видит **пустой** ответ или **перемешанный** текст — почти всегда баг
**на стороне клиента**, не прокси и не апстрима. Этот раздел — короткий чек-лист.

### Пустой `content`

Самые частые причины:

| Сценарий | Где искать данные | Что показывает `finish_reason` |
|---|---|---|
| Модель «думает» | `choices[0].message.reasoning_content` (нестрим) или `choices[0].delta.reasoning_content` (стрим) | `stop` |
| Модель вызвала tool | `choices[0].message.tool_calls[]` | `tool_calls` |
| Бюджет токенов исчерпан мыслями | `reasoning_content` есть, `content` пуст | `length` |
| Content filter (apstream-policy) | — | `content_filter` |

**Правило:** рендерьте `content || reasoning_content`, обрабатывайте `tool_calls`,
проверяйте `finish_reason`. UI, который показывает только `message.content`, будет
периодически выглядеть «пустым» — это ожидаемо.

Прокси **нормализует имя поля reasoning** к OpenAI-стандарту: даже если апстрим
(vLLM) отдаёт `reasoning`, клиенту приходит `reasoning_content`. Подробности
upstream-поведения — в `docs/vllm-developer-guide.md` §5.

### Перемешанный текст в стриме

(Напр. «профессиональны*йчик разработ*» вместо «…ный разработчик».) Это
**рассинхрон сборки SSE на стороне клиента** — чанки склеиваются не в порядке
получения. Прокси флашит каждый чанк строго в порядке прихода из апстрима
(`for chunk := range ch`), параллелизма в этом пути нет. Если видите такое:

- проверьте, что обработка чанков клиента **строго последовательная** (без
  параллельных goroutine/promise.all);
- держите `content`, `reasoning_content` и `tool_calls` в **раздельных буферах**
  — не сливайте поток мыслей в основной текст;
- для tool-call стриминга накапливайте `function.arguments` дельты до конца
  блока, не парсите JSON по частям.

### Где смотреть со стороны оператора

Если `server.audit_log: true`, прокси пишет одну `INFO`-запись на каждый
завершённый запрос с `model`, `duration_ms`, `prompt_tokens`, `completion_tokens`,
`cached_tokens` (если > 0), `finish_reason`. Для streaming-запросов вторая запись
`event=stream_end` с теми же полями. Грепайте по `finish_reason=length` или
`finish_reason=content_filter` — это даст полный список «пустых» ответов.

`/metrics` (если `server.metrics: true`) пока не разбивает по `finish_reason`,
только по `model/provider/type=prompt|completion|cached`.

---

## 14. Чек-лист «граблей»

- [ ] `/v1` обязателен в base_url. Без него — 404 на `/models`.
- [ ] `api_key` — любая непустая строка по умолчанию; если прокси с auth — ваш реальный токен.
- [ ] Используйте `auto:<caps>` для переносимости между провайдерами.
- [ ] Reasoning — поле **`reasoning_content`** в ответе (нормализованное прокси).
- [ ] Стриминг почти обязателен для длинных ответов — клиент не будет ждать всё.
- [ ] Стабильный системный префикс → prefix-cache → меньше затрат и быстрее TTFT.
- [ ] Таймауты клиента — минуты, а не секунды, особенно для длинного контекста / локальных моделей.
- [ ] `429` с `Retry-After` — уважайте; повторный запрос до истечения вернётся с тем же кодом.
- [ ] Capability-overrides на стороне прокси: если модель не возвращается через `auto:*`, поговорите с оператором — возможно, `model_capabilities` не прописаны в его конфиге.

---

## 15. Полезные команды

```bash
# узнать какие модели доступны
curl http://gap.local:8090/v1/models | jq '.data[] | {id, capabilities}'

# проверить здоровье
curl http://gap.local:8090/healthz

# одноразовый запрос с reasoning
curl http://gap.local:8090/v1/chat/completions -H 'content-type: application/json' -d '{
  "model":"auto:reasoning",
  "reasoning_effort":"high",
  "max_tokens": 4096,
  "messages":[{"role":"user","content":"Реши задачу..."}]
}' | jq '.choices[0].message | {reasoning_content, content}'

# стриминг с прогрессом размышлений
curl -N http://gap.local:8090/v1/chat/completions -H 'content-type: application/json' -d '{
  "model":"claude-sonnet-4-6",
  "budget_tokens": 4000,
  "stream": true,
  "stream_options": {"include_usage": true},
  "messages":[{"role":"user","content":"Объясни шаг за шагом"}]
}'
```
