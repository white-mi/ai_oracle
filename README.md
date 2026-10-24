# ai_oracle

VEGA x AI Talent Hub project.

ИИ-оракул это программа, которая после закрытия рынка предсказаний ищет свидетельства и решает, каким исходом закрыть рынок. Здесь лежат аудит статьи Kota «Design and Evaluation of Multi-Agent AI Oracle Systems for Prediction Market Resolution» (arXiv:2605.30802, коммит f76754b5bcff1a1d7a50c2f1073a4b6d44bd985c репозитория Tkota10/Senior-Thesis-Multi-LLM-Resolution), пилотные прогоны на моделях Claude и каркас кода, на котором строится открытый движок проверки оракулов.

## Быстрая проверка

```bash
bash scripts/setup_kota_repo.sh        # Windows: powershell -File scripts/setup_kota_repo.ps1
pip install -r requirements.txt
mkdir -p work
python -I -X utf8 kota_audit/kota_audit_proofs.py --repo audit/repo --json work/findings.json > work/proofs_output.txt
diff --strip-trailing-cr kota_audit/findings.json work/findings.json
KOTA_REPO=audit/repo python -I -X utf8 oracle_lab/tests/test_offline.py
cd audit && python -X utf8 analysis/t05_selective.py   # скрипты audit/ запускаются из audit/
```

Ни одна из этих команд не вызывает модели и не ходит в сеть. Репозиторий статьи восстанавливается в `audit/repo` из копии в `third_party/`. Скрипт доказательств сверяет хэш коммита и точный текст каждой цитируемой строки кода, а все его числа считаются по файлам репозитория статьи.

## Состав

| Папка или файл | Что внутри |
| --- | --- |
| `kota_audit/` | Скрипт доказательств `kota_audit_proofs.py`, его вывод (`proofs_output.txt`, `findings.json`) и отчёт аудита на английском |
| `audit/analysis/` | Расчёты по опубликованным файлам статьи (`t01`-`t09`), рядом вывод каждого скрипта и кэши `*.pkl` |
| `audit/exp/` | Пилот на моделях Claude по сохранённым пакетам свидетельств статьи. Обвязка `common.py`, скрипты прогонов e1-e5, анализ e1-e7, ответы моделей `*.csv`, сырые ответы `cache_calls.jsonl`, `r1_api.py` для повтора через Message Batches API |
| `audit/agent/` | Ответы агента с живым поиском, в `haiku_clean/` 150 очищенных стенограмм |
| `audit/figs/`, `audit/independent_review/` | Рисунки отчёта и скрипты независимой проверки аудита |
| `oracle_lab/` | Каркас кода экспериментов (ключи, свидетельства, промпты, клиенты моделей и Jev, метрики, сборщик Kalshi) и 8 офлайн-тестов. Сетевые клиенты с живыми сервисами пока не проверялись |
| `replication/` | Патч к коду статьи с регрессионными тестами, перенос кэша на новые ключи, план подтверждающих прогонов и расчёт их стоимости |
| `notes/` | Заметки по литературе и проверка ссылок статьи |
| `scripts/` | Восстановление репозитория статьи в `audit/repo` для bash и PowerShell |
| `third_party/kota_repo_f76754b.bundle` | Git-копия репозитория статьи на проверенном коммите |
| `requirements.txt`, `requirements-engine.txt` | Зависимости анализа и закреплённые версии для движка на LangGraph |

## Оговорки

- Числа пилота предварительные. Каждая конфигурация запускалась один раз через командную утилиту Claude. Повтор через API подготовлен, но ещё не запускался (`replication/CONFIRM_RUNS.md`).
- Вывод `t03` и `t07` зависит от того, какой из нескольких файлов пакета одного вопроса выбирает `audit/analysis/kota.py:packets()`. Сохранённые `.out` повторяются только при том порядке файлов, в котором их считали.
- Скрипты `t03`, `t08`, `e2_analyze`, `e3_deb_analyze`, `e6_dates` и `e7_hindsight` перезаписывают свои `.pkl` и `.csv`.

## Лицензия

MIT, см. `LICENSE`. В `third_party/` лежит копия репозитория статьи под его собственной лицензией MIT (Copyright (c) 2026 Tarun Kota).
