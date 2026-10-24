# Честный повтор эксперимента статьи

- `replication_fixes.patch`: исправления к репозиторию статьи (коммит f76754b). Применение:
  `git clone https://github.com/Tkota10/Senior-Thesis-Multi-LLM-Resolution repo && cd repo && git checkout f76754b && git apply ../replication_fixes.patch && python -m pytest tests`
  Ожидаемо: 35 тестов проходят, падает только `test_integration_outputs_expected_files`, который падает и без патча (ссылается на удалённый файл).
- `sizes.py`: размеры промптов и ответов по пакетам и ответам статьи, калибровка знаков в токены по вызовам Haiku. Запускать из папки `audit/` архива (нужны `exp/common.py`, `exp/cache_calls.jsonl` и клон статьи в `audit/repo`).
- `budget.py`: бюджет по условиям повтора, читает `sizes.json`. Цены проверены 11 октября 2026 года.
- Новый флаг прогонщика B: `--evidence-render text|titles`. `titles` намеренно воспроизводит опубликованные прогоны.
