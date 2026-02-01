## Regression тестування (автоматичне відтворення “поганих” кейсів)

Цей модуль потрібен, щоб **автоматично** відтворювати конкретні проблемні зони (bbox), генерувати модель і перевіряти інваріанти:
- є рельєф (не “плоско”);
- вода не “мигає” (немає співпадіння Z з тереном у водних полігонах);
- парки/зелень не накладаються на терен (анти z-fighting);
- мости реально підняті над водою (мінімальний clearance);
- артефакти/витік геометрії по краях відсутні (кліп + watertight).

### Як запускати (CLI)

```bash
cd H:\3dMap\backend
.\venv\Scripts\python -m regression.run_case --case regression\cases\paton_bridge.json --artifacts output\regression
```

CLI збереже:
- `output/regression/<case>/report.json` — метрики
- `output/regression/<case>/*.3mf` або `*.stl` (якщо включено export)

### Як запускати з pytest

За замовчуванням “важкі” regression тести вимкнені (бо можуть тягнути мережу / кеші).

```bash
cd H:\3dMap\backend
set RUN_REGRESSION=1
.\venv\Scripts\python -m pytest -q -m regression
```


