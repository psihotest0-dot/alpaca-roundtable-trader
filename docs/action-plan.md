# План за действие

**Към 2026-11-01.** Подробности за всяка констатация: [analysis-and-update-plan.md](analysis-and-update-plan.md). Сравнение с оригинала: [comparison-with-money-bot.md](comparison-with-money-bot.md).

Легенда: ✅ готово и качено · ◐ частично · ☐ предстои.

---

## 1. Правила за работа (важат за всяка задача)

1. **От 2026-11-01 работата се качва директно в `main`** (решение на собственика; преди това имаше клон за фаза). Затова правилото за проверка по-долу (4) е задължително: `main` не бива да се чупи. Ако CI на `main` стане червен, първата работа е да се оправи.
2. **Една задача = един commit** (или кратка серия), със съобщение във вида `тип(област): какво` и обяснение защо. Commit-ите завършват с реда `Co-Authored-By: …`.
3. **След всяка завършена задача: commit и веднага `git push`.** Така няма работа само на един компютър.
4. **Преди push:** `ruff check .`, `ruff format --check .`, `mypy src`, `pytest`. При промяна на зависимости, `Dockerfile` или миграции: и CI стъпките върху свежо клониране (миграция нагоре и надолу, сглобяване на образа). Локалният pre-commit хук пуска ruff и защитите срещу тайни автоматично.
5. **Задачата е „готова“ само с проверка:** тест, команда или резултат, записан в описанието на commit-а. Не „би трябвало да работи“.
6. **Нищо от `.env`, `var/`, бази, логове, ключове и сертификати не се комитва.** Хуковете и CI го спират.
7. **Без пренаписване на историята** на публичния клон. Историята е сканирана и е чиста (виж 2).

---

## 2. Състояние на хранилището (проверено)

| Проверка | Резултат |
|---|---|
| Видимост | **Публично** хранилище (GitHub API) |
| Тайни в историята | Няма. Сканирани 74 commit-а за Alpaca ключове (`PK/AK/CK` + 18 знака), ключове във формат на AnythingLLM, private key блокове и присвоени дълги токени. *Регулярното сканиране не е гаранция. Затова се включва и GitHub secret scanning (раздел 4).* |
| Следени файлове, които не трябва | Само `var/alpaca_trading.log` (без тайни) → вече не се следи |
| CI | Зелен на `main` (последните пускания) |
| Автори | 50 от 74 commit-а са на `google-labs-jules[bot]`: бот с права за запис (виж 4) |
| Git идентичност | Локално `user.email` е примерният `ivaylo@example.com`, не е свързан с GitHub профила |

---

## 3. Част A: защита на хранилището (направено, клон `chore/repo-hardening`)

| ID | Задача | Статус | Commit |
|---|---|---|---|
| A1 | `.gitignore`: `data/` → `/data/` (скриваше пакетите `data/` от git и ruff), `/var/*` с `.gitkeep`, ключове и сертификати, WAL/SHM, `.env.*`. Спира следенето на лога. | ✅ | `78362f6` |
| A2 | `.gitattributes` (LF навсякъде) и по-строг `.dockerignore` (`venv/` копираше ~18 000 файла в образа). | ✅ | `b57f19f` |
| A3 | Ruff: 2 грешки и 7 неформатирани файла, които бяха скрити. Тестове 84/84, mypy чист. | ✅ | `9433d27` |
| A4 | Pre-commit: private key, големи файлове, EOL, ruff, **блок на `.env`/бази/логове/ключове** и **блок на Alpaca/AnythingLLM ключове**. Проверено с временни файлове: и двете защити се задействат. Хукът е инсталиран локално. | ✅ | `bffae7d` |
| A5 | CI: нова задача `hygiene` (същите проверки, не се заобикалят с `--no-verify`), права само за четене, timeout, поправен ключ на кеша. *Само локално проверено; първото реално пускане е при PR.* | ✅ | `c411acc` |
| A6 | Dependabot (pip, Actions, Docker) и `SECURITY.md` с процедура при изтекъл ключ. | ✅ | `8eaa0d4` |
| A7 | Документи: анализ и сравнение с Money_bot (поправена е грешката за Poetry), този план. | ✅ | `45678ee`, планът в следващия |

**Следва:** отвори pull request: <https://github.com/psihotest0-dot/alpaca-roundtable-trader/pull/new/chore/repo-hardening> и го слей, когато CI е зелен.

---

## 4. Част B: ръчни стъпки в GitHub (само собственикът; не мога да ги направя оттук)

Нито една от тези настройки не е проверена.

- [ ] **Settings → Code security:** включи **Secret scanning** и **Push protection**, **Dependabot alerts** и **Dependabot security updates**, **Private vulnerability reporting** (SECURITY.md го сочи).
- [ ] **Settings → Branches:** правило за `main`: задължителен pull request, задължителни проверки `hygiene` и `test`, забрана на force push и на изтриване.
- [ ] **Settings → Actions → General:** права на workflow „Read repository contents“; одобрение за pull request-и от външни сътрудници.
- [ ] **Преглед на бота Jules:** Settings → GitHub Apps / Integrations. Ако не го ползваш, махни достъпа. Ако го ползваш, ограничи го до това хранилище и не му давай право да слива в `main`.
- [ ] **Профил:** двуфакторна автентикация. По желание подписани commit-и. Задай `git config user.email` на адреса `…@users.noreply.github.com`, за да се свързват commit-ите с профила ти.
- [ ] Ако някога някой ключ е бил поставен в чат, issue или лог, **смени го** в Alpaca (отделно от git).

---

## 5. Част C: следващи задачи (в този ред)

Всяка се прави на нейния клон и се качва след завършване. Детайли за задачите: раздел 6 на [analysis-and-update-plan.md](analysis-and-update-plan.md).

| Ред | ID | Задача | Клон | Статус |
|---|---|---|---|---|
| 1 | 0.3 | `safety` само в dev зависимостите (18 пакета излизат от производствения lock), `plugins` под `[tool.mypy]` (7 ненужни `type: ignore` махнати), `.gitattributes` вече показва промените по lock файла. Poetry 1.8.2 проверен с `check --lock`. | `phase-0-tests` | ✅ `b2223e2` |
| 2 | 0.4 | **Тестове за известните дефекти** (виж таблицата по-долу): 11 теста `xfail(strict)`, всеки проверен с `--runxfail`, че пада от правилната причина, плюс контролни тестове и 5 теста за празния `test_orchestrator.py`. | `phase-0-tests` | ✅ `31ca0d9`, `8965e09`, `db061e4` |
| 3 | 1.1 | Транзакции: всяко извикване на хранилище е къса транзактия с commit, без глобален engine *(C4, C5)*. Проверено и с реални CLI процеси: `pause-entries` в един процес, `status` в друг показва `Trading Halted: True`. | `phase-1-core` | ✅ `0ce0d10` |
| 4 | 1.2 | Идентичност на ордер по `client_order_id` + миграция `c3f1a9e47d52` (запазва данните, има `downgrade`, подравнява и `orderstatus`) *(C6, M5)*. **Съществуващи бази:** `alembic upgrade head`. | `phase-1-core` | ✅ `eb87251` |
| 5 | 1.3 | Пътят на данните *(C1, C1б, C1в, M2)*: `TimeFrame(15, ...)`, `fetch_bars` без `UnboundLocalError`, един формат на баровете (`normalize_bars`), **една заявка за всички символи** вместо 44 в минута, ретраи само за временни грешки. Проверено и върху реалния Alpaca (само четене): 6 символа за 6.3 s. | `main` | ✅ `53f1bad` |
| 5а | 1.3а | `LivePriceProvider` през WebSocket (цена на живо за лимита на входа, с поправките от Money_bot) | `main` | ☐ |
| 6 | 1.4 | Дневен начален капитал и дневен лимит *(C2)* | `main` | ☐ |
| 7 | 1.5 | Договор Risk → Execution (`EntryPlan`) *(C3)* | `main` | ☐ |
| 8 | 1.6 | Проверка за защита на позициите през вложени легове, **проверена на paper акаунт** *(C7)* | `main` | ☐ |
| 9 | 1.7 | Оркестратор: `TradingCycle` + `Runner`, `SIGTERM`, reconciliation, fills | `main` | ☐ |
| 10 | 2.x | Oracle: `OracleProvider`, `APPROVE / VETO / HOLD`, Lemonade, без AnythingLLM | `phase-2-oracle` | ☐ |
| 11 | 3.x | Бектест: общ код с live, реални данни, A/B на стратегиите, shadow ветонирани сигнали | `phase-3-backtest` | ☐ |
| 12 | 4.x | Стратегия и риск (ATR, филтри, PDT) | `phase-4-strategy` | ☐ |
| 13 | 5.x | Новини: първо Alpaca News API | `phase-5-news` | ☐ |
| 14 | 6.x | Наблюдаемост и операции, `validate-config`, Docker | `phase-6-ops` | ☐ |
| 15 | 7.x | Paper пробен период и `paper-readiness.md` с доказателства | `phase-7-paper` | ☐ |

### Тестове за известните дефекти: коя поправка кой тест превключва

Тестовете са в `tests/integration/` и са маркирани `xfail(strict=True)`. Така CI остава зелен, а когато дефектът бъде поправен, тестът **започва да пада** (XPASS при strict). Тогава се трие маркерът `xfail`, а тестът остава като защита.

| Задача | Дефект | Тест, който трябва да стане зелен |
|---|---|---|
| 1.1 ✅ | C4, C5 | `test_pause_entries_survives_process_exit`, `test_emergency_halt_survives_process_exit`, `test_reading_state_does_not_block_other_writers` (вече зелени, маркерите са махнати) |
| 1.2 ✅ | C6 | `test_broker_confirmation_replaces_the_inflight_placeholder`, `test_successful_live_submission_does_not_halt_trading` (вече зелени) |
| 1.3 ✅ | C1б, C1в, C1 | `test_cycle_requests_bars_with_valid_timeframes`, `test_fetch_bars_returns_the_bars_alpaca_sent`, `test_bars_loaded_through_the_data_layer_flow_into_generate_signals` (вече зелени) |
| 1.4 | C2 | `test_cycle_risk_engine_approves_a_healthy_signal` |
| 1.5 | C3 | `test_execution_engine_accepts_what_the_risk_engine_approves` |
| 1.4 + 1.5 | краен резултат (останаха C2 и C3; C4 е поправен) | `test_dry_run_cycle_persists_a_simulated_order` |

Защитни тестове, които минават още сега и трябва да останат зелени: `test_every_enum_survives_a_database_round_trip` (урокът от Money_bot: `BUY` се връщаше като `SELL`), `test_control_*` (самите фикстури произвеждат валиден сигнал).

Две бележки за Фаза 1: тестът за C1 симулира изхода на `fetch_bars` с `reset_index(level=0, drop=True)`, докато C1в не е поправен. След поправката е добре да мине през самия `fetch_bars`. А тестовете в `test_runtime_cycle.py` ползват заобикалка за C1б (`TimeFrame.__mul__`), за да стигнат до следващите дефекти. Заобикалката се маха при задача 1.3.

**Условия за преминаване между фазите:**
- след ред 2: CI зелен, новите тестове падат от правилните причини (проверено с `--runxfail`);
- след ред 9: цяла сесия dry-run на paper ключове без изключения, `pause-entries` оцелява между процеси;
- след ред 11: положителна очаквана стойност извън извадката, иначе не се продължава към ред 12 и 15;
- преди каквото и да е **live**: ред 15 завършен и отделно решение на собственика.
