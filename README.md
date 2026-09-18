# BG — هوش مصنوعی تخته‌نرد با یادگیری تقویتی

این پروژه یک بازیکن تخته‌نرد است که **از صفر و با self-play** (بازی کردن با نسخه‌های خودش) آموزش می‌بیند. کد کاملاً Python است و بعد از آموزش می‌توان با یک صفحه‌ی گرافیکی pygame مقابل آن بازی کرد.

> منظور این پروژه از `BG`، **Backgammon / تخته‌نرد** است. اگر بازی دیگری مدنظر است، باید محیط و قوانین همان بازی جایگزین شود.

## چه چیزی ساخته شده است؟

- محیط کامل تخته‌نرد شامل:
  - دو تاس معمولی و تاس جفت چهارحرکتی؛
  - ورود مهره از bar؛
  - زدن مهره‌ی تک حریف؛
  - قانون اجبار استفاده از بیشترین تعداد تاس ممکن؛
  - قانون تاس بزرگ‌تر وقتی فقط یک تاس قابل استفاده است؛
  - قوانین دقیق بیرون آوردن مهره‌ها.
- آموزش self-play با **PPO** و action masking.
- یک شبکه‌ی عصبی **Residual MLP Actor-Critic**:
  - ورودی کوچک و ساختاریافته‌ی تخته‌نرد، نه تصویر؛
  - سه بلوک residual با ۲۵۶ نورون؛
  - یک head برای احتمال حرکت و یک head برای ارزش وضعیت؛
  - حرکت‌های غیرقانونی قبل از نمونه‌برداری از سیاست حذف می‌شوند.
- اجرای موازی چندین بازی برای رسیدن به میلیون‌ها تصمیم آموزشی.
- ذخیره‌ی `latest.pt`، checkpointهای دوره‌ای و `training.csv`.
- ارزیابی مقابل بازیکن تصادفی یا self-play.
- رابط گرافیکی مناسب برای بازی انسان با مدل آموزش‌دیده.
- تست‌های قوانین و محیط.

## چرا PPO و Residual MLP؟

در تخته‌نرد، نتیجه‌ی حرکت به تاس و تصمیم‌های متوالی همان نوبت وابسته است و تعداد حرکت‌های قانونی در هر وضعیت متغیر است. PPO با یک سیاست stochastic برای چنین محیطی انتخاب مناسبی است. محیط، حرکت‌های غیرقانونی را mask می‌کند تا شبکه هیچ‌وقت از میان حرکت‌های نامعتبر انتخاب نکند.

چون ورودی، صفحه‌ی دوبعدی تصویری نیست و فقط ۲۴ خانه، bar، مهره‌های خارج‌شده و تاس‌ها را توصیف می‌کند، شبکه‌ی MLP از CNN مناسب‌تر و بسیار سبک‌تر است. وضعیت برای بازیکنِ نوبت‌دار canonical می‌شود؛ بنابراین یک شبکه هم White و هم Black را یاد می‌گیرد.

این مدل یک پروژه‌ی پژوهشی/آموزشی است و ادعای هم‌سطح بودن با موتورهای حرفه‌ای تخته‌نرد ندارد. قدرت آن مستقیماً به تعداد تصمیم‌های آموزشی و سخت‌افزار شما بستگی دارد.

## ساختار پروژه

```text
bg/
  constants.py     کدگذاری action و اندازه‌های محیط
  game.py          قوانین خالص تخته‌نرد و تولید حرکت
  env.py           محیط self-play و observation/action mask
  model.py         شبکه Actor-Critic و بارگذاری checkpoint
  training.py      حلقه‌ی PPO، rollout موازی و ذخیره مدل
  evaluate.py      بازی مدل مقابل random یا خودش
  ui.py            رابط گرافیکی pygame
  cli.py           دستورات train / evaluate / play
  __main__.py      اجرای python -m bg

tests/
  test_game.py
  test_env.py
```

## اجرای کامل روی Windows، از clone تا بازی

### ۱) نصب ابزارهای لازم

1. Git for Windows را نصب کنید: <https://git-scm.com/download/win>
2. Python نسخه‌ی 3.10 یا جدیدتر نصب کنید. برای این پروژه Python 3.11 پیشنهاد می‌شود و هنگام نصب، گزینه‌ی **Add Python to PATH** را فعال کنید.
3. PowerShell را باز کنید.

اگر اجرای اسکریپت PowerShell بسته بود، فقط برای پنجره‌ی فعلی این دستور را بزنید:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### ۲) clone کردن کد

کد این session روی branch زیر GitHub قرار می‌گیرد:

```powershell
git clone -b arena/01a0b49e-bg https://github.com/Mooli-web/BG.git
cd BG
```

اگر branch بعداً در `main` merge شده بود، clone معمولی نیز کافی است:

```powershell
git clone https://github.com/Mooli-web/BG.git
cd BG
```

### ۳) ساخت virtual environment و نصب وابستگی‌ها

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

`requirements-dev.txt` علاوه بر وابستگی‌های اجرای برنامه، ابزار تست `pytest` را هم نصب می‌کند. اگر فقط اجرای برنامه را می‌خواهید، `requirements.txt` کافی است.

اگر دستور `py` روی سیستم شما وجود ندارد، به‌جای آن از `python` استفاده کنید:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

برای استفاده از GPU انویدیا، ابتدا نسخه‌ی مناسب PyTorch همان کارت و درایور را از صفحه‌ی رسمی PyTorch نصب کنید و بعد بقیه‌ی requirements را نصب کنید. روی CPU نیز پروژه اجرا می‌شود، اما آموزش میلیون‌ها تصمیم طولانی‌تر خواهد بود.

### ۴) تست نصب

در حالی که `.venv` فعال است:

```powershell
python -m pytest
python -m bg --help
```

باید تست‌ها سبز شوند و سه دستور `train`، `evaluate` و `play` را ببینید.

### ۵) یک اجرای کوتاه برای اطمینان

قبل از آموزش طولانی، یک تست چند هزار تصمیمی انجام دهید:

```powershell
python -m bg train --total-steps 10000 --num-envs 4 --rollout-steps 128 --checkpoint-dir checkpoints\smoke
```

در پایان باید فایل زیر ساخته شده باشد:

```text
checkpoints\smoke\latest.pt
```

### ۶) آموزش اصلی چندمیلیونی

اجرای پیش‌فرض یعنی **۵٬۰۰۰٬۰۰۰ تصمیم حرکت مهره**، نه پنج میلیون بازی کامل. هر تصمیم می‌تواند یک حرکت از نوبت فعلی باشد و چند تصمیم پشت سر هم یک نوبت تاس را کامل می‌کنند.

برای شروع پیشنهادی روی CPU:

```powershell
python -m bg train `
  --total-steps 5000000 `
  --num-envs 8 `
  --rollout-steps 256 `
  --device cpu `
  --checkpoint-dir checkpoints\run-5m
```

روی GPU یا CPU قوی‌تر می‌توانید parallel environment را بیشتر کنید:

```powershell
python -m bg train --total-steps 10000000 --num-envs 16 --rollout-steps 256 --device auto --checkpoint-dir checkpoints\run-10m
```

در PowerShell علامت backtick یعنی ادامه‌ی همان دستور در خط بعد. اگر خواستید همه را یک‌خطی بنویسید:

```powershell
python -m bg train --total-steps 5000000 --num-envs 8 --rollout-steps 256 --device cpu --checkpoint-dir checkpoints\run-5m
```

خروجی‌های مهم:

- `checkpoints\run-5m\latest.pt`: آخرین مدل؛
- `checkpoints\run-5m\checkpoint_XXXXXXXXXXXX.pt`: checkpointهای دوره‌ای؛
- `checkpoints\run-5m\training.csv`: loss و آمار آموزش.

`--total-steps` هدف کلی است. مثلاً اگر تا ۵ میلیون آموزش داده‌اید و می‌خواهید تا ۱۰ میلیون ادامه دهید:

```powershell
python -m bg train `
  --resume checkpoints\run-5m\latest.pt `
  --total-steps 10000000 `
  --num-envs 8 `
  --rollout-steps 256 `
  --device auto `
  --checkpoint-dir checkpoints\run-10m
```

در اجرای resume بهتر است `--hidden-size` و `--residual-blocks` را مانند اجرای اول نگه دارید. اگر هدف فقط ادامه‌ی همان پوشه است، می‌توانید `--checkpoint-dir checkpoints\run-5m` بگذارید.

### ۷) ارزیابی مدل بدون باز کردن GUI

مقابل بازیکن تصادفی:

```powershell
python -m bg evaluate --checkpoint checkpoints\run-5m\latest.pt --games 100
```

برای ارزیابی با نمونه‌برداری تصادفی از سیاست:

```powershell
python -m bg evaluate --checkpoint checkpoints\run-5m\latest.pt --games 100 --stochastic
```

برای اجرای مدل مقابل خودش:

```powershell
python -m bg evaluate --checkpoint checkpoints\run-5m\latest.pt --games 20 --self-play
```

در ارزیابی مقابل random، مقدار `ai_win_rate` معیار ساده‌ای برای مقایسه‌ی runهای مختلف است. برای مقایسه‌ی علمی‌تر، seed و تعداد بازی یکسان استفاده کنید.

### ۸) بازی انسان مقابل شبکه

```powershell
python -m bg play --checkpoint checkpoints\run-5m\latest.pt --human white
```

یا با مهره‌ی سیاه:

```powershell
python -m bg play --checkpoint checkpoints\run-5m\latest.pt --human black
```

کنترل‌ها:

- روی مهره‌ی مشخص‌شده کلیک کنید؛
- اگر برای آن مهره دو تاس ممکن بود، کلید `1` تا `6` را بزنید؛
- اگر هیچ حرکتی ممکن نبود، `Space` یا دکمه‌ی `SPACE Pass` را بزنید؛
- کلید `R` بازی جدید و `Esc` خروج است.

## تنظیم تعداد تمرین

پیشنهاد عملی:

| هدف | دستور | کاربرد |
|---|---:|---|
| تست نصب | `10,000` | فقط اطمینان از صحت اجرا |
| نمونه‌ی اولیه | `500,000` تا `1,000,000` | بررسی روند آموزش |
| مدل قابل بازی | `5,000,000` تا `10,000,000` | شروع ارزیابی جدی |
| آزمایش قوی‌تر | `20,000,000+` | کیفیت بهتر با زمان بیشتر |

کیفیت در self-play به‌صورت یکنواخت با تعداد steps زیاد نمی‌شود؛ بنابراین `training.csv` و win rate مقابل random را بررسی کنید. عدد مناسب به CPU/GPU و هدف شما بستگی دارد. اگر آموزش ناپایدار شد، `--reward-shaping 0.001` را آزمایش کنید؛ حالت پیش‌فرض `0.0` است و فقط پاداش برد/باخت را استفاده می‌کند.

## خطاهای معمول Windows

- **`No module named ...`**: ابتدا `.venv` را فعال کنید و دوباره `python -m pip install -r requirements.txt` را اجرا کنید.
- **پنجره‌ی pygame باز نمی‌شود**: دستور `play` را روی خود Windows و در یک desktop اجرا کنید، نه محیط بدون نمایشگر یا SSH.
- **CUDA error**: ابتدا با `--device cpu` مطمئن شوید پروژه سالم است؛ سپس نسخه‌ی PyTorch سازگار با درایور GPU را نصب کنید.
- **آموزش کند است**: `--num-envs` را کمی افزایش دهید، یا از GPU استفاده کنید. در لپ‌تاپ، اجرای ۵ میلیون step ممکن است زمان قابل‌توجهی ببرد.
- **فایل checkpoint پیدا نشد**: مسیر کامل یا نسبی را دقیق بدهید؛ مسیر نسبی از همان پوشه‌ای است که PowerShell در آن `cd BG` کرده است.

فایل‌های وزن شبکه در `.gitignore` هستند و به GitHub ارسال نمی‌شوند؛ پس از آموزش، پوشه‌ی `checkpoints` روی کامپیوتر خودتان باقی می‌ماند.
