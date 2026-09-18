# BG — هوش مصنوعی تخته‌نرد با یادگیری تقویتی

این پروژه یک بازیکن تخته‌نرد است که با **Self-Play و PPO** آموزش می‌بیند. پیشنهاد استفاده‌ی فعلی این است:

- **آموزش و ارزیابی متنی:** Google Colab، به‌صورت پیش‌فرض با CPU؛ GPU اختیاری؛
- **بازی گرافیکی انسان مقابل مدل:** کامپیوتر Windows با pygame.

> منظور `BG` در این پروژه، Backgammon / تخته‌نرد است.

## امکانات

- قوانین تخته‌نرد شامل تاس جفت، Bar، زدن مهره، بیشترین تعداد حرکت و قانون تاس بزرگ‌تر؛
- جلوگیری از بیرون آوردن مهره تا زمانی که هر ۱۵ مهره‌ی بازیکن به خانه‌ی خودش نرسیده باشند؛
- محیط Self-Play با action masking؛
- الگوریتم PPO؛
- شبکه‌ی Residual MLP Actor-Critic با سه بلوک ۲۵۶ نورونی؛
- اجرای موازی چندین بازی برای آموزش چندمیلیونی؛
- checkpoint قابل ذخیره در Google Drive و ادامه‌ی آموزش؛
- ارزیابی مقابل بازیکن تصادفی بدون نیاز به pygame؛
- رابط گرافیکی استاندارد برای Windows؛
- مکث قابل تنظیم بین حرکت‌های AI، نمایش تاس کامل، تاس مصرف‌شده و آخرین حرکت؛
- نوار زنده‌ی `AI evaluation` از دید شبکه؛
- تست‌های قوانین و محیط.

## نکته‌ی مهم درباره‌ی checkpoint قدیمی

در نسخه‌ی اولیه یک خطای قانونی وجود داشت که اجازه می‌داد بازیکن قبل از ورود تمام مهره‌ها به خانه، مهره‌ای را خارج کند. این خطا اصلاح شده است.

بنابراین checkpointهایی که قبل از این اصلاح آموزش داده شده‌اند برای آموزش معتبر مناسب نیستند. آموزش جدید را بدون `--resume` و در یک پوشه‌ی جدید شروع کنید.

## ساختار پروژه

```text
bg/
  constants.py     کدگذاری action و اندازه‌های محیط
  game.py          قوانین تخته‌نرد و تولید حرکت
  env.py           محیط Self-Play و action mask
  model.py         شبکه Actor-Critic و بارگذاری checkpoint
  training.py      حلقه PPO و ذخیره مدل
  evaluate.py      ارزیابی مقابل random یا self-play
  ui.py            رابط گرافیکی pygame
  cli.py           دستورات train / evaluate / play

notebooks/
  BG_Colab_Train.ipynb    notebook آماده‌ی آموزش در Colab

tests/
  test_game.py
  test_env.py

requirements-colab.txt    وابستگی‌های آموزش Colab
requirements-play.txt     وابستگی‌های کامپیوتر Windows
```

# مسیر پیشنهادی: آموزش در Google Colab

## روش سریع: باز کردن notebook

Notebook آماده در این مسیر قرار دارد:

```text
notebooks/BG_Colab_Train.ipynb
```

می‌توانید از این لینک آن را در Colab باز کنید:

<https://colab.research.google.com/github/Mooli-web/BG/blob/arena/01a0b49e-bg/notebooks/BG_Colab_Train.ipynb>

در Colab:

1. notebook را باز کنید؛
2. سلول‌ها را به‌ترتیب اجرا کنید؛
3. Google Drive را mount کنید؛
4. آموزش را شروع کنید.

نسخه‌ی فعلی notebook به‌صورت پیش‌فرض روی **CPU** تنظیم شده است. برای این محیط کوچک، تولید حرکت‌های قانونی با Python بخش مهمی از زمان را مصرف می‌کند و یک GPU ضعیف الزاماً سریع‌تر نیست. همچنین CPU از خطاهای مربوط به GPU و قطع شدن CUDA جلوگیری می‌کند.

Notebook به‌صورت خودکار repository را clone می‌کند، وابستگی‌های آموزش را نصب می‌کند، checkpoint را داخل Google Drive ذخیره می‌کند و آموزش را chunk به chunk ادامه می‌دهد.

اگر بعداً خواستید GPU را آزمایش کنید، داخل notebook مقدار زیر را تغییر دهید:

```python
DEVICE = 'auto'
```

## روش دستی در Colab

### ۱. انتخاب CPU یا GPU

برای CPU، لازم نیست GPU فعال کنید. فقط مطمئن شوید Runtime عادی Python است.

برای آزمایش GPU اختیاری:

```text
Runtime → Change runtime type → Hardware accelerator → GPU
```

بررسی وضعیت:

```python
!nvidia-smi
```

### ۲. دریافت کد

```python
!git clone -b arena/01a0b49e-bg https://github.com/Mooli-web/BG.git /content/BG
%cd /content/BG
```

اگر قبلاً repository را clone کرده‌اید:

```python
%cd /content/BG
!git fetch origin arena/01a0b49e-bg
!git checkout arena/01a0b49e-bg
!git pull origin arena/01a0b49e-bg
```

### ۳. نصب وابستگی‌های مخصوص آموزش

برای آموزش pygame لازم نیست:

```python
%pip install -q -r requirements-colab.txt
```

بررسی GPU و PyTorch:

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

اگر می‌خواهید با CPU کار کنید، `CUDA available: False` طبیعی است و مشکلی نیست. فقط اگر `DEVICE = 'auto'` انتخاب کرده‌اید باید GPU فعال باشد.

### ۴. اتصال Google Drive

اگر checkpoint را فقط در `/content` ذخیره کنید، با قطع شدن Colab از بین می‌رود. پس Google Drive را وصل کنید:

```python
from google.colab import drive
import os

drive.mount('/content/drive')

CHECKPOINT_DIR = '/content/drive/MyDrive/BG_RL/checkpoints_fixed_rules'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print(CHECKPOINT_DIR)
```

### ۵. اجرای یک chunk آموزش از صفر

برای جلوگیری از از دست رفتن چند ساعت کار هنگام قطع شدن Colab، آموزش را به chunkهای ۲۵۰ هزار تصمیمی تقسیم کنید. این اجرای اول از checkpoint قدیمی استفاده نمی‌کند و روی CPU اجرا می‌شود:

```python
import subprocess
import sys

command = [
    sys.executable, '-m', 'bg', 'train',
    '--total-steps', '250000',
    '--num-envs', '8',
    '--rollout-steps', '128',
    '--device', 'cpu',
    '--checkpoint-dir', CHECKPOINT_DIR,
    '--save-interval', '5',
    '--log-interval', '10',
    '--seed', '7',
]

print(' '.join(command))
subprocess.run(command, check=True)
```

Notebook آماده همین کار را خودکار انجام می‌دهد؛ اگر Colab قطع شد، کافی است همان سلول chunk را دوباره اجرا کنید.

فایل‌های مهم در Google Drive:

```text
BG_RL/checkpoints_fixed_rules/latest.pt
BG_RL/checkpoints_fixed_rules/training.csv
BG_RL/checkpoints_fixed_rules/checkpoint_XXXXXXXXXXXX.pt
```

`latest.pt` مدل فعلی است و `training.csv` آمار آموزش را نگه می‌دارد.

تنظیم پیشنهادی پایدار برای CPU همین `8` محیط و `128` rollout است. اگر سرعت کافی بود، `--num-envs` را به `16` افزایش دهید. GPU در این پروژه اختیاری است و الزاماً به‌دلیل کوچک بودن شبکه سریع‌تر نیست.

### ۶. ادامه‌ی آموزش بعد از قطع Colab

ابتدا دوباره سلول‌های clone، نصب وابستگی و اتصال Drive را اجرا کنید. اگر chunk قبلی تا ۲۵۰ هزار step رسیده بود، هدف chunk بعدی را ۵۰۰ هزار بگذارید:

```python
command = [
    sys.executable, '-m', 'bg', 'train',
    '--resume', os.path.join(CHECKPOINT_DIR, 'latest.pt'),
    '--total-steps', '500000',
    '--num-envs', '8',
    '--rollout-steps', '128',
    '--device', 'cpu',
    '--checkpoint-dir', CHECKPOINT_DIR,
    '--save-interval', '5',
    '--log-interval', '10',
]

subprocess.run(command, check=True)
```

برای chunkهای بعدی مقدار `--total-steps` را به ۷۵۰۰۰۰، ۱۰۰۰۰۰۰ و ... افزایش دهید. Notebook این مقدارها را خودکار مدیریت می‌کند. اگر هدف نهایی را `10000000` بگذارید، آموزش از checkpoint فعلی تا ۱۰ میلیون step ادامه پیدا می‌کند.

### ۷. ارزیابی در Colab

ارزیابی بدون رابط گرافیکی و بدون pygame انجام می‌شود:

```python
command = [
    sys.executable, '-m', 'bg', 'evaluate',
    '--checkpoint', os.path.join(CHECKPOINT_DIR, 'latest.pt'),
    '--games', '100',
    '--device', 'cpu',
]

subprocess.run(command, check=True)
```

### ۸. دانلود مدل برای Windows

بعد از پایان آموزش:

```python
from google.colab import files

files.download(os.path.join(CHECKPOINT_DIR, 'latest.pt'))
```

اگر فایل بزرگ بود، به‌جای download مستقیم، از Google Drive روی کامپیوتر sync یا download کنید.

# اجرای بازی روی Windows

در Windows فقط checkpoint را می‌گیریم و بازی می‌کنیم؛ آموزش روی کامپیوتر انجام نمی‌شود.

## ۱. اگر پوشه‌ی قبلی را پاک کرده‌اید

در PowerShell، از یک مسیر دلخواه این دستورات را اجرا کنید:

```powershell
git clone -b arena/01a0b49e-bg https://github.com/Mooli-web/BG.git
cd BG
```

اگر قبلاً پوشه‌ای با نام `BG` وجود دارد و می‌خواهید از صفر clone کنید:

```powershell
cd ..
Remove-Item -Recurse -Force .\BG
git clone -b arena/01a0b49e-bg https://github.com/Mooli-web/BG.git
cd BG
```

دستور حذف را فقط زمانی اجرا کنید که مطمئن هستید داخل پوشه فایل مهمی ندارید.

## ۲. نصب وابستگی‌های اجرای محلی

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-play.txt
```

اگر PowerShell اجازه‌ی فعال‌سازی نداد:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

برای اطمینان:

```powershell
python -m bg --help
```

## ۳. قرار دادن checkpoint دانلودشده

فایل `latest.pt` دانلودشده از Colab را در این مسیر قرار دهید:

```text
BG\checkpoints\latest.pt
```

اگر پوشه وجود ندارد:

```powershell
New-Item -ItemType Directory -Force checkpoints
```

یا می‌توانید checkpoint را در هر مسیر دیگری قرار دهید و همان مسیر را به دستور بدهید.

## ۴. ارزیابی روی Windows

```powershell
python -m bg evaluate --checkpoint checkpoints\latest.pt --games 100 --device cpu
```

اگر نسخه‌ی PyTorch نصب‌شده GPU را درست شناسایی می‌کند:

```powershell
python -m bg evaluate --checkpoint checkpoints\latest.pt --games 100 --device auto
```

## ۵. اجرای بازی گرافیکی

```powershell
python -m bg play --checkpoint checkpoints\latest.pt --human white --device cpu --ai-delay 1500
```

یا با مهره‌ی سیاه:

```powershell
python -m bg play --checkpoint checkpoints\latest.pt --human black --device cpu --ai-delay 1500
```

`--ai-delay 1500` یعنی AI بین هر حرکت ۱.۵ ثانیه مکث می‌کند. اگر سرعت بیشتری خواستید:

```powershell
python -m bg play --checkpoint checkpoints\latest.pt --human white --device cpu --ai-delay 800
```

در رابط گرافیکی نمایش داده می‌شود:

- چیدمان استاندارد تخته‌نرد؛
- تاس کامل نوبت؛
- تاس‌های مصرف‌شده؛
- آخرین حرکت AI؛
- تعداد مهره‌های خارج‌شده و روی Bar؛
- نوار ارزیابی AI از `-1` تا `+1`.

## کنترل‌های بازی

- روی مهره‌ی مشخص‌شده کلیک کنید؛
- اگر برای یک مهره چند تاس ممکن بود، روی تاس موردنظر در پنل سمت راست کلیک کنید یا کلید همان عدد را بزنید؛
- اگر هیچ حرکت قانونی وجود نداشت، `Space` را بزنید؛
- کلید `R`: بازی جدید؛
- کلید `Esc`: خروج.

# تعداد پیشنهادی آموزش

| تعداد تصمیم | کاربرد |
|---:|---|
| `10,000` | تست اجرای Colab |
| `500,000` تا `1,000,000` | مدل اولیه |
| `5,000,000` تا `10,000,000` | مدل قابل بازی |
| `20,000,000+` | آزمایش قوی‌تر |

هر `step` یک تصمیم حرکت مهره است، نه یک بازی کامل. برای شروع بهتر است ابتدا `100000` یا `500000` step اجرا کنید، نتیجه را ارزیابی کنید و بعد آموزش را ادامه دهید.

# تست توسعه‌دهنده

اگر روی محیطی هستید که وابستگی‌ها نصب شده‌اند:

```powershell
python -m pytest
```

وضعیت مورد انتظار فعلی:

```text
11 passed
```
