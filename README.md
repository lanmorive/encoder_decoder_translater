# Transformer Translator (DE → EN)

Энкодер-декодер трансформер (архитектура "Attention is All You Need") для перевода с немецкого на английский на датасете Multi30k.

## Файлы

- `translater.py` — архитектура модели: encoder/decoder, multi-head attention, positional encoding, `make_model()`.
- `tokenizer.py` — обучение и кэширование BPE-токенизаторов для src (de) и tgt (en).
- `trainer.py` — загрузка данных, цикл обучения, label smoothing, warm-up scheduler, greedy decode, сохранение модели.

## Установка

```bash
pip install torch datasets tokenizers tqdm
```

## Обучение

```bash
python3 -c "
import torch
from trainer import train_translation
torch.manual_seed(0)
train_translation(n_layers=6, d_model=512, d_ff=2048, h=8, n_epochs=20, batch_size=128)
"
```

Параметр `device` определяется автоматически (`cuda`, если доступна, иначе `cpu`). Можно задать явно: `train_translation(..., device="cuda:0")`.

После обучения:
- печатаются примеры перевода из валидационного сплита;
- веса модели сохраняются в `checkpoints/model.pt`.

## Загрузка сохранённой модели

```python
import torch
from translater import make_model

model = make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8)
model.load_state_dict(torch.load("checkpoints/model.pt"))
model.eval()
```

`src_vocab`/`tgt_vocab` и гиперпараметры должны совпадать с теми, что использовались при обучении. Размеры словарей можно получить из сохранённых токенизаторов в `.cache/tokenizer_de.json` и `.cache/tokenizer_en.json`.

## Данные

Датасет [Multi30k](https://huggingface.co/datasets/bentrevett/multi30k) загружается автоматически через `datasets` при первом запуске и кэшируется.
