# CLIP_witch

## Run Command

```bash
python -m src.main --name "full" --train_data (train file path) --image_key file --epochs 30 --batch_size 512 --distributed --device_ids 0 1 --weight_decay 0.2 --num_warmup_steps 2000 --distributed_init_method tcp://127.0.0.1:5433
```
