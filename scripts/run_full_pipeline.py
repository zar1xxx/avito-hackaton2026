"""Скрипт-оркестратор полного нейросетевого пайплайна (end-to-end).

Последовательно выполняет все этапы от проверки данных до готового сабмита:
1. Экспресс-проверка структуры данных и memmap (quick_check_real_data.py)
2. Smoke-тест обучения на ограниченном числе лент (train.py --max-feeds)
3. Полное обучение базовой модели (train.py)
4. Fine-tuning на объединенной выборке train + val (finetune.py)
5. Генерация честных OOF-предсказаний на val (из базовой модели без утечки!)
   и финального сабмита на test (из дообученной модели) + проверка validate_submission.py.

Запуск одной командой:
  python scripts/run_full_pipeline.py --data "D:\avito_data"
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch


def run_step(step_num: int, title: str, cmd: list[str], dry_run: bool = False) -> float:
    """Выполняет один шаг пайплайна с таймером и четкой обработкой ошибок."""
    cmd_str = " ".join(cmd)
    print("\n" + "#" * 75)
    print(f"# ШАГ {step_num}: {title}")
    print(f"# Команда: {cmd_str}")
    print("#" * 75)

    if dry_run:
        print("[DRY-RUN] Команда не запускалась.")
        return 0.0

    t0 = time.perf_counter()
    res = subprocess.run(cmd, check=False)
    elapsed = time.perf_counter() - t0

    if res.returncode != 0:
        print("\n" + "!" * 75)
        print(f" [КРИТИЧЕСКАЯ ОШИБКА] Пайплайн остановлен на Шаге {step_num}: '{title}'")
        print(f" Код возврата процесса: {res.returncode} (время до падения: {elapsed:.2f}с)")
        print("!" * 75)
        sys.exit(res.returncode)

    print(f"\n[OK] Шаг {step_num} ('{title}') успешно завершен за {elapsed:.2f}с.")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Полный end-to-end пайплайн обучения и инференса NN")
    parser.add_argument("--data", required=True, help="Путь к папке с данными (train/val/test)")
    parser.add_argument("--out", default="models/nn/pipeline_run", help="Корневая папка для результатов")
    parser.add_argument("--smoke-feeds", type=int, default=5000, help="Кол-во лент для smoke-теста")
    parser.add_argument("--epochs", type=int, default=10, help="Число эпох базового обучения")
    parser.add_argument("--finetune-epochs", type=int, default=3, help="Число эпох дообучения на train+val")
    parser.add_argument("--batch-size", type=int, default=64, help="Размер батча")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate для базового обучения")
    parser.add_argument("--finetune-lr", type=float, default=2e-4, help="Learning rate для fine-tuning")
    parser.add_argument("--val-weight", type=float, default=2.0, help="Вес val-сэмплов при дообучении")
    parser.add_argument("--mixup-alpha", type=float, default=0.0, help="Alpha для MixUp (0 = выкл)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Температура логитов в ListNet")
    parser.add_argument("--patience", type=int, default=3, help="EarlyStopping patience")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-parts", type=int, default=None, help="Ограничить парты (для быстрого прогона)")
    parser.add_argument("--max-feeds", type=int, default=None, help="Ограничить ленты базового обучения (опционально)")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-smoke", action="store_true", help="Пропустить smoke-тест")
    parser.add_argument("--dry-run", action="store_true", help="Только показать команды без запуска")
    args = parser.parse_args()

    py = sys.executable
    data_dir = Path(args.data)
    out_dir = Path(args.out)

    smoke_dir = out_dir / "01_smoke_test"
    base_dir = out_dir / "02_base_model"
    ft_dir = out_dir / "03_finetuned_model"
    oof_path = out_dir / "oof_val.parquet"
    sub_path = out_dir / "submission.parquet"

    print("=" * 75)
    print(" ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА АВГУСТА (AVITO RANKING NN)")
    print(f" Данные: {data_dir}")
    print(f" Выходная папка: {out_dir}")
    print("=" * 75)

    step_timings: dict[str, float] = {}
    total_start = time.perf_counter()

    # Шаг 1: Быстрая проверка формата данных
    cmd_check = [py, "-u", str(ROOT / "scripts" / "quick_check_real_data.py"), "--data", str(data_dir)]
    step_timings["1. Check real data"] = run_step(1, "Проверка структуры данных", cmd_check, args.dry_run)

    # Шаг 2: Smoke-тест обучения на малом числе лент
    if not args.skip_smoke:
        cmd_smoke = [
            py, "-u", str(ROOT / "src" / "nn" / "train.py"),
            "--data", str(data_dir),
            "--max-feeds", str(args.smoke_feeds),
            "--epochs", "2",
            "--batch-size", str(min(args.batch_size, 32)),
            "--num-workers", str(args.num_workers),
            "--seed", str(args.seed),
            "--out", str(smoke_dir),
        ]
        if args.max_parts is not None:
            cmd_smoke.extend(["--max-parts", str(args.max_parts)])
        step_timings["2. Smoke test"] = run_step(2, f"Smoke-тест ({args.smoke_feeds} лент)", cmd_smoke, args.dry_run)

    # Шаг 3: Полное обучение базовой модели
    cmd_train = [
        py, "-u", str(ROOT / "src" / "nn" / "train.py"),
        "--data", str(data_dir),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--patience", str(args.patience),
        "--mixup-alpha", str(args.mixup_alpha),
        "--temperature", str(args.temperature),
        "--num-workers", str(args.num_workers),
        "--seed", str(args.seed),
        "--out", str(base_dir),
    ]
    if args.max_parts is not None:
        cmd_train.extend(["--max-parts", str(args.max_parts)])
    if args.max_feeds is not None:
        cmd_train.extend(["--max-feeds", str(args.max_feeds)])

    step_timings["3. Full base train"] = run_step(3, "Полное обучение базовой модели", cmd_train, args.dry_run)

    base_ckpt = base_dir / "best_model.pt"
    if not args.dry_run and not base_ckpt.exists():
        base_ckpt = base_dir / "last_model.pt"

    # Шаг 4: Fine-tuning на объединенной выборке train + val
    cmd_finetune = [
        py, "-u", str(ROOT / "src" / "nn" / "finetune.py"),
        "--data", str(data_dir),
        "--model", str(base_ckpt),
        "--val-weight", str(args.val_weight),
        "--epochs", str(args.finetune_epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.finetune_lr),
        "--num-workers", str(args.num_workers),
        "--out", str(ft_dir),
    ]
    if args.max_parts is not None:
        cmd_finetune.extend(["--max-parts", str(args.max_parts)])
    if args.max_feeds is not None:
        cmd_finetune.extend(["--max-feeds", str(args.max_feeds)])

    step_timings["4. Fine-tuning"] = run_step(4, "Fine-tuning (train + val)", cmd_finetune, args.dry_run)

    ft_ckpt = ft_dir / "best_model_finetuned.pt"
    if not args.dry_run and not ft_ckpt.exists():
        ft_ckpt = ft_dir / "last_model_finetuned.pt"

    # Шаг 5А: Честный OOF на val (СТРОГО по базовой модели, которая не видела val при обучении!)
    cmd_oof = [
        py, "-u", str(ROOT / "src" / "nn" / "predict.py"),
        "--data", str(data_dir),
        "--split", "val",
        "--model", str(base_ckpt),
        "--batch-size", str(args.batch_size),
        "--out", str(oof_path),
    ]
    if args.max_parts is not None:
        cmd_oof.extend(["--max-parts", str(args.max_parts)])
    step_timings["5a. Predict Val OOF"] = run_step(5, "Генерация честного OOF на val (для ансамбля)", cmd_oof, args.dry_run)

    # Шаг 5Б: Финальный сабмит на test (из дообученной модели)
    cmd_sub = [
        py, "-u", str(ROOT / "src" / "nn" / "predict.py"),
        "--data", str(data_dir),
        "--split", "test",
        "--model", str(ft_ckpt),
        "--batch-size", str(args.batch_size),
        "--out", str(sub_path),
    ]
    if args.max_parts is not None:
        cmd_sub.extend(["--max-parts", str(args.max_parts)])
    step_timings["5b. Predict Test Sub"] = run_step(6, "Генерация финального сабмита на test", cmd_sub, args.dry_run)

    # Шаг 6: Валидация сабмита, если в папке с данными есть sample_submission.parquet
    sample_sub = data_dir / "sample_submission.parquet"
    if sample_sub.exists() or args.dry_run:
        cmd_val = [
            py, str(ROOT / "scripts" / "validate_submission.py"),
            str(sample_sub),
            str(sub_path),
        ]
        step_timings["6. Validate Sub"] = run_step(7, "Проверка формата сабмита", cmd_val, args.dry_run)

    # Итоговая сводка
    total_elapsed = time.perf_counter() - total_start
    print("\n" + "=" * 75)
    print(" ИТОГИ РАБОТЫ ПОЛНОГО ПАЙПЛАЙНА")
    print("=" * 75)
    for k, v in step_timings.items():
        print(f"  {k:25s}: {v:.2f}с")
    print("-" * 75)

    if not args.dry_run:
        base_ndcg = torch.load(base_ckpt, map_location="cpu").get("best_ndcg", 0.0)
        ft_ndcg = torch.load(ft_ckpt, map_location="cpu").get("best_ndcg", 0.0)
        print(f"  Лучший Val NDCG базовой модели:      {base_ndcg:.4f}")
        print(f"  Лучший Val NDCG после Fine-Tuning:   {ft_ndcg:.4f}")
        print(f"  Честный OOF для ансамбля Дани:       {oof_path}")
        print(f"  Финальный сабмит (готов к отправке): {sub_path}")

    print(f"\n[DONE] Весь пайплайн успешно выполнен за {total_elapsed:.2f}с!")
    print("=" * 75)


if __name__ == "__main__":
    main()
