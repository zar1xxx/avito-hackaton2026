"""Build an executed, portable EDA notebook from the exact aggregate summary.

No Jupyter installation is required to build it. Notebook cells are executed
locally with stdout and matplotlib outputs captured in standard nbformat 4.
Only aggregate statistics are bundled; no individual user/item IDs are saved.
"""
from contextlib import redirect_stdout
import base64
import io
import json
from pathlib import Path
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip() + "\n"}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": textwrap.dedent(source).strip() + "\n"}


def build(summary_path: Path, output_path: Path):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("complete"):
        raise ValueError("EDA summary must be complete")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = output_path.parent / "eda_summary.json"
    snapshot.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    train, val, test = (summary['splits'][name] for name in ('train', 'val', 'test'))
    seen = {row['entity']: row['fraction_of_right'] for row in summary['overlaps']
            if row['left'] == 'train' and row['right'] == 'val'}
    cells = [
        markdown("""
        # EDA полного датасета Авито

        Полный проход по train / val / test: все шарды, ленты, кандидаты и события
        истории. Выборка не использовалась. Статистики считаются до обучения моделей;
        качество модели здесь не измеряется. Время показано в UTC.

        Подготовка: `python -m scripts.prepare_data --data competition --out data/prepared --eda-out runs/eda`.
        Перестроение отчёта: `python -m scripts.build_eda_notebook`.
        Рядом с ноутбуком лежит снимок агрегатов `eda_summary.json` для просмотра без сырых данных.
        """),
        code("""
        import json
        from pathlib import Path
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        ROOT = Path.cwd()
        if not (ROOT / "pyproject.toml").exists():
            ROOT = ROOT.parent
        source = globals().get("_eda_summary_path")
        if source is None:
            source = next(path for path in [Path("eda_summary.json"), ROOT / "runs/eda/summary.json",
                                           ROOT / "notebooks/eda_summary.json"] if path.exists())
        eda = json.loads(source.read_text(encoding="utf-8"))
        assert eda["complete"]
        splits = eda["splits"]
        plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": .2})
        print("Scope:", eda["scope"])
        print("Template keys:", eda["test_template_key_coverage"])
        print("Missing IDs:", eda["missing_id_policy"])
        """),
        markdown("""
        ## Объём, целевые действия и даты

        CTR — доля позитивных кандидатов, а не доля лент. Долю лент без позитивов
        показываем отдельно: по эталону такая лента получает NDCG=1. У test нет
        опубликованных меток, поэтому соответствующие показатели отсутствуют.
        """),
        code("""
        rows = []
        for name, s in splits.items():
            rows.append({"split": name, "feeds": s["feeds"], "candidates": s["candidates"],
                         "positives": s["positives"], "CTR": s["ctr"],
                         "feeds_without_positives_%": 100 * s["feeds_without_positives"] / s["feeds"] if s["positives"] is not None else None,
                         "unique_users": s["unique_users"], "unique_candidate_items": s["unique_candidate_items"],
                         "start_UTC": s["dates_utc"]["min"], "end_UTC": s["dates_utc"]["max"]})
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        """),
        markdown("""
        ## Размеры лент и историй

        Квантили точные: восстановлены из полного гистограммного счётчика с линейной
        интерполяцией, как `numpy.quantile`. На графике числа лент показаны в долях;
        логарифмическая ось Y помогает увидеть редкие размеры.
        """),
        code("""
        rows = []
        for name, s in splits.items():
            for field in ("feed_size", "history_size"):
                rows.append({"split": name, "distribution": field,
                             **{k: v for k, v in s[field].items() if k != "histogram"}})
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        for ax, field, title in zip(axes, ["feed_size", "history_size"], ["Candidates per feed", "History events per feed"]):
            for name, s in splits.items():
                hist = s[field]["histogram"]
                x = sorted(map(int, hist))
                y = [hist[str(i)] / s["feeds"] for i in x]
                ax.plot(x, y, label=name, alpha=.8)
            ax.set(title=title, xlabel="Count", ylabel="Fraction of feeds", yscale="log")
            ax.legend()
        plt.show()
        """),
        markdown("""
        ## CTR по исходной позиции в train

        В train сохранена продовая позиция; val/test получены со случайным показом.
        График описывает наблюдаемую зависимость, в которой одновременно участвуют
        порядок показа и качество кандидатов. Сам по себе он не оценивает propensity
        и не задаёт готовые IPW-веса. Нумерация с 0, после фильтрации есть пропуски.
        """),
        code("""
        position = pd.DataFrame([{"position": int(k), **v} for k, v in splits["train"]["ctr_by_position"].items()]).sort_values("position")
        print(position.head(20).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        shown = position[position.position <= 100]
        axes[0].plot(shown.position, 100 * shown.ctr)
        axes[0].set(title="Train: CTR by original position (0-100)", xlabel="Position", ylabel="CTR, %")
        axes[1].plot(position.position, position.impressions)
        axes[1].set(title="Train: impressions by original position", xlabel="Position", ylabel="Candidates", yscale="log")
        plt.show()
        """),
        markdown("""
        ## Пропуски эмбеддингов и отсутствующие ID

        `-1` — отсутствие эмбеддинга: его нельзя использовать как индекс массива.
        Нулевые user/item ID считаются отсутствующими и исключены из числа уникальных
        объектов и пересечений. Количество таких вхождений приведено отдельно.
        """),
        code("""
        rows = []
        for name, s in splits.items():
            rows.append({"split": name,
                         "candidate_missing": s["candidate_missing_embeddings"],
                         "candidate_missing_%": 100 * s["candidate_missing_embedding_rate"],
                         "history_missing": s["history_missing_embeddings"],
                         "history_missing_%": 100 * s["history_missing_embedding_rate"],
                         **{f"zero_{key}_ids": value for key, value in s["zero_ids"].items()}})
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        """),
        markdown("""
        ## Повторные товары внутри ленты

        Некоторые item_id повторяются в одной train-ленте с различными позициями
        и признаками. Все исходные вхождения сохранены; метки не агрегировались.
        Поэтому уникальный ключ строки — `(split, feed_id, candidate_index)`.
        `candidate_index` — технический номер в исходном списке, его не следует
        включать в модельные признаки. Группы item-признаков соединяют по
        `(split, feed_id, item_id)` как many-to-one; признаки конкретного
        вхождения — по ключу с candidate_index. Подробные повторные строки
        сохранены локально в `runs/eda/duplicate_candidates/`.
        """),
        code("""
        rows = [{"split": name, **{field: s[field] for field in
                 ("duplicate_item_pairs", "duplicate_item_feeds", "duplicate_item_rows_excess", "conflicting_target_pairs")}}
                for name, s in splits.items()]
        print(pd.DataFrame(rows).to_string(index=False))
        """),
        markdown("""
        ## Пересечения пользователей и товаров

        Точные пересечения уникальных ID, без выборки и приближённых оценок.
        Кандидаты и товары из истории показаны отдельно. `fraction_of_right` отвечает,
        какая доля уникальных ID правого split уже была в левом. Это доля уникальных
        объектов, а не доля строк или показов.
        """),
        code("""
        overlap = pd.DataFrame(eda["overlaps"])
        print(overlap.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        """),
        markdown("""
        ## Даты и изменение распределения признаков

        Между периодами есть временные зазоры. Сводки f1–f5 вычислены по всем
        кандидатам; стандартное отклонение — генеральное (`ddof=0`).
        """),
        code("""
        rows = []
        for name, s in splits.items():
            for day, values in s["by_day_utc"].items():
                rows.append({"split": name, "date_UTC": day, **values})
        daily = pd.DataFrame(rows)
        print(daily.to_string(index=False))
        features = pd.DataFrame([{"split": name, "feature": feature, **values}
                                 for name, s in splits.items() for feature, values in s["features"].items()])
        print("\\nFeature distributions:")
        print(features.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
        fig, ax = plt.subplots(figsize=(10, 3.5), constrained_layout=True)
        for name in splits:
            part = daily[daily.split == name]
            ax.plot(pd.to_datetime(part.date_UTC), part.feeds, marker="o", label=name)
        ax.set(title="Feed volume by date (UTC)", xlabel="Date", ylabel="Feeds", yscale="log")
        ax.legend()
        fig.autofmt_xdate()
        plt.show()
        """),
        markdown("""
        ## Категории и типы событий

        Коды событий категориальные; по одному номеру нельзя приписывать им
        бизнес-смысл. Частоты считаются по всем вхождениям событий в опубликованные
        истории: одна и та же активность может повторяться в историях разных лент.
        """),
        code("""
        for field in ("history_event_types", "history_event_sources"):
            table = pd.DataFrame({name: pd.Series({int(k): v for k, v in s[field].items()}) for name, s in splits.items()}).fillna(0).astype("int64").sort_index()
            print("\\n" + field + " (event occurrences):")
            print(table.to_string())
        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        for name, s in splits.items():
            hist = {int(k): v / s["candidates"] for k, v in s["candidate_categories"].items()}
            keys = sorted(hist)
            ax.plot(keys, [hist[k] for k in keys], marker=".", label=name)
        ax.set(title="Candidate category distribution", xlabel="Category code", ylabel="Fraction of candidates")
        ax.legend()
        plt.show()
        """),
        markdown("""
        ## Табличные данные для команды

        `data/prepared/<split>/candidates/*.parquet`: одна строка на кандидата;
        ключ `(split, feed_id, candidate_index)`, item_id, f1–f5, категория, индекс эмбеддинга,
        user_id, timestamp; target только для train/val, pos только для train.

        `data/prepared/<split>/feeds/*.parquet`: одна строка на ленту; история,
        user_embedding, timestamp, user_id, feed_size. Связь по `(split, feed_id)`.
        Истории не размножены на каждого кандидата. Индексы эмбеддингов относятся
        к таблице **своего split** в `competition/<split>/item_embeddings/`.

        Готовность и перечень частей: `data/prepared/manifest.json`.
        Набор принимается только при `complete=true`. При прерывании запускать
        подготовку с `--resume`: неизменённые части повторно не записываются,
        агрегаты EDA пересчитываются полностью.

        Для ленивого чтения:
        ```python
        from src.common.preparation import scan_candidates
        val = scan_candidates("data/prepared", "val")
        # val.select("feed_id", "item_id", "f1", "target").collect()
        ```

        Это базовые таблицы с исходными признаками. Косинусы к истории, OOF-статистики
        и обученные признаки добавляются отдельными группами по ключам.
        Признаки из будущих историй здесь не строились.
        """),
        markdown(f"""
        ## Выводы для следующих экспериментов

        - В train **{train['candidates']:,}** кандидатов, в val **{val['candidates']:,}**,
          в test **{test['candidates']:,}**. Обучение на всём train требует чтения
          по частям или отдельного планирования памяти; для быстрых экспериментов
          выбирать целые ленты.
        - Доля позитивов: train **{train['ctr']:.4%}**, val **{val['ctr']:.4%}**.
          Train и val собраны разными способами, поэтому одного совпадения
          распределения признаков недостаточно: модель нужно сравнивать на val.
        - **{val['feeds_without_positives'] / val['feeds']:.2%}** лент val не имеют
          позитивов и получают NDCG=1 при любом ранжировании. Вместе с общей метрикой
          полезно смотреть качество на лентах с позитивами, сохраняя эталонную
          общую метрику для сравнения решений.
        - **{seen['users']:.2%}** уникальных пользователей val встречались в train;
          **{seen['candidate_items']:.2%}** уникальных товаров-кандидатов val
          встречались среди кандидатов train. Стоит отдельно оценивать метрику
          для знакомых и новых ID. Эти проценты относятся к уникальным ID,
          а не к доле показов.
        - Отсутствующие эмбеддинги кандидатов: train
          **{train['candidate_missing_embedding_rate']:.4%}**, val
          **{val['candidate_missing_embedding_rate']:.4%}**, test
          **{test['candidate_missing_embedding_rate']:.4%}**. Обработку `-1`
          и признак наличия эмбеддинга сохранять в обоих модельных пайплайнах.
        - Зависимость CTR от позиции показана выше; сравнить способы дебайаса
          по псевдо-Private. Исторические таргетные статистики считать по прошлому
          или OOF. Смысл event_type исследовать отдельно: по частотам кодов он
          не определяется.
        """),
    ]
    namespace = {"__name__": "__main__", "_eda_summary_path": snapshot.resolve()}
    figures = summary_path.parent / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    original_show = plt.show
    figure_number = 0
    try:
        # show() is captured below; saved PNGs match the embedded notebook outputs.
        plt.show = lambda *args, **kwargs: None
        execution_count = 0
        for cell in cells:
            if cell["cell_type"] != "code":
                continue
            execution_count += 1
            cell["execution_count"] = execution_count
            stream = io.StringIO()
            with redirect_stdout(stream):
                exec(compile(cell["source"], "<eda-notebook>", "exec"), namespace)
            if stream.getvalue():
                cell["outputs"].append({"output_type": "stream", "name": "stdout", "text": stream.getvalue()})
            for number in plt.get_fignums():
                figure_number += 1
                figure = plt.figure(number)
                buffer = io.BytesIO()
                figure.savefig(buffer, format="png", dpi=120)
                png = buffer.getvalue()
                (figures / f"figure_{figure_number:02d}.png").write_bytes(png)
                cell["outputs"].append({"output_type": "display_data", "metadata": {},
                                        "data": {"image/png": base64.b64encode(png).decode("ascii"),
                                                 "text/plain": f"EDA figure {figure_number}"}})
                plt.close(figure)
    finally:
        plt.show = original_show
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3.11", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": sys.version.split()[0]},
    }, "cells": cells}
    for index, cell in enumerate(cells):
        cell["id"] = f"eda-{index:02d}"
    output_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Notebook: {output_path}; executed cells: {execution_count}; figures: {figure_number}")


if __name__ == "__main__":
    build(ROOT / "runs/eda/summary.json", ROOT / "notebooks/01_data_eda.ipynb")
