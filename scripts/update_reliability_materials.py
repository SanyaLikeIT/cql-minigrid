"""Generate measured result prose after the full ten-seed study completes."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text())


def update():
    complete=read('reports/reliability/complete.json')
    assert complete['status']=='complete'
    baseline=read('reports/reliability/baseline/summary.json')
    historical=read('reports/summary.json')
    ablation=read('reports/reliability/ablation/summary.json')
    nav=read('reports/reliability/navigation/summary.json')
    lines=['# Ten-seed measured results','',
        'Training seeds 0–9 are the independent experimental units. All intervals',
        'are unadjusted 95% Student t intervals across models, conditional on the',
        'fixed data and environment sets; they are not episode-level intervals.','',
        '| Dataset | Method | Historical 3-seed success % | Ten-seed mean % | Sample SD % | 95% t CI % |',
        '|---|---|---:|---:|---:|---|']
    for r in baseline['aggregate']:
        old=next(o for o in historical['aggregate'] if o['dataset']==r['dataset'] and o['method']==r['method'])
        s=r['success_rate'];lo,hi=s['ci95']
        lines.append(f"| {r['dataset']} | {r['method'].upper()} | {100*old['success_rate']['mean']:.2f} | {100*s['mean']:.2f} | {100*s['std']:.2f} | [{100*lo:.2f}, {100*hi:.2f}] |")
    lines+=['','## Paired effects','',
            '| Dataset | Contrast | Mean success difference (pp) | Sample SD (pp) | 95% t CI (pp) |',
            '|---|---|---:|---:|---|']
    conclusion=[]
    for r in baseline['paired_differences']:
        s=r['statistics']['success_rate'];lo,hi=s['ci95']
        lines.append(f"| {r['dataset']} | {r['contrast']} | {100*s['mean']:+.2f} | {100*s['std']:.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |")
        finding='the interval includes zero' if lo<=0<=hi else ('the interval is above zero' if lo>0 else 'the interval is below zero')
        conclusion.append(f"{r['dataset']} {r['contrast']}: {finding}.")
    lines+=['',*conclusion,'','These exploratory comparisons do not establish universal superiority.','',
            '## Fresh navigation test','',
            '| Policy | Success mean ± sample SD % | 95% t CI % | Collision % | Timeout % | Stall % | Successful actions |',
            '|---|---:|---|---:|---:|---:|---:|']
    for r in nav['aggregate']:
        s=r['success_rate'];lo,hi=s['ci95']
        lines.append(f"| {r['candidate']} | {100*s['mean']:.2f} ± {100*s['std']:.2f} | [{100*lo:.2f}, {100*hi:.2f}] | {100*r['collision_rate']['mean']:.2f} | {100*r['timeout_rate']['mean']:.2f} | {100*r['stall_rate']['mean']:.2f} | {r['successful_steps_mean']['mean']:.2f} |")
    lines+=['','All ten original and ten improved models use common fresh seeds 40000–40199.',
            'Historical navigation test results on 30000–30199 are preserved separately.',
            f"The validation-locked configuration is `{complete['selected']}`.",'','## Execution','',
            f"Unique checkpoints used: {complete['unique_checkpoints']}; reused: {complete['reused_models']}; newly trained: {complete['newly_trained_models']}.",
            f"Baseline: {complete['baseline_evaluation_episodes']} episodes; alpha study: {complete['ablation_evaluation_episodes']} logical episodes; navigation validation: {complete['navigation_validation_episodes']}; fresh navigation test: {complete['fresh_navigation_test_episodes']}.",
            f"Measured runner time including resume verification: {complete.get('total_execution_seconds',complete['session_seconds']):.2f} seconds.",
            'Alpha 0/1 conditions reuse equivalent baseline models for new seeds; logical',
            'episode counts include their appearances in both comparisons. Historical',
            'three-seed checkpoints and all four frozen NPZ files retain their hashes.','',
            'Full distributions, conditional valid-model counts, paired effects, CSVs',
            'and per-model plots are in `reports/reliability/`. The alpha study contains',
            '120 CQL conditions and ten balanced-data BC runs, without test-driven tuning.','',
            '![Baseline](../reports/reliability/baseline/metrics.png)',
            '![Alpha](../reports/reliability/ablation/alpha_metrics.png)',
            '![Navigation](../reports/reliability/navigation/metrics.png)']
    (ROOT/'docs/reliability_results.md').write_text('\n'.join(lines)+'\n')
    p=ROOT/'README.md';text=p.read_text()
    marker='## Historical three-seed results'
    if '## Expanded measured results' in text:
        start=text.index('## Expanded measured results');end=text.index(marker,start);text=text[:start]+text[end:]
    text=text.replace(marker,'## Expanded measured results\n\n'+
        f"The completed extension contains 60 baseline models, 130 alpha-study conditions,\n80 navigation candidates and 4,000 fresh navigation test episodes.\n"+
        f"It reuses {complete['reused_models']} checkpoints and trains {complete['newly_trained_models']} additional models.\n"+
        '[Measured results, confidence intervals and historical comparison](docs/reliability_results.md).\n\n'+marker)
    text=text.replace('artifacts/navigation-geometry/best.pt','artifacts/reliability/navigation/best.pt').replace('artifacts/navigation-geometry/evaluation-config.json','artifacts/reliability/navigation/evaluation-config.json').replace('the 200 fresh test seeds','the 200 new test seeds 40000–40199')
    p.write_text(text)
    ru=['ПРОЕКТ: OFFLINE REINFORCEMENT LEARNING В MINIGRID','',
        'Назначение проекта — сравнить обучение агента по фиксированным переходам:',
        'offline DQN, Conservative Q-Learning (CQL) и Behavioral Cloning (BC).',
        'Текущая версия расширяет исследования до 10 независимых обучающих сидов',
        '0–9. Исторические результаты трёх сидов сохранены отдельно и не выдаются',
        'за результаты десяти запусков.','',
        'ОКРУЖЕНИЕ И РЕАЛИЗАЦИЯ','',
        'MiniGrid-Dynamic-Obstacles-8x8-v0; наблюдение — 837 float32 значений:',
        'полная символическая карта 8×8, направление агента и прошедшее время.',
        'Действия: поворот влево, поворот вправо, шаг вперёд. Обычный шаг даёт 0,',
        'достижение цели — 1 − 0.9 × elapsed_steps / 256. Попытка заблокированного',
        'шага вперёд завершает эпизод с наградой −1. Поворот на месте допустим;',
        'поэтому прежняя модель могла долго вращаться, не совершая столкновение.',
        'Лимит — 256 действий. Успех, столкновение и timeout учитываются отдельно.','',
        'Сеть: два скрытых слоя по 128 ReLU нейронов и три выхода. Adam 0.0003,',
        'batch 256, gamma 0.99, 5000 градиентных обновлений, target update 250.',
        'DQN использует Huber TD loss; CQL добавляет alpha × среднее разности',
        'logsumexp значений всех действий и значения действия из датасета.',
        'BC обучается классифицировать записанное действие через cross-entropy.',
        'Оба признака завершения блокируют bootstrap. Математика алгоритмов',
        'при расширении числа сидов не изменялась.','',
        'ДАННЫЕ И ПРОТОКОЛ','',
        'PPO собрал исходные strong/mixed файлы с 200035/200019 переходами.',
        'Balanced содержит 200025 переходов с квотами 50/25/25% по переходам.',
        'Guided содержит 50012 переходов BFS-учителя. Все четыре исходных NPZ',
        'проверяются по SHA-256 и не перегенерировались для новых сидов.',
        'Учитель используется исключительно при сборе демонстраций. Нейросеть',
        'во время применения не вызывает BFS, маску действий, принудительные',
        'повороты или эвристику обхода столкновений.','',
        'Baseline: 60 моделей (2 датасета × 3 метода × 10 сидов), 6000 эпизодов',
        'на общих исходных test seeds 20000–20099. Alpha study: 3 датасета ×',
        '4 alpha (0, 0.1, 1, 5) × 10 сидов = 120 условий CQL и 10 balanced BC.',
        'Совместимые исходные веса используются повторно; новые alpha=0/1',
        'strong/mixed условия ссылаются на эквивалентные DQN/CQL baseline модели.',
        'Исторические дубли, уже существовавшие до расширения, не переобучались.','',
        'Навигация: 8 конфигураций × 10 сидов. Каждый запуск получает исходные',
        'веса Mixed CQL того же обучающего сида и 5000 дополнительных обновлений.',
        'Сохранены штрафы шага 0/0.002/0.005, guided data без геометрии и с ней.',
        'Геометрия добавляет 50 признаков из уже доступного наблюдения; новые',
        'колонки первого слоя инициализируются нулями для сохранения старой',
        'политики перед дообучением. Выбор использует только validation',
        '10000–10063 и исходные quality gates. Затем выбор записывается в',
        'test_lock.json до открытия новых сидов 40000–40199. Старая проверка',
        '30000–30199 остаётся исторической. Новая проверка: 10 исходных и',
        '10 улучшенных моделей × 200 общих окружений = 4000 эпизодов.','',
        'СТАТИСТИКА И ОГРАНИЧЕНИЯ','',
        'Единица статистического анализа — независимо обученная модель.',
        'Отчёты включают среднее, выборочное SD, медиану, минимум/максимум,',
        '95% интервал Стьюдента и индивидуальные значения всех обучающих сидов.',
        'Парные CQL−DQN и CQL−BC эффекты рассчитываются по совпадающим сидам.',
        'Эпизоды одной модели не подменяют независимые обучающие запуски.',
        'Интервалы не обрезаются по границам вероятности и не корректируются',
        'за множественные сравнения; сравнения имеют исследовательский характер.',
        'Для длины успешного эпизода исключаются только модели без успехов;',
        'действительное число моделей указано. Неуспешные модели остаются во',
        'всех безусловных метриках. Отдельно измеряются stall rate (серия ≥16',
        'действий без изменения позиции) и длины стационарных серий.',
        'Данные фиксированы, PPO-источник имеет один обучающий сид. Выводы',
        'относятся к этому протоколу, а не ко всем датасетам и окружениям.','',
        'ФАКТИЧЕСКИЕ РЕЗУЛЬТАТЫ','',
        f"Повторно использовано checkpoints: {complete['reused_models']}.",
        f"Обучено новых моделей: {complete['newly_trained_models']}.",
        f"Выбранная validation-конфигурация: {complete['selected']}.",
        f"Время выполнения runner: {complete.get('total_execution_seconds',complete['session_seconds']):.2f} секунд.",'',
        'Таблицы ниже сформированы непосредственно из сохранённых результатов.','']
    ru+=lines[6:]
    ru+=['','СТРУКТУРА И ТЕХНОЛОГИИ','',
        'Python 3.12; PyTorch; Gymnasium/MiniGrid; Stable-Baselines3 PPO; NumPy;',
        'SciPy (интервалы Стьюдента); Matplotlib; Pillow; imageio-ffmpeg;',
        'ReportLab/PyMuPDF (PDF); Jupyter/ipykernel (notebook); pytest.',
        'Точные версии и optional dependencies записаны в pyproject.toml.',
        'src/cql_minigrid — окружение, сбор данных, алгоритмы, оценка и runner.',
        'configs — настройки, схема наблюдений и разделение сидов.',
        'artifacts/reliability — новые веса, восстановленные контролы, полные',
        'эпизодные записи, прогресс, идентичность данных/кода и test lock.',
        'reports/reliability — JSON, CSV, Markdown, интервалы и графики всех сидов.',
        'docs — протоколы, обновлённый PDF и исполненный численный notebook.',
        'scripts — воспроизводимое построение графиков/PDF и аудит полноты.',
        'tests — существующие проверки и регрессии статистики/разделения сидов.',
        'Незавершённые запуски возобновляются прежней командой runner. Файлы',
        'checkpoint и JSON записываются атомарно; несовместимые config, данные',
        'или численный код отклоняются. CPU и один поток сохранены для сравнения.',
        'Notebook использует реальные функции потерь проекта на синтетическом',
        'переходе, показывает градиенты и изменение Q, затем выполняет реальную',
        'политику; синтетический пример отделён от экспериментальных результатов.',
        'PDF — английские 16:9 слайды с цветами #F7F4EE/#302D2A/#79504B,',
        'Liberation Serif/Open Sans. Слайд 5 содержит только формулы; a_D —',
        'действие датасета, b — отдельный индекс суммы консервативного штрафа.']
    (ROOT/'PROJECT_DESCRIPTION_RU.txt').write_text('\n'.join(ru)+'\n')

if __name__=='__main__':update()
