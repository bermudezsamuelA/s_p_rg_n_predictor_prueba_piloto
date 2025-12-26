# scripts/full_diagnostics.py
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path('.')
OUT = ROOT / 'data' / 'diagnostics'
OUT.mkdir(parents=True, exist_ok=True)

def load_paths():
    return {
        'super': ROOT / 'data' / 'analysis' / 'supergana_preparado.csv',
        'triple': ROOT / 'data' / 'analysis_triplegana' / 'triplegana_preparado.csv'
    }

def counts_and_nans(df):
    return {
        'shape': df.shape,
        'duplicates_fecha_hora': int(df.duplicated(subset=['fecha','hora']).sum()),
        'nans': df[['fecha','hora','numero','signo']].isna().sum().to_dict()
    }

def inspect_bins(df, bins=20):
    nums = df['numero'].astype(int).values
    freq, edges = np.histogram(nums, bins=bins)
    expected = len(nums) / bins
    low_expected = [i for i,e in enumerate(freq) if e < 5]
    return {
        'bins': bins,
        'edges': [int(x) for x in edges],
        'freq': freq.tolist(),
        'expected_per_bin': expected,
        'bins_with_obs_lt_5': low_expected
    }

def discarded_rows_report(path, df):
    # If analyzer exported filas_descartadas, include them; otherwise detect rows with NaN
    discarded_file = path.parent / 'filas_descartadas.csv'
    if discarded_file.exists():
        d = pd.read_csv(discarded_file)
        d.to_csv(OUT / f'{path.stem}_filas_descartadas.csv', index=False)
        return {'from_file': True, 'count': len(d)}
    else:
        mask = df[['fecha','numero','hora','signo']].isna().any(axis=1)
        d = df[mask]
        if not d.empty:
            d.to_csv(OUT / f'{path.stem}_filas_descartadas_detected.csv', index=False)
        return {'from_file': False, 'count': int(mask.sum())}

def summary_stats(df):
    ult = df['ultimo_digito'].value_counts().sort_index().to_dict()
    par = df['paridad'].value_counts().to_dict()
    corr = df.select_dtypes(include=[np.number]).corr()['numero'].drop('numero').to_dict()
    return {'ultimo_digito': ult, 'paridad': par, 'corr_with_numero': corr}

def run_all():
    paths = load_paths()
    report = {}
    for key, path in paths.items():
        if not path.exists():
            report[key] = {'error': 'file not found', 'path': str(path)}
            continue
        df = pd.read_csv(path)
        report[key] = {}
        report[key]['counts'] = counts_and_nans(df)
        report[key]['bins20'] = inspect_bins(df, bins=20)
        report[key]['bins10'] = inspect_bins(df, bins=10)
        report[key]['discarded'] = discarded_rows_report(path, df)
        report[key]['summary'] = summary_stats(df)
    # Save report
    with open(OUT / 'diagnostics_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("Diagnostics saved to", OUT)

if __name__ == '__main__':
    run_all()
