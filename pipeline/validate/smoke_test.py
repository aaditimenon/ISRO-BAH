import pandas as pd
from pathlib import Path

def run_smoke_test(catalogue_path):
    """
    Validate recovery of known benchmark planets:
    - WASP-121b (TIC 22529346, Period ~1.27d)
    - L 98-59 (TIC 307210830, Periods ~2.25d, 3.69d, 7.45d)
    - TOI-270 (TIC 259377017, Periods ~3.36d, 5.66d, 11.38d)
    """
    benchmarks = {
        'WASP-121b': {'tic_id': 22529346, 'period': 1.2749, 'recovered': False},
        'L 98-59': {'tic_id': 307210830, 'period': 2.25, 'recovered': False}, # checks first planet
        'TOI-270': {'tic_id': 259377017, 'period': 3.36, 'recovered': False} # checks first planet
    }
    
    path = Path(catalogue_path)
    if not path.exists():
        print(f"Warning: Catalogue not found at {catalogue_path}. Return unrecovered benchmarks.")
        return benchmarks
        
    try:
        df = pd.read_parquet(path)
        
        for name, info in benchmarks.items():
            # Find signals for this TIC ID
            star_signals = df[df['tic_id'] == info['tic_id']]
            if len(star_signals) > 0:
                # Check if any signal has a period within 2% of the benchmark period
                for _, row in star_signals.iterrows():
                    diff = abs(row['period'] - info['period']) / info['period']
                    if diff < 0.02 and row['disposition'] != 'DISCARD':
                        info['recovered'] = True
                        info['found_period'] = row['period']
                        info['found_sde'] = row['sde']
                        break
                        
    except Exception as e:
        print(f"Error during smoke test validation: {e}")
        
    return benchmarks
