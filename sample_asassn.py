from pathlib import Path
from collections import defaultdict
import re
import shutil

import pandas as pd

PERCENT = 0.30
SEED = 42

V_CSV = 'asassn_catalog_full.csv'
G_CSV = 'asassn_variables_x.csv'
V_FOLDER = 'asassnvarlc_vband_complete'
G_FOLDER = 'g_band_lcs'

STAR_RE = re.compile(r'J\d{6}(?:\.\d+)?[+-]\d{6}(?:\.\d+)?', re.I)
GAIA_RE = re.compile(r'\d{12,20}')
EXTENSIONS = ('.dat', '.csv', '.txt', '.tsv', '.lc', '.ascii', '.fits', '.parquet', '.gz')


def name_id(value):
    match = STAR_RE.search(str(value))
    return re.sub(r'[^A-Z0-9+-]', '', match.group().upper()) if match else ''


def gaia_id(value):
    match = GAIA_RE.search(str(value))
    return match.group() if match else ''


def simple_id(value):
    value = str(value).strip().upper()
    return re.sub(r'[^A-Z0-9-]', '', value) if value and value not in ('NAN', 'NONE') else ''


def read_catalog(path, band):
    available = pd.read_csv(path, nrows=0).columns.tolist()
    wanted = {'id', 'asassn_name', 'source_id', 'variable_type', 'ml_classification',
              'gdr2_id', 'edr3_source_id'}
    keep = [c for c in available if c.lower().strip() in wanted]
    df = pd.read_csv(path, usecols=keep, dtype=str, keep_default_na=False)
    df.columns = [c.lower().strip() for c in df.columns]
    # The V catalog's ID is a UUID, while the g catalog's ID is the ASASSN name.
    star_col = 'asassn_name' if band == 'v' else 'id'
    class_col = 'variable_type' if band == 'v' else 'ml_classification'
    if star_col not in df or class_col not in df:
        raise ValueError(f'{path.name}: expected {star_col} and {class_col}. Found {available}')
    for col in ('id', 'source_id', 'asassn_name', 'edr3_source_id', 'gdr2_id'):
        if col not in df:
            df[col] = ''
    df['star'] = df[star_col].map(name_id)
    df['gaia'] = df['edr3_source_id'].map(gaia_id)
    df['type'] = df[class_col].str.strip()
    return df


def unique_lookup(values):
    result = {}
    for i, value in enumerate(values):
        if not value:
            continue
        if value in result:
            result[value] = -1
        else:
            result[value] = i
    return result


def pair_catalogs(v, g):
    by_name = unique_lookup(g['star'])
    by_gaia = unique_lookup(g['gaia'])
    pairs = []
    used_g = set()
    mismatched = 0
    for i, r in enumerate(v.itertuples(index=False)):
        if not r.type:
            continue
        candidates = set()
        if r.star and r.star in by_name:
            candidates.add(by_name[r.star])
        if r.gaia and r.gaia in by_gaia:
            candidates.add(by_gaia[r.gaia])
        if len(candidates) != 1:
            continue
        j = candidates.pop()
        if j < 0 or j in used_g:
            continue
        s = g.iloc[j]
        mismatched += bool(s['type'] and r.type != s['type'])
        pairs.append((i, j, r.star or s['star'], r.type, s['type']))
        used_g.add(j)
    print(f'Catalog matches: {len(pairs):,} (different class labels: {mismatched:,})', flush=True)
    return pd.DataFrame(pairs, columns=['v_row', 'g_row', 'star', 'variable_type', 'g_type'])


def identifiers(row):
    result = [row['star'], row['gaia']]
    result += [simple_id(row[c]) for c in ('id', 'source_id', 'gdr2_id')]
    return {x for x in result if x}


def file_identifiers(path, folder):
    relative = path.relative_to(folder)
    found = set()
    star = name_id(relative.as_posix())
    if star:
        found.add(star)
    for part in relative.parts:
        name = part
        while any(name.lower().endswith(ext) for ext in EXTENSIONS):
            name = name.rsplit('.', 1)[0]
        if name.lower().startswith('light_curve_'):
            name = name[12:]
        clean = simple_id(name)
        if clean:
            found.add(clean)
        gaia = gaia_id(name)
        if gaia:
            found.add(gaia)
    return found


def find_files(folder, needed):
    if not folder.is_dir():
        raise FileNotFoundError(f'Missing folder: {folder}')
    result = defaultdict(list)
    seen = 0
    examples = []
    for path in folder.rglob('*'):
        if not path.is_file() or not path.name.lower().endswith(EXTENSIONS):
            continue
        seen += 1
        if len(examples) < 4:
            examples.append(str(path.relative_to(folder)))
        for key in file_identifiers(path, folder) & needed:
            result[key].append(path)
    print(f'{folder.name}: {seen:,} files, {len(result):,} recognized identifiers', flush=True)
    print('  Example filenames:', examples, flush=True)
    return result


def get_paths(row, index):
    paths = {}
    for key in identifiers(row):
        for file in index.get(key, []):
            paths[str(file)] = file
    return list(paths.values())


def copy_catalog_rows(original, output, rows):
    first = True
    for chunk in pd.read_csv(original, chunksize=50000, dtype=str):
        subset = chunk.loc[chunk.index.isin(rows)]
        if not subset.empty:
            subset.to_csv(output, mode='w' if first else 'a', index=False, header=first)
            first = False


def main():
    root = Path(__file__).resolve().parent
    out = root / 'data'
    if not 0 < PERCENT <= 1:
        raise ValueError('Set PERCENT to something between 0 and 1')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('data/ already contains files. Rename or clear it first.')

    v = read_catalog(root / V_CSV, 'v')
    g = read_catalog(root / G_CSV, 'g')
    print(f'V catalog: {len(v):,} stars; g catalog: {len(g):,} stars', flush=True)
    pairs = pair_catalogs(v, g)
    if pairs.empty:
        raise RuntimeError('No catalog matches by ASASSN name or Gaia EDR3 ID')

    v_rows = v.iloc[pairs.v_row]
    g_rows = g.iloc[pairs.g_row]
    v_needed = set().union(*(identifiers(row) for _, row in v_rows.iterrows()))
    g_needed = set().union(*(identifiers(row) for _, row in g_rows.iterrows()))
    v_index = find_files(root / V_FOLDER, v_needed)
    g_index = find_files(root / G_FOLDER, g_needed)

    matched = []
    v_only = g_only = 0
    for p in pairs.itertuples(index=False):
        v_paths = get_paths(v.iloc[p.v_row], v_index)
        g_paths = get_paths(g.iloc[p.g_row], g_index)
        v_only += bool(v_paths and not g_paths)
        g_only += bool(g_paths and not v_paths)
        if v_paths and g_paths:
            matched.append((p.v_row, p.g_row, p.star, p.variable_type, p.g_type, v_paths, g_paths))
    print(f'Matched stars with files in BOTH bands: {len(matched):,}', flush=True)
    print(f'  With V files only: {v_only:,}; with g files only: {g_only:,}', flush=True)
    if not matched:
        raise RuntimeError('Catalogs overlap, but no paired light-curve files were found. See filename examples above.')

    table = pd.DataFrame(matched, columns=['v_row', 'g_row', 'star', 'variable_type',
                                           'g_type', 'v_files', 'g_files'])
    sample = pd.concat([
        part.sample(n=max(1, round(len(part) * PERCENT)), random_state=SEED)
        for _, part in table.groupby('variable_type')
    ]).sort_values('v_row')

    (out / 'vband').mkdir(parents=True)
    (out / 'gband').mkdir(parents=True)
    manifest = []
    for n, row in enumerate(sample.itertuples(index=False), 1):
        locations = []
        for dest, files, source_folder in (
            ('vband', row.v_files, root / V_FOLDER),
            ('gband', row.g_files, root / G_FOLDER),
        ):
            copied = []
            for original in files:
                target = out / dest / original.relative_to(source_folder)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, target)
                copied.append(str(target.relative_to(out)))
            locations.append(';'.join(copied))
        manifest.append((row.star, row.variable_type, row.g_type, *locations))
        if n % 2000 == 0:
            print(f'Copied {n:,} / {len(sample):,}', flush=True)

    pd.DataFrame(manifest, columns=['star', 'v_class', 'g_class', 'vband_files', 'gband_files']).to_csv(
        out / 'manifest.csv', index=False)
    copy_catalog_rows(root / V_CSV, out / 'vband_catalog.csv', set(sample.v_row))
    copy_catalog_rows(root / G_CSV, out / 'gband_catalog.csv', set(sample.g_row))
    counts = pd.concat([table.variable_type.value_counts().rename('matched'),
                        sample.variable_type.value_counts().rename('selected')], axis=1).fillna(0)
    counts['matched_pct'] = (100 * counts.matched / len(table)).round(3)
    counts['selected_pct'] = (100 * counts.selected / len(sample)).round(3)
    counts.to_csv(out / 'class_distribution.csv', index_label='variable_type')
    print(f'Done: {len(sample):,} of {len(table):,} paired stars copied to data/', flush=True)


if __name__ == '__main__':
    main()
