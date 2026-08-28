#!/usr/bin/env python3
"""Regenerate the final PLOS ONE story figures (Fig1-5).

Run from the repository root. Requires `results/` (training + statistics outputs,
see README) and the standard scientific Python stack (numpy, matplotlib, seaborn,
scipy, statsmodels).

Figure generation order matters (later scripts finalize Fig3/4/5):
  1. figs_story_v2.py        -> Fig1, Fig2  (+ interim Fig3/4/5, overwritten below)
  2. figs_story_fix345.py    -> Fig3, Fig4  (final)
  3. fig5_value.py           -> Fig5        (final)
Outputs are written to figures_final/ as PNG+PDF+TIF (300 dpi).
"""
import os, sys, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'src', 'figures')
SCRIPTS = ['figs_story_v2.py', 'figs_story_fix345.py', 'fig5_value.py']
NAMES = {1:'Fig1', 2:'Fig2', 3:'Fig3', 4:'Fig4', 5:'Fig5'}

def main():
    if not os.path.isdir(os.path.join(HERE, 'results')):
        print('[!] results/ not found; run training + statistics first (see README).')
        sys.exit(1)
    os.makedirs(os.path.join(HERE, 'figures_final'), exist_ok=True)
    for script in SCRIPTS:
        p = os.path.join(SRC, script)
        if not os.path.exists(p):
            print(f'[!] missing {p}'); sys.exit(1)
        print(f'Running {script} ...')
        r = subprocess.run([sys.executable, p], cwd=HERE, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-1200:]); print(r.stderr[-1200:]); sys.exit(1)
    print('Figure regeneration complete ->', os.path.join(HERE, 'figures_final'))

if __name__ == '__main__':
    main()
