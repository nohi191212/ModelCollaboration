import os
"""Resolve only completed strict students or explicitly named formal controls."""
import json
from pathlib import Path

ROOT=Path(os.environ.get('ICASSP_DATA_ROOT', str(Path(__file__).resolve().parents[2])))

def student_artifacts(size,control=None):
    if control is None:
        assert size in ['1M','2M','4M','8M']
        source=ROOT/'outputs/mini_siglip_strict_20260910_300ep'/size
        done=json.loads((source/'completion.json').read_text())
        assert done['status']=='complete' and done['epochs']==300
        checkpoint=source/'epoch_300.pt';feature_key=size
    else:
        assert control and control not in ['.','..'] and '/' not in control and '\\' not in control
        source=ROOT/'outputs/router_exploration_25pct_20260910/distillation_controls/students'/control
        done=json.loads((source/'completion.json').read_text())
        assert done['status']=='complete' and done['smoke'] is False
        checkpoint=source/'final.pt';feature_key='control_'+control
    assert checkpoint.is_file() and (source/'student_tokenizer.json').is_file()
    config=json.loads((source/'config.json').read_text())
    return {'directory':source,'checkpoint':checkpoint,'feature_key':feature_key,'parameters':config['parameters']}
