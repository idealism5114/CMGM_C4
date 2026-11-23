"""Standalone source launcher; works without installing or importing old CMGM."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from cmgm_c4.cli import main

if __name__=='__main__':raise SystemExit(main())
