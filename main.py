import multiprocessing as mp

from src.gui import run_app


if __name__ == "__main__":
    mp.freeze_support()
    run_app()
