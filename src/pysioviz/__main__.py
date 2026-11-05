"""Module execution entry point for PysioViz.

Enables running the application via:
    uv run -m pysioviz -p <path_to_session>
    python -m pysioviz -p <path_to_session>
"""

from pysioviz.main import main

if __name__ == '__main__':
    main()
