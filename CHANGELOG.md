# MADHATdm Changelog

## [2.1] – 2025-07-23

### Added
- Improved logging: uses Python's `logging` module throughout, with log messages for progress, warnings, and errors.
- .gitignore file to exclude logs, cache files, and data outputs from version control.
- `-m/--models` and `-s/--sets` flags via `argparse` for running model–set combinations from the command line interface.
- File verification: checks that all model and set files exist before running analysis, and logs missing files
- Explanatory comments in key functions and logic blocks for maintainability.
- MAX_CONVERGENCE_ITERATIONS global to prevent potential infinite loops.

### Changed
- Replaced manual `open()/split()` loops for reading input files with `np.loadtxt` calls, improving the speed of combined analysis.
- Output files can be tab-separated (default: same as MADHATv2), or space-separated (more human-readable) with a customizable `output_fmt` string.
- Extracted main workflow into a standalone `main()` function to make the code more modular and testable.
- Merged duplicate blocks computing `P_S_bar` and `P_B_bar` into a single reusable function `compute_P_bar()`.
- Functionalized the beta calculation (`compute_beta()`) and other probability routines (e.g., `compute_P_sig()` and `prune_P_bar()`) to be more consistent and easier to debug.
- Removed variables `P_S_bar_length` and `P_B_bar_length`; the code now dynamically tracks those lengths.

### Fixed
- Corrected background summation order: The “floor” operation is now applied to individual model weights before summing rather than summing first and flooring afterward. This aligns the implementation with the procedure described in [Reference arXiv:2401.05327].