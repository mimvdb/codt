# CODTree: Continuous Optimal Decision Trees
Finds optimal classification and regression trees for data sets with continuous features.

## Quickstart
For python bindings: install [`uv`](https://github.com/astral-sh/uv) (faster pip replacement) and run `uv run examples/simple.py`.

For rust CLI: run `cargo run --release -- -f ../datasets/regression/concrete.csv -d 3 -s bfs-balance-small-lb squared-error`

## Profiling
To profile the program using samply follow the following steps.
- Install samply from https://github.com/mstange/samply
- Allow perf events until reboot with `echo '1' | sudo tee /proc/sys/kernel/perf_event_paranoid`
- Compile with `cargo build --profile profiling`
- Record with `samply record ./target/profiling/codt-cli -f ../datasets/regression/concrete.csv -d 3 -s bfs-balance-small-lb squared-error`

## Python bindings
The python bindings use [PyO3](https://github.com/PyO3/pyo3) and [maturin](https://github.com/PyO3/maturin).

The quickstart executes an example in release mode. Alternatively, run `uv run --config-setting 'build-args=--profile=dev' examples/simple.py` to build in debug mode for faster build times.

## Multithreading
You can use the `RAYON_NUM_THREADS` environment variable to control the maximum number of threads used.
