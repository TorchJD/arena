echo "Installing environment with torchjd at ref main..."
./setup_env.sh main
echo "Done. Computing objective values..."
uv run python -m scripts.compute main "UPGrad()" agg runtime
echo "Done, result saved in results/"

echo "Installing environment with torchjd at ref qp_solver..."
./setup_env.sh qp_solver
echo "Done. Computing objective values for UPGrad()..."
uv run python -m scripts.compute qp_solver "UPGrad()" agg runtime
echo "Done, result saved in results/"

echo "Done. Computing objective values UPGrad(max_iter=100, eps=0)..."
uv run python -m scripts.compute qp_solver "UPGrad(max_iter=100, eps=0)" agg runtime
echo "Done, result saved in results/"
