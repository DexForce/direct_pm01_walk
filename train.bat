rem python scripts\rsl_rl\train.py --task=Template-Direct-Pm01-Walk-Direct-v0 --max_iterations=30000  > log.txt
python scripts\rsl_rl\train.py --task=Template-Direct-Pm01-Walk-Direct-v0 --num_envs=10000 --max_iterations=60000  --load_run=2025-11-02_08-32-08 --num_envs=20 --resume --headless > log.txt
