echo "pull from server 2..."
# rsync -av --include-from="sync/pull_include.txt" --exclude="/*" c2:~/research/humanoid_symmetry/g1-ppo/ ./
rsync -av c2:~/research/humanoid_symmetry/g1-ppo/legged_gym/logs/ ./legged_gym/logs/