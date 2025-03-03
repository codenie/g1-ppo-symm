echo "pull from server 2..."
rsync -av --include-from="sync/pull_include.txt" --exclude="/*" c2:~/research/humanoid_symmetry/g1-ppo/ ./