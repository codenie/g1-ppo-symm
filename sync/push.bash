echo "push to server 2..."
rsync -av --exclude-from="sync/push_exclude.txt" ./ c2:~/research/humanoid_symmetry/g1-ppo/
