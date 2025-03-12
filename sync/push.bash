# echo "push to server 2..."
# rsync -av --exclude-from="sync/push_exclude.txt" ./ c2:~/research/humanoid_symmetry/g1-ppo/

echo "push to cz server 2..."
rsync -av --exclude-from="sync/push_exclude.txt" ./ cz_s2:~/research/humanoid_symmetry/g1-ppo-symm/
